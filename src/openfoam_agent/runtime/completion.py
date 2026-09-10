"""Request termination and output proof, separate from physical goal validation."""
from __future__ import annotations
import gzip
import hashlib
import math
import re
from pathlib import Path
from openfoam_agent.contracts.models import (
    CompletionContract,
    ExecutionBoundContract,
    ResidualThreshold,
    ResultAcceptanceContract,
    RuntimeContract,
)
from openfoam_agent.contracts.regions import region_layouts
from openfoam_agent.verification.foam_semantics import parse_top_level_assignments


def _control_entries(workspace):
    path = workspace.resolve_case_path("system/controlDict", must_exist=True)
    entries, complete = parse_top_level_assignments(path.read_text(encoding="utf-8"))
    return entries, complete


def _entry_text(entries, key, default=None):
    raw = entries.get(key, default)
    if isinstance(raw, list):
        raw = " ".join(str(x) for x in raw)
    if raw is None:
        return None
    return str(raw).strip().rstrip(";").strip()


def _entry_number(entries, key, default=None):
    raw = _entry_text(entries, key, default)
    if raw is None:
        return None
    try:
        value = float(raw)
    except ValueError:
        return None
    return value if math.isfinite(value) else None


def _required_result_fields(plan):
    return list(dict.fromkeys(p[2:] for p in plan.required_case_files if p.startswith("0/") and len(p) > 2))


def _engineering_default_controls(plan):
    """Extract only controller-safe runtime facts from Agent-owned defaults.

    The compiler never invents CFD targets. It recognizes values the Agent already
    selected, while leaving prose-only/non-deterministic criteria advisory.
    """
    max_iterations = None
    residuals: dict[str, float] = {}
    advisory: list[str] = []
    matched = False
    for item in plan.engineering_defaults:
        parameter = item.parameter.casefold()
        value = item.value.strip()
        if not any(token in parameter for token in ("iteration", "convergence", "residual", "steady")):
            continue
        matched = True
        # A phrase such as "Maximum 2000 iterations" is an Agent-selected execution
        # upper bound, not a Python-selected engineering value.
        m = re.search(r"(?i)\b(?:maximum|max(?:imum)?\s*)?(\d{1,9})\s+(?:outer\s+)?iterations?\b", value)
        if m:
            candidate = int(m.group(1))
            if candidate > 0:
                max_iterations = candidate if max_iterations is None else min(max_iterations, candidate)

        # Parse explicit scientific-notation residual pairs after the first residual
        # marker, e.g. "target residuals U 1e-7 and p 1e-6". Values not expressed in
        # a machine-safe form remain advisory rather than being guessed.
        lower = value.casefold()
        pos = lower.find("residual")
        if pos >= 0:
            segment = value[pos:]
            for field, raw in re.findall(
                r"\b([A-Za-z][A-Za-z0-9_.-]*)\s*(?:[:=<]+\s*)?((?:\d+(?:\.\d*)?|\.\d+)[eE][-+]?\d+)\b",
                segment,
            ):
                try:
                    threshold = float(raw)
                except ValueError:
                    continue
                if math.isfinite(threshold) and threshold > 0:
                    residuals[field] = threshold
        advisory.append(f"{item.parameter}: {value}")
    return max_iterations, residuals, advisory, matched


def compile_runtime_contract(plan, workspace, *, wall_seconds=None) -> RuntimeContract:
    """Compile execution bounds and result acceptance from the immutable plan/case.

    This compiler is deterministic. It may formalize values already selected by the
    Agent or literal in controlDict, but it never chooses new CFD engineering values.
    Incomplete result-acceptance criteria are review information, not solve blockers.
    """
    entries, complete = _control_entries(workspace)
    fields = _required_result_fields(plan)
    control_start = _entry_number(entries, "startTime", 0.0) if complete else None
    control_end = _entry_number(entries, "endTime") if complete else None
    start_from = _entry_text(entries, "startFrom", "startTime") if complete else None
    default_max_iterations, default_residuals, advisory_defaults, matched_defaults = _engineering_default_controls(plan)

    explicit = plan.completion
    restart = plan.execution.parallel.restart if plan.execution is not None else None
    warnings: list[str] = []
    if restart is not None:
        from .restart import verify_restart
        verify_restart(workspace, plan)

    if explicit is not None:
        mode = explicit.mode
        start_time = float(restart.time_name) if restart is not None else explicit.start_time
        end_time = explicit.end_time
        if mode == "transient":
            if not complete:
                warnings.append("controlDict termination is not fully literal; explicit completion remains authoritative for result review.")
            else:
                if start_from != "startTime" and restart is None:
                    raise ValueError("Restart completion needs explicit native restart qualification; not verified.")
                for key, expected in (("startTime", start_time), ("endTime", end_time)):
                    actual = _entry_number(entries, key)
                    if expected is not None and actual is not None and not math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-12):
                        raise ValueError(f"Completion contract disagrees with literal controlDict {key}.")
        max_iterations = None
        if mode == "steady":
            if control_start is not None and control_end is not None and control_end > control_start:
                span = control_end - control_start
                if math.isclose(span, round(span), rel_tol=0, abs_tol=1e-9):
                    max_iterations = int(round(span))
            if max_iterations is None:
                max_iterations = default_max_iterations
        bound_source = "plan_completion" if mode == "transient" else ("mixed" if max_iterations is not None else "plan_completion")
        execution_bound = ExecutionBoundContract(
            mode=mode,
            start_time=start_time,
            end_time=end_time if mode == "transient" else None,
            max_iterations=max_iterations,
            wall_seconds=wall_seconds,
            minimum_steps=explicit.minimum_steps,
            required_result_fields=explicit.required_result_fields or fields,
            source=bound_source,
            warnings=warnings,
        )
        acceptance_complete = (
            bool(explicit.residual_thresholds) if mode == "steady"
            else bool(explicit.custom_evidence) if mode == "custom"
            else True
        )
        acceptance_warnings = []
        if mode == "steady" and not explicit.residual_thresholds:
            acceptance_warnings.append("No explicit steady residual acceptance thresholds were supplied; bounded execution is still authorized and results require review.")
        if mode == "custom" and not explicit.custom_evidence:
            acceptance_warnings.append("Custom result acceptance has no deterministic evaluator; bounded execution is still authorized and results require review.")
        result_acceptance = ResultAcceptanceContract(
            residual_thresholds=list(explicit.residual_thresholds),
            consecutive_samples=explicit.consecutive_samples,
            criteria_complete=acceptance_complete,
            advisory_criteria=([explicit.custom_evidence] if explicit.custom_evidence else []),
            warnings=acceptance_warnings,
            source="plan_completion",
        )
        return RuntimeContract(execution_bound=execution_bound, result_acceptance=result_acceptance)

    mode = plan.temporal_behavior
    start_time = float(restart.time_name) if restart is not None else (control_start if control_start is not None else 0.0)
    end_time = None
    max_iterations = None
    bound_source = "runtime_policy" if wall_seconds is not None else "controlDict"

    if mode == "transient":
        if complete and start_from not in {None, "startTime"}:
            raise ValueError("Restart start time must be supplied by the qualified restart transaction.")
        end_time = control_end
        if end_time is None:
            warnings.append("Literal transient endTime could not be compiled; wall-time protection remains the execution bound and result completion requires review.")
        else:
            bound_source = "controlDict"
    elif mode == "steady":
        if control_start is not None and control_end is not None and control_end > control_start:
            span = control_end - control_start
            if math.isclose(span, round(span), rel_tol=0, abs_tol=1e-9):
                max_iterations = int(round(span))
                bound_source = "controlDict"
        if default_max_iterations is not None:
            if max_iterations is None:
                max_iterations = default_max_iterations
                bound_source = "engineering_defaults"
            elif max_iterations != default_max_iterations:
                warnings.append(
                    f"controlDict iteration span ({max_iterations}) differs from Agent engineering default ({default_max_iterations}); the literal case bound governs execution."
                )
                bound_source = "mixed"
        if max_iterations is None:
            warnings.append("No literal/Agent iteration bound was compiled; the runtime wall-time limit is the hard execution bound.")
    else:
        if default_max_iterations is not None:
            max_iterations = default_max_iterations
            bound_source = "engineering_defaults"
        else:
            warnings.append("Custom execution has no deterministic iteration bound; the runtime wall-time limit is the hard execution bound.")

    execution_bound = ExecutionBoundContract(
        mode=mode,
        start_time=start_time,
        end_time=end_time,
        max_iterations=max_iterations,
        wall_seconds=wall_seconds,
        minimum_steps=1,
        required_result_fields=fields,
        source=bound_source,
        warnings=warnings,
    )

    residual_thresholds = [ResidualThreshold(field=field, value=value) for field, value in default_residuals.items()]
    acceptance_warnings: list[str] = []
    criteria_complete = False
    if mode == "steady":
        # Engineering-default prose may contain additional QoI stability/conservation
        # criteria that this compiler deliberately does not pretend to evaluate.
        # Residuals are a machine-checkable subset, so inferred acceptance remains
        # incomplete and therefore reviewable even when those residuals pass.
        criteria_complete = False
        if residual_thresholds:
            acceptance_warnings.append("A steady residual subset was machine-compiled from Agent defaults; additional convergence/flow criteria remain advisory and require result review.")
        else:
            acceptance_warnings.append("Steady residual thresholds were not machine-compiled; solve may run within its execution bound and numerical convergence requires result review.")
    elif mode == "transient":
        # End-time attainment is execution completion, not physical/numerical quality.
        acceptance_warnings.append("No independent transient numerical/physical result-acceptance criteria were compiled; result review remains required.")
    else:
        acceptance_warnings.append("Custom result acceptance is not deterministically specified; result review remains required.")

    result_acceptance = ResultAcceptanceContract(
        residual_thresholds=residual_thresholds,
        consecutive_samples=3,
        criteria_complete=criteria_complete,
        advisory_criteria=advisory_defaults if matched_defaults else [],
        warnings=acceptance_warnings,
        source="engineering_defaults" if matched_defaults else "none",
    )
    return RuntimeContract(execution_bound=execution_bound, result_acceptance=result_acceptance)


def completion_contract(plan, workspace):
    """Backward-compatible completion view compiled from the new runtime contract.

    New runtime execution must use ``compile_runtime_contract`` so bounded execution
    and result acceptance remain separate. This adapter remains for restart/tests and
    older callers that consume CompletionContract directly.
    """
    runtime = compile_runtime_contract(plan, workspace)
    bound, acceptance = runtime.execution_bound, runtime.result_acceptance
    return CompletionContract(
        mode=bound.mode,
        resolution_state=("resolved" if (bound.mode == "transient" and bound.end_time is not None) or (bound.mode == "steady" and bool(acceptance.residual_thresholds)) else "intent"),
        start_time=bound.start_time,
        end_time=bound.end_time,
        minimum_steps=bound.minimum_steps,
        residual_thresholds=list(acceptance.residual_thresholds),
        consecutive_samples=acceptance.consecutive_samples,
        required_result_fields=list(bound.required_result_fields),
    )


def snapshot_result_outputs(workspace, contract):
    """Capture pre-run file identities; an old field is not new execution evidence."""
    records = {}
    result_fields = (
        contract.execution_bound.required_result_fields
        if isinstance(contract, RuntimeContract)
        else contract.required_result_fields
    )
    for directory in workspace.case_dir.iterdir():
        if directory.is_symlink() or not directory.is_dir(): continue
        try: value = float(directory.name)
        except ValueError: continue
        if not math.isfinite(value): continue
        for field in result_fields:
            field = field.removeprefix("0/")
            for suffix in ("", ".gz"):
                path = workspace.resolve_result_path(f"{directory.name}/{field}{suffix}")
                if path.is_file():
                    st = path.stat()
                    records[path.relative_to(workspace.case_dir).as_posix()] = (st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns)
        if len(records) > 100000: raise ValueError("Too many result fields for bounded freshness tracking.")
    return records


def verify_result_outputs(workspace, plan, contract, last_time, *, before=None):
    """Inspect declared ASCII/gzip-ASCII fields with bounded reads.

    Binary OpenFOAM fields need a native reader and are deliberately NOT declared
    verified by this Python text check. Header/finite payload is not solution accuracy.
    """
    failures, records = [], []
    declared_fields = (
        contract.execution_bound.required_result_fields
        if isinstance(contract, RuntimeContract)
        else contract.required_result_fields
    )
    fields = declared_fields or [p[2:] for p in plan.required_case_files if p.startswith("0/")]
    if not fields:
        return False, ["No required result fields are declared; output completeness is unverified."], records
    if last_time is None:
        return False, ["No final time exists for result validation."], records
    candidates = []
    for p in workspace.case_dir.iterdir():
        if p.is_symlink() or not p.is_dir():
            continue
        try:
            value = float(p.name)
        except ValueError:
            continue
        if math.isfinite(value) and math.isclose(value, last_time, rel_tol=1e-8, abs_tol=1e-10):
            candidates.append(p)
    if len(candidates) != 1:
        return False, ["A unique final-time result directory is missing (write the completion time explicitly)."], records
    final = candidates[0]
    for field in fields:
        if field.startswith("0/"):
            field = field[2:]
        relative = f"{final.name}/{field}"
        try:
            path = workspace.resolve_result_path(relative)
            if not path.exists():
                path = workspace.resolve_result_path(relative + ".gz", must_exist=True)
            st = path.stat()
            identity = (st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns)
            if before is not None and before.get(path.relative_to(workspace.case_dir).as_posix()) == identity:
                raise ValueError("Result field is unchanged from before this run; stale output is not execution evidence.")
            if st.st_size > 256 * 1024 * 1024:
                raise ValueError("Result field exceeds bounded inspection size.")
            digest = hashlib.sha256()
            opener = gzip.open if path.suffix == ".gz" else open
            nonfinite = False; binary = False; seen_internal = False; numeric_payload = False
            expanded = 0
            with opener(path, "rb") as handle:
                while True:
                    raw = handle.readline(65537)
                    if not raw:
                        break
                    expanded += len(raw)
                    if expanded > 256 * 1024 * 1024 or len(raw) > 65536:
                        raise ValueError("Expanded result field/line exceeds bounded inspection size.")
                    digest.update(raw)
                    line = raw.decode("utf-8", errors="strict")
                    binary |= bool(re.search(r"\bformat\s+binary\s*;", line))
                    nonfinite |= bool(re.search(r"(?<![A-Za-z])(?:nan|[-+]?inf(?:inity)?)(?![A-Za-z])", line, re.I))
                    seen_internal |= bool(re.search(r"\binternalField\b", line))
                    if seen_internal and re.search(r"[-+]?\d+(?:\.\d*)?(?:[eE][-+]?\d+)?", line):
                        numeric_payload = True
            if binary:
                raise ValueError("Binary fields require native finite-value validation; not verified by the text reader.")
            if nonfinite or not seen_internal or not numeric_payload:
                raise ValueError("Field has non-finite values or no identifiable internal numeric field.")
            records.append({"path": path.relative_to(workspace.case_dir).as_posix(),
                            "expanded_sha256": digest.hexdigest(), "bytes_inspected": expanded,
                            "scope": "ASCII internal-field presence and lexical finite-value check"})
        except (ValueError, OSError, UnicodeError) as exc:
            failures.append(f"Result field {relative}: {exc}")
    return not failures, failures, records
