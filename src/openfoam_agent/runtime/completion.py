"""Request termination and output proof, separate from physical goal validation."""
from __future__ import annotations
import gzip
import hashlib
import math
import re
from pathlib import Path
from openfoam_agent.contracts.models import CompletionContract
from openfoam_agent.contracts.regions import region_layouts
from openfoam_agent.verification.foam_semantics import parse_top_level_assignments


def completion_contract(plan, workspace):
    if plan.completion is not None:
        contract = plan.completion
        restart = plan.execution.parallel.restart if plan.execution is not None else None
        if restart is not None:
            from .restart import verify_restart
            verify_restart(workspace, plan)
            contract = contract.model_copy(update={"start_time": float(restart.time_name)})
        if contract.mode == "transient":
            path = workspace.resolve_case_path("system/controlDict", must_exist=True)
            entries, complete = parse_top_level_assignments(path.read_text(encoding="utf-8"))
            if not complete:
                raise ValueError("Dynamic completion settings need native verification; not automatically approved.")
            for key, expected in (("startTime", contract.start_time), ("endTime", contract.end_time)):
                raw = entries.get(key, "0" if key == "startTime" else "")
                if isinstance(raw, list): raw = " ".join(raw)
                if not math.isclose(float(str(raw).strip().rstrip(";")), expected, rel_tol=1e-10, abs_tol=1e-12):
                    raise ValueError(f"Completion contract disagrees with literal controlDict {key}.")
            if str(entries.get("startFrom", "startTime")).strip().rstrip(";") != "startTime":
                raise ValueError("Restart completion needs explicit native restart qualification; not verified.")
        return contract
    if plan.temporal_behavior != "transient":
        raise ValueError("A steady/custom run requires an explicit convergence/completion contract.")
    path = workspace.resolve_case_path("system/controlDict", must_exist=True)
    entries, complete = parse_top_level_assignments(path.read_text(encoding="utf-8"))
    if not complete:
        raise ValueError("Dynamic controlDict termination cannot be inferred; supply a completion contract.")
    # This fallback interprets literal execution settings, not the user's physics.
    def number(key, default=None):
        raw = entries.get(key, default)
        if isinstance(raw, list):
            raw = " ".join(raw)
        if raw is None:
            raise ValueError(f"Missing literal {key} in completion settings.")
        return float(str(raw).strip().rstrip(";").strip())
    start_from = str(entries.get("startFrom", "startTime")).strip().rstrip(";")
    if start_from != "startTime":
        raise ValueError("Restart start time must be supplied explicitly in a completion contract.")
    fields = [p[2:] for p in plan.required_case_files if p.startswith("0/")]
    return CompletionContract(mode="transient", start_time=number("startTime", "0"),
                              end_time=number("endTime"), required_result_fields=fields)


def snapshot_result_outputs(workspace, contract):
    """Capture pre-run file identities; an old field is not new execution evidence."""
    records = {}
    for directory in workspace.case_dir.iterdir():
        if directory.is_symlink() or not directory.is_dir(): continue
        try: value = float(directory.name)
        except ValueError: continue
        if not math.isfinite(value): continue
        for field in contract.required_result_fields:
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
    fields = contract.required_result_fields or [p[2:] for p in plan.required_case_files if p.startswith("0/")]
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
