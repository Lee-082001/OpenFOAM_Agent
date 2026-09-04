from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass, field
from pathlib import Path

from openfoam_agent.schemas.common import ToolResult
from openfoam_agent.schemas.engineering import EngineeringPlan, MeshEvidence
from openfoam_agent.schemas.intake import CFDIntakeSpec
from openfoam_agent.tools.openfoam import OpenFOAMTools
from openfoam_agent.tools.workspace import CaseWorkspace, WorkspaceSafetyError
from openfoam_agent.verification.foam_semantics.parser import parse_named_dictionary_assignments


_SOLVER_ENTRY = re.compile(r"(?m)^\s*solver\s+(?P<solver>[A-Za-z][A-Za-z0-9_]*)\s*;")
_CELL_COUNT = re.compile(r"^\s*cells:\s*(\d+)\s*$", re.MULTILINE | re.IGNORECASE)
_NON_ORTHO = re.compile(r"Mesh non-orthogonality Max:\s*([-+0-9.eE]+)", re.IGNORECASE)
_SKEW = re.compile(r"Max skewness\s*=\s*([-+0-9.eE]+)", re.IGNORECASE)
_NEGATIVE = (
    re.compile(r"cells with negative volume\s*:\s*(\d+)", re.IGNORECASE),
    re.compile(r"negative volume cells\s*:\s*(\d+)", re.IGNORECASE),
)
_MESH_OK = re.compile(r"^\s*Mesh OK\.\s*$", re.MULTILINE)
_DATA_EXTENSIONS = {".stl", ".obj", ".off", ".vtk", ".csv", ".dat", ".eMesh"}
_NUMBER_TOKEN = re.compile(
    r"(?<![A-Za-z0-9_.])[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?(?![A-Za-z0-9_.])"
)
_BLOCK_COMMENT = re.compile(r"/\*.*?\*/", re.DOTALL)
_LINE_COMMENT = re.compile(r"//[^\n\r]*")


@dataclass
class SafetyCheckResult:
    valid: bool
    failures: list[str] = field(default_factory=list)
    tool_results: list[ToolResult] = field(default_factory=list)


class DeterministicSafetyGate:
    """Validate only safety, provenance, integrity and native-tool evidence.

    There are deliberately no rules mapping physics/motion types to a solver,
    mesh method, BC keyword, dynamic-mesh implementation or numerical scheme.
    """

    def __init__(self, tools: OpenFOAMTools, workspace: CaseWorkspace):
        self.tools = tools
        self.workspace = workspace

    def validate_plan(self, plan: EngineeringPlan, intake: CFDIntakeSpec) -> SafetyCheckResult:
        failures = list(self.workspace.validate_all_content())
        if plan.confirmed_intake_sha256 != intake.digest():
            failures.append(
                "Engineering plan is not bound to the exact confirmed intake digest."
            )

        expected_fact_ids = {
            fact.id for fact in intake.facts if fact.category != "context"
        }
        actual_fact_ids = set(plan.confirmed_fact_ids)
        if actual_fact_ids != expected_fact_ids:
            missing = sorted(expected_fact_ids - actual_fact_ids)
            extra = sorted(actual_fact_ids - expected_fact_ids)
            failures.append(
                f"Engineering plan fact provenance mismatch; missing={missing}, extra={extra}."
            )

        binding_ids = {item.fact_id for item in plan.confirmed_fact_bindings}
        if binding_ids != expected_fact_ids:
            missing = sorted(expected_fact_ids - binding_ids)
            extra = sorted(binding_ids - expected_fact_ids)
            failures.append(
                f"Engineering plan fact implementation binding mismatch; missing={missing}, extra={extra}."
            )
        for binding in plan.confirmed_fact_bindings:
            fact = intake.fact(binding.fact_id)
            for relative in binding.case_files:
                try:
                    bound_path = self.workspace.resolve_case_path(relative)
                except WorkspaceSafetyError as exc:
                    failures.append(f"Invalid implementation binding case:{relative}: {exc}")
                    continue
                if not bound_path.is_file():
                    failures.append(
                        f"Confirmed fact binding {binding.fact_id} references missing case file {relative}."
                    )
            for assertion in binding.case_assertions:
                try:
                    path = self.workspace.resolve_case_path(assertion.path, must_exist=True)
                    content = path.read_text(encoding="utf-8", errors="replace")
                except (WorkspaceSafetyError, FileNotFoundError) as exc:
                    failures.append(
                        f"Semantic assertion for {binding.fact_id} cannot read {assertion.path}: {exc}"
                    )
                    continue
                if assertion.entry_path:
                    observed = _foam_dictionary_entry_value(content, assertion.entry_path)
                    if observed is None:
                        failures.append(
                            f"Semantic assertion for {binding.fact_id} cannot resolve entry {assertion.entry_path!r} in {assertion.path}."
                        )
                        continue
                    if not assertion.expected_value:
                        failures.append(
                            f"Semantic assertion for {binding.fact_id} has no expected value for entry {assertion.entry_path!r}."
                        )
                        continue
                    if _normalize_foam_value(observed) != _normalize_foam_value(assertion.expected_value):
                        failures.append(
                            f"Semantic assertion for {binding.fact_id} expected {assertion.entry_path}={assertion.expected_value!r} "
                            f"in {assertion.path}, observed {observed!r}."
                        )
                    continue
                if assertion.anchor:
                    if not _contains_semantic_snippet(content, assertion.anchor):
                        failures.append(
                            f"Semantic assertion for {binding.fact_id} anchor is not present in {assertion.path}: {assertion.anchor!r}."
                        )
                    continue
                if not assertion.contains:
                    failures.append(
                        f"Semantic assertion for {binding.fact_id} has no artifact locator to verify."
                    )
                    continue
                for snippet in assertion.contains:
                    if not _contains_semantic_snippet(content, snippet):
                        failures.append(
                            f"Semantic assertion for {binding.fact_id} is not present in {assertion.path}: {snippet!r}."
                        )

            if binding.numeric_relation is not None:
                if fact is None:
                    failures.append(
                        f"Numeric semantic assertion references unknown confirmed fact {binding.fact_id}."
                    )
                else:
                    failures.extend(
                        self._validate_numeric_relation(binding.fact_id, fact.value, binding.numeric_relation)
                    )

        # Selected high-impact invariants must carry machine-checkable implementation
        # evidence.  Python does not decide the CFD implementation; the Agent chooses
        # the snippets/relation and Python verifies them against the current case.
        if intake.semantic_contract_version == "2":
            binding_by_id = {item.fact_id: item for item in plan.confirmed_fact_bindings}
            for fact in intake.facts:
                if fact.category == "context":
                    continue
                binding = binding_by_id.get(fact.id)
                if binding is None:
                    continue
                if fact.category in {"classification", "temporal"} and not binding.case_assertions:
                    failures.append(
                        f"Confirmed {fact.category} fact {fact.id} requires at least one case semantic assertion."
                    )
                if fact.source == "user" and fact.category in {"physics", "scale", "property"}:
                    numeric_targets = _finite_numbers(fact.value)
                    if len(numeric_targets) == 1 and binding.numeric_relation is None:
                        failures.append(
                            f"Numeric confirmed fact {fact.id} requires a machine-checkable numeric relation assertion."
                        )

        detected = self.tools.detected_foundation_version()
        if detected and plan.openfoam_version != detected:
            failures.append(
                f"Engineering plan targets OpenFOAM {plan.openfoam_version}, "
                f"but runtime reports {detected}."
            )

        control_path = self.workspace.resolve_case_path("system/controlDict")
        execution = plan.execution
        driver = execution.driver if execution is not None else "foamRun"
        runner = getattr(self.tools, "runner", None)
        if detected and runner is not None:
            driver_status = runner.executable_status(driver)
            if not bool(driver_status.get("available")):
                failures.append(
                    f"Selected OpenFOAM execution driver {driver!r} is not available in the trusted sourced installation."
                )

        if not control_path.is_file():
            failures.append(f"system/controlDict is required for bounded {driver} execution.")
        else:
            control = control_path.read_text(encoding="utf-8", errors="replace")
            if execution is None or driver == "foamRun":
                expected_solver = plan.solver if execution is None else execution.solver_module
                match = _SOLVER_ENTRY.search(control)
                if match is None:
                    failures.append(
                        "system/controlDict must declare a solver entry so the approved "
                        "single-region execution can be checked against the runtime case."
                    )
                elif expected_solver is not None and match.group("solver") != expected_solver:
                    failures.append("system/controlDict solver disagrees with the EngineeringPlan execution spec.")
            elif driver == "foamMultiRun":
                actual, complete = parse_named_dictionary_assignments(control, "regionSolvers")
                expected = {item.region: item.solver_module for item in execution.regions}
                if not actual:
                    failures.append("system/controlDict must declare regionSolvers for foamMultiRun execution.")
                elif not complete:
                    failures.append(
                        "system/controlDict regionSolvers contains dynamic/indeterminate entries; "
                        "deterministic multi-region execution semantics could not be proven."
                    )
                elif actual != expected:
                    failures.append(
                        "system/controlDict regionSolvers disagrees with the EngineeringPlan execution spec."
                    )

        return SafetyCheckResult(valid=not failures, failures=failures)

    def _validate_numeric_relation(self, fact_id: str, fact_value: str, relation) -> list[str]:
        failures: list[str] = []
        if not relation.numerator:
            failures.append(
                f"Numeric semantic assertion for {fact_id} requires at least one numerator term."
            )
            return failures
        if not math.isfinite(relation.relative_tolerance) or not (
            0.0 < relation.relative_tolerance <= 0.05
        ):
            failures.append(
                f"Numeric semantic assertion for {fact_id} has an invalid relative tolerance."
            )
            return failures
        targets = _finite_numbers(fact_value)
        if len(targets) != 1:
            failures.append(
                f"Numeric semantic assertion for {fact_id} requires exactly one numeric target in the confirmed fact value."
            )
            return failures
        target = targets[0]
        numerator = 1.0
        denominator = 1.0
        for side, terms in (("numerator", relation.numerator), ("denominator", relation.denominator)):
            for term in terms:
                try:
                    path = self.workspace.resolve_case_path(term.path, must_exist=True)
                    content = path.read_text(encoding="utf-8", errors="replace")
                except (WorkspaceSafetyError, FileNotFoundError) as exc:
                    failures.append(
                        f"Numeric semantic evidence for {fact_id} cannot read {term.path}: {exc}"
                    )
                    continue
                token_value = _numeric_value_from_term(content, term)
                if token_value is None:
                    failures.append(
                        f"Numeric semantic evidence for {fact_id} cannot resolve a scalar from {term.path}."
                    )
                    continue
                if not math.isfinite(term.multiplier) or abs(term.multiplier) > 1e9:
                    failures.append(
                        f"Numeric semantic evidence for {fact_id} has an invalid multiplier."
                    )
                    continue
                effective_value = token_value * term.multiplier
                if side == "numerator":
                    numerator *= effective_value
                else:
                    denominator *= effective_value
        if denominator == 0.0:
            failures.append(f"Numeric semantic assertion for {fact_id} divides by zero.")
            return failures
        computed = numerator / denominator
        if not math.isclose(
            computed,
            target,
            rel_tol=relation.relative_tolerance,
            abs_tol=max(1e-12, abs(target) * relation.relative_tolerance),
        ):
            failures.append(
                f"Numeric semantic assertion for {fact_id} recomputes to {computed}, "
                f"not the confirmed target {target}."
            )
        return failures

    def validate_native_inputs(self) -> SafetyCheckResult:
        failures = list(self.workspace.validate_all_content())
        tool_results: list[ToolResult] = []
        if failures:
            return SafetyCheckResult(False, failures, tool_results)

        for relative in self.workspace.list_authored():
            path = self.workspace.resolve_case_path(relative, must_exist=True)
            if path.suffix in _DATA_EXTENSIONS:
                continue
            result = self.tools.foam_dictionary_validate(path, cwd=self.workspace.case_dir)
            tool_results.append(result)
            if not result.success:
                excerpt = _combined_output(result)[-1200:]
                failures.append(f"foamDictionary rejected {relative}: {excerpt}")
        return SafetyCheckResult(not failures, failures, tool_results)

    def verify_seal(self, plan: EngineeringPlan, seal) -> None:
        self.workspace.verify_seal(seal, plan)


def parse_check_mesh_evidence(result: ToolResult) -> MeshEvidence:
    text = _combined_output(result)
    negative: int | None = None
    for pattern in _NEGATIVE:
        matched = pattern.search(text)
        if matched:
            negative = int(matched.group(1))
            break
    cells = _int_match(_CELL_COUNT, text)
    non_ortho = _float_match(_NON_ORTHO, text)
    skew = _float_match(_SKEW, text)
    warnings: list[str] = []
    if cells is None:
        warnings.append("checkMesh output did not expose a cell count.")
    if non_ortho is None:
        warnings.append("checkMesh output did not expose maximum non-orthogonality.")
    if skew is None:
        warnings.append("checkMesh output did not expose maximum skewness.")
    return MeshEvidence(
        command_succeeded=result.success,
        mesh_ok=bool(_MESH_OK.search(text)),
        cell_count=cells,
        max_non_orthogonality=non_ortho,
        max_skewness=skew,
        negative_volume_cells=negative,
        raw_log_sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
        warnings=warnings,
    )


def _combined_output(result: ToolResult) -> str:
    return "\n".join(part for part in (result.stdout, result.stderr) if part)


def _numeric_tokens(text: str) -> list[tuple[str, float]]:
    result: list[tuple[str, float]] = []
    for match in _NUMBER_TOKEN.finditer(text):
        token = match.group(0)
        value = float(token)
        if math.isfinite(value):
            result.append((token, value))
    return result


def _finite_numbers(text: str) -> list[float]:
    return [value for _, value in _numeric_tokens(text)]


def _active_foam_content(content: str) -> str:
    return _LINE_COMMENT.sub("", _BLOCK_COMMENT.sub("", content))


def _normalize_foam_value(value: str) -> str:
    text = " ".join(str(value or "").strip().rstrip(";").split())
    return text


def _foam_dictionary_entries(content: str) -> dict[str, str]:
    """Extract simple leaf entry paths from ordinary OpenFOAM dictionaries.

    This is intentionally a small structural reader, not a CFD interpreter. It is
    sufficient for deterministic serializer output and common 0/constant/system
    dictionaries. List-heavy raw geometry can use compact anchors instead.
    """

    active = _active_foam_content(content)
    entries: dict[str, str] = {}
    stack: list[str] = []
    pending: str | None = None
    statement = ""
    for raw_line in active.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line == "{":
            if pending:
                stack.append(pending)
                pending = None
            continue
        if line.startswith("}"):
            if stack:
                stack.pop()
            pending = None
            statement = ""
            continue
        if line.endswith("{"):
            key = line[:-1].strip()
            if key:
                stack.append(key)
            pending = None
            continue
        if ";" not in line:
            # A bare word followed by ``{`` on the next line is a dictionary block.
            if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_:+-]*", line):
                pending = line
                statement = ""
                continue
            statement = (statement + " " + line).strip()
            continue
        statement = (statement + " " + line).strip()
        while ";" in statement:
            chunk, statement = statement.split(";", 1)
            chunk = chunk.strip()
            if not chunk:
                continue
            parts = chunk.split(None, 1)
            if len(parts) != 2:
                continue
            key, value = parts
            if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_:+-]*", key):
                continue
            path = ".".join([*stack, key])
            entries[path] = value.strip()
        pending = None
    return entries


def _foam_dictionary_entry_value(content: str, entry_path: str) -> str | None:
    return _foam_dictionary_entries(content).get(entry_path)


def _anchor_window(content: str, anchor: str, occurrence: int) -> str | None:
    active = _active_foam_content(content)
    normalized_anchor = " ".join(anchor.split())
    if not normalized_anchor:
        return None
    matches: list[str] = []
    for line in active.splitlines():
        normalized_line = " ".join(line.split())
        if normalized_anchor in normalized_line:
            matches.append(normalized_line)
    if occurrence < len(matches):
        return matches[occurrence]
    # Fallback for anchors spanning line breaks. Keep only a compact normalized window.
    normalized = " ".join(active.split())
    starts: list[int] = []
    start = 0
    while True:
        idx = normalized.find(normalized_anchor, start)
        if idx < 0:
            break
        starts.append(idx)
        start = idx + max(1, len(normalized_anchor))
    if occurrence >= len(starts):
        return None
    idx = starts[occurrence]
    return normalized[max(0, idx - 80): idx + len(normalized_anchor) + 80]


def _numeric_value_from_term(content: str, term) -> float | None:
    source: str | None = None
    if term.entry_path:
        source = _foam_dictionary_entry_value(content, term.entry_path)
    elif term.anchor:
        source = _anchor_window(content, term.anchor, term.occurrence)
    elif term.excerpt:
        # v2.15 compatibility path.
        if not _contains_semantic_snippet(content, term.excerpt):
            return None
        if term.value_token:
            try:
                value = float(term.value_token)
            except ValueError:
                return None
            return value if math.isfinite(value) else None
        source = term.excerpt
    if source is None:
        return None
    numbers = _numeric_tokens(source)
    if term.number_index >= len(numbers):
        return None
    return numbers[term.number_index][1]


def _contains_semantic_snippet(content: str, snippet: str) -> bool:
    """Whitespace-insensitive containment for Agent-carried case evidence.

    OpenFOAM dictionary serialization may change indentation/newlines without
    changing a value.  Treat whitespace as formatting so semantic assertions do
    not fail merely because Python rendered the same dictionary differently.
    """

    active_content = _active_foam_content(content)
    normalized_content = " ".join(active_content.split())
    normalized_snippet = " ".join(snippet.split())
    return bool(normalized_snippet) and normalized_snippet in normalized_content


def _int_match(pattern: re.Pattern[str], text: str) -> int | None:
    matched = pattern.search(text)
    return int(matched.group(1)) if matched else None


def _float_match(pattern: re.Pattern[str], text: str) -> float | None:
    matched = pattern.search(text)
    return float(matched.group(1)) if matched else None
