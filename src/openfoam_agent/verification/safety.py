from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path

from openfoam_agent.schemas.common import ToolResult
from openfoam_agent.schemas.engineering import EngineeringPlan, MeshEvidence
from openfoam_agent.schemas.intake import CFDIntakeSpec
from openfoam_agent.tools.openfoam import OpenFOAMTools
from openfoam_agent.tools.workspace import CaseWorkspace, WorkspaceSafetyError


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

        detected = self.tools.detected_foundation_version()
        if detected and plan.openfoam_version != detected:
            failures.append(
                f"Engineering plan targets OpenFOAM {plan.openfoam_version}, "
                f"but runtime reports {detected}."
            )

        control_path = self.workspace.resolve_case_path("system/controlDict")
        if not control_path.is_file():
            failures.append("system/controlDict is required for bounded foamRun execution.")
        else:
            control = control_path.read_text(encoding="utf-8", errors="replace")
            match = _SOLVER_ENTRY.search(control)
            if match is None:
                failures.append(
                    "system/controlDict must declare a solver entry so the approved "
                    "EngineeringPlan can be checked against the runtime case."
                )
            elif match.group("solver") != plan.solver:
                failures.append(
                    "system/controlDict solver disagrees with the EngineeringPlan."
                )

        return SafetyCheckResult(valid=not failures, failures=failures)

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


def _int_match(pattern: re.Pattern[str], text: str) -> int | None:
    matched = pattern.search(text)
    return int(matched.group(1)) if matched else None


def _float_match(pattern: re.Pattern[str], text: str) -> float | None:
    matched = pattern.search(text)
    return float(matched.group(1)) if matched else None
