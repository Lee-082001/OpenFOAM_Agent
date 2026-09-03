from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from openfoam_agent.schemas.engineering import EngineeringPlan
from openfoam_agent.tools.openfoam import OpenFOAMTools
from openfoam_agent.tools.workspace import CaseWorkspace
from openfoam_agent.verification.foam_semantics import (
    BoundaryFieldInterpreter,
    ResolutionStatus,
    parse_boundary_selectors,
    parse_mesh_boundary,
)


_CORE_SYSTEM_FILES = ("system/controlDict", "system/fvSchemes", "system/fvSolution")
_FIELD_DIR = "0/"
# Narrow executable constraint types whose effective field patch type must match
# the mesh patch type. Ordinary patch/wall boundaries intentionally are excluded.
_CONSTRAINT_PATCH_TYPES = frozenset({"empty", "wedge", "symmetry", "symmetryPlane", "cyclic", "cyclicAMI"})


@dataclass
class PreSolveValidationResult:
    valid: bool
    failures: list[str] = field(default_factory=list)
    checked_files: list[str] = field(default_factory=list)
    mesh_patches: list[str] = field(default_factory=list)
    mesh_patch_types: dict[str, str] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    boundary_resolutions: dict[str, dict[str, str]] = field(default_factory=dict)


class PreSolveCompletenessGate:
    """Validate solver-input completeness without making CFD engineering choices.

    The Engineering Agent declares solver-specific required case files. Python only
    verifies that those declarations exist, parse as OpenFOAM dictionaries where
    applicable, and that declared initial fields cover every mesh boundary patch.
    """

    def __init__(self, tools: OpenFOAMTools, workspace: CaseWorkspace) -> None:
        self.tools = tools
        self.workspace = workspace

    def validate(self, plan: EngineeringPlan) -> PreSolveValidationResult:
        return self.validate_required_case_files(plan.required_case_files)

    def validate_required_case_files(
        self,
        required_case_files: list[str],
    ) -> PreSolveValidationResult:
        """Validate an Agent-declared solver-input file set without a full plan.

        This supports short engineering sequences such as write -> dictionary check ->
        pre-solve readiness while keeping the final EngineeringPlan validation intact.
        """

        failures: list[str] = []
        warnings: list[str] = []
        boundary_resolutions: dict[str, dict[str, str]] = {}
        required = list(dict.fromkeys([*_CORE_SYSTEM_FILES, *required_case_files]))
        field_files = [item for item in required_case_files if item.startswith(_FIELD_DIR)]
        if not field_files:
            failures.append(
                "EngineeringPlan.required_case_files must declare the solver-required initial field files under 0/."
            )

        for relative in required:
            path = self.workspace.resolve_case_path(relative)
            if not path.is_file():
                failures.append(f"Required solve input is missing: {relative}")
                continue
            if self._should_dictionary_validate(path):
                result = self.tools.foam_dictionary_validate(path, cwd=self.workspace.case_dir)
                if not result.success:
                    excerpt = "\n".join(part for part in (result.stdout, result.stderr) if part)[-1200:]
                    failures.append(f"foamDictionary rejected required solve input {relative}: {excerpt}")

        boundary_path = self.workspace.resolve_case_path("constant/polyMesh/boundary")
        mesh = None
        mesh_patches: list[str] = []
        mesh_patch_types: dict[str, str] = {}
        if not boundary_path.is_file():
            failures.append("constant/polyMesh/boundary is missing; mesh patch coverage cannot be verified.")
        else:
            boundary_text = boundary_path.read_text(encoding="utf-8", errors="replace")
            mesh = parse_mesh_boundary(boundary_text)
            mesh_patches = mesh.names
            mesh_patch_types = mesh.patch_types
            if not mesh_patches:
                failures.append("No mesh boundary patches could be parsed from constant/polyMesh/boundary.")

        if mesh is not None and mesh.patches:
            interpreter = BoundaryFieldInterpreter()
            for relative in field_files:
                path = self.workspace.resolve_case_path(relative)
                if not path.is_file():
                    continue
                text = path.read_text(encoding="utf-8", errors="replace")
                selectors = parse_boundary_selectors(text)
                resolutions = interpreter.resolve_all(mesh, selectors)
                boundary_resolutions[relative] = {
                    patch_name: resolution.match_kind.value
                    for patch_name, resolution in resolutions.items()
                }

                missing = sorted(
                    patch_name
                    for patch_name, resolution in resolutions.items()
                    if resolution.status == ResolutionStatus.MISSING
                )
                if missing:
                    failures.append(
                        f"Boundary coverage mismatch in {relative}; missing patchField entries: {missing} (no effective OpenFOAM selector matched)"
                    )

                indeterminate = [
                    resolution
                    for resolution in resolutions.values()
                    if resolution.status == ResolutionStatus.INDETERMINATE
                ]
                for resolution in indeterminate:
                    warnings.append(
                        f"Boundary coverage is indeterminate in {relative} for patch {resolution.patch.name}: "
                        f"{resolution.reason}. Python did not prove this patch missing."
                    )

                for patch_name, resolution in sorted(resolutions.items()):
                    if resolution.status != ResolutionStatus.RESOLVED:
                        continue
                    mesh_type = resolution.patch.patch_type
                    field_type = resolution.effective_field_type
                    if not field_type:
                        continue
                    if (
                        mesh_type in _CONSTRAINT_PATCH_TYPES
                        or field_type in _CONSTRAINT_PATCH_TYPES
                    ) and mesh_type != field_type:
                        via = resolution.match_kind.value
                        selector = resolution.selector.key.raw if resolution.selector is not None else "<OpenFOAM auto rule>"
                        failures.append(
                            "Boundary constraint-type mismatch in "
                            f"{relative} for patch {patch_name}: mesh={mesh_type}, field={field_type} "
                            f"(resolved via {via} {selector}). "
                            "Constraint patches such as empty/wedge/symmetry/cyclic must match before foamRun."
                        )
                if "internalField" not in text:
                    failures.append(f"Required initial field {relative} does not declare internalField.")
                if "dimensions" not in text:
                    failures.append(f"Required initial field {relative} does not declare dimensions.")

        return PreSolveValidationResult(
            valid=not failures,
            failures=failures,
            checked_files=required,
            mesh_patches=mesh_patches,
            mesh_patch_types=mesh_patch_types,
            warnings=warnings,
            boundary_resolutions=boundary_resolutions,
        )

    @staticmethod
    def _should_dictionary_validate(path: Path) -> bool:
        return path.suffix.lower() not in {".stl", ".obj", ".off", ".vtk", ".csv", ".dat", ".emesh"}
