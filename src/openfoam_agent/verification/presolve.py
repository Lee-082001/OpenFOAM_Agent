from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from openfoam_agent.contracts.regions import region_layouts

from openfoam_agent.schemas.engineering import EngineeringPlan
from openfoam_agent.tools.foam_file import validate_foam_file_header
from openfoam_agent.tools.openfoam import OpenFOAMTools
from openfoam_agent.tools.workspace import CaseWorkspace
from openfoam_agent.verification.foam_semantics import (
    BoundaryFieldInterpreter,
    ResolutionStatus,
    parse_boundary_selectors,
    parse_mesh_boundary,
    parse_top_level_assignments,
)


_CORE_SYSTEM_FILES = ("system/controlDict", "system/fvSchemes", "system/fvSolution")
_FIELD_DIR = "0/"
# Narrow executable constraint types whose effective field patch type must match
# the mesh patch type. Ordinary patch/wall boundaries intentionally are excluded.
_CONSTRAINT_PATCH_TYPES = frozenset({"empty", "wedge", "symmetry", "symmetryPlane", "cyclic", "cyclicAMI"})


@dataclass
class PreSolveValidationResult:
    valid: bool
    regions: dict[str, dict[str, object]] = field(default_factory=dict)
    failures: list[str] = field(default_factory=list)
    checked_files: list[str] = field(default_factory=list)
    mesh_patches: list[str] = field(default_factory=list)
    mesh_patch_types: dict[str, str] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    boundary_resolutions: dict[str, dict[str, str]] = field(default_factory=dict)
    file_header_classes: dict[str, str] = field(default_factory=dict)


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
        try:
            layouts = region_layouts(plan)
        except ValueError as exc:
            return PreSolveValidationResult(valid=False, failures=[str(exc)])
        result = self._validate_layouts(plan.required_case_files, layouts)
        for interface in plan.interfaces:
            left = result.regions.get(interface.region, {})
            right = result.regions.get(interface.neighbour_region, {})
            if interface.patch not in left.get("mesh_patches", []):
                result.failures.append(f"Interface patch missing: {interface.region}/{interface.patch}.")
            if interface.neighbour_patch not in right.get("mesh_patches", []):
                result.failures.append(f"Interface neighbour patch missing: {interface.neighbour_region}/{interface.neighbour_patch}.")
            for field_name in interface.fields:
                for region in (interface.region, interface.neighbour_region):
                    if f"0/{region}/{field_name}" not in result.checked_files:
                        result.failures.append(f"Interface field not declared: 0/{region}/{field_name}.")
            # Membership is deterministic; correct physical coupling is not inferred
            # from matching patch names. Native and balance evidence remain separate.
            result.warnings.append(
                f"Interface {interface.region}/{interface.patch} membership checked; "
                "physical coupling and flux conservation require native/QoI verification."
            )
        result.valid = not result.failures
        return result

    def validate_required_case_files(self, required_case_files: list[str]) -> PreSolveValidationResult:
        try:
            return self._validate_layouts(required_case_files, region_layouts(required_files=required_case_files))
        except ValueError as exc:
            return PreSolveValidationResult(valid=False, failures=[str(exc)])

    def _validate_layouts(self, required_case_files, layouts) -> PreSolveValidationResult:
        result = PreSolveValidationResult(valid=True)
        for layout in layouts:
            partial = self._validate_region(required_case_files, layout)
            result.regions[layout.region] = {
                "valid": partial.valid, "mesh_patches": partial.mesh_patches,
                "mesh_patch_types": partial.mesh_patch_types,
                "checked_files": partial.checked_files,
                "solver_module": layout.solver_module,
            }
            result.failures.extend(partial.failures)
            result.warnings.extend(partial.warnings)
            result.checked_files.extend(p for p in partial.checked_files if p not in result.checked_files)
            prefix = layout.region + "/" if layout.region else ""
            result.mesh_patches.extend(prefix + p for p in partial.mesh_patches)
            result.mesh_patch_types.update({prefix+k: v for k,v in partial.mesh_patch_types.items()})
            result.boundary_resolutions.update(partial.boundary_resolutions)
            result.file_header_classes.update(partial.file_header_classes)
        result.valid = not result.failures
        return result

    def _validate_region(self, required_case_files, layout) -> PreSolveValidationResult:
        failures: list[str] = []
        warnings: list[str] = []
        boundary_resolutions: dict[str, dict[str, str]] = {}
        core = ["system/controlDict", f"{layout.system_dir}/fvSchemes", f"{layout.system_dir}/fvSolution"]
        # Validate shared root controls plus this region, never demand root dummy fv*.
        selected = [p for p in required_case_files
                    if (len(PurePosixPath(p).parts) == 2
                        or str(PurePosixPath(p).parent) in {layout.field_dir, layout.system_dir, layout.constant_dir})]
        required = list(dict.fromkeys([*core, *selected,
                        *(f"{layout.field_dir}/{name}" for name in layout.required_fields)]))
        field_files = [item for item in required if str(PurePosixPath(item).parent) == layout.field_dir]
        if not field_files:
            failures.append(
                "EngineeringPlan.required_case_files must declare the solver-required initial field files under 0/."
            )

        file_header_classes: dict[str, str] = {}
        for relative in required:
            path = self.workspace.resolve_case_path(relative)
            if not path.is_file():
                failures.append(f"Required solve input is missing: {relative}")
                continue
            if self._should_dictionary_validate(path):
                text = path.read_text(encoding="utf-8", errors="replace")
                header = validate_foam_file_header(
                    relative,
                    text,
                    expected_class=("dictionary" if relative.startswith("system/") else None),
                )
                file_header_classes[relative] = header.header.class_name
                failures.extend(header.failures)
                warnings.extend(header.warnings)
                if header.valid:
                    result = self.tools.foam_dictionary_validate(path, cwd=self.workspace.case_dir)
                    if not result.success:
                        excerpt = "\n".join(part for part in (result.stdout, result.stderr) if part)[-1200:]
                        failures.append(f"foamDictionary rejected required solve input {relative}: {excerpt}")

        boundary_relative = f"{layout.mesh_dir}/boundary"
        boundary_path = self.workspace.resolve_case_path(boundary_relative)
        mesh = None
        mesh_patches: list[str] = []
        mesh_patch_types: dict[str, str] = {}
        if not boundary_path.is_file():
            failures.append(f"{boundary_relative} is missing; mesh patch coverage cannot be verified.")
        else:
            boundary_text = boundary_path.read_text(encoding="utf-8", errors="replace")
            mesh = parse_mesh_boundary(boundary_text)
            mesh_patches = mesh.names
            mesh_patch_types = mesh.patch_types
            if not mesh_patches:
                failures.append(f"No mesh boundary patches could be parsed from {boundary_relative}.")

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
                            "Constraint patches such as empty/wedge/symmetry/cyclic must match before the selected OpenFOAM runtime."
                        )
                field_entries, field_projection_complete = parse_top_level_assignments(text)
                for required_key in ("internalField", "dimensions"):
                    if required_key in field_entries:
                        continue
                    if field_projection_complete:
                        failures.append(
                            f"Required initial field {relative} does not declare {required_key}."
                        )
                    else:
                        warnings.append(
                            f"Required initial field {relative} has no literal top-level {required_key}, "
                            "but dynamic OpenFOAM directives/expansions make the effective value "
                            "indeterminate; Python did not prove it missing."
                        )

        return PreSolveValidationResult(
            valid=not failures,
            failures=failures,
            checked_files=required,
            mesh_patches=mesh_patches,
            mesh_patch_types=mesh_patch_types,
            warnings=warnings,
            boundary_resolutions=boundary_resolutions,
            file_header_classes=file_header_classes,
        )

    @staticmethod
    def _should_dictionary_validate(path: Path) -> bool:
        return path.suffix.lower() not in {".stl", ".obj", ".off", ".vtk", ".csv", ".dat", ".emesh"}
