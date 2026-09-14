from __future__ import annotations

from pathlib import PurePosixPath

from openfoam_agent.engineering.native_region_contract import declared_named_regions
from .models import RegionCaseLayout


def _manifest_field_names(required: list[str], region: str) -> list[str]:
    field_dir = f"0/{region}" if region else "0"
    return [
        PurePosixPath(path).name
        for path in required
        if str(PurePosixPath(path).parent) == field_dir
    ]


def region_layouts(plan=None, required_files: list[str] | None = None) -> list[RegionCaseLayout]:
    """Return region layout projection without inventing topology from file paths.

    For a full EngineeringPlan, execution.regions is the named-region source of truth.
    region_layouts may carry additional file-layout metadata but must describe the same
    topology/solver assignments. `required_files` alone can validate only a root case;
    named regions require explicit plan authority.
    """
    required = list(required_files if required_files is not None else (plan.required_case_files if plan else []))

    if plan is None:
        named_paths = [
            path for path in required
            if len(PurePosixPath(path).parts) >= 3
            and PurePosixPath(path).parts[0] in {"0", "system", "constant"}
        ]
        if named_paths:
            raise ValueError(
                "Named-region file layout cannot be inferred from required_case_files alone; an EngineeringPlan execution topology is required."
            )
        return [RegionCaseLayout(region="", required_fields=_manifest_field_names(required, ""))]

    execution = getattr(plan, "execution", None)
    assignments = {
        item.region: item.solver_module
        for item in (execution.regions if execution is not None else [])
    }
    names = declared_named_regions(plan)

    explicit = list(getattr(plan, "region_layouts", None) or [])
    if explicit:
        by_name = {item.region: item for item in explicit}
        if len(by_name) != len(explicit):
            raise ValueError("Duplicate region layouts.")
        if names and set(by_name) != set(names):
            raise ValueError("Region layout projection does not exactly match execution.regions topology.")
        for name, layout in by_name.items():
            expected_solver = assignments.get(name)
            if expected_solver is not None and layout.solver_module != expected_solver:
                raise ValueError("Region layout solver disagrees with execution assignment.")
        layouts = explicit
    elif names:
        layouts = [
            RegionCaseLayout(
                region=name,
                solver_module=assignments.get(name),
                required_fields=_manifest_field_names(required, name),
            )
            for name in names
        ]
    else:
        layouts = [
            RegionCaseLayout(
                region="",
                solver_module=(getattr(execution, "solver_module", None) if execution is not None else None),
                required_fields=_manifest_field_names(required, ""),
            )
        ]

    declared = {layout.region for layout in layouts if layout.region}
    for path in required:
        parts = PurePosixPath(path).parts
        candidate = ""
        if len(parts) >= 3 and parts[0] == "0":
            candidate = parts[1]
        elif len(parts) == 3 and parts[0] in {"system", "constant"}:
            candidate = parts[1]
        if candidate and candidate not in declared:
            raise ValueError(
                f"Required file {path!r} references undeclared region {candidate!r}."
            )
    if declared and any(len(PurePosixPath(path).parts) == 2 and path.startswith("0/") for path in required):
        raise ValueError("Named-region execution cannot mix root initial fields without an explicit root region contract.")
    return layouts


def region_mesh_digest(workspace, region: str) -> str:
    from .mesh_dependencies import MeshDependencyGraph
    return MeshDependencyGraph(workspace).digest(region)


def validate_design(plan, intake) -> list[str]:
    """Design-stage hard checks only.

    The controller verifies immutable intent closure and safe artifact references. It
    does not use fact bindings to claim implementation truth; that is established by
    explicit assertions/native validation after authoring.
    """
    failures: list[str] = []
    ids = {item.id for item in intake.facts if item.category != "context"}
    if set(plan.confirmed_fact_ids) != ids:
        failures.append("Design does not exactly preserve active confirmed fact IDs.")
    if {item.fact_id for item in plan.confirmed_fact_bindings} != ids:
        failures.append("Design bindings do not exactly cover confirmed facts.")

    try:
        region_layouts(plan)
    except ValueError as exc:
        failures.append(str(exc))

    required = set(plan.required_case_files)
    for binding in plan.confirmed_fact_bindings:
        refs = set(binding.case_files) | {item.path for item in binding.case_assertions}
        if binding.numeric_relation is not None:
            refs |= {
                item.path
                for item in [*binding.numeric_relation.numerator, *binding.numeric_relation.denominator]
            }
        if refs - required:
            failures.append(
                f"Design binding {binding.fact_id} references undeclared files: {sorted(refs - required)}"
            )
    return failures
