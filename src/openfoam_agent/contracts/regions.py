from __future__ import annotations
from pathlib import PurePosixPath
from .models import RegionCaseLayout


def region_layouts(plan=None, required_files: list[str] | None = None) -> list[RegionCaseLayout]:
    required = list(required_files if required_files is not None else (plan.required_case_files if plan else []))
    inferred = set()
    for text in required:
        parts = PurePosixPath(text).parts
        if len(parts) >= 3 and parts[0] == "0":
            inferred.add(parts[1])
        if len(parts) == 3 and parts[0] == "system" and parts[2] in {"fvSchemes", "fvSolution"}:
            inferred.add(parts[1])
    assignments = {x.region: x.solver_module for x in (plan.execution.regions if plan and plan.execution else [])}
    if plan and plan.region_layouts:
        layouts = plan.region_layouts
        names = {x.region for x in layouts}
        if len(names) != len(layouts):
            raise ValueError("Duplicate region layouts.")
        if assignments and names != set(assignments):
            raise ValueError("Region layout and execution assignments do not exactly match.")
        if inferred - names:
            raise ValueError("Required files reference undeclared regions.")
        for layout in layouts:
            if assignments and layout.solver_module != assignments.get(layout.region):
                raise ValueError("Region layout solver disagrees with execution assignment.")
        return layouts
    names = sorted(set(assignments) | inferred) or [""]
    if inferred and any(len(PurePosixPath(p).parts) == 2 and p.startswith("0/") for p in required):
        raise ValueError("Mixed root and named-region initial fields require explicit region layouts.")
    return [RegionCaseLayout(region=name, solver_module=assignments.get(name),
            required_fields=[PurePosixPath(p).name for p in required
                             if str(PurePosixPath(p).parent) == (f"0/{name}" if name else "0")]) for name in names]


def region_mesh_digest(workspace, region: str) -> str:
    from .mesh_dependencies import MeshDependencyGraph
    return MeshDependencyGraph(workspace).digest(region)


def validate_design(plan, intake) -> list[str]:
    failures = []
    ids = {x.id for x in intake.facts if x.category != "context"}
    if set(plan.confirmed_fact_ids) != ids:
        failures.append("Design does not exactly preserve active confirmed fact IDs.")
    if {x.fact_id for x in plan.confirmed_fact_bindings} != ids:
        failures.append("Design bindings do not exactly cover confirmed facts.")
    required = set(plan.required_case_files)
    for binding in plan.confirmed_fact_bindings:
        refs = set(binding.case_files) | {x.path for x in binding.case_assertions}
        if binding.numeric_relation is not None:
            refs |= {x.path for x in [*binding.numeric_relation.numerator, *binding.numeric_relation.denominator]}
        if refs - required:
            failures.append(f"Design binding {binding.fact_id} references undeclared files: {sorted(refs - required)}")
    try:
        layouts = region_layouts(plan)
        for layout in layouts:
            if layout.region:
                for name in ("fvSchemes", "fvSolution"):
                    if f"{layout.system_dir}/{name}" not in required:
                        failures.append(f"Design lacks {layout.system_dir}/{name}.")
                if not any(p.startswith(layout.field_dir + "/") for p in required):
                    failures.append(f"Design lacks initial fields for region {layout.region}.")
        names = {x.region for x in layouts}
        for interface in plan.interfaces:
            if interface.region == interface.neighbour_region or {interface.region, interface.neighbour_region} - names:
                failures.append("Design interface references invalid region topology.")
    except ValueError as exc:
        failures.append(str(exc))
    return failures
