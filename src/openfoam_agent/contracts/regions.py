from __future__ import annotations
from pathlib import PurePosixPath
from .models import RegionCaseLayout
from openfoam_agent.contracts.execution_scopes import execution_scopes


def _required_fields(required: list[str], region: str) -> list[str]:
    parent = f"0/{region}" if region else "0"
    return [
        PurePosixPath(path).name
        for path in required
        if str(PurePosixPath(path).parent) == parent
    ]


def region_layouts(plan=None, required_files: list[str] | None = None) -> list[RegionCaseLayout]:
    """Legacy RegionCaseLayout projection from the canonical execution scopes.

    New controller logic must consume execution_scopes directly. This wrapper exists
    for old validators/extensions and cannot create a second topology authority.
    """
    required = list(required_files if required_files is not None else (plan.required_case_files if plan else []))
    return [
        RegionCaseLayout(
            region=scope.name or "",
            solver_module=scope.solver_module,
            required_fields=_required_fields(required, scope.name or ""),
        )
        for scope in execution_scopes(plan)
    ]


def region_mesh_digest(workspace, region: str) -> str:
    from .mesh_dependencies import MeshDependencyGraph
    return MeshDependencyGraph(workspace).digest(region)


def validate_design(plan, intake) -> list[str]:
    """Design hard checks: identity closure plus truthful optional implementation claims."""
    failures = []
    ids = {item.id for item in intake.facts if item.category != "context"}
    if set(plan.confirmed_fact_ids) != ids:
        failures.append("Design does not exactly preserve active confirmed fact IDs.")
    binding_ids = {item.fact_id for item in plan.confirmed_fact_bindings}
    if binding_ids - ids:
        failures.append("Design contains implementation bindings for unknown confirmed facts.")
    required = set(plan.required_case_files)
    for binding in plan.confirmed_fact_bindings:
        refs = set(binding.case_files) | {item.path for item in binding.case_assertions}
        if binding.numeric_relation is not None:
            refs |= {item.path for item in [*binding.numeric_relation.numerator, *binding.numeric_relation.denominator]}
        if refs - required:
            failures.append(
                f"Design binding {binding.fact_id} references undeclared files: {sorted(refs - required)}"
            )
    return failures
