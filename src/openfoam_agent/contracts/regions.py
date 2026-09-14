from __future__ import annotations

from pathlib import PurePosixPath

from .models import RegionCaseLayout


def _manifest_region_names(required: list[str]) -> set[str]:
    names: set[str] = set()
    for text in required:
        parts = PurePosixPath(text).parts
        if len(parts) >= 3 and parts[0] == "0":
            names.add(parts[1])
        elif len(parts) == 3 and parts[0] in {"system", "constant"}:
            # A three-part system/constant path may be region-scoped. Only treat it
            # as region evidence when the middle component is not a known root
            # directory. This is validation evidence, never topology authority.
            if parts[1] not in {"polyMesh", "triSurface", "geometry"}:
                names.add(parts[1])
    return names


def _manifest_required_fields(required: list[str], region: str) -> list[str]:
    parent = f"0/{region}" if region else "0"
    return list(
        dict.fromkeys(
            PurePosixPath(path).name
            for path in required
            if str(PurePosixPath(path).parent) == parent
        )
    )


def region_layouts(plan=None, required_files: list[str] | None = None) -> list[RegionCaseLayout]:
    """Return canonical case layouts without creating CFD topology from file paths.

    v5.0 authority rule:
    - For an EngineeringPlan with ``execution.regions``, execution is the sole
      authority for named-region topology and solver assignment.
    - ``region_layouts`` and the required-file manifest are projections that must be
      consistent with that topology; they never override it.
    - Legacy/no-execution callers may still project layouts from explicit
      ``region_layouts`` or, as a compatibility fallback, from required-file paths.
    """

    required = list(
        required_files
        if required_files is not None
        else (plan.required_case_files if plan is not None else [])
    )
    inferred = _manifest_region_names(required)

    execution = getattr(plan, "execution", None) if plan is not None else None
    assignments = {
        item.region: item.solver_module
        for item in (list(getattr(execution, "regions", None) or []))
    }

    if assignments:
        names = list(assignments)
        explicit_layouts = list(getattr(plan, "region_layouts", None) or [])
        if explicit_layouts:
            explicit_names = [item.region for item in explicit_layouts]
            if len(explicit_names) != len(set(explicit_names)):
                raise ValueError("Duplicate region layouts.")
            if set(explicit_names) != set(names):
                raise ValueError(
                    "Region layout names disagree with the authoritative execution region topology."
                )
        undeclared = inferred - set(names)
        if undeclared:
            raise ValueError(
                "Required files reference regions outside the authoritative execution topology: "
                + ", ".join(sorted(undeclared))
            )
        if any(
            len(PurePosixPath(path).parts) == 2 and path.startswith("0/")
            for path in required
        ):
            raise ValueError(
                "Root initial fields cannot be mixed into an execution-declared named-region case."
            )
        return [
            RegionCaseLayout(
                region=name,
                solver_module=assignments[name],
                required_fields=_manifest_required_fields(required, name),
            )
            for name in names
        ]

    explicit_layouts = list(getattr(plan, "region_layouts", None) or []) if plan is not None else []
    if explicit_layouts:
        names = [item.region for item in explicit_layouts]
        if len(names) != len(set(names)):
            raise ValueError("Duplicate region layouts.")
        undeclared = inferred - set(names)
        if undeclared:
            raise ValueError("Required files reference undeclared regions.")
        return [
            RegionCaseLayout(
                region=item.region,
                solver_module=item.solver_module,
                required_fields=_manifest_required_fields(required, item.region)
                or list(item.required_fields),
            )
            for item in explicit_layouts
        ]

    # Compatibility-only fallback for old plans that predate an explicit execution
    # topology. File paths may project layout here, but this path is not used to
    # override a modern Agent-owned execution decision.
    names = sorted(inferred) or [""]
    if inferred and any(
        len(PurePosixPath(path).parts) == 2 and path.startswith("0/")
        for path in required
    ):
        raise ValueError("Mixed root and named-region initial fields require explicit region layouts.")
    return [
        RegionCaseLayout(
            region=name,
            solver_module=None,
            required_fields=_manifest_required_fields(required, name),
        )
        for name in names
    ]


def region_mesh_digest(workspace, region: str) -> str:
    from .mesh_dependencies import MeshDependencyGraph

    return MeshDependencyGraph(workspace).digest(region)


def validate_design(plan, intake) -> list[str]:
    """Design-stage hard checks only.

    v5.0 keeps frozen-intake identity closure deterministic while leaving CFD
    implementation truth to authoring/native validation. Region topology is checked
    through :func:`region_layouts`, which treats execution as authoritative.
    """

    failures: list[str] = []
    ids = {item.id for item in intake.facts if item.category != "context"}
    if set(plan.confirmed_fact_ids) != ids:
        failures.append("Design does not exactly preserve active confirmed fact IDs.")
    if {item.fact_id for item in plan.confirmed_fact_bindings} != ids:
        failures.append("Design bindings do not exactly cover confirmed facts.")

    required = set(plan.required_case_files)
    for binding in plan.confirmed_fact_bindings:
        refs = set(binding.case_files) | {item.path for item in binding.case_assertions}
        if binding.numeric_relation is not None:
            refs |= {
                item.path
                for item in [
                    *binding.numeric_relation.numerator,
                    *binding.numeric_relation.denominator,
                ]
            }
        if refs - required:
            failures.append(
                f"Design binding {binding.fact_id} references undeclared files: "
                f"{sorted(refs - required)}"
            )

    try:
        region_layouts(plan)
    except ValueError as exc:
        failures.append(str(exc))
    return failures
