from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from openfoam_agent.engineering.native_region_contract import (
    declared_named_regions,
    invocation_region,
    mesh_validation_commands,
    scoped_arguments,
    system_dictionary_scopes,
)
from openfoam_agent.schemas.engineering import ExecuteCasePlanAction, NativeOpenFOAMCommand
from openfoam_agent.tools.native_contracts import (
    command_permitted,
    native_tool_contract,
    required_dictionary,
)


# Files that are authored inputs but are not OpenFOAM dictionaries/fields. They are
# validated by their actual consumer (surfaceCheck, mesh utility, solver) instead of
# the FoamFile/header parser.
_NON_DICTIONARY_EXTENSIONS = {
    ".stl", ".obj", ".off", ".vtk", ".vtp", ".csv", ".dat", ".emesh", ".gz",
}
_SURFACE_EXTENSIONS = {".stl", ".obj", ".off", ".vtk", ".vtp"}


@dataclass(frozen=True)
class CaseBuildGraph:
    """Controller-compiled build graph for one frozen case candidate.

    The LLM owns engineering content. Python owns manifest coverage, validation target
    discovery and executable action ordering. Region topology comes only from the
    Agent-selected execution/layout contract; manifest paths cannot create topology.
    """

    required_paths: tuple[str, ...]
    authored_paths: tuple[str, ...]
    missing_required_paths: tuple[str, ...]
    dictionary_paths: tuple[str, ...]
    surface_paths: tuple[str, ...]
    native_pipeline: tuple[NativeOpenFOAMCommand, ...]
    failures: tuple[str, ...]
    warnings: tuple[str, ...]

    @property
    def valid(self) -> bool:
        return not self.failures


def _dedupe_invocations(items: list[NativeOpenFOAMCommand]) -> list[NativeOpenFOAMCommand]:
    unique: list[NativeOpenFOAMCommand] = []
    seen: set[tuple[str, tuple[str, ...], str]] = set()
    for item in items:
        token = (item.command, tuple(item.arguments), item.role)
        if token in seen:
            continue
        seen.add(token)
        unique.append(item)
    return unique


def compile_case_build_graph(
    execution: ExecuteCasePlanAction,
    candidate_bundle: dict[str, str],
) -> CaseBuildGraph:
    """Compile one authoritative authoring/validation/native graph.

    ``EngineeringPlan.required_case_files`` is the solve-input manifest owned by the
    Agent design. Model validation lists remain hints only. Python may bind an authored
    native dictionary to its exact deterministic consumer, but it never infers CFD
    region topology or a meshing strategy from arbitrary filenames.
    """

    required = tuple(dict.fromkeys(execution.plan.required_case_files))
    authored = tuple(candidate_bundle.keys())
    authored_set = set(authored)
    missing = tuple(path for path in required if path not in authored_set)
    failures: list[str] = []
    warnings: list[str] = []

    if not required:
        failures.append(
            "EngineeringPlan.required_case_files is empty; the Agent must declare at least one solve-input path before authoring."
        )
    if missing:
        failures.append(
            "Required case manifest is not fully authored: " + ", ".join(missing)
        )

    try:
        declared_regions = declared_named_regions(execution.plan)
    except ValueError as exc:
        failures.append(f"Invalid named-region topology: {exc}")
        declared_regions = []

    dictionary_paths = tuple(
        path
        for path in authored
        if Path(path).suffix.lower() not in _NON_DICTIONARY_EXTENSIONS
    )
    surface_paths = tuple(
        path
        for path in authored
        if Path(path).suffix.lower() in _SURFACE_EXTENSIONS
    )

    if execution.native_pipeline:
        hints = list(execution.native_pipeline)
    else:
        hints = [
            NativeOpenFOAMCommand(command=command, role="mesh")
            for command in execution.mesh_commands
        ]

    check_hints = [item for item in hints if item.command == "checkMesh"]
    strategy = _dedupe_invocations(
        [
            item for item in hints
            if item.command not in {"foamDictionary", "surfaceCheck", "checkMesh"}
        ]
    )

    # Infer only exact dictionary consumers. A region-scoped dictionary is executable
    # only when its region is already declared by execution/layout authority.
    try:
        strategy_scopes = {(item.command, invocation_region(item)) for item in strategy}
        block_scopes = system_dictionary_scopes(authored, "blockMeshDict")
        snappy_scopes = system_dictionary_scopes(authored, "snappyHexMeshDict")
    except ValueError as exc:
        failures.append(f"Invalid native region scope: {exc}")
        strategy_scopes = set()
        block_scopes = []
        snappy_scopes = []

    for region in [*block_scopes, *snappy_scopes]:
        if region and region not in declared_regions:
            failures.append(
                f"Authored mesh dictionary targets undeclared region {region!r}; case paths cannot create CFD region topology."
            )

    inferred_blocks = [
        NativeOpenFOAMCommand(
            command="blockMesh",
            arguments=scoped_arguments(region),
            role="mesh",
        )
        for region in block_scopes
        if (not region or region in declared_regions)
        and ("blockMesh", region) not in strategy_scopes
    ]
    strategy = inferred_blocks + strategy
    strategy_scopes.update(("blockMesh", region) for region in block_scopes)

    for region in snappy_scopes:
        if region and region not in declared_regions:
            continue
        if ("snappyHexMesh", region) in strategy_scopes:
            continue
        invocation = NativeOpenFOAMCommand(
            command="snappyHexMesh",
            arguments=scoped_arguments(region, "-overwrite"),
            role="mesh",
        )
        same_scope_blocks = [
            index for index, item in enumerate(strategy)
            if item.command == "blockMesh" and invocation_region(item) == region
        ]
        insert_at = same_scope_blocks[-1] + 1 if same_scope_blocks else len(strategy)
        strategy.insert(insert_at, invocation)
        strategy_scopes.add(("snappyHexMesh", region))

    executable_strategy: list[NativeOpenFOAMCommand] = []
    for invocation in strategy:
        try:
            region = invocation_region(invocation)
            if region and region not in declared_regions:
                warnings.append(
                    f"Dropped native hint {invocation.command}: region {region!r} is not in the Agent-declared execution topology."
                )
                continue
            required_dict = required_dictionary(invocation.command, invocation.arguments)
        except ValueError as exc:
            failures.append(f"Invalid native invocation {invocation.command}: {exc}")
            continue
        if not command_permitted(invocation.command, "authoring"):
            warnings.append(f"Dropped native hint {invocation.command}: command is not permitted in authoring phase.")
            continue
        if required_dict and required_dict not in authored_set:
            warnings.append(
                f"Dropped stale native hint {invocation.command}: required authored input {required_dict} is absent."
            )
            continue
        executable_strategy.append(invocation)

    valid_check_hints: list[NativeOpenFOAMCommand] = []
    for item in _dedupe_invocations(check_hints):
        try:
            region = invocation_region(item)
        except ValueError as exc:
            failures.append(f"Invalid checkMesh region scope: {exc}")
            continue
        if declared_regions:
            if not region or region not in declared_regions:
                warnings.append(
                    "Dropped checkMesh hint outside declared named-region topology: "
                    + (region or "<root>")
                )
                continue
        elif region:
            warnings.append(
                f"Dropped checkMesh hint for undeclared named region {region!r}."
            )
            continue
        valid_check_hints.append(item)

    checks = mesh_validation_commands(execution.plan, valid_check_hints)
    pipeline = tuple(executable_strategy + checks)

    for hinted in list(execution.validate_dictionaries) + list(execution.surface_checks):
        if hinted not in authored_set:
            warnings.append(
                f"Ignored stale validation hint for unauthored path: {hinted}"
            )

    return CaseBuildGraph(
        required_paths=required,
        authored_paths=authored,
        missing_required_paths=missing,
        dictionary_paths=dictionary_paths,
        surface_paths=surface_paths,
        native_pipeline=pipeline,
        failures=tuple(dict.fromkeys(failures)),
        warnings=tuple(dict.fromkeys(warnings)),
    )
