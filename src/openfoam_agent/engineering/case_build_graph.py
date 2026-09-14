from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from openfoam_agent.engineering.native_scope import (
    dedupe_invocations,
    final_check_mesh_commands,
    invocation_region,
    scoped_arguments,
    system_dictionary_scopes,
    validate_scopes_against_plan,
)
from openfoam_agent.schemas.engineering import ExecuteCasePlanAction, NativeOpenFOAMCommand
from openfoam_agent.tools.native_contracts import (
    command_permitted,
    required_dictionary,
)


_NON_DICTIONARY_EXTENSIONS = {
    ".stl", ".obj", ".off", ".vtk", ".vtp", ".csv", ".dat", ".emesh", ".gz",
}
_SURFACE_EXTENSIONS = {".stl", ".obj", ".off", ".vtk", ".vtp"}


@dataclass(frozen=True)
class CaseBuildGraph:
    """Controller-compiled build graph for one frozen case candidate.

    The Agent owns engineering content and strategy. Python owns manifest coverage,
    executable prerequisites, deterministic consumer ordering and final validation
    scope. Region names are never inferred from file paths when an execution topology
    exists.
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


def compile_case_build_graph(
    execution: ExecuteCasePlanAction,
    candidate_bundle: dict[str, str],
) -> CaseBuildGraph:
    """Compile one authoritative authoring/validation/native graph."""

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
        failures.append("Required case manifest is not fully authored: " + ", ".join(missing))

    dictionary_paths = tuple(
        path for path in authored if Path(path).suffix.lower() not in _NON_DICTIONARY_EXTENSIONS
    )
    surface_paths = tuple(
        path for path in authored if Path(path).suffix.lower() in _SURFACE_EXTENSIONS
    )

    if execution.native_pipeline:
        hints = list(execution.native_pipeline)
    else:
        hints = [
            NativeOpenFOAMCommand(command=command, role="mesh")
            for command in execution.mesh_commands
        ]

    check_hints = [item for item in hints if item.command == "checkMesh"]
    strategy = dedupe_invocations(
        item
        for item in hints
        if item.command not in {"foamDictionary", "surfaceCheck", "checkMesh"}
    )

    try:
        strategy_scopes = {(item.command, invocation_region(item)) for item in strategy}
        block_scopes = system_dictionary_scopes(authored, "blockMeshDict")
        snappy_scopes = system_dictionary_scopes(authored, "snappyHexMeshDict")
        invalid_scopes = validate_scopes_against_plan(
            execution.plan, [*block_scopes, *snappy_scopes]
        )
        if invalid_scopes:
            failures.append(
                "Authored mesh dictionaries target regions outside the authoritative execution topology: "
                + ", ".join(invalid_scopes)
            )
    except ValueError as exc:
        failures.append(f"Invalid native region scope: {exc}")
        strategy_scopes = set()
        block_scopes = []
        snappy_scopes = []

    inferred_blocks = [
        NativeOpenFOAMCommand(
            command="blockMesh",
            arguments=scoped_arguments(region),
            role="mesh",
        )
        for region in block_scopes
        if ("blockMesh", region) not in strategy_scopes
    ]
    strategy = inferred_blocks + strategy
    strategy_scopes.update(("blockMesh", region) for region in block_scopes)

    for region in snappy_scopes:
        if ("snappyHexMesh", region) in strategy_scopes:
            continue
        invocation = NativeOpenFOAMCommand(
            command="snappyHexMesh",
            arguments=scoped_arguments(region, "-overwrite"),
            role="mesh",
        )
        same_scope_blocks = [
            index
            for index, item in enumerate(strategy)
            if item.command == "blockMesh" and invocation_region(item) == region
        ]
        insert_at = same_scope_blocks[-1] + 1 if same_scope_blocks else len(strategy)
        strategy.insert(insert_at, invocation)
        strategy_scopes.add(("snappyHexMesh", region))

    executable_strategy: list[NativeOpenFOAMCommand] = []
    for invocation in strategy:
        try:
            required_dict = required_dictionary(invocation.command, invocation.arguments)
        except ValueError as exc:
            failures.append(f"Invalid native invocation {invocation.command}: {exc}")
            continue
        if not command_permitted(invocation.command, "authoring"):
            warnings.append(
                f"Dropped native hint {invocation.command}: command is not permitted in authoring phase."
            )
            continue
        if required_dict and required_dict not in authored_set:
            warnings.append(
                f"Dropped stale native hint {invocation.command}: required authored input {required_dict} is absent."
            )
            continue
        executable_strategy.append(invocation)

    try:
        checks, scope_warnings = final_check_mesh_commands(execution.plan, check_hints)
        warnings.extend(scope_warnings)
    except ValueError as exc:
        failures.append(f"Invalid checkMesh topology: {exc}")
        checks = []

    pipeline = tuple(executable_strategy + checks)

    for hinted in list(execution.validate_dictionaries) + list(execution.surface_checks):
        if hinted not in authored_set:
            warnings.append(f"Ignored stale validation hint for unauthored path: {hinted}")

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
