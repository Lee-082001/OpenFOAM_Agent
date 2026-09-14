from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from openfoam_agent.schemas.engineering import ExecuteCasePlanAction, NativeOpenFOAMCommand
from openfoam_agent.tools.native_contracts import (
    command_permitted,
    native_command_region,
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

# Native utility prerequisites/effects are centralized in native_tool_contracts.py.


@dataclass(frozen=True)
class CaseBuildGraph:
    """Controller-compiled build graph for one frozen case candidate.

    The LLM owns engineering content. Python owns manifest coverage, validation target
    discovery and executable action ordering. This prevents a stale model-authored
    validation list from referring to a file that was never authored.
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


def _dict_override(arguments: list[str]) -> str | None:
    for flag in ("-dict", "-dictFile"):
        if flag in arguments:
            index = arguments.index(flag)
            if index + 1 < len(arguments):
                return arguments[index + 1]
    return None


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


def _system_dictionary_scopes(authored: tuple[str, ...], filename: str) -> list[str]:
    """Return root/named-region scopes that actually authored one system dictionary."""
    scopes: list[str] = []
    for path in authored:
        parts = PurePosixPath(path).parts
        if parts == ("system", filename):
            scope = ""
        elif len(parts) == 3 and parts[0] == "system" and parts[2] == filename:
            scope = parts[1]
        else:
            continue
        if scope not in scopes:
            scopes.append(scope)
    return scopes


def _scoped_arguments(region: str, *extra: str) -> list[str]:
    return (["-region", region] if region else []) + list(extra)


def _invocation_region(invocation: NativeOpenFOAMCommand) -> str:
    return native_command_region(invocation.arguments) or ""


def _declared_named_regions(plan) -> list[str]:
    """Project Agent-declared named regions without inventing CFD topology."""
    names: list[str] = []
    for layout in list(getattr(plan, "region_layouts", None) or []):
        name = str(getattr(layout, "region", "") or "")
        if name and name not in names:
            names.append(name)
    execution = getattr(plan, "execution", None)
    for assignment in list(getattr(execution, "regions", None) or []):
        name = str(getattr(assignment, "region", "") or "")
        if name and name not in names:
            names.append(name)
    if names:
        return names
    for path in list(getattr(plan, "required_case_files", None) or []):
        parts = PurePosixPath(path).parts
        if len(parts) >= 3 and parts[0] == "0":
            name = parts[1]
        elif len(parts) == 3 and parts[0] == "system" and parts[2] in {
            "fvSchemes", "fvSolution", "blockMeshDict", "snappyHexMeshDict"
        }:
            name = parts[1]
        else:
            continue
        if name and name not in names:
            names.append(name)
    return names


def compile_case_build_graph(
    execution: ExecuteCasePlanAction,
    candidate_bundle: dict[str, str],
) -> CaseBuildGraph:
    """Compile one authoritative authoring/validation/native graph.

    ``EngineeringPlan.required_case_files`` is the only solve-input manifest. LLM
    fields such as ``validate_dictionaries`` and ``surface_checks`` are compatibility
    hints and never create actions independently. Native mesh commands are strategy
    hints; Python removes documentary probes, verifies strong prerequisites, infers
    obvious mesh consumers from authored dictionaries, and owns final checkMesh order.
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

    # Documentary probes are not action-authority. surfaceCheck is recompiled from
    # actual authored surfaces below; checkMesh is re-owned by the controller so it
    # can never run before a later mesh mutation.
    check_hints = [item for item in hints if item.command == "checkMesh"]
    strategy = [
        item for item in hints
        if item.command not in {"foamDictionary", "surfaceCheck", "checkMesh"}
    ]
    strategy = _dedupe_invocations(strategy)

    # Infer only exact dictionary consumers. Region names come from authored paths
    # or the Agent-owned plan; Python does not invent regions or meshing strategy.
    try:
        strategy_scopes = {(item.command, _invocation_region(item)) for item in strategy}
        block_scopes = _system_dictionary_scopes(authored, "blockMeshDict")
        snappy_scopes = _system_dictionary_scopes(authored, "snappyHexMeshDict")
    except ValueError as exc:
        failures.append(f"Invalid native region scope: {exc}")
        strategy_scopes = set()
        block_scopes = []
        snappy_scopes = []

    inferred_blocks = [
        NativeOpenFOAMCommand(
            command="blockMesh",
            arguments=_scoped_arguments(region),
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
            arguments=_scoped_arguments(region, "-overwrite"),
            role="mesh",
        )
        same_scope_blocks = [
            index for index, item in enumerate(strategy)
            if item.command == "blockMesh" and _invocation_region(item) == region
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
        contract = native_tool_contract(invocation.command)
        if not command_permitted(invocation.command, "authoring"):
            warnings.append(f"Dropped native hint {invocation.command}: command is not permitted in authoring phase.")
            continue
        if required_dict and required_dict not in authored_set:
            # Native lists are only strategy hints. A stale hint must not regain
            # authority merely because it names a real executable; drop it and let
            # actual bundle consumers/checkMesh reveal whether more preprocessing is
            # needed. This is the native analogue of ignoring stale dictionary hints.
            warnings.append(
                f"Dropped stale native hint {invocation.command}: required authored input {required_dict} is absent."
            )
            continue
        executable_strategy.append(invocation)

    # Final checkMesh follows the declared solve-region topology. A named multi-region
    # case must not fall back to constant/polyMesh at the case root.
    checks = _dedupe_invocations(check_hints)
    named_regions = _declared_named_regions(execution.plan)
    if named_regions:
        by_region: dict[str, NativeOpenFOAMCommand] = {}
        for item in checks:
            try:
                region = _invocation_region(item)
            except ValueError as exc:
                failures.append(f"Invalid checkMesh region scope: {exc}")
                continue
            if not region or region not in named_regions:
                warnings.append(
                    "Dropped checkMesh hint outside declared named-region topology: "
                    + (region or "<root>")
                )
                continue
            by_region.setdefault(region, item)
        checks = [
            by_region.get(region)
            or NativeOpenFOAMCommand(
                command="checkMesh",
                arguments=_scoped_arguments(region),
                role="mesh_validation",
            )
            for region in named_regions
        ]
    elif not checks:
        checks = [NativeOpenFOAMCommand(command="checkMesh", role="mesh_validation")]

    # Surface checking is controller-owned and derived from actual artifacts. It is
    # represented separately by SurfaceCheckAction, not duplicated in native_pipeline.
    pipeline = tuple(executable_strategy + checks)

    # Validation mirrors can still be useful diagnostics in reports, but stale paths
    # are explicitly ignored rather than executed.
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
