from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from openfoam_agent.schemas.engineering import ExecuteCasePlanAction, NativeOpenFOAMCommand


# Files that are authored inputs but are not OpenFOAM dictionaries/fields. They are
# validated by their actual consumer (surfaceCheck, mesh utility, solver) instead of
# the FoamFile/header parser.
_NON_DICTIONARY_EXTENSIONS = {
    ".stl", ".obj", ".off", ".vtk", ".vtp", ".csv", ".dat", ".emesh", ".gz",
}
_SURFACE_EXTENSIONS = {".stl", ".obj", ".off", ".vtk", ".vtp"}

# Strong utility prerequisites. These are only used when the command has its normal
# dictionary contract. Commands not listed here remain model-selected strategy hints
# and are still constrained by SafeRunner/provider policy later.
_DEFAULT_DICT_PREREQUISITES = {
    "blockMesh": "system/blockMeshDict",
    "snappyHexMesh": "system/snappyHexMeshDict",
    "topoSet": "system/topoSetDict",
    "setFields": "system/setFieldsDict",
    "createPatch": "system/createPatchDict",
    "decomposePar": "system/decomposeParDict",
    "extrudeMesh": "system/extrudeMeshDict",
}


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

    # Infer only commands whose relationship to an authored dictionary is exact and
    # deterministic. More complex utilities remain explicit engineering strategy.
    commands = [item.command for item in strategy]
    if "system/blockMeshDict" in authored_set and "blockMesh" not in commands:
        strategy.insert(0, NativeOpenFOAMCommand(command="blockMesh", role="mesh"))
        commands.insert(0, "blockMesh")
    if "system/snappyHexMeshDict" in authored_set and "snappyHexMesh" not in commands:
        insert_at = commands.index("blockMesh") + 1 if "blockMesh" in commands else len(strategy)
        strategy.insert(
            insert_at,
            NativeOpenFOAMCommand(command="snappyHexMesh", arguments=["-overwrite"], role="mesh"),
        )
        commands.insert(insert_at, "snappyHexMesh")

    executable_strategy: list[NativeOpenFOAMCommand] = []
    for invocation in strategy:
        required_dict = _dict_override(invocation.arguments) or _DEFAULT_DICT_PREREQUISITES.get(invocation.command)
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

    checks = _dedupe_invocations(check_hints)
    if not checks:
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
