from __future__ import annotations

from dataclasses import dataclass
from pathlib import PurePosixPath
import re
from typing import Literal

Phase = Literal["authoring", "repair", "runtime_repair", "strategy_revision", "postprocess", "runtime", "restart"]


@dataclass(frozen=True)
class NativeToolContract:
    command: str
    effect: str
    required_dictionary: str | None = None
    permitted_phases: tuple[Phase, ...] = ("authoring", "repair", "runtime_repair", "strategy_revision")
    controller_finalizer: bool = False


# Controller-owned native metadata.  This is the single place where an engineering
# utility's strong dictionary prerequisite and phase ownership are declared.
_REGISTRY: dict[str, NativeToolContract] = {
    "blockMesh": NativeToolContract("blockMesh", "mesh", "system/blockMeshDict"),
    "snappyHexMesh": NativeToolContract("snappyHexMesh", "mesh", "system/snappyHexMeshDict"),
    "surfaceFeatureExtract": NativeToolContract("surfaceFeatureExtract", "mesh", "system/surfaceFeatureExtractDict"),
    "createBaffles": NativeToolContract("createBaffles", "mesh"),
    "splitMeshRegions": NativeToolContract("splitMeshRegions", "mesh"),
    "setsToZones": NativeToolContract("setsToZones", "mesh"),
    "surfaceTransformPoints": NativeToolContract("surfaceTransformPoints", "mesh"),
    "topoSet": NativeToolContract("topoSet", "mesh", "system/topoSetDict"),
    "setFields": NativeToolContract("setFields", "initialization", "system/setFieldsDict"),
    "createPatch": NativeToolContract("createPatch", "mesh", "system/createPatchDict"),
    "decomposePar": NativeToolContract("decomposePar", "decomposition", "system/decomposeParDict"),
    "extrudeMesh": NativeToolContract("extrudeMesh", "mesh", "system/extrudeMeshDict"),
    "checkMesh": NativeToolContract("checkMesh", "validation", controller_finalizer=True),
    "surfaceCheck": NativeToolContract("surfaceCheck", "validation"),
    "foamDictionary": NativeToolContract("foamDictionary", "validation"),
    "potentialFoam": NativeToolContract("potentialFoam", "initialization"),
    "renumberMesh": NativeToolContract("renumberMesh", "mesh"),
    "transformPoints": NativeToolContract("transformPoints", "mesh"),
    "gmshToFoam": NativeToolContract("gmshToFoam", "mesh"),
    "fluentMeshToFoam": NativeToolContract("fluentMeshToFoam", "mesh"),
    "foamToC": NativeToolContract("foamToC", "query"),
    "foamListTimes": NativeToolContract("foamListTimes", "query"),
    "mapFields": NativeToolContract("mapFields", "initialization"),
    "foamCleanCase": NativeToolContract("foamCleanCase", "destructive", permitted_phases=()),
    "foamCleanPolyMesh": NativeToolContract("foamCleanPolyMesh", "destructive", permitted_phases=()),
    "foamPostProcess": NativeToolContract("foamPostProcess", "postprocess", permitted_phases=("postprocess",)),
    "foamRun": NativeToolContract("foamRun", "solve", permitted_phases=("runtime",)),
    "foamMultiRun": NativeToolContract("foamMultiRun", "solve", permitted_phases=("runtime",)),
    "reconstructPar": NativeToolContract("reconstructPar", "reconstruction", permitted_phases=("runtime", "restart")),
    "reconstructParMesh": NativeToolContract("reconstructParMesh", "reconstruction", permitted_phases=("runtime", "restart")),
}


def native_tool_contract(command: str) -> NativeToolContract:
    contract = _REGISTRY.get(command)
    if contract is not None:
        return contract
    # Unknown commands are not given hidden authority.  Preserve the central effect
    # classification so callers can reject/route them deterministically.
    return NativeToolContract(command, "unknown", permitted_phases=())


_REGION_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9_.-]*$")


def native_command_region(arguments: list[str] | tuple[str, ...]) -> str | None:
    """Return one validated ``-region`` target from a native invocation.

    Region scope is deterministic execution metadata, not a CFD design choice.
    """
    args = list(arguments)
    positions = [i for i, value in enumerate(args) if value == "-region"]
    if "-allRegions" in args and positions:
        raise ValueError("Native command cannot combine -region with -allRegions.")
    if len(positions) > 1:
        raise ValueError("Native command may declare -region at most once.")
    if not positions:
        return None
    index = positions[0]
    if index + 1 >= len(args):
        raise ValueError("Native command -region is missing its region name.")
    region = args[index + 1]
    if not _REGION_NAME.fullmatch(region):
        raise ValueError(f"Unsafe native region name: {region!r}")
    return region


def required_dictionary(command: str, arguments: list[str] | tuple[str, ...]) -> str | None:
    args = list(arguments)
    for flag in ("-dict", "-dictFile"):
        if flag in args:
            index = args.index(flag)
            if index + 1 < len(args):
                return args[index + 1]
    required = native_tool_contract(command).required_dictionary
    region = native_command_region(args)
    if required and region:
        parts = PurePosixPath(required).parts
        if len(parts) == 2 and parts[0] == "system":
            return f"system/{region}/{parts[1]}"
    return required


def command_permitted(command: str, phase: Phase) -> bool:
    return phase in native_tool_contract(command).permitted_phases


def registered_effects() -> dict[str, str]:
    return {name: contract.effect for name, contract in _REGISTRY.items()}
