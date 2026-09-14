"""Backward-compatible exports for the v5 execution-scope controller.

New controller code should import ``openfoam_agent.contracts.execution_scopes``.
This module remains only so persisted tests/extensions using the old helper path do
not become an independent topology authority.
"""
from __future__ import annotations

from pathlib import PurePosixPath

from openfoam_agent.contracts.execution_scopes import (
    execution_scopes,
    final_mesh_validation_commands,
    scope_arguments_for,
    scope_for_native_arguments,
)
from openfoam_agent.schemas.engineering import NativeOpenFOAMCommand
from openfoam_agent.tools.native_contracts import native_command_region


_MESH_SYSTEM_DICTIONARIES = {
    "blockMeshDict",
    "snappyHexMeshDict",
    "surfaceFeatureExtractDict",
    "createPatchDict",
    "topoSetDict",
    "setFieldsDict",
    "decomposeParDict",
}


def system_dictionary_scopes(paths, filename: str) -> list[str]:
    scopes: list[str] = []
    for path in paths:
        parts = PurePosixPath(path).parts
        if parts == ("system", filename):
            name = ""
        elif len(parts) == 3 and parts[0] == "system" and parts[2] == filename:
            name = parts[1]
        else:
            continue
        if name not in scopes:
            scopes.append(name)
    return scopes


def scoped_arguments(region: str, *extra: str) -> list[str]:
    # Compatibility only: generic code should use scope_arguments_for(command, scope).
    return (["-region", region] if region else []) + list(extra)


def invocation_region(invocation: NativeOpenFOAMCommand) -> str:
    return native_command_region(invocation.arguments) or ""


def declared_named_regions(plan) -> list[str]:
    return [scope.name for scope in execution_scopes(plan) if scope.name is not None]


def mesh_affecting_path(path: str) -> bool:
    parts = PurePosixPath(path).parts
    if not parts:
        return False
    if parts[0] == "system" and parts[-1] in _MESH_SYSTEM_DICTIONARIES and len(parts) in {2, 3}:
        return True
    if parts[0] == "constant":
        if len(parts) >= 2 and parts[1] in {"polyMesh", "triSurface", "geometry"}:
            return True
        if len(parts) >= 3 and parts[2] in {"polyMesh", "triSurface", "geometry"}:
            return True
    return False


def final_check_mesh_commands(plan) -> list[NativeOpenFOAMCommand]:
    return final_mesh_validation_commands(plan)
