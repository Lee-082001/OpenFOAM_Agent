from __future__ import annotations

from pathlib import PurePosixPath
from typing import Iterable

from openfoam_agent.schemas.engineering import NativeOpenFOAMCommand
from openfoam_agent.tools.native_contracts import native_command_region


_MESH_DICTIONARIES = frozenset({
    "blockMeshDict",
    "snappyHexMeshDict",
    "surfaceFeatureExtractDict",
    "createPatchDict",
    "topoSetDict",
    "setFieldsDict",
    "decomposeParDict",
    "extrudeMeshDict",
})


def dedupe_invocations(items: Iterable[NativeOpenFOAMCommand]) -> list[NativeOpenFOAMCommand]:
    out: list[NativeOpenFOAMCommand] = []
    seen: set[tuple[str, tuple[str, ...], str]] = set()
    for item in items:
        token = (item.command, tuple(item.arguments), item.role)
        if token in seen:
            continue
        seen.add(token)
        out.append(item)
    return out


def scoped_arguments(region: str, *extra: str) -> list[str]:
    return (["-region", region] if region else []) + list(extra)


def invocation_region(invocation: NativeOpenFOAMCommand) -> str:
    return native_command_region(invocation.arguments) or ""


def declared_named_regions(plan) -> list[str]:
    """Return the controller-validated named-region topology.

    Modern plans use execution.regions as the sole authority. ``region_layouts`` is a
    compatibility fallback only for legacy plans that do not carry execution-region
    assignments. Required-file paths never create topology here.
    """

    execution = getattr(plan, "execution", None)
    assignments = list(getattr(execution, "regions", None) or [])
    if assignments:
        names = [str(item.region) for item in assignments]
        if len(names) != len(set(names)):
            raise ValueError("Execution contains duplicate named regions.")
        return names

    layouts = list(getattr(plan, "region_layouts", None) or [])
    names = [str(getattr(item, "region", "") or "") for item in layouts]
    names = [name for name in names if name]
    if len(names) != len(set(names)):
        raise ValueError("Region layouts contain duplicate named regions.")
    return names


def system_dictionary_scopes(paths: Iterable[str], filename: str) -> list[str]:
    """Return root/named-region scopes that actually contain one system dictionary."""

    scopes: list[str] = []
    for path in paths:
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


def validate_scopes_against_plan(plan, scopes: Iterable[str]) -> list[str]:
    """Reject region-scoped artifacts outside an authoritative named topology."""

    declared = declared_named_regions(plan)
    if not declared:
        return []
    allowed = set(declared)
    return sorted({scope for scope in scopes if scope and scope not in allowed})


def final_check_mesh_commands(
    plan,
    hints: Iterable[NativeOpenFOAMCommand] = (),
) -> tuple[list[NativeOpenFOAMCommand], list[str]]:
    """Compile final checkMesh invocations from authoritative topology.

    Model-provided checkMesh actions are hints only. The controller owns final ordering
    and scope, but never invents region names.
    """

    warnings: list[str] = []
    checks = dedupe_invocations(item for item in hints if item.command == "checkMesh")
    named_regions = declared_named_regions(plan)
    if named_regions:
        by_region: dict[str, NativeOpenFOAMCommand] = {}
        for item in checks:
            region = invocation_region(item)
            if not region or region not in named_regions:
                warnings.append(
                    "Dropped checkMesh hint outside authoritative named-region topology: "
                    + (region or "<root>")
                )
                continue
            by_region.setdefault(region, item)
        return [
            by_region.get(region)
            or NativeOpenFOAMCommand(
                command="checkMesh",
                arguments=scoped_arguments(region),
                role="mesh_validation",
            )
            for region in named_regions
        ], warnings

    root = next((item for item in checks if not invocation_region(item)), None)
    return [root or NativeOpenFOAMCommand(command="checkMesh", role="mesh_validation")], warnings


def mesh_affecting_path(path: str) -> bool:
    """Whether a case-relative artifact can invalidate mesh evidence.

    This is path/effect bookkeeping only; it does not choose a meshing strategy.
    Region-scoped system dictionaries and polyMesh paths are treated exactly like their
    root equivalents.
    """

    parts = PurePosixPath(path).parts
    if not parts:
        return False
    if parts[0] == "system" and parts[-1] in _MESH_DICTIONARIES and len(parts) in {2, 3}:
        return True
    if parts[0] == "constant":
        if len(parts) >= 2 and parts[1] in {"triSurface", "polyMesh", "geometry"}:
            return True
        if len(parts) >= 3 and parts[2] in {"triSurface", "polyMesh", "geometry"}:
            return True
    return False
