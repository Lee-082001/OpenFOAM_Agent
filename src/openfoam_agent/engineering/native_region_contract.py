from __future__ import annotations

from pathlib import PurePosixPath

from openfoam_agent.schemas.engineering import NativeOpenFOAMCommand
from openfoam_agent.tools.native_contracts import native_command_region


def scoped_arguments(region: str, *extra: str) -> list[str]:
    """Bind an already-declared region to native argv without choosing topology."""
    return (["-region", region] if region else []) + list(extra)


def invocation_region(invocation: NativeOpenFOAMCommand) -> str:
    return native_command_region(invocation.arguments) or ""


def system_dictionary_scopes(paths: tuple[str, ...] | list[str] | set[str], filename: str) -> list[str]:
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


def declared_named_regions(plan) -> list[str]:
    """Return the canonical Agent-declared named-region topology.

    `execution.regions` is the execution-topology authority. `region_layouts` is a
    compatibility projection and may confirm the same names, but case-file paths never
    create regions. This prevents a manifest spelling from becoming hidden CFD topology.
    """
    execution = getattr(plan, "execution", None)
    execution_regions = [
        str(getattr(item, "region", "") or "")
        for item in list(getattr(execution, "regions", None) or [])
        if str(getattr(item, "region", "") or "")
    ]
    if len(execution_regions) != len(set(execution_regions)):
        raise ValueError("Execution specification contains duplicate named regions.")

    layout_regions = [
        str(getattr(item, "region", "") or "")
        for item in list(getattr(plan, "region_layouts", None) or [])
        if str(getattr(item, "region", "") or "")
    ]
    if len(layout_regions) != len(set(layout_regions)):
        raise ValueError("Region layout projection contains duplicate named regions.")

    if execution_regions:
        if layout_regions and set(layout_regions) != set(execution_regions):
            raise ValueError("Region layout projection disagrees with execution.regions topology.")
        return execution_regions
    return layout_regions


def mesh_validation_commands(plan, existing_hints: list[NativeOpenFOAMCommand] | None = None) -> list[NativeOpenFOAMCommand]:
    """Compile final checkMesh commands from declared execution topology only."""
    hints = list(existing_hints or [])
    regions = declared_named_regions(plan)
    if not regions:
        root = next((item for item in hints if invocation_region(item) == ""), None)
        return [root or NativeOpenFOAMCommand(command="checkMesh", role="mesh_validation")]

    by_region: dict[str, NativeOpenFOAMCommand] = {}
    for item in hints:
        region = invocation_region(item)
        if region in regions:
            by_region.setdefault(region, item)
    return [
        by_region.get(region)
        or NativeOpenFOAMCommand(
            command="checkMesh",
            arguments=scoped_arguments(region),
            role="mesh_validation",
        )
        for region in regions
    ]
