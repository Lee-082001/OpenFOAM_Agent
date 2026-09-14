from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from openfoam_agent.engineering.native_scope import (
    dedupe_invocations,
    final_check_mesh_commands,
    invocation_region,
    mesh_affecting_path,
    scoped_arguments,
    system_dictionary_scopes,
    validate_scopes_against_plan,
)
from openfoam_agent.schemas.engineering import NativeOpenFOAMCommand
from openfoam_agent.tools.foam_file import validate_foam_file_header
from openfoam_agent.tools.native_contracts import (
    command_permitted,
    native_tool_contract,
    required_dictionary,
)

_NON_DICTIONARY_EXTENSIONS = {
    ".stl", ".obj", ".off", ".vtk", ".vtp", ".csv", ".dat", ".emesh", ".gz"
}
_SURFACE_EXTENSIONS = {".stl", ".obj", ".off", ".vtk", ".vtp"}


@dataclass(frozen=True)
class CaseDeltaGraph:
    """Controller-owned graph for edits to an already committed case.

    v5.0 uses the same region/native-scope rules as initial authoring. Repair hints are
    non-authoritative; Python derives only deterministic consumers of changed artifacts
    and final checkMesh scope from the frozen execution topology.
    """

    changed_files: dict[str, str]
    drop_paths: tuple[str, ...]
    effective_paths: tuple[str, ...]
    missing_required_paths: tuple[str, ...]
    dictionary_paths: tuple[str, ...]
    surface_paths: tuple[str, ...]
    native_pipeline: tuple[NativeOpenFOAMCommand, ...]
    validate_pre_solve: bool
    failures: tuple[str, ...]
    warnings: tuple[str, ...]

    @property
    def valid(self) -> bool:
        return not self.failures


def compile_case_delta_graph(
    workspace,
    plan,
    *,
    patches: Iterable[tuple[str, str, str]] = (),
    replacements: Iterable[tuple[str, str]] = (),
    drop_paths: Iterable[str] = (),
    native_hints: Iterable[NativeOpenFOAMCommand] = (),
    mesh_commands: Iterable[str] = (),
    validate_dictionaries: Iterable[str] = (),
    surface_checks: Iterable[str] = (),
    validate_pre_solve: bool = True,
    phase: str = "repair",
) -> CaseDeltaGraph:
    failures: list[str] = []
    warnings: list[str] = []
    requested_drops = tuple(dict.fromkeys(str(path) for path in drop_paths))

    effective: dict[str, str | None] = {}
    for path in workspace.list_authored():
        try:
            effective[path] = workspace.read_text(
                path, max_chars=workspace.max_file_bytes + 1
            )
        except (FileNotFoundError, OSError, UnicodeError) as exc:
            failures.append(f"Unable to inspect committed case file {path}: {exc}")
    for path in plan.required_case_files:
        if path in effective:
            continue
        try:
            target = workspace.resolve_case_path(path, must_exist=True)
        except (FileNotFoundError, OSError):
            continue
        try:
            effective[path] = target.read_text(encoding="utf-8", errors="replace")
        except (OSError, UnicodeError) as exc:
            failures.append(f"Unable to inspect required case file {path}: {exc}")

    effective_before_drop = set(effective)
    active_drops: list[str] = []
    for path in requested_drops:
        if path not in effective_before_drop:
            warnings.append(f"Ignored stale drop hint for absent path: {path}")
            continue
        active_drops.append(path)
    drops = tuple(active_drops)

    changed: dict[str, str] = {}
    replacement_paths: set[str] = set()
    for path, content in replacements:
        if path in changed and changed[path] != content:
            failures.append(f"Conflicting replacement content for {path}.")
            continue
        changed[path] = content
        replacement_paths.add(path)

    for path, old, new in patches:
        if path in replacement_paths:
            failures.append(f"Case delta cannot patch and replace {path} in the same turn.")
            continue
        if path in drops:
            failures.append(f"Case delta cannot patch and drop {path} in the same turn.")
            continue
        current = effective.get(path)
        if isinstance(current, str) and current.endswith("\n... [truncated]"):
            failures.append(f"Patch target is too large for exact in-memory repair: {path}")
            continue
        if current is None:
            failures.append(f"Patch target is absent from the committed case: {path}")
            continue
        if not old:
            failures.append(f"Patch old text must not be empty: {path}")
            continue
        count = current.count(old)
        if count != 1:
            failures.append(f"Exact patch requires one match in {path}; observed {count}.")
            continue
        effective[path] = current.replace(old, new, 1)
        changed[path] = str(effective[path])

    for path in drops:
        if path in changed:
            failures.append(f"Case delta cannot replace and drop {path} in the same turn.")
        effective.pop(path, None)
    for path, content in changed.items():
        effective[path] = content

    effective_paths = tuple(sorted(effective))
    effective_set = set(effective_paths)
    missing = tuple(path for path in plan.required_case_files if path not in effective_set)
    if missing:
        failures.append(
            "Required case manifest would be incomplete after repair: " + ", ".join(missing)
        )

    failures.extend(workspace.validate_candidate_bundle(changed, drop_paths=drops))
    for path, content in changed.items():
        if not validate_pre_solve or Path(path).suffix.lower() in _NON_DICTIONARY_EXTENSIONS:
            continue
        header = validate_foam_file_header(
            path,
            content,
            expected_class=("dictionary" if path.startswith("system/") else None),
        )
        failures.extend(f"{path}: {failure}" for failure in header.failures)

    changed_paths = set(changed)
    all_delta_paths = changed_paths | set(drops)
    dictionary_paths = tuple(
        sorted(
            path
            for path in changed_paths
            if Path(path).suffix.lower() not in _NON_DICTIONARY_EXTENSIONS
        )
    )
    surface_paths = tuple(
        sorted(
            path
            for path in changed_paths
            if Path(path).suffix.lower() in _SURFACE_EXTENSIONS
        )
    )

    hints = list(native_hints)
    if not hints:
        hints = [
            NativeOpenFOAMCommand(command=str(command), role="mesh")
            for command in mesh_commands
        ]
    check_hints = [item for item in hints if item.command == "checkMesh"]
    strategy: list[NativeOpenFOAMCommand] = []
    for item in dedupe_invocations(hints):
        if item.command in {"foamDictionary", "surfaceCheck", "checkMesh"}:
            continue
        if not command_permitted(item.command, phase):
            warnings.append(
                f"Dropped native hint {item.command}: command is not permitted in {phase} phase."
            )
            continue
        try:
            required = required_dictionary(item.command, item.arguments)
        except ValueError as exc:
            failures.append(f"Invalid native invocation {item.command}: {exc}")
            continue
        if required and required not in effective_set:
            warnings.append(
                f"Dropped stale native hint {item.command}: required input {required} is absent after delta."
            )
            continue
        strategy.append(item)

    try:
        scopes = {(item.command, invocation_region(item)) for item in strategy}
        changed_block_scopes = system_dictionary_scopes(changed_paths, "blockMeshDict")
        changed_snappy_scopes = system_dictionary_scopes(changed_paths, "snappyHexMeshDict")
        invalid_scopes = validate_scopes_against_plan(
            plan, [*changed_block_scopes, *changed_snappy_scopes]
        )
        if invalid_scopes:
            failures.append(
                "Repair changes mesh dictionaries outside the authoritative execution topology: "
                + ", ".join(invalid_scopes)
            )
    except ValueError as exc:
        failures.append(f"Invalid repair region scope: {exc}")
        scopes = set()
        changed_block_scopes = []
        changed_snappy_scopes = []

    inferred_blocks = [
        NativeOpenFOAMCommand(
            command="blockMesh",
            arguments=scoped_arguments(region),
            role="mesh",
        )
        for region in changed_block_scopes
        if ("blockMesh", region) not in scopes
    ]
    strategy = inferred_blocks + strategy
    scopes.update(("blockMesh", region) for region in changed_block_scopes)

    for region in changed_snappy_scopes:
        if ("snappyHexMesh", region) in scopes:
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
        scopes.add(("snappyHexMesh", region))

    mesh_changed = any(mesh_affecting_path(path) for path in all_delta_paths)
    mesh_changed = mesh_changed or any(
        native_tool_contract(item.command).effect
        in {"mesh", "decomposition", "initialization"}
        for item in strategy
    )
    if mesh_changed:
        try:
            checks, scope_warnings = final_check_mesh_commands(plan, check_hints)
            warnings.extend(scope_warnings)
            strategy = [item for item in strategy if item.command != "checkMesh"] + checks
        except ValueError as exc:
            failures.append(f"Invalid repair checkMesh topology: {exc}")

    for hinted in list(validate_dictionaries) + list(surface_checks):
        if hinted not in effective_set:
            warnings.append(f"Ignored stale validation hint for absent path after delta: {hinted}")

    return CaseDeltaGraph(
        changed_files=changed,
        drop_paths=drops,
        effective_paths=effective_paths,
        missing_required_paths=missing,
        dictionary_paths=dictionary_paths,
        surface_paths=surface_paths,
        native_pipeline=tuple(strategy),
        validate_pre_solve=bool(validate_pre_solve),
        failures=tuple(dict.fromkeys(failures)),
        warnings=tuple(dict.fromkeys(warnings)),
    )
