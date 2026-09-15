from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from openfoam_agent.tools.native_contracts import native_tool_contract, required_dictionary, command_permitted
from openfoam_agent.schemas.engineering import NativeOpenFOAMCommand
from openfoam_agent.contracts.execution_scopes import (
    affected_mesh_scope_keys,
    auto_consumers_for_delta,
    execution_scopes,
    final_mesh_validation_commands,
    scope_key,
)
from openfoam_agent.engineering.case_build_graph import _stable_dependency_order
from openfoam_agent.tools.foam_file import validate_foam_file_header

_NON_DICTIONARY_EXTENSIONS = {".stl", ".obj", ".off", ".vtk", ".vtp", ".csv", ".dat", ".emesh", ".gz"}
_SURFACE_EXTENSIONS = {".stl", ".obj", ".off", ".vtk", ".vtp"}


@dataclass(frozen=True)
class CaseDeltaGraph:
    """Controller-owned graph for edits to an already committed case.

    LLM repair payloads describe content deltas and native strategy hints only.  The
    controller builds the effective case in memory, proves required-file coverage and
    content safety before mutation, and compiles validators/native consumers from that
    effective state.  Stale model validation paths never become executable actions.
    """

    changed_files: dict[str, str]
    drop_paths: tuple[str, ...]
    effective_paths: tuple[str, ...]
    missing_required_paths: tuple[str, ...]
    dictionary_paths: tuple[str, ...]
    surface_paths: tuple[str, ...]
    native_pipeline: tuple[NativeOpenFOAMCommand, ...]
    validate_pre_solve: bool
    affected_mesh_scope_keys: tuple[str, ...]
    reused_mesh_scope_keys: tuple[str, ...]
    revalidation_domains: tuple[str, ...]
    failures: tuple[str, ...]
    warnings: tuple[str, ...]

    @property
    def valid(self) -> bool:
        return not self.failures


def _dedupe_native(items: Iterable[NativeOpenFOAMCommand]) -> list[NativeOpenFOAMCommand]:
    out: list[NativeOpenFOAMCommand] = []
    seen: set[tuple[str, tuple[str, ...]]] = set()
    for item in items:
        token = (item.command, tuple(item.arguments))
        if token in seen:
            continue
        seen.add(token)
        out.append(item)
    return out



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
    drops: tuple[str, ...] = ()

    # Build an effective path/content view from controller-tracked files plus every
    # required solve input that physically exists.  Native-generated polyMesh files do
    # not need to be materialized as text to establish required input coverage.
    effective: dict[str, str | None] = {}
    for path in workspace.list_authored():
        try:
            effective[path] = workspace.read_text(path, max_chars=workspace.max_file_bytes + 1)
        except (FileNotFoundError, OSError, UnicodeError) as exc:
            failures.append(f"Unable to inspect committed case file {path}: {exc}")
    for path in plan.required_case_files:
        if path in effective:
            continue
        try:
            target = workspace.resolve_case_path(path, must_exist=True)
        except (FileNotFoundError, OSError) :
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

    # Normalize replacement ownership. Exact repeated values are harmless; conflicting
    # values for one path are a semantic delta conflict and are never silently chosen.
    changed: dict[str, str] = {}
    replacement_paths: set[str] = set()
    for path, content in replacements:
        if path in changed and changed[path] != content:
            failures.append(f"Conflicting replacement content for {path}.")
            continue
        changed[path] = content
        replacement_paths.add(path)

    # Patches are applied in-memory to the effective candidate. No workspace mutation
    # occurs until the complete delta graph has passed all deterministic checks.
    patched_paths: set[str] = set()
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
        patched_paths.add(path)

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
        failures.append("Required case manifest would be incomplete after repair: " + ", ".join(missing))

    # Preflight every prospective write with the same safety/content policy as the real
    # workspace. Aggregate size validation is also performed before commit.
    bundle_failures = workspace.validate_candidate_bundle(changed, drop_paths=drops)
    failures.extend(bundle_failures)
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
    dictionary_paths = tuple(sorted(
        path for path in changed_paths
        if Path(path).suffix.lower() not in _NON_DICTIONARY_EXTENSIONS
    ))
    surface_paths = tuple(sorted(
        path for path in changed_paths
        if Path(path).suffix.lower() in _SURFACE_EXTENSIONS
    ))

    hints = list(native_hints)
    if not hints:
        hints = [NativeOpenFOAMCommand(command=str(command), role="mesh") for command in mesh_commands]
    strategy: list[NativeOpenFOAMCommand] = []
    for item in _dedupe_native(hints):
        if item.command in {"foamDictionary", "surfaceCheck", "checkMesh"}:
            continue
        contract = native_tool_contract(item.command)
        if not command_permitted(item.command, phase):
            warnings.append(f"Dropped native hint {item.command}: command is not permitted in {phase} phase.")
            continue
        required = required_dictionary(item.command, item.arguments)
        if required and required not in effective_set:
            warnings.append(f"Dropped stale native hint {item.command}: required input {required} is absent after delta.")
            continue
        strategy.append(item)

    changed_mesh_inputs = set(changed_paths) | set(drops)
    try:
        inferred = auto_consumers_for_delta(plan, effective_set, changed_mesh_inputs, strategy)
    except ValueError as exc:
        failures.append(f"Could not compile scope-bound delta consumers: {exc}")
        inferred = []
    strategy = _dedupe_native([*strategy, *inferred])

    try:
        affected_scopes = affected_mesh_scope_keys(
            workspace, plan, changed_mesh_inputs, strategy
        )
    except ValueError as exc:
        failures.append(f"Could not bind mesh delta to execution scopes: {exc}")
        affected_scopes = set()

    if affected_scopes:
        strategy = [
            item for item in strategy
            if not native_tool_contract(item.command).controller_finalizer
        ]
        strategy.extend(
            final_mesh_validation_commands(plan, only_scope_keys=affected_scopes)
        )

    # Repair/revision execution uses the same producer->consumer dependency order as
    # initial case authoring. Independent commands retain the Agent's relative order.
    finalizers = [item for item in strategy if native_tool_contract(item.command).controller_finalizer]
    consumers = [item for item in strategy if not native_tool_contract(item.command).controller_finalizer]
    strategy, ordering_failures = _stable_dependency_order(plan, _dedupe_native(consumers))
    strategy.extend(finalizers)
    failures.extend(ordering_failures)

    all_scope_keys = {scope_key(scope) for scope in execution_scopes(plan)}
    reused_scopes = all_scope_keys - set(affected_scopes)
    domains: list[str] = []
    if dictionary_paths:
        domains.append("dictionary")
    if surface_paths:
        domains.append("surface")
    if affected_scopes:
        domains.append("mesh")
    if validate_pre_solve:
        domains.append("pre_solve")

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
        affected_mesh_scope_keys=tuple(sorted(affected_scopes)),
        reused_mesh_scope_keys=tuple(sorted(reused_scopes)),
        revalidation_domains=tuple(domains),
        failures=tuple(dict.fromkeys(failures)),
        warnings=tuple(dict.fromkeys(warnings)),
    )
