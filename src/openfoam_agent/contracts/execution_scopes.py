from __future__ import annotations

from dataclasses import dataclass
import re
from pathlib import PurePosixPath

from openfoam_agent.schemas.engineering import ExecutionScope, NativeOpenFOAMCommand
from openfoam_agent.tools.native_contracts import (
    native_command_region,
    native_tool_contract,
    registered_contracts,
    required_dictionary,
)

ROOT_SCOPE_KEY = "root"
_REGION_SCOPE_PREFIX = "region:"
_REGION_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9_.-]*$")


@dataclass(frozen=True)
class ScopePaths:
    key: str
    name: str | None
    field_dir: str
    constant_dir: str
    system_dir: str


def scope_key(scope: ExecutionScope) -> str:
    return ROOT_SCOPE_KEY if scope.name is None else f"{_REGION_SCOPE_PREFIX}{scope.name}"


def scope_from_key(key: str) -> ExecutionScope:
    if key == ROOT_SCOPE_KEY:
        return ExecutionScope(name=None)
    if key.startswith(_REGION_SCOPE_PREFIX):
        name = key[len(_REGION_SCOPE_PREFIX):]
        if not _REGION_NAME.fullmatch(name):
            raise ValueError(f"Invalid persisted execution scope key: {key!r}")
        return ExecutionScope(name=name)
    raise ValueError(f"Unknown execution scope key: {key!r}")


def scope_paths(scope: ExecutionScope) -> ScopePaths:
    # This is namespace rendering, not a CFD topology branch. ``scope.name`` is data
    # selected by the Agent; Python only renders the corresponding OpenFOAM namespace.
    if scope.name is None:
        return ScopePaths(scope_key(scope), None, "0", "constant", "system")
    return ScopePaths(
        scope_key(scope),
        scope.name,
        f"0/{scope.name}",
        f"constant/{scope.name}",
        f"system/{scope.name}",
    )


def execution_scopes(plan_or_execution) -> list[ExecutionScope]:
    """Return normalized scopes without classifying a case as single/multi-region.

    New v5 executions carry ``scopes``. The remainder is a centralized legacy loader
    only; all downstream controller code consumes the same scope list.
    """
    execution = getattr(plan_or_execution, "execution", None)
    if execution is None and hasattr(plan_or_execution, "driver"):
        execution = plan_or_execution
    scopes = list(getattr(execution, "scopes", None) or []) if execution is not None else []
    if scopes:
        return scopes

    regions = list(getattr(execution, "regions", None) or []) if execution is not None else []
    if regions:
        return [
            ExecutionScope(
                name=item.region,
                solver_module=item.solver_module,
                solver_provider_id=item.provider_id,
            )
            for item in regions
        ]
    solver = getattr(execution, "solver_module", None) if execution is not None else None
    provider = getattr(execution, "solver_provider_id", None) if execution is not None else None
    if solver or provider:
        return [ExecutionScope(name=None, solver_module=solver, solver_provider_id=provider)]
    if execution is not None:
        return [ExecutionScope(name=None)]

    layouts = list(getattr(plan_or_execution, "region_layouts", None) or [])
    if layouts:
        return [
            ExecutionScope(
                name=(str(item.region) if str(getattr(item, "region", "") or "") else None),
                solver_module=getattr(item, "solver_module", None),
                solver_provider_id=None,
            )
            for item in layouts
        ]
    return [
        ExecutionScope(
            name=None,
            solver_module=getattr(plan_or_execution, "solver", None),
            solver_provider_id=getattr(plan_or_execution, "solver_provider_id", None),
        )
    ]


def scope_by_key(plan_or_execution) -> dict[str, ExecutionScope]:
    scopes = execution_scopes(plan_or_execution)
    result = {scope_key(scope): scope for scope in scopes}
    if len(result) != len(scopes):
        raise ValueError("Execution topology contains duplicate scope identities.")
    return result


def scope_for_native_arguments(plan_or_execution, arguments) -> ExecutionScope:
    region = native_command_region(arguments)
    scopes = execution_scopes(plan_or_execution)
    matches = [scope for scope in scopes if scope.name == region]
    if len(matches) != 1:
        target = ROOT_SCOPE_KEY if region is None else f"region:{region}"
        raise ValueError(
            f"Native invocation cannot be bound to exactly one declared execution scope: {target}"
        )
    return matches[0]


def scope_arguments_for(command: str, scope: ExecutionScope) -> list[str]:
    """Render a tool's deterministic scope selector from package-data contract."""
    contract = native_tool_contract(command)
    if scope.name is None:
        return []
    template = tuple(getattr(contract, "scope_arguments", ()) or ())
    if not template:
        raise ValueError(f"Native tool {command!r} has no named-scope invocation contract.")
    return [token.replace("{scope_name}", scope.name) for token in template]


def render_scope_template(template: str, scope: ExecutionScope) -> str:
    paths = scope_paths(scope)
    replacements = {
        "{scope_key}": paths.key,
        "{scope_name}": scope.name or "",
        "{field_dir}": paths.field_dir,
        "{constant_dir}": paths.constant_dir,
        "{system_dir}": paths.system_dir,
    }
    result = str(template)
    for key, value in replacements.items():
        result = result.replace(key, value)
    return str(PurePosixPath(result))


def _path_overlap(left: str, right: str) -> bool:
    left = str(PurePosixPath(left)).rstrip("/")
    right = str(PurePosixPath(right)).rstrip("/")
    return left == right or left.startswith(right + "/") or right.startswith(left + "/")


def _contract_arguments(command: str, scope: ExecutionScope) -> list[str]:
    contract = native_tool_contract(command)
    return [
        *scope_arguments_for(command, scope),
        *list(getattr(contract, "default_arguments", ()) or ()),
    ]


def auto_mesh_consumers(plan, available_paths) -> list[tuple[ExecutionScope, NativeOpenFOAMCommand]]:
    """Compile deterministic consumers from actual authored tool dictionaries.

    The data contract decides which utilities are safe to infer from their dictionary;
    Python does not select a meshing strategy from geometry/physics.
    """
    paths = set(str(item) for item in available_paths)
    contracts = sorted(
        (item for item in registered_contracts() if getattr(item, "controller_auto_consumer", False)),
        key=lambda item: (getattr(item, "execution_order", 500), item.command),
    )
    result: list[tuple[ExecutionScope, NativeOpenFOAMCommand]] = []
    for scope in execution_scopes(plan):
        for contract in contracts:
            args = _contract_arguments(contract.command, scope)
            required_path = required_dictionary(contract.command, args)
            if not required_path:
                continue
            if required_path not in paths:
                continue
            result.append((scope, NativeOpenFOAMCommand(
                command=contract.command,
                arguments=args,
                role="mesh",
            )))
    return result


def final_mesh_validation_commands(plan, *, only_scope_keys: set[str] | None = None) -> list[NativeOpenFOAMCommand]:
    """Compile controller finalizers uniformly over declared execution scopes."""
    finalizers = sorted(
        (item for item in registered_contracts() if item.controller_finalizer),
        key=lambda item: (getattr(item, "execution_order", 10000), item.command),
    )
    result: list[NativeOpenFOAMCommand] = []
    for scope in execution_scopes(plan):
        if only_scope_keys is not None and scope_key(scope) not in only_scope_keys:
            continue
        for contract in finalizers:
            result.append(NativeOpenFOAMCommand(
                command=contract.command,
                arguments=_contract_arguments(contract.command, scope),
                role="mesh_validation",
            ))
    return result


def affected_mesh_scope_keys(workspace, plan, changed_paths, native_invocations=()) -> set[str]:
    """Return scopes whose *observed* mesh dependency DAG intersects a delta.

    Existing native operations, not broad filename classes, define freshness. This is
    why changing ``physicalProperties`` does not stale checkMesh evidence, while a
    changed blockMeshDict or polyMesh does. New mesh invocations supplied by the delta
    additionally invalidate the scope they will mutate.
    """
    from openfoam_agent.contracts.mesh_dependencies import MeshDependencyGraph

    changed = {str(PurePosixPath(path)) for path in changed_paths}
    graph = MeshDependencyGraph(workspace)
    affected: set[str] = set()
    scopes = execution_scopes(plan)
    for scope in scopes:
        dependencies, _ = graph.dependencies(scope.name or "")
        if any(_path_overlap(path, dep) for path in changed for dep in dependencies):
            affected.add(scope_key(scope))

    for invocation in native_invocations:
        contract = native_tool_contract(invocation.command)
        if contract.effect not in {"mesh", "decomposition"}:
            continue
        region = native_command_region(invocation.arguments)
        if region is None:
            if len(scopes) == 1:
                affected.add(scope_key(scopes[0]))
            else:
                affected.update(scope_key(scope) for scope in scopes)
        else:
            matches = [scope for scope in scopes if scope.name == region]
            if len(matches) != 1:
                raise ValueError(f"Native mesh invocation targets undeclared scope region:{region}")
            affected.add(scope_key(matches[0]))
    return affected


def auto_consumers_for_delta(plan, available_paths, changed_paths) -> list[NativeOpenFOAMCommand]:
    """Return only auto-consumers whose declared mesh inputs intersect a delta."""
    changed = {str(PurePosixPath(path)) for path in changed_paths}
    result: list[NativeOpenFOAMCommand] = []
    for scope, invocation in auto_mesh_consumers(plan, available_paths):
        contract = native_tool_contract(invocation.command)
        inputs = [render_scope_template(item, scope) for item in contract.mesh_dependency_inputs]
        if any(_path_overlap(path, dep) for path in changed for dep in inputs):
            result.append(invocation)
    return result


def scope_mesh_digest(workspace, scope: ExecutionScope) -> str:
    from openfoam_agent.contracts.mesh_dependencies import MeshDependencyGraph
    return MeshDependencyGraph(workspace).digest(scope.name or "")


def current_mesh_evidence_failures(state, plan, workspace, *, max_mesh_cells: int) -> list[str]:
    """Validate mesh evidence uniformly for every declared execution scope."""
    failures: list[str] = []
    evidence_map = dict(getattr(state, "mesh_evidence_by_scope", {}) or {})
    manifest_map = dict(getattr(state, "mesh_manifest_by_scope", {}) or {})
    for scope in execution_scopes(plan):
        key = scope_key(scope)
        evidence = evidence_map.get(key)
        label = scope.name or "root"
        if evidence is None:
            failures.append(f"Passing checkMesh evidence is missing for execution scope {label}.")
            continue
        if not evidence.passed:
            failures.append(f"checkMesh evidence did not pass for execution scope {label}.")
            continue
        if evidence.cell_count is not None and evidence.cell_count > max_mesh_cells:
            failures.append(
                f"Mesh cell count {evidence.cell_count} for execution scope {label} exceeds bounded policy limit {max_mesh_cells}."
            )
        expected = scope_mesh_digest(workspace, scope)
        if manifest_map.get(key) != expected:
            failures.append(f"Mesh evidence is stale for execution scope {label}.")
    return failures


def sync_legacy_mesh_state(state) -> None:
    """Project the unified map into legacy fields without making them authority."""
    evidence_map = dict(getattr(state, "mesh_evidence_by_scope", {}) or {})
    manifest_map = dict(getattr(state, "mesh_manifest_by_scope", {}) or {})
    state.mesh_evidence = evidence_map.get(ROOT_SCOPE_KEY)
    state.region_mesh_evidence = {
        key[len(_REGION_SCOPE_PREFIX):]: value
        for key, value in evidence_map.items()
        if key.startswith(_REGION_SCOPE_PREFIX)
    }
    state.region_mesh_manifests = {
        key[len(_REGION_SCOPE_PREFIX):]: value
        for key, value in manifest_map.items()
        if key.startswith(_REGION_SCOPE_PREFIX)
    }
