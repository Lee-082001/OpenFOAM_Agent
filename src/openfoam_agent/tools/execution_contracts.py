from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Literal


ScopeNamespacePolicy = Literal["root_only", "named_only", "any"]
ScopeSolverBindingPolicy = Literal["required", "optional", "forbidden"]
ControlBindingMode = Literal["none", "scalar", "named_map"]
PlanSolverMirror = Literal["driver", "scope_solver"]


@dataclass(frozen=True)
class ExecutionDriverContract:
    driver: str
    provider_ids: tuple[str, ...]
    min_scopes: int
    max_scopes: int | None
    scope_namespace: ScopeNamespacePolicy
    scope_solver_binding: ScopeSolverBindingPolicy
    control_binding_mode: ControlBindingMode
    control_binding_entry: str | None
    runtime_arguments: tuple[str, ...]
    plan_solver_mirror: PlanSolverMirror


def _data_path() -> Path:
    return Path(__file__).resolve().parents[1] / "data" / "execution_driver_contracts.json"


def _parse_contract(item: dict, *, default_driver: str = "*") -> ExecutionDriverContract:
    cardinality = dict(item.get("scope_cardinality") or {})
    binding = dict(item.get("control_dict_binding") or {})
    max_scopes = cardinality.get("max")
    return ExecutionDriverContract(
        driver=str(item.get("driver") or default_driver),
        provider_ids=tuple(str(x) for x in item.get("provider_ids", []) if str(x)),
        min_scopes=int(cardinality.get("min", 1)),
        max_scopes=(None if max_scopes is None else int(max_scopes)),
        scope_namespace=str(item.get("scope_namespace", "any")),
        scope_solver_binding=str(item.get("scope_solver_binding", "optional")),
        control_binding_mode=str(binding.get("mode", "none")),
        control_binding_entry=(str(binding["entry"]) if binding.get("entry") else None),
        runtime_arguments=tuple(str(x) for x in item.get("runtime_arguments", [])),
        plan_solver_mirror=str(item.get("plan_solver_mirror", "driver")),
    )


def _load() -> tuple[ExecutionDriverContract, tuple[ExecutionDriverContract, ...]]:
    raw = json.loads(_data_path().read_text(encoding="utf-8"))
    if raw.get("schema_version") != "1.0":
        raise ValueError("Unsupported execution driver contract schema version.")
    default = _parse_contract(dict(raw.get("default_direct_application") or {}))
    drivers = tuple(_parse_contract(dict(item)) for item in raw.get("drivers", []))
    names = [item.driver for item in drivers]
    if len(names) != len(set(names)):
        raise ValueError("Duplicate execution driver contract.")
    return default, drivers


_DEFAULT_CONTRACT, _DRIVER_CONTRACTS = _load()


def execution_driver_contract(driver: str, provider_id: str | None = None) -> ExecutionDriverContract:
    """Resolve execution syntax/shape metadata without choosing a driver.

    The Agent has already selected ``driver`` and ``provider_id``. This function only
    returns the deterministic execution contract associated with that selection.
    Unknown/direct solver applications use the generic single-root direct contract.
    """

    provider_id = str(provider_id or "")
    by_provider = [item for item in _DRIVER_CONTRACTS if provider_id and provider_id in item.provider_ids]
    if len(by_provider) > 1:
        raise ValueError(f"Execution provider {provider_id!r} maps to multiple contracts.")
    if by_provider:
        contract = by_provider[0]
        if contract.driver != driver:
            raise ValueError(
                f"Execution driver {driver!r} disagrees with provider contract {contract.driver!r}."
            )
        return contract
    by_name = [item for item in _DRIVER_CONTRACTS if item.driver == driver]
    if len(by_name) > 1:
        raise ValueError(f"Execution driver {driver!r} maps to multiple contracts.")
    if by_name:
        return by_name[0]
    return ExecutionDriverContract(
        driver=driver,
        provider_ids=(),
        min_scopes=_DEFAULT_CONTRACT.min_scopes,
        max_scopes=_DEFAULT_CONTRACT.max_scopes,
        scope_namespace=_DEFAULT_CONTRACT.scope_namespace,
        scope_solver_binding=_DEFAULT_CONTRACT.scope_solver_binding,
        control_binding_mode=_DEFAULT_CONTRACT.control_binding_mode,
        control_binding_entry=_DEFAULT_CONTRACT.control_binding_entry,
        runtime_arguments=_DEFAULT_CONTRACT.runtime_arguments,
        plan_solver_mirror=_DEFAULT_CONTRACT.plan_solver_mirror,
    )


def _scope_name(scope) -> str | None:
    name = getattr(scope, "name", None)
    if name is None:
        return None
    text = str(name).strip()
    return text or None


def validate_execution_contract(execution) -> list[str]:
    contract = execution_driver_contract(execution.driver, execution.driver_provider_id)
    scopes = list(getattr(execution, "scopes", None) or [])
    failures: list[str] = []
    count = len(scopes)
    if count < contract.min_scopes:
        failures.append(
            f"Execution provider {execution.driver_provider_id} requires at least {contract.min_scopes} scope(s); observed {count}."
        )
    if contract.max_scopes is not None and count > contract.max_scopes:
        failures.append(
            f"Execution provider {execution.driver_provider_id} permits at most {contract.max_scopes} scope(s); observed {count}."
        )
    names = [_scope_name(scope) for scope in scopes]
    if contract.scope_namespace == "root_only" and any(name is not None for name in names):
        failures.append(f"Execution provider {execution.driver_provider_id} accepts only the root case scope.")
    if contract.scope_namespace == "named_only" and any(name is None for name in names):
        failures.append(f"Execution provider {execution.driver_provider_id} requires named case scopes.")

    for scope in scopes:
        solver = getattr(scope, "solver_module", None)
        provider = getattr(scope, "solver_provider_id", None)
        has_solver = bool(solver or provider)
        complete = bool(solver and provider)
        if has_solver and not complete:
            failures.append("Execution scope solver_module and solver_provider_id must be supplied together.")
        if contract.scope_solver_binding == "required" and not complete:
            failures.append(
                f"Execution provider {execution.driver_provider_id} requires a solver binding for every scope."
            )
        if contract.scope_solver_binding == "forbidden" and has_solver:
            failures.append(
                f"Execution provider {execution.driver_provider_id} is a direct solver application and cannot carry a nested scope solver binding."
            )
    return list(dict.fromkeys(failures))


def _render_argument(token: str, scope) -> str:
    if token == "{scope_solver}":
        solver = getattr(scope, "solver_module", None)
        if not solver:
            raise ValueError("Execution contract requires a scope solver but none is bound.")
        return str(solver)
    return token


def runtime_arguments(execution) -> list[str]:
    contract = execution_driver_contract(execution.driver, execution.driver_provider_id)
    scopes = list(getattr(execution, "scopes", None) or [])
    if validate_execution_contract(execution):
        raise ValueError("Execution topology does not satisfy the selected driver contract.")
    scope = scopes[0] if scopes else None
    prefix = [_render_argument(token, scope) for token in contract.runtime_arguments]
    return [*prefix, *list(execution.arguments)]


def sealed_solver_mirror(execution) -> tuple[str, str]:
    contract = execution_driver_contract(execution.driver, execution.driver_provider_id)
    if contract.plan_solver_mirror == "driver":
        return execution.driver, execution.driver_provider_id
    scopes = list(getattr(execution, "scopes", None) or [])
    if not scopes:
        raise ValueError("Execution driver contract requires a scope solver mirror but no scope exists.")
    scope = scopes[0]
    solver = getattr(scope, "solver_module", None)
    provider = getattr(scope, "solver_provider_id", None)
    if not solver or not provider:
        raise ValueError("Execution driver contract requires a complete scope solver mirror.")
    return str(solver), str(provider)


def provider_requirements(execution) -> list[tuple[str, str, str]]:
    """Return Agent-selected provider requirements as (id, expected name, label)."""
    requirements = [(execution.driver_provider_id, execution.driver, "execution driver")]
    for scope in list(getattr(execution, "scopes", None) or []):
        solver = getattr(scope, "solver_module", None)
        provider = getattr(scope, "solver_provider_id", None)
        if solver and provider:
            scope_name = _scope_name(scope) or "root"
            requirements.append((str(provider), str(solver), f"scope solver {scope_name}"))
    return requirements


def control_dict_binding(execution) -> tuple[str, str | None, object]:
    """Return (mode, entry, expected) for deterministic controlDict validation."""
    contract = execution_driver_contract(execution.driver, execution.driver_provider_id)
    entry = contract.control_binding_entry
    scopes = list(getattr(execution, "scopes", None) or [])
    if contract.control_binding_mode == "none":
        return "none", entry, None
    if contract.control_binding_mode == "scalar":
        if len(scopes) != 1:
            raise ValueError("Scalar execution binding requires exactly one scope.")
        return "scalar", entry, getattr(scopes[0], "solver_module", None)
    if contract.control_binding_mode == "named_map":
        expected = {}
        for scope in scopes:
            name = _scope_name(scope)
            if name is None:
                raise ValueError("Named-map execution binding requires named scopes.")
            expected[name] = getattr(scope, "solver_module", None)
        return "named_map", entry, expected
    raise ValueError(f"Unsupported control binding mode: {contract.control_binding_mode}")


def contract_summary(driver: str, provider_id: str | None = None) -> dict[str, object]:
    contract = execution_driver_contract(driver, provider_id)
    return {
        "driver": contract.driver,
        "min_scopes": contract.min_scopes,
        "max_scopes": contract.max_scopes,
        "scope_namespace": contract.scope_namespace,
        "scope_solver_binding": contract.scope_solver_binding,
        "control_binding": {
            "mode": contract.control_binding_mode,
            "entry": contract.control_binding_entry,
        },
        "plan_solver_mirror": contract.plan_solver_mirror,
    }
