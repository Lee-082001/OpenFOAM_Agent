from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path, PurePosixPath
import json
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
    scope_arguments: tuple[str, ...] = ()
    mesh_dependency_inputs: tuple[str, ...] = ()
    mesh_dependency_outputs: tuple[str, ...] = ()
    controller_auto_consumer: bool = False
    execution_order: int = 500
    default_arguments: tuple[str, ...] = ()


# Controller-owned executable contracts are auditable package data, not hidden CFD
# strategy code. They describe effects/prerequisites only; they never select a tool.
def _load_registry() -> dict[str, NativeToolContract]:
    path = Path(__file__).resolve().parents[1] / "data" / "native_tool_contracts.json"
    raw = json.loads(path.read_text(encoding="utf-8"))
    registry: dict[str, NativeToolContract] = {}
    for item in raw.get("commands", []):
        command = str(item["command"])
        if command in registry:
            raise ValueError(f"Duplicate native tool contract: {command}")
        registry[command] = NativeToolContract(
            command=command,
            effect=str(item["effect"]),
            required_dictionary=item.get("required_dictionary"),
            permitted_phases=tuple(item.get("permitted_phases", [])),
            controller_finalizer=bool(item.get("controller_finalizer", False)),
            scope_arguments=tuple(item.get("scope_arguments", [])),
            mesh_dependency_inputs=tuple(item.get("mesh_dependency_inputs", [])),
            mesh_dependency_outputs=tuple(item.get("mesh_dependency_outputs", [])),
            controller_auto_consumer=bool(item.get("controller_auto_consumer", False)),
            execution_order=int(item.get("execution_order", 500)),
            default_arguments=tuple(item.get("default_arguments", [])),
        )
    return registry


_REGISTRY = _load_registry()


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


def registered_contracts() -> tuple[NativeToolContract, ...]:
    """Return immutable controller execution contracts for generic graph compilers."""
    return tuple(_REGISTRY[name] for name in sorted(_REGISTRY))
