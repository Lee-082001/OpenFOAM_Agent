from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from openfoam_agent.contracts.execution import ExecutionApproval
from openfoam_agent.contracts.models import ResourceLimits
from openfoam_agent.tools.native_contracts import native_tool_contract, registered_effects


class ExecutionPolicyError(RuntimeError):
    pass


class ProcessBudgetExceeded(ExecutionPolicyError):
    pass


# Trusted application code owns effects through the central NativeToolContract registry.
COMMAND_EFFECTS: dict[str, str] = registered_effects()


def command_effect(command: str, arguments: list[str], installation: Any = None) -> str:
    if arguments and all(x in {"-help", "-help-full", "--help", "-version"} for x in arguments):
        return "query"
    if command == "foamListTimes" and "-rm" in arguments:
        return "destructive"
    if command == "decomposePar" and "-force" in arguments:
        return "destructive"
    if command == "foamDictionary" and any(x in {"-set", "-remove", "-add"} for x in arguments):
        return "write"
    if command in COMMAND_EFFECTS:
        return COMMAND_EFFECTS[command]
    if installation is not None:
        for item in installation.executables:
            if item.name == command and item.category in {"execution_driver", "solver_application"}:
                return "solve"
    return "unknown"


def deny_unapproved_engineering_solve(command: str, arguments: list[str], installation: Any = None) -> None:
    effect = command_effect(command, arguments, installation)
    if effect in {"solve", "destructive", "unknown", "reconstruction"}:
        raise ExecutionPolicyError(
            f"Native {command} has effect={effect}; engineering prepare/repair cannot authorize it. "
            "Main calculation requires the approved runtime execution path."
        )


@dataclass
class ProcessBudget:
    limit: int = 40
    records: list[dict[str, Any]] = field(default_factory=list)
    ledger_path: Path | None = None
    _lock: threading.RLock = field(default_factory=threading.RLock, repr=False)

    def __post_init__(self):
        if self.ledger_path is not None and self.ledger_path.exists():
            if self.ledger_path.is_symlink():
                raise ExecutionPolicyError("Process ledger may not be a symlink.")
            payload = json.loads(self.ledger_path.read_text(encoding="utf-8"))
            rows = payload.get("records")
            if not isinstance(rows, list) or [x.get("id") for x in rows] != list(range(1, len(rows)+1)):
                raise ExecutionPolicyError("Process ledger is malformed; budget will not be reset.")
            self.records = rows

    @property
    def used(self) -> int:
        return sum(x.get("pid") is not None for x in self.records)

    @property
    def attempts(self) -> int:
        return len(self.records)

    def _persist(self) -> None:
        if self.ledger_path is None:
            return
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.ledger_path.with_suffix(".tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump({"limit": self.limit, "records": self.records}, handle, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o600)
        os.replace(temporary, self.ledger_path)

    def reserve(self, command: list[str], effect: str) -> dict[str, Any]:
        with self._lock:
            if self.attempts >= self.limit:
                raise ProcessBudgetExceeded(f"Actual subprocess budget exhausted ({self.attempts}/{self.limit}); no process created.")
            row: dict[str, Any] = {"id": self.attempts + 1, "command": command, "effect": effect,
                                   "status": "spawn_intent", "pid": None, "started_unix": time.time()}
            self.records.append(row)
            self._persist()
            return row

    def update(self, row: dict[str, Any], **updates: Any) -> None:
        with self._lock:
            row.update(updates)
            self._persist()



@dataclass
class ValidationExecutionContext:
    """Exact, serial, short-lived pre-approval consumer validation context."""

    expected_command: list[str]
    case_dir: Path
    workspace_root: Path
    max_wall_seconds: int = 60

    def validate(self, command: list[str], cwd: Path, timeout: int, ranks: int = 1) -> None:
        expected_dir = self.case_dir.resolve()
        root = self.workspace_root.resolve()
        if cwd.resolve() != expected_dir:
            raise ExecutionPolicyError("Validation execution cwd differs from the Python-owned shadow case.")
        if expected_dir == root or root not in expected_dir.parents:
            raise ExecutionPolicyError("Validation execution must stay inside the bounded workspace.")
        if ranks != 1:
            raise ExecutionPolicyError("Zero-step consumer validation is serial only.")
        if timeout > self.max_wall_seconds:
            raise ExecutionPolicyError("Validation execution exceeds the bounded validation wall time.")
        if command != self.expected_command:
            raise ExecutionPolicyError("Validation execution argv differs from the Python-owned exact command.")

@dataclass
class ExecutionContext:
    approval: ExecutionApproval
    plan: Any
    seal: Any
    workspace: Any
    phase: str = "runtime"

    def validate(self, command: list[str], cwd: Path, timeout: int, ranks: int = 1) -> None:
        if self.phase != "runtime":
            raise ExecutionPolicyError("Calculation is only permitted in the runtime phase.")
        try:
            self.approval.check_plan(self.plan)
            self.approval.check_repair_files(self.seal)
            self.workspace.verify_seal(self.seal, self.plan)
        except (ValueError, RuntimeError) as exc:
            raise ExecutionPolicyError(str(exc)) from exc
        if cwd != self.workspace.case_dir.resolve():
            raise ExecutionPolicyError("Execution cwd differs from the approved sealed case.")
        limits = self.approval.resource_limits
        if timeout > limits.wall_seconds or ranks > limits.max_ranks:
            raise ExecutionPolicyError("Execution exceeds approved wall-time/rank limits.")
        execution = self.approval.execution
        expected = [str(execution["driver"])]
        if execution["driver"] == "foamRun":
            expected += ["-solver", str(execution["solver_module"])]
        expected += execution.get("arguments", [])
        parallel = execution.get("parallel", {})
        if ranks != parallel.get("ranks", 1):
            raise ExecutionPolicyError("MPI rank count differs from approved execution.")
        normalized = [command[0]]
        i = 1
        while i < len(command):
            if command[i] == "-case":
                if i + 1 >= len(command) or Path(command[i + 1]).resolve() != cwd:
                    raise ExecutionPolicyError("Case override differs from the approved cwd.")
                i += 2
            elif command[i] == "-parallel" and ranks > 1:
                i += 1
            else:
                normalized.append(command[i]); i += 1
        if normalized != expected:
            raise ExecutionPolicyError("Actual argv differs from the user-approved execution specification.")
