"""Versioned, atomic checkpoints. Unknown in-flight work is never replayed."""
from __future__ import annotations
import hashlib
import json
import os
import time
from pathlib import Path
from pydantic import BaseModel
from openfoam_agent.schemas.engineering import EngineeringPlan, ExecuteCasePlanAction, EngineeringEvent, TypedBlockMeshFile
from openfoam_agent.tools.safe_runner import SafeRunner
from .state import CFDState
from .states import State


class CheckpointError(RuntimeError):
    pass


_SNAPSHOT_FIELDS = (
    "_checkmesh_mesh_manifest", "_presolve_case_manifest", "_presolve_required_case_files",
    "_pending_execution_plan", "_draft_design_plan", "_draft_authoring_brief",
    "_pending_candidate_execution", "_pending_candidate_failed_paths", "_structured_block_mesh",
    "_phase_prompt_counts", "_phase_context_snapshots", "_evidence_gap_ledger",
    "_retrieval_cycles", "_evidence_retrieval_disabled", "_active_transaction",
)
_MODELS = {"_pending_execution_plan": EngineeringPlan, "_draft_design_plan": EngineeringPlan,
           "_pending_candidate_execution": ExecuteCasePlanAction, "_structured_block_mesh": TypedBlockMeshFile}


def _jsonable(value):
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(x) for x in (sorted(value, key=str) if isinstance(value, set) else value)]
    return value


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


def environment_fingerprint(tools):
    snapshot = tools.environment_snapshot()
    runner = getattr(tools, "runner", None)
    if isinstance(runner, SafeRunner):
        snapshot = {"installation": snapshot, "native_environment": runner.sanitized_environment(),
                    "executable_metadata": []}
        for item in runner.installation.executables:
            try:
                path = runner.resolve_trusted_executable(item.name)
                stat = path.stat()
                snapshot["executable_metadata"].append([item.name, stat.st_size, stat.st_mtime_ns])
            except (OSError, ValueError, RuntimeError):
                snapshot["executable_metadata"].append([item.name, "unavailable"])
    return _digest(snapshot)


class CheckpointStore:
    SCHEMA_VERSION = 1

    def __init__(self, agent):
        self.agent = agent
        self.path = agent.workspace.root / "checkpoint.json"

    def save(self, state, *, reason):
        workspace = self.agent.workspace
        snapshot = {name: _jsonable(getattr(self.agent, name, None)) for name in _SNAPSHOT_FIELDS}
        state.engineering_checkpoint = snapshot
        runner = getattr(self.agent.tools, "runner", None)
        if isinstance(runner, SafeRunner):
            state.native_process_records = _jsonable(runner.budget.records)
        payload = {"schema_version": self.SCHEMA_VERSION, "reason": reason, "saved_unix": time.time(),
                   "workspace": str(workspace.root), "environment_sha256": environment_fingerprint(self.agent.tools),
                   "inputs": [x.model_dump(mode="json") for x in workspace.execution_file_seals()],
                   "state": state.model_dump(mode="json")}
        envelope = {"payload": payload, "sha256": _digest(payload)}
        temporary = self.path.with_suffix(".tmp")
        if self.path.is_symlink() or temporary.is_symlink():
            raise CheckpointError("Checkpoint path cannot be a symlink.")
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(envelope, stream, ensure_ascii=False, sort_keys=True)
            stream.flush(); os.fsync(stream.fileno())
        os.chmod(temporary, 0o600)
        os.replace(temporary, self.path)
        if os.name == "posix":
            fd = os.open(workspace.root, os.O_RDONLY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
        return self.path

    def load(self):
        if self.path.is_symlink() or not self.path.is_file():
            raise CheckpointError("A regular checkpoint.json is required.")
        if self.path.stat().st_size > 64_000_000:
            raise CheckpointError("Checkpoint exceeds the bounded state size limit.")
        envelope = json.loads(self.path.read_text(encoding="utf-8"))
        payload = envelope.get("payload", {})
        if envelope.get("sha256") != _digest(payload):
            raise CheckpointError("Checkpoint checksum mismatch.")
        if payload.get("schema_version") != self.SCHEMA_VERSION:
            raise CheckpointError("Unsupported checkpoint version; implicit migration is not allowed.")
        workspace = self.agent.workspace
        if payload.get("workspace") != str(workspace.root):
            raise CheckpointError("Checkpoint relocation requires an explicit fork; paths will not be guessed.")
        if payload.get("environment_sha256") != environment_fingerprint(self.agent.tools):
            raise CheckpointError("Native environment changed; evidence must be revalidated before resume.")
        state = CFDState.model_validate(payload["state"])
        if state.intake_confirmed:
            state.assert_confirmed_intake()
        workspace._authored_paths = {x["path"] for x in payload["inputs"] if x["origin"] == "agent"}
        workspace._asset_paths = {x["path"] for x in payload["inputs"] if x["origin"] == "user_asset"}
        current = [x.model_dump(mode="json") for x in workspace.execution_file_seals()]
        if current != payload["inputs"]:
            raise CheckpointError("Case inputs changed since the checkpoint; no action was replayed.")
        snapshot = state.engineering_checkpoint
        if snapshot.get("_active_transaction"):
            raise CheckpointError("Interrupted multi-action transaction needs explicit reconciliation; no automatic replay.")
        if state.pending_action and state.pending_action.get("status") != "completed":
            raise CheckpointError("An action has uncertain completion; partial work is preserved, not replayed.")
        runner = getattr(self.agent.tools, "runner", None)
        records = runner.budget.records if isinstance(runner, SafeRunner) else state.native_process_records
        if any(row.get("status") in {"running", "spawn_intent"} for row in records):
            raise CheckpointError("A recorded native process may be in flight; manual reconciliation required. No stale PID was killed.")
        if isinstance(runner, SafeRunner) and records != state.native_process_records:
            raise CheckpointError("Process ledger advanced beyond the checkpoint; no process was replayed.")
        if state.pending_action and state.pending_action.get("event"):
            event = EngineeringEvent.model_validate(state.pending_action["event"])
            if event not in state.engineering_events:
                state.engineering_events.append(event)
        state.pending_action = None
        for name in _SNAPSHOT_FIELDS:
            value = snapshot.get(name)
            if value is not None and name in _MODELS:
                value = _MODELS[name].model_validate(value)
            if name in {"_presolve_required_case_files", "_pending_candidate_failed_paths"} and value is not None:
                value = tuple(value)
            if value is not None:
                setattr(self.agent, name, value)
        if state.case_seal is not None and state.engineering_plan is not None:
            workspace.verify_seal(state.case_seal, state.engineering_plan)
        # Fresh process environment is a new execution session: never silently
        # carry a /solve approval into a resumed run.
        state.solve_approved = False
        state.execution_approval = None
        if state.current_state == State.SIMULATION:
            state.transition(State.SOLVE_READY, "Checkpoint restored; fresh /solve approval is required.")
        elif state.current_state == State.RUNTIME_REPAIR:
            state.transition(State.ENGINEERING_REVIEW_REQUIRED, "Interrupted runtime repair needs review.")
        self.agent._resume_pending = True
        return state
