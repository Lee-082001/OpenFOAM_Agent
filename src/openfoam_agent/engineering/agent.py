from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

from openfoam_agent.agents.intake import confirmed_intake_definition
from openfoam_agent.llm.context import (
    build_bounded_json_prompt,
    compact_event_for_model,
    structured_request_metrics,
)
from openfoam_agent.llm.prompts import (
    ENGINEERING_SYSTEM_PROMPT,
    PREPARE_SYSTEM_PROMPT,
    PREPARE_DECISION_ONLY_SYSTEM_PROMPT,
    CASE_PLAN_RETRY_SYSTEM_PROMPT,
    REPAIR_SYSTEM_PROMPT,
    REVISION_SYSTEM_PROMPT,
    RUNTIME_REPAIR_SYSTEM_PROMPT,
    STRATEGY_REVISION_SYSTEM_PROMPT,
    FINALIZATION_SYSTEM_PROMPT,
)
from openfoam_agent.llm.protocol import StructuredLLM
from openfoam_agent.progress import (
    NullProgressReporter,
    ProgressEvent,
    ProgressReporter,
    action_importance,
    describe_action,
)
from openfoam_agent.schemas.feedback import RevisionFileChange, RevisionRecord
from openfoam_agent.schemas.engineering import (
    BlockAction,
    CaseBundleFile,
    DeleteCaseFileAction,
    EngineeringBudgetExtension,
    EngineeringEvent,
    EngineeringPlan,
    EngineeringSequenceAction,
    ExecuteCasePlanAction,
    FinalizationTurn,
    PrepareTurn,
    PrepareDecisionOnlyTurn,
    CasePlanRetryTurn,
    CandidateCasePlanRepairAction,
    GatherEvidenceAction,
    RepairCasePlanAction,
    RuntimeCaseRepairAction,
    RepairTurn,
    RevisionTurn,
    RuntimeRepairTurn,
    StrategyRevisionAction,
    StrategyRevisionTurn,
    EngineeringTurn,
    ObservedEngineeringEvidence,
    canonical_engineering_evidence_id,
    FinishPreviewAction,
    InspectEnvironmentAction,
    ListCaseFilesAction,
    ReadCaseFileAction,
    ReadReferenceAction,
    PatchCaseFileAction,
    RetrySolverAction,
    RunMeshCommandAction,
    SearchCapabilitiesAction,
    SearchReferencesAction,
    SurfaceCheckAction,
    ValidateDictionaryAction,
    ValidatePreSolveAction,
    WriteCaseFileAction,
)
from openfoam_agent.tools.capability_catalog import CapabilityCatalog
from openfoam_agent.tools.diagnostics import diagnose_openfoam_failure
from openfoam_agent.tools.foam_file import validate_foam_file_header
from openfoam_agent.tools.openfoam import OpenFOAMTools
from openfoam_agent.tools.foam_serializer import (
    FoamSerializationError,
    serialize_block_mesh,
    serialize_foam_dictionary,
)
from openfoam_agent.tools.references import OpenFOAMReferenceIndex
from openfoam_agent.tools.workspace import CaseWorkspace, WorkspaceSafetyError
from openfoam_agent.verification.presolve import PreSolveCompletenessGate
from openfoam_agent.verification.safety import (
    DeterministicSafetyGate,
    parse_check_mesh_evidence,
)
from openfoam_agent.workflow.state import CFDState
from openfoam_agent.workflow.states import State
from openfoam_agent.schemas.simulation import RuntimeRepairDecision


@dataclass
class EngineeringPolicy:
    # Soft preparation budget. Reaching this boundary does not automatically
    # terminate the run: deterministic progress evidence can extend the window.
    max_agent_steps: int = 20
    hard_max_agent_steps: int = 40
    step_extension: int = 10
    progress_window: int = 8

    # Final plan submission is deliberately separated from tool work so a
    # successful checkMesh at a budget boundary cannot dead-end the run.
    max_finalization_steps: int = 3

    # Resource budgets are independent from LLM-turn budgets. Python only
    # bounds execution/retry cost; it does not make CFD design decisions.
    max_native_commands: int = 40
    max_mesh_repair_cycles: int = 6
    max_runtime_repair_steps: int = 10

    # v2.9: LLM-turn budgets and deterministic action budgets are separate. A single
    # LLM turn may authorize a full execution plan or a short bounded sequence.
    max_tool_actions: int = 160
    max_runtime_repair_tool_actions: int = 48

    # Complete-plan authoring failures happen before any case mutation. Keep this
    # retry class separately bounded so a model cannot burn the full Engineering
    # budget repeatedly regenerating an unsafe/unserializable bundle.
    max_case_plan_authoring_retries: int = 3

    # v2.13: retrieval is driven by explicit evidence gaps rather than free-form
    # repeated search turns. These are hard fuses, not the normal stopping rule;
    # novelty/stagnation is tracked per gap.
    max_prepare_retrieval_cycles: int = 3
    max_runtime_retrieval_cycles: int = 2

    observation_history: int = 12
    max_observation_chars: int = 12_000
    model_event_excerpt_chars: int = 2_500
    max_model_prompt_chars: int = 60_000
    max_model_feedback_items: int = 8
    max_mesh_cells: int = 5_000_000
    require_solve_ready_gate: bool = False

    # v2.9: when the capability graph is small, preload deterministic provider
    # evidence into the first engineering prompt so solver selection does not
    # require an extra LLM -> search_capabilities -> LLM round trip.
    preload_capabilities: bool = False
    max_preloaded_capabilities: int = 24

    # v2.10 token controls. Kept opt-in at the library-policy level for API
    # compatibility; the production CLI enables both.
    compact_phase_schemas: bool = False
    state_delta_context: bool = False

    def __post_init__(self) -> None:
        integer_fields = {
            "max_agent_steps": self.max_agent_steps,
            "hard_max_agent_steps": self.hard_max_agent_steps,
            "step_extension": self.step_extension,
            "progress_window": self.progress_window,
            "max_finalization_steps": self.max_finalization_steps,
            "max_native_commands": self.max_native_commands,
            "max_mesh_repair_cycles": self.max_mesh_repair_cycles,
            "max_runtime_repair_steps": self.max_runtime_repair_steps,
            "max_tool_actions": self.max_tool_actions,
            "max_runtime_repair_tool_actions": self.max_runtime_repair_tool_actions,
            "max_case_plan_authoring_retries": self.max_case_plan_authoring_retries,
            "max_prepare_retrieval_cycles": self.max_prepare_retrieval_cycles,
            "max_runtime_retrieval_cycles": self.max_runtime_retrieval_cycles,
            "observation_history": self.observation_history,
            "max_observation_chars": self.max_observation_chars,
            "model_event_excerpt_chars": self.model_event_excerpt_chars,
            "max_model_prompt_chars": self.max_model_prompt_chars,
            "max_model_feedback_items": self.max_model_feedback_items,
            "max_mesh_cells": self.max_mesh_cells,
            "max_preloaded_capabilities": self.max_preloaded_capabilities,
        }
        for name, value in integer_fields.items():
            if value < 1:
                raise ValueError(f"{name} must be >= 1")
        if self.hard_max_agent_steps < self.max_agent_steps:
            raise ValueError("hard_max_agent_steps must be >= max_agent_steps")


@dataclass
class RepairOutcome:
    decision: RuntimeRepairDecision
    plan: EngineeringPlan | None = None
    reason: str = ""

    @property
    def retry(self) -> bool:
        """Backward-compatible convenience for the runtime orchestrator/tests."""

        return self.decision == RuntimeRepairDecision.RETRY_SOLVER


_MESH_TOPOLOGY_MUTATING_COMMANDS = frozenset({"blockMesh", "snappyHexMesh", "createPatch"})


class CFDEngineeringAgent:
    """Single agent that owns CFD design, implementation and failure repair."""

    def __init__(
        self,
        llm: StructuredLLM,
        *,
        workspace: str | Path,
        capability_db: str | Path,
        tools: OpenFOAMTools | None = None,
        policy: EngineeringPolicy | None = None,
        progress: ProgressReporter | None = None,
    ) -> None:
        self.llm = llm
        self.workspace = CaseWorkspace(workspace)
        self.tools = tools or OpenFOAMTools.for_workspace(self.workspace.root)
        self.catalog = CapabilityCatalog(capability_db)
        self.references = OpenFOAMReferenceIndex()
        self.safety = DeterministicSafetyGate(self.tools, self.workspace)
        self.presolve = PreSolveCompletenessGate(self.tools, self.workspace)
        self.policy = policy or EngineeringPolicy()
        self.progress = progress or NullProgressReporter()
        self._checkmesh_mesh_manifest: str | None = None
        self._presolve_case_manifest: str | None = None
        self._presolve_required_case_files: tuple[str, ...] | None = None
        self._pending_execution_plan: EngineeringPlan | None = None
        self._pending_candidate_execution: ExecuteCasePlanAction | None = None
        self._pending_candidate_failed_paths: tuple[str, ...] = ()
        self._phase_prompt_counts: dict[str, int] = {}
        self._phase_context_snapshots: dict[str, dict[str, str | None]] = {}
        self._evidence_gap_ledger: dict[str, dict[str, dict[str, object]]] = {}
        self._retrieval_cycles: dict[str, int] = {}

    def prepare(self, state: CFDState, *, native_execution: bool = True) -> CFDState:
        state.assert_confirmed_intake()
        self._evidence_gap_ledger["prepare"] = {}
        self._retrieval_cycles["prepare"] = 0
        if native_execution and not self._checkmesh_preflight(state, phase="preflight"):
            return state
        state.engineering_round_start_index = len(state.engineering_events)
        state.transition(State.ENGINEERING, "Confirmed CFD definition handed to CFDEngineeringAgent.")
        self.progress.emit(
            ProgressEvent(
                phase="engineering",
                message="확정된 CFD 정의로 autonomous engineering 시작",
                status="start",
                metrics={
                    "llmSoftBudget": self.policy.max_agent_steps,
                    "llmHardCap": self.policy.hard_max_agent_steps,
                    "toolBudget": self.policy.max_tool_actions,
                    "nativeBudget": self.policy.max_native_commands,
                },
            )
        )

        current_limit = min(self.policy.max_agent_steps, self.policy.hard_max_agent_steps)
        step = 1
        while True:
            while step <= current_limit:
                turn = self._generate_turn(
                    state,
                    step=step,
                    local_step=step,
                    current_step_limit=current_limit,
                    phase="prepare",
                    native_execution=native_execution,
                )
                terminal = self._execute_prepare_decision(
                    state,
                    turn.action,
                    llm_step=step,
                    progress_phase="engineering",
                    progress_step=step,
                    progress_limit=current_limit,
                    native_execution=native_execution,
                )
                if terminal:
                    return state
                step += 1

            # If the case is already validated, stop tool work and provide a
            # dedicated bounded finalization window before considering extension.
            if self._ready_for_finalization(state, native_execution=native_execution):
                return self._run_finalization_window(
                    state,
                    native_execution=native_execution,
                    start_step=step,
                )

            if current_limit >= self.policy.hard_max_agent_steps:
                break

            progress, progress_reason = self._progress_allows_extension(state)
            if not progress:
                state.transition(
                    State.ENGINEERING_BLOCKED,
                    "Engineering soft budget reached without new deterministic progress "
                    f"evidence in the last {self.policy.progress_window} steps: {progress_reason}",
                )
                return state

            previous_limit = current_limit
            current_limit = min(
                self.policy.hard_max_agent_steps,
                current_limit + self.policy.step_extension,
            )
            state.engineering_budget_extensions.append(
                EngineeringBudgetExtension(
                    boundary_step=previous_limit,
                    previous_limit=previous_limit,
                    new_limit=current_limit,
                    reason=progress_reason,
                )
            )
            self.progress.emit(
                ProgressEvent(
                    phase="engineering",
                    message="deterministic progress evidence 확인; engineering budget 연장",
                    status="info",
                    metrics={"from": previous_limit, "to": current_limit},
                )
            )

        state.transition(
            State.ENGINEERING_BLOCKED,
            "Engineering hard step budget exhausted "
            f"({self.policy.hard_max_agent_steps}) before a current validated case was ready for finalization.",
        )
        return state

    def revise_from_feedback(self, state: CFDState, *, native_execution: bool = True) -> CFDState:
        state.assert_confirmed_intake()
        proposal = state.active_revision_proposal
        if state.current_state != State.REVISION_READY or proposal is None:
            raise ValueError("Human-feedback revision requires REVISION_READY with an active proposal.")
        if proposal.requires_intake_revision:
            raise ValueError("Confirmed user facts must be revised through intake before case engineering.")
        if state.engineering_plan is None or state.case_seal is None:
            raise ValueError("Human-feedback revision requires an existing sealed case and plan.")
        if state.engineering_plan.digest() != proposal.baseline_plan_sha256:
            raise WorkspaceSafetyError("Engineering plan changed after the human revision proposal was created.")
        if state.case_seal.manifest_sha256 != proposal.baseline_manifest_sha256:
            raise WorkspaceSafetyError("Case seal changed after the human revision proposal was created.")

        if native_execution and not self._checkmesh_preflight(state, phase="preflight"):
            return state

        self.workspace.adopt_seal(state.case_seal)
        self.safety.verify_seal(state.engineering_plan, state.case_seal)
        if state.mesh_evidence is not None and state.mesh_evidence.passed:
            self._checkmesh_mesh_manifest = self.workspace.mesh_manifest_digest()

        revision_id = f"rev-{len(state.revision_history) + 1:04d}"
        state.pending_revision_archive_path = self.workspace.archive_revision_outputs(revision_id)
        self.safety.verify_seal(state.engineering_plan, state.case_seal)

        for feedback in state.human_feedback:
            if feedback.feedback_id in proposal.feedback_ids:
                feedback.status = "revision_in_progress"
        state.engineering_round_start_index = len(state.engineering_events)

        # Prior numerical evidence remains auditable through feedback/revision history,
        # but it must never appear as evidence for the newly revised, unsolved case.
        state.solve_approved = False
        state.simulation = None
        state.runtime_report = None
        state.simulation_attempts = 0
        state.last_runtime_log_excerpt = None
        state.postprocessing_events = []
        state.force_coefficient_analysis = None
        state.postprocessing_report = None
        state.transition(
            State.ENGINEERING,
            f"User confirmed revision proposal {proposal.proposal_id}; sealed case handed back to CFDEngineeringAgent.",
        )
        self.progress.emit(
            ProgressEvent(
                phase="revision",
                message=f"human-feedback revision 시작: {proposal.proposal_id}",
                status="start",
                metrics={
                    "llmSoftBudget": self.policy.max_agent_steps,
                    "llmHardCap": self.policy.hard_max_agent_steps,
                    "toolBudget": self.policy.max_tool_actions,
                },
            )
        )

        current_limit = min(self.policy.max_agent_steps, self.policy.hard_max_agent_steps)
        local_step = 1
        base_step = len(state.engineering_events)
        while True:
            while local_step <= current_limit:
                global_step = base_step + local_step
                turn = self._generate_turn(
                    state,
                    step=global_step,
                    local_step=local_step,
                    current_step_limit=current_limit,
                    phase="human_revision",
                    native_execution=native_execution,
                )
                terminal = self._execute_prepare_decision(
                    state,
                    turn.action,
                    llm_step=global_step,
                    progress_phase="revision",
                    progress_step=local_step,
                    progress_limit=current_limit,
                    native_execution=native_execution,
                )
                if terminal:
                    return state
                local_step += 1

            if self._ready_for_finalization(state, native_execution=native_execution):
                return self._run_finalization_window(
                    state,
                    native_execution=native_execution,
                    start_step=base_step + local_step,
                    phase="human_revision_finalize",
                )

            if current_limit >= self.policy.hard_max_agent_steps:
                break
            progress, progress_reason = self._progress_allows_extension(state)
            if not progress:
                state.transition(
                    State.ENGINEERING_BLOCKED,
                    "Human-feedback revision soft budget reached without new deterministic progress "
                    f"evidence in the last {self.policy.progress_window} steps: {progress_reason}",
                )
                return state
            previous_limit = current_limit
            current_limit = min(
                self.policy.hard_max_agent_steps,
                current_limit + self.policy.step_extension,
            )
            state.engineering_budget_extensions.append(
                EngineeringBudgetExtension(
                    boundary_step=previous_limit,
                    previous_limit=previous_limit,
                    new_limit=current_limit,
                    reason=f"human-feedback revision: {progress_reason}",
                )
            )
            self.progress.emit(
                ProgressEvent(
                    phase="revision",
                    message="revision progress evidence 확인; engineering budget 연장",
                    metrics={"from": previous_limit, "to": current_limit},
                )
            )

        state.transition(
            State.ENGINEERING_BLOCKED,
            f"Human-feedback revision hard step budget exhausted ({self.policy.hard_max_agent_steps}).",
        )
        return state

    def _execute_prepare_decision(
        self,
        state: CFDState,
        action: object,
        *,
        llm_step: int,
        progress_phase: str,
        progress_step: int,
        progress_limit: int,
        native_execution: bool,
    ) -> bool:
        """Execute one LLM decision, which may contain a bounded action sequence."""

        if isinstance(action, CandidateCasePlanRepairAction):
            return self._execute_candidate_case_plan_repair(
                state,
                action,
                llm_step=llm_step,
                progress_phase=progress_phase,
                progress_step=progress_step,
                progress_limit=progress_limit,
                native_execution=native_execution,
            )

        if isinstance(action, RepairCasePlanAction):
            return self._execute_prepare_repair_plan(
                state,
                action,
                llm_step=llm_step,
                progress_phase=progress_phase,
                native_execution=native_execution,
            )

        if isinstance(action, StrategyRevisionAction):
            return self._execute_strategy_revision(
                state,
                action,
                llm_step=llm_step,
                progress_phase=progress_phase,
                native_execution=native_execution,
            )

        if isinstance(action, ExecuteCasePlanAction):
            self._mark_evidence_gaps_satisfied("prepare")
            return self._execute_case_plan(
                state,
                action,
                llm_step=llm_step,
                progress_phase=progress_phase,
                progress_step=progress_step,
                progress_limit=progress_limit,
                native_execution=native_execution,
            )

        if isinstance(action, EngineeringSequenceAction):
            return self._execute_prepare_sequence(
                state,
                action,
                llm_step=llm_step,
                progress_phase=progress_phase,
                progress_step=progress_step,
                progress_limit=progress_limit,
                native_execution=native_execution,
            )

        if self._tool_action_count(state) >= self.policy.max_tool_actions:
            state.transition(
                State.ENGINEERING_BLOCKED,
                f"Engineering deterministic action budget exhausted ({self.policy.max_tool_actions}).",
            )
            return True

        self._emit_action_started(
            progress_phase,
            action,
            step=progress_step,
            limit=progress_limit,
        )
        event, terminal = self._dispatch_prepare(
            state,
            action,
            step=llm_step,
            native_execution=native_execution,
        )
        state.engineering_events.append(event)
        self._emit_engineering_event(
            progress_phase,
            event,
            step=progress_step,
            limit=progress_limit,
            state=state,
        )
        return terminal

    @staticmethod
    def _candidate_failure_paths(
        failures: list[str],
        candidate_bundle: dict[str, str],
    ) -> list[str]:
        """Extract implicated authored paths from deterministic preflight diagnostics."""
        matched: list[str] = []
        for path in candidate_bundle:
            if any(path in failure for failure in failures):
                matched.append(path)
        return matched[:12]

    def _candidate_repair_context(self) -> dict[str, object] | None:
        """Return a bounded capsule for repairing the retained in-memory candidate."""
        candidate = self._pending_candidate_execution
        if candidate is None:
            return None
        failed = set(self._pending_candidate_failed_paths)
        manifest: list[dict[str, object]] = []
        artifacts: list[dict[str, object]] = []
        for item in candidate.files:
            manifest.append({"path": item.path, "kind": "raw", "chars": len(item.content)})
            if item.path in failed:
                artifacts.append(
                    {"path": item.path, "kind": "raw", "content": item.content[:40_000]}
                )
        for item in candidate.typed_dictionaries:
            manifest.append(
                {
                    "path": item.path,
                    "kind": "typed_dictionary",
                    "foam_class": item.foam_class,
                    "entries": len(item.entries),
                }
            )
            if item.path in failed:
                artifacts.append(
                    {
                        "path": item.path,
                        "kind": "typed_dictionary",
                        "foam_class": item.foam_class,
                        "entries": [entry.model_dump(mode="json") for entry in item.entries],
                    }
                )
        return {
            "goal": candidate.goal,
            "failed_paths": list(self._pending_candidate_failed_paths),
            "manifest": manifest,
            "failed_artifacts": artifacts,
            "pipeline": {
                "validate_dictionaries": candidate.validate_dictionaries,
                "surface_checks": candidate.surface_checks,
                "mesh_commands": candidate.mesh_commands,
                "required_case_files": candidate.required_case_files,
            },
            "plan_capsule": {
                "solver": candidate.plan.solver,
                "solver_provider_id": candidate.plan.solver_provider_id,
                "required_case_files": candidate.plan.required_case_files,
                "confirmed_intake_sha256": candidate.plan.confirmed_intake_sha256,
            },
        }

    def _apply_candidate_case_plan_repair(
        self,
        repair: CandidateCasePlanRepairAction,
    ) -> ExecuteCasePlanAction:
        """Apply a model-authored delta to the retained candidate without touching workspace."""
        candidate = self._pending_candidate_execution
        if candidate is None:
            raise WorkspaceSafetyError("No retained candidate case plan exists to repair.")

        raw = {item.path: item for item in candidate.files}
        typed = {item.path: item for item in candidate.typed_dictionaries}
        block_mesh = candidate.block_mesh

        for path in repair.drop_paths:
            raw.pop(path, None)
            typed.pop(path, None)
            if block_mesh is not None and block_mesh.path == path:
                block_mesh = None

        for patch in repair.patches:
            if patch.path in raw:
                current = raw[patch.path].content
            elif patch.path in typed:
                current = serialize_foam_dictionary(typed[patch.path])
            else:
                raise WorkspaceSafetyError(
                    f"Candidate patch target does not exist: {patch.path}"
                )
            if current.count(patch.old) != 1:
                raise WorkspaceSafetyError(
                    f"Candidate patch old fragment must occur exactly once in {patch.path}."
                )
            content = current.replace(patch.old, patch.new, 1)
            raw[patch.path] = CaseBundleFile(path=patch.path, content=content)
            typed.pop(patch.path, None)

        for item in repair.replacement_files:
            raw[item.path] = item
            typed.pop(item.path, None)
            if block_mesh is not None and block_mesh.path == item.path:
                block_mesh = None

        for item in repair.typed_dictionaries:
            typed[item.path] = item
            raw.pop(item.path, None)
            if block_mesh is not None and block_mesh.path == item.path:
                block_mesh = None


        data = candidate.model_dump(mode="python")
        data["files"] = [item.model_dump(mode="python") for item in raw.values()]
        data["typed_dictionaries"] = [item.model_dump(mode="python") for item in typed.values()]
        data["block_mesh"] = block_mesh.model_dump(mode="python") if block_mesh is not None else None
        return ExecuteCasePlanAction.model_validate(data)

    def _execute_candidate_case_plan_repair(
        self,
        state: CFDState,
        repair: CandidateCasePlanRepairAction,
        *,
        llm_step: int,
        progress_phase: str,
        progress_step: int,
        progress_limit: int,
        native_execution: bool,
    ) -> bool:
        """Repair the retained candidate, then rerun transactional whole-plan preflight."""
        if not (repair.patches or repair.replacement_files or repair.typed_dictionaries or repair.drop_paths):
            event = self._event(
                llm_step,
                repair.type,
                False,
                "Candidate repair contained no file change; return a real delta or block.",
            )
            state.engineering_events.append(event)
            self._emit_engineering_event(
                f"{progress_phase}-candidate-repair",
                event,
                step=progress_step,
                limit=progress_limit,
                state=state,
            )
            return False
        try:
            candidate = self._apply_candidate_case_plan_repair(repair)
        except (WorkspaceSafetyError, FoamSerializationError, ValueError) as exc:
            event = self._event(llm_step, repair.type, False, str(exc))
            state.engineering_events.append(event)
            self._emit_engineering_event(
                f"{progress_phase}-candidate-repair",
                event,
                step=progress_step,
                limit=progress_limit,
                state=state,
            )
            return False

        self._pending_candidate_execution = candidate
        event = self._event(
            llm_step,
            repair.type,
            True,
            "Applied delta to retained candidate; re-running whole-bundle authoring preflight.",
        )
        state.engineering_events.append(event)
        self._emit_engineering_event(
            f"{progress_phase}-candidate-repair",
            event,
            step=progress_step,
            limit=progress_limit,
            state=state,
        )
        return self._execute_case_plan(
            state,
            candidate,
            llm_step=llm_step,
            progress_phase=progress_phase,
            progress_step=progress_step,
            progress_limit=progress_limit,
            native_execution=native_execution,
        )

    def _execute_case_plan(
        self,
        state: CFDState,
        execution: ExecuteCasePlanAction,
        *,
        llm_step: int,
        progress_phase: str,
        progress_step: int,
        progress_limit: int,
        native_execution: bool,
    ) -> bool:
        """Execute a complete LLM-authored case plan without intermediate LLM calls.

        The high-level plan is deliberately expanded into the existing primitive
        actions. This preserves the exact same sandbox, path checks, OpenFOAM
        command allowlists, budgets, checkMesh evidence parser, pre-solve gate and
        final plan/CaseSeal validation used by ordinary actions.
        """

        actions: list[object] = []
        rendered_files: list[tuple[str, str]] = [
            (item.path, item.content) for item in execution.files
        ]
        for item in execution.typed_dictionaries:
            try:
                content = serialize_foam_dictionary(item)
            except FoamSerializationError as exc:
                event = self._event(
                    llm_step,
                    "typed_dictionary_serialize",
                    False,
                    f"Typed dictionary serialization failed for {item.path}: {exc}",
                )
                blocked = self._record_case_plan_authoring_failure(state, event)
                self._emit_engineering_event(
                    f"{progress_phase}-execution-plan",
                    event,
                    step=progress_step,
                    limit=progress_limit,
                    state=state,
                )
                # Authoring failures are transactional: no candidate case file has
                # been committed. Retain the complete candidate in memory so the next
                # turn can repair only the implicated candidate entry instead of
                # regenerating the whole case as a large Structured Output object.
                self._pending_candidate_execution = execution
                self._pending_candidate_failed_paths = (item.path,)
                self._pending_execution_plan = None
                return blocked
            rendered_files.append((item.path, content))

        if execution.block_mesh is not None:
            try:
                rendered_files.append((execution.block_mesh.path, serialize_block_mesh(execution.block_mesh)))
            except (FoamSerializationError, ValueError) as exc:
                event = self._event(
                    llm_step,
                    "block_mesh_serialize",
                    False,
                    f"Typed blockMesh serialization failed: {exc}",
                )
                blocked = self._record_case_plan_authoring_failure(state, event)
                self._emit_engineering_event(
                    f"{progress_phase}-execution-plan", event, step=progress_step, limit=progress_limit, state=state
                )
                self._pending_candidate_execution = execution
                self._pending_candidate_failed_paths = ("system/blockMeshDict",)
                self._pending_execution_plan = None
                return blocked

        # v2.11: all-or-nothing authoring preflight.  Validate path/content/library
        # policy and aggregate authored size for the *entire* candidate bundle before
        # the first workspace mutation.  This prevents an early rejected file from
        # leaving only the preceding files behind and triggering a long missing-file
        # repair cascade.
        candidate_bundle = {path: content for path, content in rendered_files}
        bundle_failures = self.workspace.validate_candidate_bundle(candidate_bundle)

        # v3.0.2: solve-critical OpenFOAM files must satisfy the IOobject-facing
        # FoamFile contract before *any* candidate file is committed. This closes the
        # gap where foamDictionary accepted headerless content and blockMesh/foamRun
        # discovered the malformed header later, one file at a time.
        header_targets = list(dict.fromkeys([
            "system/controlDict",
            "system/fvSchemes",
            "system/fvSolution",
            *execution.required_case_files,
            *execution.validate_dictionaries,
        ]))
        for path in header_targets:
            content = candidate_bundle.get(path)
            if content is None:
                continue
            suffix = Path(path).suffix.lower()
            if suffix in {".stl", ".obj", ".off", ".vtk", ".csv", ".dat", ".emesh"}:
                continue
            header = validate_foam_file_header(
                path,
                content,
                expected_class=("dictionary" if path.startswith("system/") else None),
            )
            bundle_failures.extend(
                f"{path}: {failure}" for failure in header.failures
            )

        if bundle_failures:
            event = self._event(
                llm_step,
                "case_bundle_preflight",
                False,
                "Case bundle rejected before commit; no candidate case files were written.",
                "\n".join(f"- {failure}" for failure in bundle_failures),
            )
            blocked = self._record_case_plan_authoring_failure(state, event)
            self._emit_engineering_event(
                f"{progress_phase}-execution-plan",
                event,
                step=progress_step,
                limit=progress_limit,
                state=state,
            )
            self._pending_candidate_execution = execution
            self._pending_candidate_failed_paths = tuple(
                self._candidate_failure_paths(bundle_failures, candidate_bundle)
            )
            self._pending_execution_plan = None
            return blocked

        # Only after every candidate file passes deterministic authoring preflight do
        # we start mutating the workspace.
        self._pending_candidate_execution = None
        self._pending_candidate_failed_paths = ()
        # Since all file writes precede dictionary or
        # native execution in the expanded plan, later OpenFOAM failures always see a
        # complete authored bundle and can use true delta RepairTurn semantics.
        for path, content in rendered_files:
            actions.append(
                WriteCaseFileAction(
                    type="write_case_file",
                    path=path,
                    content=content,
                    rationale="",
                )
            )

        self._pending_execution_plan = execution.plan
        for path in execution.validate_dictionaries:
            actions.append(
                ValidateDictionaryAction(
                    type="validate_dictionary",
                    path=path,
                    rationale="",
                )
            )
        for path in execution.surface_checks:
            actions.append(
                SurfaceCheckAction(
                    type="surface_check",
                    path=path,
                    rationale="",
                )
            )
        for command in execution.mesh_commands:
            actions.append(
                RunMeshCommandAction(
                    type="run_mesh_command",
                    command=command,
                    rationale="",
                )
            )
        actions.append(
            ValidatePreSolveAction(
                type="validate_pre_solve",
                required_case_files=execution.required_case_files,
                rationale="",
            )
        )
        actions.append(
            FinishPreviewAction(
                type="finish_preview",
                plan=execution.plan,
                rationale="",
            )
        )

        execution_id = f"{progress_phase}:execution-plan:{llm_step:04d}"
        total = len(actions)
        self.progress.emit(
            ProgressEvent(
                phase=f"{progress_phase}-execution-plan",
                message=f"deterministic execution plan 시작: {execution.goal}",
                status="start",
                step=progress_step,
                limit=progress_limit,
                metrics={"actions": total, "files": len(execution.files) + len(execution.typed_dictionaries) + (1 if execution.block_mesh is not None else 0)},
            )
        )

        for index, member in enumerate(actions, start=1):
            if self._tool_action_count(state) >= self.policy.max_tool_actions:
                event = self._event(
                    llm_step,
                    getattr(member, "type", "unknown"),
                    False,
                    f"Engineering deterministic action budget exhausted ({self.policy.max_tool_actions}); execution plan stopped.",
                )
                event = self._tag_execution_plan_event(
                    event, execution, execution_id, index, total
                )
                state.engineering_events.append(event)
                self._emit_engineering_event(
                    f"{progress_phase}-execution-plan",
                    event,
                    step=index,
                    limit=total,
                    state=state,
                )
                state.transition(
                    State.ENGINEERING_BLOCKED,
                    f"Engineering deterministic action budget exhausted ({self.policy.max_tool_actions}).",
                )
                return True

            self._emit_action_started(
                f"{progress_phase}-execution-plan",
                member,
                step=index,
                limit=total,
            )
            event, terminal = self._dispatch_prepare(
                state,
                member,
                step=llm_step,
                native_execution=native_execution,
            )
            event = self._tag_execution_plan_event(
                event, execution, execution_id, index, total
            )
            state.engineering_events.append(event)
            self._emit_engineering_event(
                f"{progress_phase}-execution-plan",
                event,
                step=index,
                limit=total,
                state=state,
            )

            if not event.success:
                self.progress.emit(
                    ProgressEvent(
                        phase=f"{progress_phase}-execution-plan",
                        message=f"execution plan 중단: {execution.goal}",
                        status="failure",
                        metrics={"executed": index, "planned": total},
                    )
                )
                return terminal
            if terminal:
                self.progress.emit(
                    ProgressEvent(
                        phase=f"{progress_phase}-execution-plan",
                        message=f"execution plan 완료: {execution.goal}",
                        status="success",
                        metrics={"executed": index, "planned": total},
                    )
                )
                self._pending_execution_plan = None
                return True

        return False

    def _repair_actions(
        self,
        state: CFDState,
        repair: RepairCasePlanAction,
        *,
        runtime: bool,
    ) -> tuple[list[object], EngineeringPlan]:
        """Expand one delta-only repair into existing deterministic primitive actions."""

        actions: list[object] = []
        for patch in repair.patches:
            actions.append(PatchCaseFileAction(type="patch_case_file", patch=patch))
        for item in repair.replacement_files:
            actions.append(
                WriteCaseFileAction(
                    type="write_case_file", path=item.path, content=item.content, rationale=""
                )
            )
        for item in repair.typed_dictionaries:
            actions.append(
                WriteCaseFileAction(
                    type="write_case_file",
                    path=item.path,
                    content=serialize_foam_dictionary(item),
                    rationale="",
                )
            )
        for path in repair.validate_dictionaries:
            actions.append(
                ValidateDictionaryAction(type="validate_dictionary", path=path, rationale="")
            )
        for path in repair.surface_checks:
            actions.append(SurfaceCheckAction(type="surface_check", path=path, rationale=""))
        for command in repair.mesh_commands:
            actions.append(RunMeshCommandAction(type="run_mesh_command", command=command, rationale=""))

        plan = repair.updated_plan or state.engineering_plan or self._pending_execution_plan
        if plan is None:
            raise WorkspaceSafetyError(
                "Delta repair has no baseline EngineeringPlan. Return execute_case_plan instead."
            )
        if repair.validate_pre_solve:
            actions.append(
                ValidatePreSolveAction(
                    type="validate_pre_solve",
                    required_case_files=plan.required_case_files,
                    rationale="",
                )
            )
        if runtime or repair.retry_solver:
            actions.append(RetrySolverAction(type="retry_solver", plan=plan, rationale=""))
        else:
            actions.append(FinishPreviewAction(type="finish_preview", plan=plan, rationale=""))
        return actions, plan

    def _execute_prepare_repair_plan(
        self,
        state: CFDState,
        repair: RepairCasePlanAction,
        *,
        llm_step: int,
        progress_phase: str,
        native_execution: bool,
    ) -> bool:
        if not (repair.patches or repair.replacement_files or repair.typed_dictionaries or repair.updated_plan is not None):
            event = self._event(
                llm_step,
                repair.type,
                False,
                "Repair plan contained no artifact or EngineeringPlan change; return a real delta or block.",
            )
            state.engineering_events.append(event)
            return False
        try:
            actions, _ = self._repair_actions(state, repair, runtime=False)
        except (WorkspaceSafetyError, FoamSerializationError) as exc:
            event = self._event(llm_step, repair.type, False, str(exc))
            state.engineering_events.append(event)
            return False
        sequence_id = f"{progress_phase}:repair-plan:{llm_step:04d}"
        total = len(actions)
        for index, member in enumerate(actions, start=1):
            if self._tool_action_count(state) >= self.policy.max_tool_actions:
                state.transition(
                    State.ENGINEERING_BLOCKED,
                    f"Engineering deterministic action budget exhausted ({self.policy.max_tool_actions}).",
                )
                return True
            event, terminal = self._dispatch_prepare(
                state, member, step=llm_step, native_execution=native_execution
            )
            event = event.model_copy(
                update={
                    "sequence_id": sequence_id,
                    "sequence_goal": repair.diagnosis,
                    "sequence_index": index,
                    "sequence_length": total,
                }
            )
            state.engineering_events.append(event)
            self._emit_engineering_event(
                f"{progress_phase}-repair-plan",
                event,
                step=index,
                limit=total,
                state=state,
            )
            if not event.success:
                return terminal
            if terminal:
                self._pending_execution_plan = None
                return True
        return False

    def _execute_strategy_revision(
        self,
        state: CFDState,
        revision: StrategyRevisionAction,
        *,
        llm_step: int,
        progress_phase: str,
        native_execution: bool,
    ) -> bool:
        """Apply a meshing-strategy delta after tool incompatibility/no-progress.

        Python does not choose the replacement strategy. It only applies the Agent's
        explicit delta, invalidates stale mesh evidence and executes the new pipeline.
        """
        plan = revision.updated_plan or state.engineering_plan or self._pending_execution_plan
        if plan is None:
            event = self._event(
                llm_step, revision.type, False,
                "Strategy revision has no baseline EngineeringPlan; provide updated_plan or block.",
            )
            state.engineering_events.append(event)
            return False

        actions: list[object] = []
        for path in revision.drop_paths:
            actions.append(DeleteCaseFileAction(type="delete_case_file", path=path, rationale=""))
        for patch in revision.patches:
            actions.append(PatchCaseFileAction(type="patch_case_file", patch=patch))
        for item in revision.replacement_files:
            actions.append(WriteCaseFileAction(type="write_case_file", path=item.path, content=item.content, rationale=""))
        for item in revision.typed_dictionaries:
            actions.append(WriteCaseFileAction(type="write_case_file", path=item.path, content=serialize_foam_dictionary(item), rationale=""))
        if revision.block_mesh is not None:
            actions.append(WriteCaseFileAction(type="write_case_file", path=revision.block_mesh.path, content=serialize_block_mesh(revision.block_mesh), rationale=""))
        for path in revision.validate_dictionaries:
            actions.append(ValidateDictionaryAction(type="validate_dictionary", path=path, rationale=""))
        for path in revision.surface_checks:
            actions.append(SurfaceCheckAction(type="surface_check", path=path, rationale=""))
        for command in revision.mesh_commands:
            actions.append(RunMeshCommandAction(type="run_mesh_command", command=command, rationale=""))
        if revision.validate_pre_solve:
            actions.append(ValidatePreSolveAction(type="validate_pre_solve", required_case_files=plan.required_case_files, rationale=""))
        actions.append(FinishPreviewAction(type="finish_preview", plan=plan, rationale=""))

        sequence_id = f"{progress_phase}:strategy-revision:{llm_step:04d}"
        total = len(actions)
        for index, member in enumerate(actions, start=1):
            if self._tool_action_count(state) >= self.policy.max_tool_actions:
                state.transition(State.ENGINEERING_BLOCKED, f"Engineering deterministic action budget exhausted ({self.policy.max_tool_actions}).")
                return True
            event, terminal = self._dispatch_prepare(state, member, step=llm_step, native_execution=native_execution)
            event = event.model_copy(update={
                "sequence_id": sequence_id,
                "sequence_goal": revision.diagnosis,
                "sequence_index": index,
                "sequence_length": total,
            })
            state.engineering_events.append(event)
            self._emit_engineering_event(f"{progress_phase}-strategy-revision", event, step=index, limit=total, state=state)
            if not event.success:
                return terminal
            if terminal:
                self._pending_execution_plan = None
                return True
        return False

    def _runtime_repair_actions(
        self,
        state: CFDState,
        repair: RuntimeCaseRepairAction,
    ) -> tuple[list[object], EngineeringPlan]:
        """Expand grouped runtime edits into sequential deterministic primitives."""
        plan = state.engineering_plan
        if plan is None:
            raise WorkspaceSafetyError("Runtime repair requires the approved EngineeringPlan.")
        actions: list[object] = []
        for group in repair.file_patches:
            for edit in group.edits:
                actions.append(
                    PatchCaseFileAction(
                        type="patch_case_file",
                        patch={"path": group.path, "old": edit.old, "new": edit.new},
                    )
                )
        for item in repair.replacement_files:
            actions.append(
                WriteCaseFileAction(type="write_case_file", path=item.path, content=item.content, rationale="")
            )
        for item in repair.typed_dictionaries:
            actions.append(
                WriteCaseFileAction(
                    type="write_case_file",
                    path=item.path,
                    content=serialize_foam_dictionary(item),
                    rationale="",
                )
            )
        for path in repair.validate_dictionaries:
            actions.append(ValidateDictionaryAction(type="validate_dictionary", path=path, rationale=""))
        for path in repair.surface_checks:
            actions.append(SurfaceCheckAction(type="surface_check", path=path, rationale=""))
        for command in repair.mesh_commands:
            actions.append(RunMeshCommandAction(type="run_mesh_command", command=command, rationale=""))
        if repair.validate_pre_solve:
            actions.append(
                ValidatePreSolveAction(
                    type="validate_pre_solve",
                    required_case_files=plan.required_case_files,
                    rationale="",
                )
            )
        if repair.retry_solver:
            actions.append(RetrySolverAction(type="retry_solver", plan=plan, rationale=""))
        return actions, plan

    def _execute_runtime_repair_plan(
        self,
        state: CFDState,
        repair: RuntimeCaseRepairAction | RepairCasePlanAction,
        *,
        approved_solver: str,
        llm_step: int,
        native_execution: bool,
        runtime_event_start: int,
    ) -> RepairOutcome | None:
        if isinstance(repair, RuntimeCaseRepairAction) and not (
            repair.file_patches or repair.replacement_files or repair.typed_dictionaries
        ):
            reason = (
                "Runtime repair contained no case-file change; automatic solver retry without "
                "a concrete repair is not authorized."
            )
            state.engineering_events.append(self._event(llm_step, repair.type, False, reason))
            return None
        if isinstance(repair, RepairCasePlanAction) and not (
            repair.patches or repair.replacement_files or repair.typed_dictionaries or repair.updated_plan is not None
        ):
            reason = "Runtime repair contained no executable delta."
            state.engineering_events.append(self._event(llm_step, repair.type, False, reason))
            return None
        self._mark_evidence_gaps_satisfied("runtime_repair")
        try:
            if isinstance(repair, RuntimeCaseRepairAction):
                actions, plan = self._runtime_repair_actions(state, repair)
            else:
                actions, plan = self._repair_actions(state, repair, runtime=True)
        except FoamSerializationError as exc:
            state.engineering_events.append(
                self._event(llm_step, repair.type, False, f"Runtime repair serialization failed: {exc}")
            )
            return None
        except WorkspaceSafetyError as exc:
            reason = f"Runtime repair blocked by workspace safety: {exc}"
            return RepairOutcome(RuntimeRepairDecision.BLOCKED, reason=reason)
        if plan.solver != approved_solver:
            return RepairOutcome(
                RuntimeRepairDecision.NEEDS_USER_REVIEW,
                reason="Runtime repair attempted to change the user-approved solver.",
            )
        sequence_id = f"runtime-repair:repair-plan:{llm_step:04d}"
        total = len(actions)
        for index, member in enumerate(actions, start=1):
            if len(state.engineering_events) - runtime_event_start >= self.policy.max_runtime_repair_tool_actions:
                return RepairOutcome(
                    RuntimeRepairDecision.BLOCKED,
                    reason=f"Runtime repair deterministic action budget exhausted ({self.policy.max_runtime_repair_tool_actions}).",
                )
            outcome: RepairOutcome | None = None
            if isinstance(member, RetrySolverAction):
                event, outcome = self._dispatch_retry_solver(
                    state,
                    member,
                    approved_solver=approved_solver,
                    step=llm_step,
                    native_execution=native_execution,
                )
            else:
                event = self._dispatch_tool_action(
                    member,
                    step=llm_step,
                    native_execution=native_execution,
                    phase="runtime_repair",
                    state=state,
                )
            event = event.model_copy(
                update={
                    "sequence_id": sequence_id,
                    "sequence_goal": repair.diagnosis,
                    "sequence_index": index,
                    "sequence_length": total,
                }
            )
            state.engineering_events.append(event)
            if not event.success:
                return outcome
            if outcome is not None:
                return outcome
        return None

    @staticmethod
    def _tag_execution_plan_event(
        event: EngineeringEvent,
        execution: ExecuteCasePlanAction,
        execution_id: str,
        index: int,
        total: int,
    ) -> EngineeringEvent:
        return event.model_copy(
            update={
                "sequence_id": execution_id,
                "sequence_goal": execution.goal,
                "sequence_index": index,
                "sequence_length": total,
            }
        )

    def _execute_prepare_sequence(
        self,
        state: CFDState,
        sequence: EngineeringSequenceAction,
        *,
        llm_step: int,
        progress_phase: str,
        progress_step: int,
        progress_limit: int,
        native_execution: bool,
    ) -> bool:
        sequence_id = f"{progress_phase}:{llm_step:04d}"
        self.progress.emit(
            ProgressEvent(
                phase=f"{progress_phase}-sequence",
                message=f"sequence 시작: {sequence.goal}",
                status="start",
                step=progress_step,
                limit=progress_limit,
                metrics={"actions": len(sequence.actions)},
            )
        )

        for index, member in enumerate(sequence.actions, start=1):
            if self._tool_action_count(state) >= self.policy.max_tool_actions:
                event = self._event(
                    llm_step,
                    member.type,
                    False,
                    f"Engineering deterministic action budget exhausted ({self.policy.max_tool_actions}); sequence stopped.",
                )
                event = self._tag_sequence_event(event, sequence, sequence_id, index)
                state.engineering_events.append(event)
                self._emit_engineering_event(
                    f"{progress_phase}-sequence",
                    event,
                    step=index,
                    limit=len(sequence.actions),
                    state=state,
                )
                state.transition(
                    State.ENGINEERING_BLOCKED,
                    f"Engineering deterministic action budget exhausted ({self.policy.max_tool_actions}).",
                )
                return True

            self._emit_action_started(
                f"{progress_phase}-sequence",
                member,
                step=index,
                limit=len(sequence.actions),
            )
            event, terminal = self._dispatch_prepare(
                state,
                member,
                step=llm_step,
                native_execution=native_execution,
            )
            event = self._tag_sequence_event(event, sequence, sequence_id, index)
            state.engineering_events.append(event)
            self._emit_engineering_event(
                f"{progress_phase}-sequence",
                event,
                step=index,
                limit=len(sequence.actions),
                state=state,
            )
            if not event.success:
                self.progress.emit(
                    ProgressEvent(
                        phase=f"{progress_phase}-sequence",
                        message=f"sequence 중단: {sequence.goal}",
                        status="failure",
                        metrics={"executed": index, "planned": len(sequence.actions)},
                    )
                )
                return terminal
            if terminal:
                self.progress.emit(
                    ProgressEvent(
                        phase=f"{progress_phase}-sequence",
                        message=f"sequence 완료: {sequence.goal}",
                        status="success",
                        metrics={"executed": index, "planned": len(sequence.actions)},
                    )
                )
                return True

        self.progress.emit(
            ProgressEvent(
                phase=f"{progress_phase}-sequence",
                message=f"sequence 완료: {sequence.goal}",
                status="success",
                metrics={"executed": len(sequence.actions)},
            )
        )
        return False

    @staticmethod
    def _tag_sequence_event(
        event: EngineeringEvent,
        sequence: EngineeringSequenceAction,
        sequence_id: str,
        index: int,
    ) -> EngineeringEvent:
        return event.model_copy(
            update={
                "sequence_id": sequence_id,
                "sequence_goal": sequence.goal,
                "sequence_index": index,
                "sequence_length": len(sequence.actions),
            }
        )

    def _checkmesh_preflight(self, state: CFDState, *, phase: str) -> bool:
        """Fail fast before LLM engineering when trusted checkMesh is unavailable.

        This intentionally checks only checkMesh. Solver/mesh-strategy-specific tools remain
        agent-observed capabilities so their absence can be handled during engineering.
        """
        preflight = getattr(self.tools, "check_mesh_preflight", None)
        if not callable(preflight):
            # Injected test/dry tool doubles may not expose host executable discovery.
            return True
        status = preflight()
        available = bool(status.get("available")) and bool(status.get("trusted", True))
        if available:
            self.progress.emit(
                ProgressEvent(
                    phase=phase,
                    message="OpenFOAM native preflight: checkMesh 확인",
                    status="success",
                    metrics={"checkMesh": "available"},
                )
            )
            return True

        reason = str(status.get("reason") or "trusted checkMesh executable is unavailable")
        state.engineering_round_start_index = len(state.engineering_events)
        state.transition(
            State.ENGINEERING_BLOCKED,
            "OpenFOAM native preflight failed before autonomous engineering: "
            f"checkMesh is unavailable or untrusted. {reason} "
            "Source the OpenFOAM environment before launching/restarting openfoam-agent.",
        )
        self.progress.emit(
            ProgressEvent(
                phase=phase,
                message="OpenFOAM native preflight: checkMesh를 찾을 수 없음",
                status="failure",
                metrics={"engineeringActions": 0},
            )
        )
        return False

    def _progress_allows_extension(self, state: CFDState) -> tuple[bool, str]:
        """Decide only whether more bounded work is justified, never what CFD work to do."""

        events = self._current_round_events(state)
        window = self.policy.progress_window
        if len(events) < window:
            return False, "insufficient recent history to prove progress"

        recent = events[-window:]
        earlier = events[:-window]
        earlier_signatures = {self._progress_signature(event) for event in earlier}

        evidence_actions = {
            "search_capabilities",
            "search_references",
            "read_reference",
            "gather_evidence",
            "write_case_file",
            "delete_case_file",
            "validate_dictionary",
            "surface_check",
            "run_mesh_command",
        }
        recent_evidence = [
            event
            for event in recent
            if event.action_type in evidence_actions
            and (event.action_type != "gather_evidence" or bool(event.observed_evidence))
        ]
        novel = [
            event
            for event in recent_evidence
            if self._progress_signature(event) not in earlier_signatures
        ]
        if not novel:
            return False, "recent actions only repeated previously observed action/result signatures"

        # A new case artifact, new tool/reference observation, or changed native
        # result is enough to justify another bounded chunk. This intentionally
        # does not judge whether the CFD choice itself is good.
        kinds = sorted({event.action_type for event in novel})
        return True, f"new evidence observed via {', '.join(kinds)}"

    @staticmethod
    def _progress_signature(event: EngineeringEvent) -> str:
        output = re.sub(
            r"(?im)^\s*(?:ExecutionTime|ClockTime)\b.*$",
            "",
            event.output_excerpt,
        )
        payload = "\n".join(
            [
                event.action_type,
                "1" if event.success else "0",
                event.summary,
                output,
                event.artifact_sha256 or "",
            ]
        )
        return hashlib.sha256(payload.encode("utf-8", errors="replace")).hexdigest()

    @staticmethod
    def _current_round_events(state: CFDState) -> list[EngineeringEvent]:
        start = min(state.engineering_round_start_index, len(state.engineering_events))
        return state.engineering_events[start:]

    def _native_command_count(self, state: CFDState) -> int:
        return sum(1 for event in self._current_round_events(state) if event.native_command_executed)

    def _tool_action_count(self, state: CFDState) -> int:
        return len(self._current_round_events(state))

    def _mesh_repair_status(self, state: CFDState) -> tuple[int, bool, bool]:
        """Return (completed/started repair cycles, failure pending, current cycle started)."""

        cycles = 0
        failure_pending = False
        current_cycle_started = False
        for event in self._current_round_events(state):
            if event.mesh_command_executed:
                if event.success:
                    failure_pending = False
                    current_cycle_started = False
                else:
                    failure_pending = True
                    current_cycle_started = False
                continue
            if (
                failure_pending
                and event.success
                and event.action_type in {"write_case_file", "delete_case_file"}
                and not current_cycle_started
            ):
                cycles += 1
                current_cycle_started = True
        return cycles, failure_pending, current_cycle_started

    def _mesh_repair_cycle_count(self, state: CFDState) -> int:
        return self._mesh_repair_status(state)[0]

    def _ready_for_finalization(self, state: CFDState, *, native_execution: bool) -> bool:
        # Do not silently grant extra retries after a rejected terminal claim.  The
        # finalization window exists only for the boundary case where ordinary tool
        # work consumed the budget and the case itself is already current/validated.
        if state.engineering_events and state.engineering_events[-1].action_type in {
            "finish_preview",
            "block",
            "retry_solver",
        }:
            return False
        if not native_execution:
            return bool(self.workspace.file_seals())
        evidence = state.mesh_evidence
        return bool(
            evidence is not None
            and evidence.passed
            and (evidence.cell_count is None or evidence.cell_count <= self.policy.max_mesh_cells)
            and self._checkmesh_mesh_manifest is not None
            and self._checkmesh_mesh_manifest == self.workspace.mesh_manifest_digest()
        )

    def _run_finalization_window(
        self,
        state: CFDState,
        *,
        native_execution: bool,
        start_step: int,
        phase: str = "prepare_finalize",
    ) -> CFDState:
        for offset in range(self.policy.max_finalization_steps):
            step = start_step + offset
            turn = self._generate_turn(
                state,
                step=step,
                local_step=offset + 1,
                current_step_limit=self.policy.max_finalization_steps,
                phase=phase,
                native_execution=native_execution,
            )
            action = turn.action
            progress_phase = "revision-finalize" if phase.startswith("human_revision") else "finalizing"
            self._emit_action_started(
                progress_phase, action, step=offset + 1, limit=self.policy.max_finalization_steps
            )

            if isinstance(action, (FinishPreviewAction, BlockAction)):
                event, terminal = self._dispatch_prepare(
                    state,
                    action,
                    step=step,
                    native_execution=native_execution,
                )
            else:
                event = self._event(
                    step,
                    getattr(action, "type", "unknown"),
                    False,
                    "Engineering tool budget is exhausted; finalization window permits only finish_preview or block.",
                )
                terminal = False

            state.engineering_events.append(event)
            self._emit_engineering_event(
                progress_phase,
                event,
                step=offset + 1,
                limit=self.policy.max_finalization_steps,
                state=state,
            )
            if terminal:
                return state

        state.transition(
            State.ENGINEERING_BLOCKED,
            f"Engineering finalization budget exhausted ({self.policy.max_finalization_steps}) after the case was validated.",
        )
        return state

    def _runtime_repair_outcome(
        self,
        state: CFDState,
        decision: RuntimeRepairDecision,
        reason: str,
        plan: EngineeringPlan | None = None,
    ) -> RepairOutcome:
        """Close the transient RUNTIME_REPAIR state before returning to the workflow.

        RUNTIME_REPAIR is an internal state owned by RuntimeOrchestrator/Engineering. It
        must never leak back to the top-level workflow. The decision enum makes every
        non-retry exit explicit instead of overloading a boolean.
        """

        if decision == RuntimeRepairDecision.RETRY_SOLVER:
            if state.current_state != State.SIMULATION:
                state.transition(State.SIMULATION, reason or "Runtime repair requested solver retry.")
        elif decision == RuntimeRepairDecision.NEEDS_USER_REVIEW:
            state.solve_approved = False
            state.transition(State.ENGINEERING_REVIEW_REQUIRED, reason)
        elif decision == RuntimeRepairDecision.STRATEGY_REVISION:
            state.solve_approved = False
            state.transition(State.ENGINEERING_REVIEW_REQUIRED, reason)
        else:
            state.solve_approved = False
            state.transition(State.ENGINEERING_BLOCKED, reason)
        return RepairOutcome(decision=decision, plan=plan, reason=reason)

    def repair_runtime(
        self,
        state: CFDState,
        *,
        runtime_log: str,
        attempt: int,
        native_execution: bool = True,
    ) -> RepairOutcome:
        if state.engineering_plan is None or state.case_seal is None:
            return self._runtime_repair_outcome(state, RuntimeRepairDecision.BLOCKED, "No approved engineering plan is available.")
        try:
            self.workspace.adopt_seal(state.case_seal)
            self.safety.verify_seal(state.engineering_plan, state.case_seal)
        except WorkspaceSafetyError as exc:
            return self._runtime_repair_outcome(state, RuntimeRepairDecision.BLOCKED, f"Approved case integrity verification failed: {exc}")
        if state.mesh_evidence is not None and state.mesh_evidence.passed:
            # The full case seal proves the runtime-repair workspace is still the exact
            # case for which this persisted checkMesh evidence was approved. Restore
            # freshness against the narrower mesh-only manifest so solver-input edits
            # do not force redundant checkMesh runs.
            self._checkmesh_mesh_manifest = self.workspace.mesh_manifest_digest()
        approved_solver = state.engineering_plan.solver
        self._evidence_gap_ledger["runtime_repair"] = {}
        self._retrieval_cycles["runtime_repair"] = 0
        state.last_runtime_log_excerpt = runtime_log[-12000:]
        state.transition(
            State.RUNTIME_REPAIR,
            f"foamRun attempt {attempt} failed; log returned to CFDEngineeringAgent.",
        )
        self.progress.emit(
            ProgressEvent(
                phase="runtime-repair",
                message=f"runtime repair cycle 시작: solver attempt={attempt}",
                status="start",
                metrics={
                    "llmTurnBudget": self.policy.max_runtime_repair_steps,
                    "toolBudget": self.policy.max_runtime_repair_tool_actions,
                },
            )
        )
        runtime_event_start = len(state.engineering_events)
        for local_step in range(1, self.policy.max_runtime_repair_steps + 1):
            step = len(state.engineering_events) + 1
            turn = self._generate_turn(
                state,
                step=step,
                local_step=local_step,
                current_step_limit=self.policy.max_runtime_repair_steps,
                phase="runtime_repair",
                runtime_log=runtime_log,
                native_execution=native_execution,
            )
            action = turn.action
            if isinstance(action, (RuntimeCaseRepairAction, RepairCasePlanAction)):
                outcome = self._execute_runtime_repair_plan(
                    state,
                    action,
                    approved_solver=approved_solver,
                    llm_step=step,
                    native_execution=native_execution,
                    runtime_event_start=runtime_event_start,
                )
                if outcome is not None:
                    return self._runtime_repair_outcome(
                        state, outcome.decision, outcome.reason, outcome.plan
                    )
                continue
            if isinstance(action, EngineeringSequenceAction):
                outcome = self._execute_runtime_sequence(
                    state,
                    action,
                    approved_solver=approved_solver,
                    llm_step=step,
                    progress_step=local_step,
                    native_execution=native_execution,
                    runtime_event_start=runtime_event_start,
                )
                if outcome is not None:
                    return self._runtime_repair_outcome(
                        state, outcome.decision, outcome.reason, outcome.plan
                    )
                continue

            if len(state.engineering_events) - runtime_event_start >= self.policy.max_runtime_repair_tool_actions:
                reason = (
                    "Runtime repair deterministic action budget exhausted "
                    f"({self.policy.max_runtime_repair_tool_actions})."
                )
                return self._runtime_repair_outcome(state, RuntimeRepairDecision.BLOCKED, reason)

            self._emit_action_started(
                "runtime-repair",
                action,
                step=local_step,
                limit=self.policy.max_runtime_repair_steps,
            )
            if isinstance(action, RetrySolverAction):
                event, outcome = self._dispatch_retry_solver(
                    state,
                    action,
                    approved_solver=approved_solver,
                    step=step,
                    native_execution=native_execution,
                )
                state.engineering_events.append(event)
                self._emit_engineering_event(
                    "runtime-repair",
                    event,
                    step=local_step,
                    limit=self.policy.max_runtime_repair_steps,
                    state=state,
                )
                if outcome is not None:
                    return self._runtime_repair_outcome(
                        state, outcome.decision, outcome.reason, outcome.plan
                    )
                continue
            if isinstance(action, BlockAction):
                event = self._event(step, action.type, True, action.reason)
                state.engineering_events.append(event)
                self._emit_engineering_event(
                    "runtime-repair",
                    event,
                    step=local_step,
                    limit=self.policy.max_runtime_repair_steps,
                    state=state,
                )
                decision = (
                    RuntimeRepairDecision.NEEDS_USER_REVIEW
                    if action.needs_user_input
                    else RuntimeRepairDecision.BLOCKED
                )
                return self._runtime_repair_outcome(state, decision, action.reason)

            event = self._dispatch_tool_action(
                action,
                step=step,
                native_execution=native_execution,
                phase="runtime_repair",
                state=state,
            )
            state.engineering_events.append(event)
            self._emit_engineering_event(
                "runtime-repair",
                event,
                step=local_step,
                limit=self.policy.max_runtime_repair_steps,
                state=state,
            )
        reason = f"Runtime repair action budget exhausted ({self.policy.max_runtime_repair_steps})."
        return self._runtime_repair_outcome(state, RuntimeRepairDecision.BLOCKED, reason)

    def _execute_runtime_sequence(
        self,
        state: CFDState,
        sequence: EngineeringSequenceAction,
        *,
        approved_solver: str,
        llm_step: int,
        progress_step: int,
        native_execution: bool,
        runtime_event_start: int,
    ) -> RepairOutcome | None:
        sequence_id = f"runtime-repair:{llm_step:04d}"
        self.progress.emit(
            ProgressEvent(
                phase="runtime-repair-sequence",
                message=f"sequence 시작: {sequence.goal}",
                status="start",
                step=progress_step,
                limit=self.policy.max_runtime_repair_steps,
                metrics={"actions": len(sequence.actions)},
            )
        )
        for index, member in enumerate(sequence.actions, start=1):
            if len(state.engineering_events) - runtime_event_start >= self.policy.max_runtime_repair_tool_actions:
                reason = (
                    "Runtime repair deterministic action budget exhausted "
                    f"({self.policy.max_runtime_repair_tool_actions})."
                )
                return self._runtime_repair_outcome(state, RuntimeRepairDecision.BLOCKED, reason)

            self._emit_action_started(
                "runtime-repair-sequence",
                member,
                step=index,
                limit=len(sequence.actions),
            )
            outcome: RepairOutcome | None = None
            if isinstance(member, RetrySolverAction):
                event, outcome = self._dispatch_retry_solver(
                    state,
                    member,
                    approved_solver=approved_solver,
                    step=llm_step,
                    native_execution=native_execution,
                )
            elif isinstance(member, FinishPreviewAction):
                event = self._event(
                    llm_step,
                    member.type,
                    False,
                    "finish_preview is not valid inside runtime repair; use retry_solver as the sequence terminator.",
                )
            else:
                event = self._dispatch_tool_action(
                    member,
                    step=llm_step,
                    native_execution=native_execution,
                    phase="runtime_repair",
                    state=state,
                )
            event = self._tag_sequence_event(event, sequence, sequence_id, index)
            state.engineering_events.append(event)
            self._emit_engineering_event(
                "runtime-repair-sequence",
                event,
                step=index,
                limit=len(sequence.actions),
                state=state,
            )
            if not event.success:
                self.progress.emit(
                    ProgressEvent(
                        phase="runtime-repair-sequence",
                        message=f"sequence 중단: {sequence.goal}",
                        status="failure",
                        metrics={"executed": index, "planned": len(sequence.actions)},
                    )
                )
                return outcome
            if outcome is not None:
                self.progress.emit(
                    ProgressEvent(
                        phase="runtime-repair-sequence",
                        message=f"sequence 완료: {sequence.goal}",
                        status="success",
                        metrics={"executed": index, "planned": len(sequence.actions)},
                    )
                )
                return outcome

        self.progress.emit(
            ProgressEvent(
                phase="runtime-repair-sequence",
                message=f"sequence 완료: {sequence.goal}",
                status="success",
                metrics={"executed": len(sequence.actions)},
            )
        )
        return None

    def _dispatch_retry_solver(
        self,
        state: CFDState,
        action: RetrySolverAction,
        *,
        approved_solver: str,
        step: int,
        native_execution: bool,
    ) -> tuple[EngineeringEvent, RepairOutcome | None]:
        result = self.safety.validate_plan(action.plan, state.intake)  # type: ignore[arg-type]
        result.failures.extend(self._validate_observed_provenance(action.plan, state))
        result.valid = not result.failures
        if action.plan.solver != approved_solver:
            result.failures.append("Runtime repair attempted to change the user-approved solver.")
            result.valid = False
        if native_execution and result.valid:
            native = self.safety.validate_native_inputs()
            result.failures.extend(native.failures)
            if self.policy.require_solve_ready_gate:
                presolve_is_current = bool(
                    self._presolve_case_manifest == self.workspace.manifest_digest()
                    and self._presolve_required_case_files
                    == tuple(action.plan.required_case_files)
                )
                if not presolve_is_current:
                    presolve = self.presolve.validate(action.plan)
                    result.failures.extend(presolve.failures)
                    if presolve.valid:
                        self._presolve_case_manifest = self.workspace.manifest_digest()
                        self._presolve_required_case_files = tuple(action.plan.required_case_files)
            if state.mesh_evidence is None or not state.mesh_evidence.passed:
                result.failures.append(
                    "A passing checkMesh result with cell-count evidence is required before an automatic solver retry."
                )
            elif state.mesh_evidence.cell_count is not None and state.mesh_evidence.cell_count > self.policy.max_mesh_cells:
                result.failures.append(
                    f"Mesh cell count {state.mesh_evidence.cell_count} exceeds bounded policy limit {self.policy.max_mesh_cells}."
                )
            elif self._checkmesh_mesh_manifest != self.workspace.mesh_manifest_digest():
                result.failures.append(
                    "Mesh-affecting inputs changed after checkMesh; re-run checkMesh before retry_solver."
                )
            result.valid = not result.failures
        if result.valid:
            state.engineering_plan = action.plan
            state.case_seal = self.workspace.seal(action.plan)
            state.transition(State.SIMULATION, "Engineering repair requested bounded solver retry.")
            return (
                self._event(
                    step,
                    action.type,
                    True,
                    "Runtime repair validated, pre-solve complete, and sealed; solver retry requested.",
                ),
                RepairOutcome(RuntimeRepairDecision.RETRY_SOLVER, action.plan),
            )
        return (
            self._event(
                step,
                action.type,
                False,
                "Solver retry was rejected by deterministic safety/pre-solve validation.",
                "\n".join(result.failures),
            ),
            None,
        )

    def _dispatch_prepare(
        self,
        state: CFDState,
        action,
        *,
        step: int,
        native_execution: bool,
    ) -> tuple[EngineeringEvent, bool]:
        if isinstance(action, FinishPreviewAction):
            validation = self.safety.validate_plan(action.plan, state.intake)  # type: ignore[arg-type]
            validation.failures.extend(self._validate_observed_provenance(action.plan, state))
            validation.valid = not validation.failures
            if native_execution and validation.valid:
                if state.mesh_evidence is None or not state.mesh_evidence.passed:
                    validation.failures.append(
                        "A successful current checkMesh result with cell-count evidence is required before solve approval."
                    )
                elif state.mesh_evidence.cell_count is not None and state.mesh_evidence.cell_count > self.policy.max_mesh_cells:
                    validation.failures.append(
                        f"Mesh cell count {state.mesh_evidence.cell_count} exceeds bounded policy limit {self.policy.max_mesh_cells}."
                    )
                elif self._checkmesh_mesh_manifest != self.workspace.mesh_manifest_digest():
                    validation.failures.append(
                        "Mesh-affecting inputs changed after the last successful checkMesh; re-run checkMesh."
                    )
                validation.valid = not validation.failures
            proposal = state.active_revision_proposal
            if validation.valid and proposal is not None and proposal.requires_case_revision:
                if self.workspace.manifest_digest() == proposal.baseline_manifest_sha256:
                    validation.failures.append(
                        "Human-feedback proposal requires a case revision, but the solver-input manifest is unchanged."
                    )
                    validation.valid = False
            presolve = None
            if native_execution and validation.valid and self.policy.require_solve_ready_gate:
                current_manifest = self.workspace.manifest_digest()
                cached_presolve = (
                    self._presolve_case_manifest == current_manifest
                    and self._presolve_required_case_files == tuple(action.plan.required_case_files)
                )
                if cached_presolve:
                    self.progress.emit(
                        ProgressEvent(
                            phase="pre-solve",
                            message="동일 case manifest의 pre-solve evidence 재사용",
                            status="success",
                        )
                    )
                else:
                    self.progress.emit(
                        ProgressEvent(
                            phase="pre-solve",
                            message="solve-ready case completeness 검증",
                            status="start",
                        )
                    )
                    presolve = self.presolve.validate(action.plan)
                    if not presolve.valid:
                        validation.failures.extend(presolve.failures)
                        validation.valid = False
                        self.progress.emit(
                            ProgressEvent(
                                phase="pre-solve",
                                message="solve-ready completeness 검증 실패",
                                status="failure",
                                details=tuple(item[:800] for item in presolve.failures[:12]),
                                metrics={"checkedFiles": len(presolve.checked_files), "meshPatches": len(presolve.mesh_patches)},
                            )
                        )
                    else:
                        self._presolve_case_manifest = current_manifest
                        self._presolve_required_case_files = tuple(action.plan.required_case_files)
                        self.progress.emit(
                            ProgressEvent(
                                phase="pre-solve",
                                message="solve-ready completeness 검증 통과",
                                status="success",
                                details=tuple(item[:800] for item in presolve.warnings[:12]),
                                metrics={
                                    "checkedFiles": len(presolve.checked_files),
                                    "meshPatches": len(presolve.mesh_patches),
                                    "semanticWarnings": len(presolve.warnings),
                                },
                            )
                        )
            if validation.valid:
                previous_plan = state.engineering_plan
                previous_seal = state.case_seal
                state.engineering_plan = action.plan
                state.case_seal = self.workspace.seal(action.plan)
                state.case_dir = str(self.workspace.case_dir)
                if proposal is not None and previous_plan is not None and previous_seal is not None:
                    state.revision_history.append(
                        self._revision_record(
                            state,
                            proposal_id=proposal.proposal_id,
                            feedback_ids=proposal.feedback_ids,
                            before_plan=previous_plan,
                            before_seal=previous_seal,
                            after_plan=action.plan,
                            after_seal=state.case_seal,
                        )
                    )
                    for feedback in state.human_feedback:
                        if feedback.feedback_id in proposal.feedback_ids:
                            feedback.status = "awaiting_rerun"
                    state.active_revision_proposal = None
                    state.pending_revision_archive_path = None
                destination = (
                    State.SOLVE_READY
                    if native_execution and self.policy.require_solve_ready_gate
                    else (State.MESH_READY if native_execution else State.CASE_PREVIEW_READY)
                )
                if native_execution and self.policy.require_solve_ready_gate:
                    state.transition(State.MESH_READY, "Current case passed checkMesh and mesh evidence gates.")
                    state.transition(State.PRE_SOLVE_VALIDATION, "Mesh-ready case entered deterministic pre-solve completeness validation.")
                state.transition(
                    destination,
                    (
                        ("Agent case passed safety/integrity, checkMesh, and pre-solve completeness gates. Solver approval is required."
                         if self.policy.require_solve_ready_gate else
                         "Agent case passed safety/integrity gates and checkMesh. Solver approval is required.")
                        if native_execution
                        else "Agent case preview passed static safety/integrity gates; native tools were not executed."
                    ),
                )
                return (
                    self._event(
                        step,
                        action.type,
                        True,
                        "Engineering plan accepted and case sealed.",
                        artifact_sha256=state.case_seal.manifest_sha256,
                    ),
                    True,
                )
            return (
                self._event(
                    step,
                    action.type,
                    False,
                    "Engineering plan rejected by deterministic safety/evidence gate.",
                    "\n".join(validation.failures),
                ),
                False,
            )
        if isinstance(action, RetrySolverAction):
            return (
                self._event(
                    step,
                    action.type,
                    False,
                    "retry_solver is only valid after a solver failure.",
                ),
                False,
            )
        if isinstance(action, BlockAction):
            state.transition(
                State.ENGINEERING_REVIEW_REQUIRED if action.needs_user_input else State.ENGINEERING_BLOCKED,
                action.reason,
            )
            return self._event(step, action.type, True, action.reason), True

        return (
            self._dispatch_tool_action(
                action,
                step=step,
                native_execution=native_execution,
                phase="prepare",
                state=state,
            ),
            False,
        )

    def _evidence_gap_status(self, phase: str) -> list[dict[str, object]]:
        ledger = self._evidence_gap_ledger.get(phase, {})
        result: list[dict[str, object]] = []
        for gap_id, item in sorted(ledger.items()):
            seen = item.get("seen_ids", set())
            result.append(
                {
                    "gap_id": gap_id,
                    "missing_evidence": item.get("missing_evidence", ""),
                    "why_required": item.get("why_required", ""),
                    "retrievals": int(item.get("retrievals", 0)),
                    "seen_evidence_count": len(seen) if isinstance(seen, set) else len(list(seen)),
                    "last_new_evidence_count": int(item.get("last_new_count", 0)),
                    "status": str(item.get("status", "open")),
                    "refines_gap_id": item.get("refines_gap_id"),
                    "superseded_by": item.get("superseded_by"),
                    "stagnant": str(item.get("status", "")) == "stagnant",
                    "satisfied": str(item.get("status", "")) == "satisfied",
                }
            )
        return result

    def _mark_evidence_gaps_satisfied(self, phase: str) -> None:
        """Close retrieved gaps when the Agent elects to proceed without more retrieval."""

        ledger = self._evidence_gap_ledger.get(phase, {})
        for entry in ledger.values():
            if str(entry.get("status", "")) == "evidence_available":
                entry["status"] = "satisfied"

    def _gather_evidence(
        self,
        action: GatherEvidenceAction,
        *,
        step: int,
        phase: str,
    ) -> EngineeringEvent:
        """Resolve explicit evidence gaps with bounded deterministic batch retrieval."""

        limit = (
            self.policy.max_runtime_retrieval_cycles
            if phase == "runtime_repair"
            else self.policy.max_prepare_retrieval_cycles
        )
        cycles = self._retrieval_cycles.get(phase, 0)
        if cycles >= limit:
            return self._event(
                step,
                action.type,
                False,
                f"Evidence retrieval hard fuse reached for {phase} ({limit} cycle(s)); use existing evidence or block.",
            )
        ledger = self._evidence_gap_ledger.setdefault(phase, {})
        performed_retrieval = False
        observed_by_id: dict[str, ObservedEngineeringEvidence] = {}
        new_observed_ids: set[str] = set()
        gap_results: list[dict[str, object]] = []

        for gap in action.gaps:
            existing = ledger.get(gap.gap_id)
            if existing is not None:
                gap_results.append(
                    {
                        "gap_id": gap.gap_id,
                        "status": "already_retrieved_blocked",
                        "message": (
                            "Each evidence gap may be retrieved once. Use the accumulated evidence, "
                            "proceed, or declare a new more-specific gap with refines_gap_id."
                        ),
                    }
                )
                continue

            parent = None
            if gap.refines_gap_id is not None:
                parent = ledger.get(gap.refines_gap_id)
                if parent is None:
                    gap_results.append(
                        {
                            "gap_id": gap.gap_id,
                            "status": "unknown_refinement_parent",
                            "message": f"refines_gap_id {gap.refines_gap_id} is not present in the evidence-gap ledger.",
                        }
                    )
                    continue
                if str(parent.get("status", "")) == "satisfied":
                    gap_results.append(
                        {
                            "gap_id": gap.gap_id,
                            "status": "satisfied_parent_blocked",
                            "message": f"Evidence gap {gap.refines_gap_id} is already satisfied and cannot be refined.",
                        }
                    )
                    continue

            entry = {
                "missing_evidence": gap.missing_evidence,
                "why_required": gap.why_required,
                "refines_gap_id": gap.refines_gap_id,
                "seen_ids": set(),
                "retrievals": 0,
                "last_new_count": 0,
                "status": "open",
            }
            ledger[gap.gap_id] = entry
            if parent is not None:
                parent["status"] = "superseded"
                parent["superseded_by"] = gap.gap_id

            performed_retrieval = True
            found: dict[str, dict[str, object]] = {}
            ref_records: dict[str, dict[str, object]] = {}
            for query in gap.capability_queries:
                for item in self.catalog.search(query, limit=8):
                    provider_id = str(item.get("provider_id", ""))
                    if not provider_id:
                        continue
                    evidence_id = canonical_engineering_evidence_id("capability", provider_id)
                    found[evidence_id] = {
                        "kind": "capability",
                        "reference": provider_id,
                        "query": query,
                        "result": item,
                    }
                    observed_by_id[evidence_id] = ObservedEngineeringEvidence(
                        evidence_id=evidence_id,
                        kind="capability",
                        reference=provider_id,
                        summary=(
                            f"Capability provider {provider_id}: {item.get('name', '')} "
                            f"({item.get('provider_type', '')}, OpenFOAM {item.get('openfoam_version', '')})"
                        )[:1200],
                    )

            for query in gap.reference_queries:
                for item in self.references.search(query, scope=gap.reference_scope, limit=6):
                    reference = str(item.get("reference", ""))
                    if not reference:
                        continue
                    evidence_id = canonical_engineering_evidence_id("openfoam_reference", reference)
                    record = {
                        "kind": "openfoam_reference",
                        "reference": reference,
                        "query": query,
                        "snippet": str(item.get("snippet", ""))[:700],
                    }
                    found[evidence_id] = record
                    ref_records.setdefault(reference, record)

            read_budget = gap.read_top_reference_matches
            for reference, record in list(ref_records.items())[:read_budget]:
                try:
                    excerpt = self.references.read(reference, start_line=1, line_count=48)[:4000]
                except (OSError, ValueError):
                    excerpt = ""
                if excerpt:
                    record["content_excerpt"] = excerpt

            for evidence_id, record in found.items():
                if record["kind"] != "openfoam_reference":
                    continue
                reference = str(record["reference"] )
                detail = str(record.get("content_excerpt") or record.get("snippet") or "")
                observed_by_id[evidence_id] = ObservedEngineeringEvidence(
                    evidence_id=evidence_id,
                    kind="openfoam_reference",
                    reference=reference,
                    summary=(f"Installed OpenFOAM reference {reference}: {detail[:1000]}")[:1200],
                )

            seen_ids = entry.setdefault("seen_ids", set())
            if not isinstance(seen_ids, set):
                seen_ids = set(seen_ids)
                entry["seen_ids"] = seen_ids
            new_ids = sorted(set(found) - seen_ids)
            seen_ids.update(found)
            new_observed_ids.update(new_ids)
            entry["retrievals"] = int(entry.get("retrievals", 0)) + 1
            entry["last_new_count"] = len(new_ids)
            entry["status"] = "evidence_available" if new_ids else "stagnant"
            gap_results.append(
                {
                    "gap_id": gap.gap_id,
                    "status": "new_evidence" if new_ids else "no_new_evidence",
                    "new_evidence_ids": new_ids,
                    "new_evidence": [found[eid] for eid in new_ids],
                    "total_seen": len(seen_ids),
                }
            )

        if performed_retrieval:
            self._retrieval_cycles[phase] = cycles + 1
        new_total = sum(len(item.get("new_evidence_ids", [])) for item in gap_results)
        stagnant = [
            item["gap_id"]
            for item in gap_results
            if item.get("status") in {
                "no_new_evidence",
                "already_retrieved_blocked",
                "unknown_refinement_parent",
                "satisfied_parent_blocked",
            }
        ]
        return self._event(
            step,
            action.type,
            True,
            f"Evidence-gap batch completed: {new_total} new evidence item(s); stagnant={len(stagnant)}.",
            _json({
                "cycle": self._retrieval_cycles.get(phase, cycles),
                "cycle_limit": limit,
                "gaps": gap_results,
            }),
            observed_evidence=[observed_by_id[eid] for eid in sorted(new_observed_ids) if eid in observed_by_id],
        )

    def _runtime_case_file_contract_scan(self, state: CFDState) -> dict[str, object]:
        """Scan solve-critical text files for systematic FoamFile contract failures.

        This is intentionally deterministic and batch-oriented. A runtime failure in one
        IOobject must not force the LLM to discover the same missing-header defect one file
        per foamRun attempt. Core system files, every current initial field, and every
        Agent-declared required solve input are scanned together.
        """

        seals = {item.path: item for item in self.workspace.file_seals()}
        candidates: list[str] = ["system/controlDict", "system/fvSchemes", "system/fvSolution"]
        if state.engineering_plan is not None:
            candidates.extend(state.engineering_plan.required_case_files)
        candidates.extend(path for path in seals if path.startswith("0/"))
        candidates.extend(path for path in seals if path.startswith("system/"))

        invalid: list[dict[str, object]] = []
        checked: list[dict[str, object]] = []
        seen: set[str] = set()
        for path in candidates:
            if path in seen or path not in seals:
                continue
            seen.add(path)
            suffix = Path(path).suffix.lower()
            if suffix in {".stl", ".obj", ".off", ".vtk", ".csv", ".dat", ".emesh"}:
                continue
            try:
                text = self.workspace.read_text(path)
            except (OSError, WorkspaceSafetyError):
                continue
            result = validate_foam_file_header(
                path,
                text,
                expected_class=("dictionary" if path.startswith("system/") else None),
            )
            record = {
                "path": path,
                "class": result.header.class_name or None,
                "object": result.header.object_name or None,
                "valid": result.valid,
            }
            checked.append(record)
            if not result.valid:
                invalid.append({**record, "failures": list(result.failures)})

        return {
            "checked_count": len(checked),
            "invalid_count": len(invalid),
            "invalid": invalid[:20],
        }

    def _runtime_relevant_case_files(self, state: CFDState, runtime_log: str | None) -> list[dict[str, object]]:
        """Return a bounded, failure-focused case slice for runtime repair."""
        seals = {item.path: item for item in self.workspace.file_seals()}
        candidates: list[str] = ["system/fvSchemes", "system/fvSolution", "system/controlDict"]
        text = runtime_log or ""
        for match in re.findall(r"(?:0|constant|system)/[A-Za-z0-9_.\/-]+", text):
            probe = match.rstrip("./")
            parts = probe.split("/")
            while parts:
                candidate = "/".join(parts)
                if candidate in seals:
                    candidates.append(candidate)
                    break
                parts.pop()
        required = state.engineering_plan.required_case_files if state.engineering_plan is not None else []
        for path in required:
            base = path.rsplit("/", 1)[-1]
            if base and re.search(rf"(?<![A-Za-z0-9_]){re.escape(base)}(?![A-Za-z0-9_])", text):
                candidates.append(path)
        result: list[dict[str, object]] = []
        seen: set[str] = set()
        for path in candidates:
            if path in seen or path not in seals:
                continue
            seen.add(path)
            try:
                content = self.workspace.read_text(path)
            except (OSError, WorkspaceSafetyError):
                continue
            result.append(
                {
                    "path": path,
                    "sha256": seals[path].sha256,
                    "content": content[:8000],
                    "truncated": len(content) > 8000,
                }
            )
            if len(result) >= 6:
                break
        return result

    def _dispatch_tool_action(
        self,
        action,
        *,
        step: int,
        native_execution: bool,
        phase: str,
        state: CFDState | None = None,
    ) -> EngineeringEvent:
        try:
            if (
                native_execution
                and state is not None
                and isinstance(
                    action,
                    (
                        ValidateDictionaryAction,
                        SurfaceCheckAction,
                        RunMeshCommandAction,
                        ValidatePreSolveAction,
                    ),
                )
                and self._native_command_count(state) >= self.policy.max_native_commands
            ):
                return self._event(
                    step,
                    getattr(action, "type", "unknown"),
                    False,
                    f"Native OpenFOAM command budget exhausted ({self.policy.max_native_commands}); no command was executed.",
                )

            if isinstance(action, GatherEvidenceAction):
                return self._gather_evidence(action, step=step, phase=phase)

            if isinstance(action, InspectEnvironmentAction):
                payload = {
                    "runtime": self.tools.environment_snapshot(),
                    "capability_graph": self.catalog.summary(),
                    "reference_roots": self.references.summary(),
                }
                return self._event(step, action.type, True, "Environment inspected.", _json(payload))

            if isinstance(action, SearchCapabilitiesAction):
                results = self.catalog.search(action.query)
                observed = [
                    ObservedEngineeringEvidence(
                        evidence_id=canonical_engineering_evidence_id(
                            "capability", str(item["provider_id"])
                        ),
                        kind="capability",
                        reference=str(item["provider_id"]),
                        summary=(
                            f"Capability provider {item['provider_id']}: {item.get('name', '')} "
                            f"({item.get('provider_type', '')}, OpenFOAM {item.get('openfoam_version', '')})"
                        )[:1200],
                    )
                    for item in results
                    if isinstance(item, dict) and item.get("provider_id")
                ]
                return self._event(
                    step,
                    action.type,
                    True,
                    f"Capability search returned {len(results)} provider(s).",
                    _json(results),
                    observed_evidence=observed,
                )

            if isinstance(action, SearchReferencesAction):
                results = self.references.search(action.query, scope=action.scope)
                observed = [
                    ObservedEngineeringEvidence(
                        evidence_id=canonical_engineering_evidence_id(
                            "openfoam_reference", str(item["reference"])
                        ),
                        kind="openfoam_reference",
                        reference=str(item["reference"]),
                        summary=(
                            f"Installed OpenFOAM reference {item['reference']}: "
                            f"{str(item.get('snippet', ''))[:700]}"
                        )[:1200],
                    )
                    for item in results
                    if isinstance(item, dict) and item.get("reference")
                ]
                return self._event(
                    step,
                    action.type,
                    True,
                    f"Reference search returned {len(results)} result(s).",
                    _json(results),
                    observed_evidence=observed,
                )

            if isinstance(action, ReadReferenceAction):
                text = self.references.read(
                    action.reference,
                    start_line=action.start_line,
                    line_count=action.line_count,
                )
                observed = [
                    ObservedEngineeringEvidence(
                        evidence_id=canonical_engineering_evidence_id(
                            "openfoam_reference", action.reference
                        ),
                        kind="openfoam_reference",
                        reference=action.reference,
                        summary=f"Read installed OpenFOAM reference {action.reference}.",
                    )
                ]
                return self._event(
                    step,
                    action.type,
                    True,
                    f"Read {action.reference}.",
                    text,
                    observed_evidence=observed,
                )

            if isinstance(action, ListCaseFilesAction):
                files = [
                    {
                        "path": item.path,
                        "sha256": item.sha256,
                        "size_bytes": item.size_bytes,
                    }
                    for item in self.workspace.file_seals()
                ]
                return self._event(step, action.type, True, "Listed agent-authored case files.", _json(files))

            if isinstance(action, ReadCaseFileAction):
                text = self.workspace.read_text(action.path)
                return self._event(step, action.type, True, f"Read {action.path}.", text)

            if isinstance(action, WriteCaseFileAction):
                if action.path.startswith("postprocessConfig/"):
                    return self._event(
                        step,
                        action.type,
                        False,
                        "postprocessConfig/ is reserved for the post-processing phase after a successful solve.",
                    )
                if state is not None:
                    cycles, failure_pending, current_cycle_started = self._mesh_repair_status(state)
                    if (
                        failure_pending
                        and not current_cycle_started
                        and cycles >= self.policy.max_mesh_repair_cycles
                    ):
                        return self._event(
                            step,
                            action.type,
                            False,
                            f"Mesh repair cycle budget exhausted ({self.policy.max_mesh_repair_cycles}); case edit was not applied.",
                        )
                mesh_affecting = self.workspace.is_mesh_affecting_path(action.path)
                digest = self.workspace.write_text(action.path, action.content)
                self._presolve_case_manifest = None
                self._presolve_required_case_files = None
                if mesh_affecting:
                    self._checkmesh_mesh_manifest = None
                    if state is not None:
                        state.mesh_evidence = None
                if phase == "runtime_repair" and state is not None:
                    # Any runtime repair mutation invalidates the user-approved case seal
                    # until retry_solver revalidates and reseals the current case.
                    state.case_seal = None
                return self._event(
                    step,
                    action.type,
                    True,
                    f"Wrote {action.path}.",
                    artifact_sha256=digest,
                )

            if isinstance(action, PatchCaseFileAction):
                patch = action.patch
                if state is not None:
                    cycles, failure_pending, current_cycle_started = self._mesh_repair_status(state)
                    if failure_pending and not current_cycle_started and cycles >= self.policy.max_mesh_repair_cycles:
                        return self._event(
                            step,
                            action.type,
                            False,
                            f"Mesh repair cycle budget exhausted ({self.policy.max_mesh_repair_cycles}); case patch was not applied.",
                        )
                mesh_affecting = self.workspace.is_mesh_affecting_path(patch.path)
                digest = self.workspace.patch_text_once(patch.path, patch.old, patch.new)
                self._presolve_case_manifest = None
                self._presolve_required_case_files = None
                if mesh_affecting:
                    self._checkmesh_mesh_manifest = None
                    if state is not None:
                        state.mesh_evidence = None
                if phase == "runtime_repair" and state is not None:
                    state.case_seal = None
                return self._event(
                    step, action.type, True, f"Patched {patch.path}.", artifact_sha256=digest
                )

            if isinstance(action, DeleteCaseFileAction):
                if state is not None:
                    cycles, failure_pending, current_cycle_started = self._mesh_repair_status(state)
                    if (
                        failure_pending
                        and not current_cycle_started
                        and cycles >= self.policy.max_mesh_repair_cycles
                    ):
                        return self._event(
                            step,
                            action.type,
                            False,
                            f"Mesh repair cycle budget exhausted ({self.policy.max_mesh_repair_cycles}); case delete was not applied.",
                        )
                mesh_affecting = self.workspace.is_mesh_affecting_path(action.path)
                self.workspace.delete(action.path)
                self._presolve_case_manifest = None
                self._presolve_required_case_files = None
                if mesh_affecting:
                    self._checkmesh_mesh_manifest = None
                    if state is not None:
                        state.mesh_evidence = None
                if phase == "runtime_repair" and state is not None:
                    state.case_seal = None
                return self._event(step, action.type, True, f"Deleted {action.path}.")

            if isinstance(action, ValidateDictionaryAction):
                if not native_execution:
                    return self._event(
                        step,
                        action.type,
                        False,
                        "Native execution is disabled; OpenFOAM file/header validation was not run.",
                    )
                target = self.workspace.resolve_case_path(action.path, must_exist=True)
                text = target.read_text(encoding="utf-8", errors="replace")
                header = validate_foam_file_header(
                    action.path,
                    text,
                    expected_class=("dictionary" if action.path.startswith("system/") else None),
                )
                if not header.valid:
                    return self._event(
                        step,
                        action.type,
                        False,
                        f"OpenFOAM file header rejected {action.path} before foamDictionary.",
                        header.render(),
                    )
                result = self.tools.foam_dictionary_validate(target, cwd=self.workspace.case_dir)
                output = _tool_output(result)
                self.workspace.write_log(f"{step:03d}.foamDictionary.log", output)
                event_output = output
                summary = f"FoamFile header and foamDictionary accepted {action.path}."
                if not result.success:
                    diagnostic = diagnose_openfoam_failure(result, command_name="foamDictionary")
                    event_output = diagnostic.render()
                    summary = (
                        f"foamDictionary returned status {result.return_code}; "
                        "native diagnostic captured."
                    )
                return self._event(
                    step,
                    action.type,
                    result.success,
                    summary,
                    event_output,
                    native_command_executed=True,
                )

            if isinstance(action, SurfaceCheckAction):
                if not native_execution:
                    return self._event(
                        step,
                        action.type,
                        False,
                        "Native execution is disabled; surfaceCheck was not run.",
                    )
                target = self.workspace.resolve_case_path(action.path, must_exist=True)
                result = self.tools.surface_check(target, cwd=self.workspace.case_dir)
                output = _tool_output(result)
                self.workspace.write_log(f"{step:03d}.surfaceCheck.log", output)
                event_output = output
                summary = f"surfaceCheck {'passed' if result.success else 'failed'} for {action.path}."
                if not result.success:
                    diagnostic = diagnose_openfoam_failure(result, command_name="surfaceCheck")
                    event_output = diagnostic.render()
                    summary = (
                        f"surfaceCheck returned status {result.return_code}; "
                        "native diagnostic captured."
                    )
                return self._event(
                    step,
                    action.type,
                    result.success,
                    summary,
                    event_output,
                    native_command_executed=True,
                )

            if isinstance(action, ValidatePreSolveAction):
                if not native_execution:
                    return self._event(
                        step,
                        action.type,
                        False,
                        "Native execution is disabled; pre-solve readiness was not validated.",
                    )
                result = self.presolve.validate_required_case_files(action.required_case_files)
                output = "\n".join(result.failures)
                if result.valid:
                    self._presolve_case_manifest = self.workspace.manifest_digest()
                    self._presolve_required_case_files = tuple(action.required_case_files)
                    output = (
                        f"checkedFiles={len(result.checked_files)}\n"
                        f"meshPatches={len(result.mesh_patches)}"
                    )
                return self._event(
                    step,
                    action.type,
                    result.valid,
                    (
                        "Pre-solve readiness validation passed."
                        if result.valid
                        else "Pre-solve readiness validation failed."
                    ),
                    output,
                    native_command_executed=True,
                )

            if isinstance(action, RunMeshCommandAction):
                if not native_execution:
                    return self._event(
                        step,
                        action.type,
                        False,
                        f"Native execution is disabled; {action.command} was not run.",
                    )
                preflight = self.safety.validate_native_inputs()
                if not preflight.valid:
                    return self._event(
                        step,
                        action.type,
                        False,
                        f"{action.command} blocked by generic syntax/safety preflight.",
                        "\n".join(preflight.failures),
                    )
                precondition_ok, precondition_reason = self._mesh_command_precondition(action.command)
                if not precondition_ok:
                    signature = f"tool_contract:{action.command}:prerequisite"
                    return self._event(
                        step,
                        "mesh_tool_precondition",
                        False,
                        f"{action.command} blocked by deterministic executable precondition.",
                        precondition_reason,
                        failure_signature=signature,
                        failure_scope="strategy",
                    )

                result = self.tools.run_mesh_command(action.command, self.workspace.case_dir)
                if action.command != "checkMesh":
                    self._presolve_case_manifest = None
                    self._presolve_required_case_files = None
                if action.command in _MESH_TOPOLOGY_MUTATING_COMMANDS:
                    # blockMesh/snappyHexMesh/createPatch can change polyMesh patch topology.
                    # Any previous checkMesh/pre-solve/field-boundary compatibility evidence
                    # is stale until the new topology is checked again. This holds even on a
                    # failed native command because partial filesystem mutation is possible.
                    self._checkmesh_mesh_manifest = None
                    if state is not None:
                        state.mesh_evidence = None
                        if phase == "runtime_repair":
                            state.case_seal = None
                output = _tool_output(result)
                self.workspace.write_log(f"{step:03d}.{action.command}.log", output)
                event_output = output
                event_success = result.success
                summary = f"{action.command} returned status {result.return_code}."
                failure_signature = None
                failure_scope = None
                if not result.success:
                    diagnostic = diagnose_openfoam_failure(result, command_name=action.command)
                    event_output = diagnostic.render()
                    failure_signature = self._native_failure_signature(
                        action.command, diagnostic.kind, diagnostic.excerpt
                    )
                    failure_scope = "local"
                    summary = (
                        f"{action.command} returned status {result.return_code}; "
                        "native diagnostic captured."
                    )
                if action.command == "checkMesh" and state is not None:
                    evidence = parse_check_mesh_evidence(result)
                    state.mesh_evidence = evidence
                    event_success = evidence.passed
                    summary = (
                        f"checkMesh returned status {result.return_code}; "
                        f"evidence {'passed' if evidence.passed else 'failed'}."
                    )
                    if evidence.passed:
                        self._checkmesh_mesh_manifest = self.workspace.mesh_manifest_digest()
                return self._event(
                    step,
                    action.type,
                    event_success,
                    summary,
                    event_output,
                    native_command_executed=True,
                    mesh_command_executed=True,
                    failure_signature=failure_signature,
                    failure_scope=failure_scope,
                )
        except (ValueError, FileNotFoundError, WorkspaceSafetyError, OSError) as exc:
            return self._event(
                step,
                getattr(action, "type", "unknown"),
                False,
                f"Tool action rejected: {type(exc).__name__}: {exc}",
            )

        return self._event(
            step,
            getattr(action, "type", "unknown"),
            False,
            f"Action is not valid in {phase} phase.",
        )

    @staticmethod
    def _revision_record(
        state: CFDState,
        *,
        proposal_id: str,
        feedback_ids: list[str],
        before_plan: EngineeringPlan,
        before_seal,
        after_plan: EngineeringPlan,
        after_seal,
    ) -> RevisionRecord:
        before = {item.path: item.sha256 for item in before_seal.files}
        after = {item.path: item.sha256 for item in after_seal.files}
        changes: list[RevisionFileChange] = []
        for path in sorted(before.keys() | after.keys()):
            old = before.get(path)
            new = after.get(path)
            if old == new:
                continue
            if old is None:
                change = "added"
            elif new is None:
                change = "removed"
            else:
                change = "modified"
            changes.append(
                RevisionFileChange(
                    path=path,
                    change=change,
                    before_sha256=old,
                    after_sha256=new,
                )
            )
        return RevisionRecord(
            revision_id=f"rev-{len(state.revision_history) + 1:04d}",
            proposal_id=proposal_id,
            feedback_ids=list(feedback_ids),
            before_plan_sha256=before_plan.digest(),
            after_plan_sha256=after_plan.digest(),
            before_manifest_sha256=before_seal.manifest_sha256,
            after_manifest_sha256=after_seal.manifest_sha256,
            archive_path=state.pending_revision_archive_path,
            file_changes=changes,
        )

    def _case_plan_authoring_failure_count(self, state: CFDState) -> int:
        return sum(
            1
            for event in self._current_round_events(state)
            if (
                not event.success
                and event.action_type in {"typed_dictionary_serialize", "case_bundle_preflight"}
            )
        )

    def _record_case_plan_authoring_failure(
        self,
        state: CFDState,
        event: EngineeringEvent,
    ) -> bool:
        """Append a pre-commit failure and stop after a small dedicated retry budget.

        Returns True when the workflow was transitioned to ENGINEERING_BLOCKED.
        """

        state.engineering_events.append(event)
        failures = self._case_plan_authoring_failure_count(state)
        if failures >= self.policy.max_case_plan_authoring_retries:
            state.transition(
                State.ENGINEERING_BLOCKED,
                "Complete case-plan authoring repeatedly failed deterministic pre-commit checks "
                f"({failures}/{self.policy.max_case_plan_authoring_retries}).",
            )
            return True
        return False

    def _case_plan_retry_required(self, state: CFDState) -> bool:
        """Return whether the previous complete-plan authoring attempt failed pre-commit.

        These failures leave the workspace intentionally untouched, but the rejected
        complete candidate remains in Python memory. The compact retry contract therefore
        accepts only a delta against that retained candidate (or block).
        """

        events = self._current_round_events(state)
        if not events:
            return False
        last = events[-1]
        return (
            not last.success
            and last.action_type in {"typed_dictionary_serialize", "case_bundle_preflight"}
            and self._pending_execution_plan is None
            and self._pending_candidate_execution is not None
        )

    def _mesh_tool_contracts(self) -> list[dict[str, object]]:
        provider = getattr(self.tools, "mesh_tool_contracts", None)
        if callable(provider):
            value = provider()
            return list(value) if isinstance(value, list) else []
        return []

    def _mesh_command_precondition(self, command: str) -> tuple[bool, str]:
        checker = getattr(self.tools, "mesh_command_precondition", None)
        if callable(checker):
            return checker(command, self.workspace.case_dir)
        return True, ""

    @staticmethod
    def _native_failure_signature(command: str, kind: str, excerpt: str) -> str:
        """Normalize a native failure enough to detect repeated no-progress repairs."""
        primary = ""
        for line in excerpt.splitlines():
            text = line.strip()
            if not text:
                continue
            lowered = text.casefold()
            if (
                "foam fatal" in lowered
                or lowered.startswith("file:")
                or lowered.startswith("from function")
                or lowered.startswith("in file")
                or text.startswith("#")
            ):
                continue
            primary = text
            break
        if not primary:
            primary = kind
        primary = re.sub(r"<[^>]+>", "<PATH>", primary)
        primary = re.sub(r"\s+", " ", primary).strip().casefold()
        digest = hashlib.sha256(f"{command}\0{kind}\0{primary}".encode("utf-8", errors="replace")).hexdigest()[:20]
        return f"native:{command}:{kind}:{digest}"

    def _strategy_revision_required(self, state: CFDState) -> bool:
        events = self._current_round_events(state)
        failures = [event for event in events if not event.success]
        if not failures:
            return False
        # Escalation applies only to the *current* failure. A later unrelated
        # validation failure must not resurrect an already-addressed strategy fault.
        last = failures[-1]
        if not last.failure_signature:
            return False
        if last.failure_scope == "strategy":
            return True
        same = [
            event for event in failures
            if event.failure_signature == last.failure_signature
        ]
        return len(same) >= 2 and last.mesh_command_executed

    def _phase_contract(self, state: CFDState, phase: str):
        if not self.policy.compact_phase_schemas:
            return EngineeringTurn, ENGINEERING_SYSTEM_PROMPT, "legacy"
        if phase in {"prepare_finalize", "human_revision_finalize"}:
            return FinalizationTurn, FINALIZATION_SYSTEM_PROMPT, "finalize"
        if phase == "runtime_repair":
            return RuntimeRepairTurn, RUNTIME_REPAIR_SYSTEM_PROMPT, "runtime_repair"
        if phase.startswith("human_revision"):
            return RevisionTurn, REVISION_SYSTEM_PROMPT, "revision"
        if phase == "prepare" and self._case_plan_retry_required(state):
            return CasePlanRetryTurn, CASE_PLAN_RETRY_SYSTEM_PROMPT, "replan"
        if phase == "prepare" and self._strategy_revision_required(state):
            return StrategyRevisionTurn, STRATEGY_REVISION_SYSTEM_PROMPT, "strategy_revision"
        if phase == "prepare" and self._pending_execution_plan is not None:
            return RepairTurn, REPAIR_SYSTEM_PROMPT, "repair"
        if phase == "prepare" and self._retrieval_cycles.get("prepare", 0) >= self.policy.max_prepare_retrieval_cycles:
            return PrepareDecisionOnlyTurn, PREPARE_DECISION_ONLY_SYSTEM_PROMPT, "prepare_decide"
        return PrepareTurn, PREPARE_SYSTEM_PROMPT, "prepare"

    def _engineering_conversation_key(self, state: CFDState, contract_phase: str) -> str:
        if contract_phase == "runtime_repair":
            return f"eng:{state.run_id}:runtime:{state.simulation_attempts}"
        if contract_phase == "revision":
            proposal = state.active_revision_proposal
            suffix = proposal.proposal_id if proposal is not None else str(len(state.revision_history))
            return f"eng:{state.run_id}:revision:{suffix}"
        return f"eng:{state.run_id}:round:{state.engineering_round_start_index}"

    def _generate_turn(
        self,
        state: CFDState,
        *,
        step: int,
        local_step: int | None = None,
        current_step_limit: int | None = None,
        phase: str,
        runtime_log: str | None = None,
        native_execution: bool = True,
    ):
        if local_step is None:
            local_step = step
        if current_step_limit is None:
            current_step_limit = (
                self.policy.max_runtime_repair_steps
                if phase == "runtime_repair"
                else self.policy.max_agent_steps
            )

        schema, system_prompt, contract_phase = self._phase_contract(state, phase)
        conversation_key = self._engineering_conversation_key(state, contract_phase)
        prompt_count = self._phase_prompt_counts.get(conversation_key, 0)
        plan_digest = state.engineering_plan.digest() if state.engineering_plan is not None else None
        manifest_digest = self.workspace.manifest_digest()
        evidence_records = [
            item.model_dump(mode="json")
            for item in self._observed_evidence_registry(state).values()
        ]
        case_files = [
            {"path": item.path, "sha256": item.sha256, "size_bytes": item.size_bytes}
            for item in self.workspace.file_seals()
        ]
        budget = {
            "llm_limit": current_step_limit,
            "llm_remaining": max(0, current_step_limit - local_step + 1),
            "initial_engineering_step_budget": self.policy.max_agent_steps,
            "current_engineering_step_limit": current_step_limit,
            "hard_engineering_step_limit": self.policy.hard_max_agent_steps,
            "steps_remaining_in_current_window": max(0, current_step_limit - local_step + 1),
            "finalization_only": phase in {"prepare_finalize", "human_revision_finalize"},
            "tool_limit": (
                self.policy.max_runtime_repair_tool_actions
                if phase == "runtime_repair"
                else self.policy.max_tool_actions
            ),
            "tool_used": self._tool_action_count(state),
            "native_limit": self.policy.max_native_commands,
            "native_used": self._native_command_count(state),
            "retrieval_cycles_used": self._retrieval_cycles.get(phase, 0),
            "retrieval_cycle_limit": (
                self.policy.max_runtime_retrieval_cycles
                if phase == "runtime_repair"
                else self.policy.max_prepare_retrieval_cycles
            ),
        }
        bindings = {
            "intake_sha256": state.intake_digest,
            "plan_sha256": plan_digest,
            "manifest_sha256": manifest_digest,
            "check_mesh_passed": bool(state.mesh_evidence and state.mesh_evidence.passed),
            "check_mesh_log_sha256": (
                state.mesh_evidence.raw_log_sha256 if state.mesh_evidence else None
            ),
        }

        # Delta mode is only safe when the backend can chain the previous response.
        supports_stateful = False
        try:
            import inspect
            supports_stateful = "conversation_key" in inspect.signature(self.llm.generate).parameters
        except (TypeError, ValueError):
            supports_stateful = False
        use_delta = bool(
            self.policy.state_delta_context
            and prompt_count > 0
            and contract_phase != "runtime_repair"
        )
        previous_snapshot = self._phase_context_snapshots.get(conversation_key, {})

        if contract_phase == "runtime_repair":
            payload: dict[str, object] = {
                "state_mode": "runtime_failure_slice",
                "phase": phase,
                "step": step,
                "confirmed_facts": [
                    {"id": fact.id, "value": fact.value, "source": fact.source}
                    for fact in (state.intake.facts if state.intake is not None else [])
                    if fact.category != "context"
                ],
                "approved_plan": (
                    {
                        "solver": state.engineering_plan.solver,
                        "solver_provider_id": state.engineering_plan.solver_provider_id,
                        "temporal_behavior": state.engineering_plan.temporal_behavior,
                        "motion_kind": state.engineering_plan.motion_kind,
                        "mesh_motion_requirement": state.engineering_plan.mesh_motion_requirement,
                        "required_case_files": state.engineering_plan.required_case_files,
                        "confirmed_fact_bindings": [
                            item.model_dump(mode="json")
                            for item in state.engineering_plan.confirmed_fact_bindings
                        ],
                    }
                    if state.engineering_plan is not None else None
                ),
                "native_failure": self._redact_local_paths(runtime_log[-6000:]) if runtime_log else None,
                "case_file_contract_scan": self._runtime_case_file_contract_scan(state),
                "relevant_case_files": self._runtime_relevant_case_files(state, runtime_log),
                "mesh_evidence": {
                    "passed": bool(state.mesh_evidence and state.mesh_evidence.passed),
                    "cell_count": state.mesh_evidence.cell_count if state.mesh_evidence else None,
                    "manifest_current": bool(
                        self._checkmesh_mesh_manifest
                        and self._checkmesh_mesh_manifest == self.workspace.mesh_manifest_digest()
                    ),
                },
                "available_evidence": evidence_records[-20:],
                "evidence_gap_status": self._evidence_gap_status("runtime_repair"),
                "bindings": bindings,
                "budget": budget,
            }
            instruction = (
                "Repair the actual runtime failure from this bounded diagnostic/file slice. "
                "Treat case_file_contract_scan as deterministic evidence: when it reports multiple "
                "invalid solve inputs, repair the whole systematic class of file-contract defects "
                "in one cycle rather than waiting for foamRun to fail on each file. "
                "Prefer repair_runtime_case. Use gather_evidence only for an explicit missing "
                "tool/version fact that the supplied files and native diagnostic cannot resolve:\n"
            )
        elif use_delta:
            payload = {
                "state_mode": "delta_from_previous_response",
                "phase": phase,
                "step": step,
                "confirmed_facts": [
                    {"id": fact.id, "value": fact.value, "source": fact.source}
                    for fact in (state.intake.facts if state.intake is not None else [])
                    if fact.category != "context"
                ],
                "baseline_plan": (
                    {
                        "case_name": state.engineering_plan.case_name,
                        "solver": state.engineering_plan.solver,
                        "solver_provider_id": state.engineering_plan.solver_provider_id,
                        "openfoam_version": state.engineering_plan.openfoam_version,
                        "temporal_behavior": state.engineering_plan.temporal_behavior,
                        "motion_kind": state.engineering_plan.motion_kind,
                        "mesh_motion_requirement": state.engineering_plan.mesh_motion_requirement,
                        "required_case_files": state.engineering_plan.required_case_files,
                        "confirmed_fact_bindings": [
                            item.model_dump(mode="json")
                            for item in state.engineering_plan.confirmed_fact_bindings
                        ],
                        "confirmed_intake_sha256": state.engineering_plan.confirmed_intake_sha256,
                    }
                    if state.engineering_plan is not None
                    else (
                        {
                            "solver": self._pending_execution_plan.solver,
                            "solver_provider_id": self._pending_execution_plan.solver_provider_id,
                            "required_case_files": self._pending_execution_plan.required_case_files,
                            "confirmed_fact_bindings": [
                                item.model_dump(mode="json")
                                for item in self._pending_execution_plan.confirmed_fact_bindings
                            ],
                            "confirmed_intake_sha256": self._pending_execution_plan.confirmed_intake_sha256,
                        }
                        if self._pending_execution_plan is not None else None
                    )
                ),
                "bindings": bindings,
                "current_case_files": case_files,
                "recent_observations": self._recent_observations_for_model(state),
                "available_evidence": evidence_records,
                "evidence_gap_status": self._evidence_gap_status(phase),
                "tool_execution_contracts": self._mesh_tool_contracts(),
                "budget": budget,
                "ready_for_finalization": (
                    self._ready_for_finalization(state, native_execution=native_execution)
                    if phase in {"prepare", "prepare_finalize", "human_revision", "human_revision_finalize"}
                    else False
                ),
                "runtime_log_excerpt": (
                    self._redact_local_paths(runtime_log[-4000:]) if runtime_log else None
                ),
                "human_feedback": [
                    {
                        "feedback_id": item.feedback_id,
                        "text": item.text,
                        "status": item.status,
                    }
                    for item in state.human_feedback[-self.policy.max_model_feedback_items :]
                ],
                "active_revision_proposal": (
                    state.active_revision_proposal.model_dump(mode="json")
                    if state.active_revision_proposal is not None else None
                ),
                "baseline_plan_decisions": (
                    [item.model_dump(mode="json") for item in state.engineering_plan.decisions]
                    if state.engineering_plan is not None and contract_phase == "revision" else []
                ),
            }
            if previous_snapshot.get("plan_sha256") != plan_digest:
                payload["current_engineering_plan"] = (
                    state.engineering_plan.model_dump(mode="json")
                    if state.engineering_plan is not None else None
                )
            instruction = (
                "This is a compact state delta/capsule. Preserve confirmed_facts and baseline_plan; "
                "use recent evidence to return only the next changed action. Do not regenerate unchanged "
                "case content:\n"
            )
        else:
            payload = {
                "state_mode": "full",
                "phase": phase,
                "step": step,
                "confirmed_intake": confirmed_intake_definition(state),
                "intake_sha256": state.intake_digest,
                "exploratory_assumptions_authorized": state.user_request.exploratory_completion_authorized,
                "environment_hint": self.tools.environment_snapshot(),
                "capability_graph_hint": self.catalog.summary(),
                "preloaded_capability_providers": (
                    self.catalog.search("", limit=self.policy.max_preloaded_capabilities)
                    if self.policy.preload_capabilities else []
                ),
                "reference_roots": self.references.summary(),
                "current_case_files": case_files,
                "current_engineering_plan": (
                    state.engineering_plan.model_dump(mode="json")
                    if state.engineering_plan is not None else None
                ),
                "pending_engineering_plan": (
                    self._pending_execution_plan.model_dump(mode="json")
                    if self._pending_execution_plan is not None else None
                ),
                "recent_observations": self._recent_observations_for_model(state),
                "cumulative_provenance": self._cumulative_provenance_summary(state),
                "available_evidence": evidence_records,
                "evidence_gap_status": self._evidence_gap_status(phase),
                "bindings": bindings,
                "deterministic_bindings": {
                    "confirmed_intake": {
                        "bound_by": "python",
                        "sha256": state.intake_digest,
                        "fact_ids": sorted(
                            fact.id for fact in state.intake.facts if fact.category != "context"
                        ) if state.intake is not None else [],
                    },
                    "check_mesh": {
                        "bound_by": "python",
                        "passed": bool(state.mesh_evidence and state.mesh_evidence.passed),
                        "cell_count": state.mesh_evidence.cell_count if state.mesh_evidence else None,
                        "raw_log_sha256": state.mesh_evidence.raw_log_sha256 if state.mesh_evidence else None,
                    },
                    "case_manifest": {"bound_by": "python", "sha256": manifest_digest},
                },
                "tool_execution_contracts": self._mesh_tool_contracts(),
                "budget": budget,
                "ready_for_finalization": (
                    self._ready_for_finalization(state, native_execution=native_execution)
                    if phase in {"prepare", "prepare_finalize", "human_revision", "human_revision_finalize"}
                    else False
                ),
                "human_feedback": [
                    item.model_dump(mode="json")
                    for item in state.human_feedback[-self.policy.max_model_feedback_items :]
                ],
                "active_revision_proposal": (
                    state.active_revision_proposal.model_dump(mode="json")
                    if state.active_revision_proposal is not None else None
                ),
                "runtime_log_excerpt": (
                    self._redact_local_paths(runtime_log[-4000:]) if runtime_log else None
                ),
            }
            if contract_phase == "prepare":
                instruction = (
                    "Choose the next engineering action. Prefer execute_case_plan when ready. If an "
                    "external tool/version fact is genuinely missing, declare one or more explicit gaps "
                    "in a single gather_evidence batch; engineering-choice unknowns are assumptions, not "
                    "retrieval gaps:\n"
                )
            elif contract_phase == "prepare_decide":
                instruction = (
                    "The bounded retrieval window is closed. Use the accumulated evidence to return "
                    "execute_case_plan, or block if faithful implementation is impossible:\n"
                )
            elif contract_phase == "replan":
                instruction = (
                    "The previous complete case bundle failed deterministic pre-commit authoring checks. "
                    "The full candidate is retained in Python memory. Return only repair_candidate_case_plan "
                    "with the minimum delta for the implicated candidate path(s), or block. No partial "
                    "workspace case was committed:\n"
                )
            elif contract_phase == "strategy_revision":
                instruction = (
                    "The current meshing strategy was invalidated by a deterministic tool contract or repeated identical native failure. "
                    "Return revise_mesh_strategy with a different compatible meshing pipeline; do not retry the invalidated command unless its prerequisite state is explicitly changed:\n"
                )
            elif contract_phase in {"repair", "revision"}:
                instruction = (
                    "Return a delta-only repair. Prefer exact patches and do not repeat unchanged files "
                    "or plan content. Use observed failure/evidence only:\n"
                )
            else:
                instruction = "Finalize or block from the validated state:\n"

        if contract_phase == "replan":
            payload["retained_candidate"] = self._candidate_repair_context()

        prompt_result = build_bounded_json_prompt(
            instruction,
            payload,
            max_chars=self.policy.max_model_prompt_chars,
        )
        metrics = structured_request_metrics(
            schema,
            prompt_result.prompt,
            system_prompt=system_prompt,
        )
        metrics["compacted"] = prompt_result.compacted
        metrics["deltaContext"] = use_delta
        metrics["contractPhase"] = contract_phase
        model_name = getattr(self.llm, "model", None)
        if isinstance(model_name, str) and model_name:
            metrics["model"] = model_name
        max_output_tokens = getattr(self.llm, "max_output_tokens", None)
        if max_output_tokens is not None:
            metrics["maxOutputTokens"] = max_output_tokens
        self.progress.emit(
            ProgressEvent(
                phase="llm-context",
                message=f"{phase} LLM context 준비",
                status="info",
                step=local_step,
                limit=current_step_limit,
                metrics=metrics,
            )
        )

        kwargs = {
            "system_prompt": system_prompt,
        }
        if supports_stateful:
            kwargs.update(
                {
                    "conversation_key": conversation_key,
                    "use_previous_response": bool(use_delta and getattr(self.llm, "store", False)),
                    "prompt_cache_key": f"ofa-eng-{contract_phase}",
                }
            )
        turn = self.llm.generate(schema, prompt_result.prompt, **kwargs)
        self._phase_prompt_counts[conversation_key] = prompt_count + 1
        feedback_digest = hashlib.sha256(
            json.dumps(
                [item.model_dump(mode="json") for item in state.human_feedback],
                ensure_ascii=True,
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()
        self._phase_context_snapshots[conversation_key] = {
            "plan_sha256": plan_digest,
            "manifest_sha256": manifest_digest,
            "feedback_sha256": feedback_digest,
        }
        usage = getattr(self.llm, "last_usage", None)
        if isinstance(usage, dict) and usage:
            usage = dict(usage)
            if isinstance(model_name, str) and model_name:
                usage["model"] = model_name
            self.progress.emit(
                ProgressEvent(
                    phase="llm-usage",
                    message=f"{phase} OpenAI token usage",
                    status="info",
                    step=local_step,
                    limit=current_step_limit,
                    metrics=usage,
                )
            )
        return turn

    def _validate_observed_provenance(
        self,
        plan: EngineeringPlan,
        state: CFDState,
    ) -> list[str]:
        """Reject LLM-selected evidence IDs that Python did not issue in this run."""

        failures: list[str] = []
        registry = self._observed_evidence_registry(state)
        capability_ids = {
            item.reference: item.evidence_id
            for item in registry.values()
            if item.kind == "capability"
        }
        if not capability_ids:
            failures.append(
                "Engineering plan has no successful capability-graph observation in this run."
            )

        provider = self.catalog.provider(plan.solver_provider_id)
        if provider is None:
            failures.append(
                f"Solver provider '{plan.solver_provider_id}' does not exist in the loaded capability graph."
            )
        else:
            if provider.provider_type not in {"solver", "generated_solver"}:
                failures.append(
                    f"Capability provider '{plan.solver_provider_id}' is not a solver provider."
                )
            if provider.name != plan.solver:
                failures.append(
                    f"Engineering plan solver '{plan.solver}' disagrees with capability provider "
                    f"'{plan.solver_provider_id}' ({provider.name})."
                )
            if provider.openfoam_version != plan.openfoam_version:
                failures.append(
                    f"Solver provider '{plan.solver_provider_id}' targets OpenFOAM "
                    f"{provider.openfoam_version}, not {plan.openfoam_version}."
                )
        if plan.solver_provider_id not in capability_ids:
            failures.append(
                f"Solver provider '{plan.solver_provider_id}' was not present in deterministic capability evidence supplied to this run."
            )

        for evidence in plan.evidence:
            if evidence.evidence_id not in registry:
                failures.append(
                    "Engineering evidence ID was not issued by the deterministic evidence "
                    f"registry in this run: {evidence.evidence_id}"
                )
        return failures

    def _observed_evidence_registry(
        self,
        state: CFDState,
    ) -> dict[str, ObservedEngineeringEvidence]:
        registry: dict[str, ObservedEngineeringEvidence] = {}

        # v2.9 fast path: a small capability graph can be deterministically exposed
        # before the first LLM call. This is still Python-issued evidence; the model
        # chooses the solver/provider, while Python later validates the choice.
        if self.policy.preload_capabilities:
            for item in self.catalog.search("", limit=self.policy.max_preloaded_capabilities):
                provider_id = str(item.get("provider_id", ""))
                if not provider_id:
                    continue
                evidence = ObservedEngineeringEvidence(
                    evidence_id=canonical_engineering_evidence_id("capability", provider_id),
                    kind="capability",
                    reference=provider_id,
                    summary=(
                        f"Preloaded capability provider {provider_id}: {item.get('name', '')} "
                        f"({item.get('provider_type', '')}, OpenFOAM {item.get('openfoam_version', '')})"
                    )[:1200],
                )
                registry[evidence.evidence_id] = evidence

        for event in state.engineering_events:
            if not event.success:
                continue
            for item in event.observed_evidence:
                registry[item.evidence_id] = item
        return dict(sorted(registry.items()))

    def _cumulative_provenance_summary(self, state: CFDState) -> dict[str, object]:
        successful = [event for event in state.engineering_events if event.success]
        provider_ids: set[str] = set()
        reference_hints: set[str] = set()
        for event in successful:
            if event.action_type == "search_capabilities":
                try:
                    payload = json.loads(event.output_excerpt)
                except (TypeError, json.JSONDecodeError):
                    payload = []
                if isinstance(payload, list):
                    for item in payload:
                        if isinstance(item, dict) and isinstance(item.get("provider_id"), str):
                            provider_ids.add(item["provider_id"])
            elif event.action_type in {"search_references", "read_reference"}:
                # Keep only compact hints; the deterministic gate still checks the
                # original full event history rather than trusting this summary.
                if event.summary:
                    reference_hints.add(self._redact_local_paths(event.summary)[:300])

        registry = self._observed_evidence_registry(state)
        provider_ids.update(
            item.reference for item in registry.values() if item.kind == "capability"
        )
        return {
            "successful_action_types": sorted({event.action_type for event in successful}),
            "observed_capability_provider_ids": sorted(provider_ids),
            "reference_observation_summaries": sorted(reference_hints)[-12:],
            "canonical_evidence_ids": list(registry)[-40:],
            "mesh_evidence_passed": bool(state.mesh_evidence and state.mesh_evidence.passed),
        }

    def _redact_event_for_model(self, event: EngineeringEvent) -> dict[str, object]:
        payload = compact_event_for_model(
            event,
            excerpt_chars=self.policy.model_event_excerpt_chars,
        )
        payload["summary"] = self._redact_local_paths(str(payload.get("summary", "")))
        payload["output_excerpt"] = self._redact_local_paths(
            str(payload.get("output_excerpt", ""))
        )
        return payload

    def _recent_observations_for_model(self, state: CFDState) -> list[dict[str, object]]:
        """Return compact LLM observations, collapsing sequence members into one record.

        Raw EngineeringEvent objects remain in CFDState for deterministic evidence,
        budgets, diagnostics and audit. Only the model projection is compacted.
        """

        groups: list[dict[str, object]] = []
        events = state.engineering_events
        index = 0
        while index < len(events):
            event = events[index]
            if not event.sequence_id:
                groups.append(self._redact_event_for_model(event))
                index += 1
                continue

            sequence_id = event.sequence_id
            members: list[EngineeringEvent] = []
            while index < len(events) and events[index].sequence_id == sequence_id:
                members.append(events[index])
                index += 1

            failed = next((item for item in members if not item.success), None)
            sequence_projection: dict[str, object] = {
                "kind": "engineering_sequence_summary",
                "step": members[0].step,
                "sequence_id": sequence_id,
                "goal": self._redact_local_paths(members[0].sequence_goal or "")[:1000],
                "success": failed is None,
                "planned_actions": members[0].sequence_length,
                "executed_actions": len(members),
                "stopped_early": bool(
                    members[0].sequence_length
                    and len(members) < members[0].sequence_length
                ),
                "actions": [
                    {
                        "action_type": item.action_type,
                        "success": item.success,
                        "summary": self._redact_local_paths(item.summary)[:500],
                        **(
                            {"artifact_sha256": item.artifact_sha256}
                            if item.artifact_sha256
                            else {}
                        ),
                    }
                    for item in members
                ],
                "native_commands_executed": sum(
                    1 for item in members if item.native_command_executed
                ),
                "mesh_commands_executed": sum(
                    1 for item in members if item.mesh_command_executed
                ),
            }
            if failed is not None and failed.output_excerpt.strip():
                sequence_projection["failure_output_excerpt"] = self._redact_local_paths(
                    failed.output_excerpt[-self.policy.model_event_excerpt_chars :]
                )
            groups.append(sequence_projection)

        return groups[-self.policy.observation_history :]

    def redact_native_observation(self, text: str) -> str:
        """Return a display/model-safe native diagnostic without changing its meaning."""
        return self._redact_local_paths(text)

    def _redact_local_paths(self, text: str) -> str:
        replacements: list[tuple[str, str]] = []
        known = [
            (str(self.workspace.case_dir), "<CASE_DIR>"),
            (str(self.workspace.root), "<WORKSPACE>"),
            (os.environ.get("WM_PROJECT_DIR", ""), "<OPENFOAM_ROOT>"),
            (os.environ.get("HOME", ""), "<HOME>"),
        ]
        for raw, marker in known:
            if raw:
                replacements.append((str(Path(raw).expanduser()), marker))
                try:
                    replacements.append((str(Path(raw).expanduser().resolve()), marker))
                except OSError:
                    pass
        redacted = text
        for raw, marker in sorted(set(replacements), key=lambda item: len(item[0]), reverse=True):
            redacted = redacted.replace(raw, marker)

        # Remove any other absolute Unix path appearing in model-bound tool/log text.
        # Preserve only the basename as a diagnostic hint. User-provided prompt text is
        # not passed through this function.
        absolute_path = re.compile(r"(?<![A-Za-z0-9_:/])/(?P<body>[A-Za-z0-9._~+-][^\s\"'<>|;()]*)")

        def replace_path(match: re.Match[str]) -> str:
            raw = "/" + match.group("body")
            core = raw.rstrip(".,:")
            suffix = raw[len(core):]
            name = Path(core).name
            marker = f"<LOCAL_PATH:{name}>" if name else "<LOCAL_PATH>"
            return marker + suffix

        return absolute_path.sub(replace_path, redacted)

    def _emit_action_started(
        self,
        phase: str,
        action: object,
        *,
        step: int,
        limit: int,
    ) -> None:
        importance = action_importance(str(getattr(action, "type", "action")))
        self.progress.emit(
            ProgressEvent(
                phase=phase,
                message=describe_action(action),
                status="start",
                step=step,
                limit=limit,
                importance=importance,
            )
        )

    def _emit_engineering_event(
        self,
        phase: str,
        event: EngineeringEvent,
        *,
        step: int,
        limit: int,
        state: CFDState | None = None,
    ) -> None:
        metrics: dict[str, object] = {}
        if event.action_type == "run_mesh_command" and state is not None and state.mesh_evidence is not None:
            evidence = state.mesh_evidence
            metrics = {
                "cells": evidence.cell_count,
                "maxNonOrtho": evidence.max_non_orthogonality,
                "maxSkew": evidence.max_skewness,
            }
        details: tuple[str, ...] = ()
        if not event.success and event.output_excerpt.strip():
            if event.action_type == "finish_preview":
                details = tuple(
                    self._redact_local_paths(line.strip())[:800]
                    for line in event.output_excerpt.splitlines()
                    if line.strip()
                )[:12]
            elif event.native_command_executed:
                details = tuple(
                    self._redact_local_paths(line.rstrip())[:800]
                    for line in event.output_excerpt.splitlines()
                    if line.strip()
                )[:24]
        self.progress.emit(
            ProgressEvent(
                phase=phase,
                message=event.summary,
                status="success" if event.success else "failure",
                step=step,
                limit=limit,
                importance=action_importance(event.action_type),
                metrics=metrics,
                details=details,
            )
        )

    def _event(
        self,
        step: int,
        action_type: str,
        success: bool,
        summary: str,
        output: str = "",
        *,
        artifact_sha256: str | None = None,
        native_command_executed: bool = False,
        mesh_command_executed: bool = False,
        failure_signature: str | None = None,
        failure_scope: str | None = None,
        observed_evidence: list[ObservedEngineeringEvidence] | None = None,
    ) -> EngineeringEvent:
        if len(output) > self.policy.max_observation_chars:
            output = output[-self.policy.max_observation_chars:]
            output = "... [truncated]\n" + output
        return EngineeringEvent(
            step=step,
            action_type=action_type,
            success=success,
            summary=summary,
            output_excerpt=output,
            artifact_sha256=artifact_sha256,
            native_command_executed=native_command_executed,
            mesh_command_executed=mesh_command_executed,
            failure_signature=failure_signature,
            failure_scope=failure_scope,
            observed_evidence=list(observed_evidence or []),
        )


def _tool_output(result) -> str:
    return "\n".join(part for part in (result.stdout, result.stderr) if part)


def _json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2)
