from __future__ import annotations

from openfoam_agent.engineering.authoring_tasks import accept_task
from openfoam_agent.engineering.design_seal import DesignSealError, materialize_engineering_plan
from openfoam_agent.engineering.repair_context import repair_episode_requires_direct_attempt
from openfoam_agent.contracts.regions import validate_design

import re

from openfoam_agent.progress import ProgressEvent
from openfoam_agent.schemas.engineering import (
    BlockAction,
    DeleteCaseFileAction,
    EngineeringEvent,
    EngineeringPlan,
    EngineeringSequenceAction,
    ExecuteCasePlanAction,
    DesignCaseAction,
    CaseAuthoringAction,
    CandidateCasePlanRepairAction,
    CandidateBlockMeshRepairAction,
    BlockMeshRepairAction,
    GatherEvidenceAction,
    RepairCasePlanAction,
    StrategyRevisionAction,
    FinishPreviewAction,
    ReadReferenceAction,
    PatchCaseFileAction,
    RetrySolverAction,
    RunMeshCommandAction,
    RunNativeOpenFOAMAction,
    SearchCapabilitiesAction,
    SearchReferencesAction,
    WriteCaseFileAction,
)
from openfoam_agent.workflow.state import CFDState
from openfoam_agent.workflow.states import State

# v4.8 action/decision phase controller.


def execute_prepare_decision(self, state, action, **kwargs):
    self._active_transaction = {"type": action.type, "step": kwargs.get("llm_step")}
    self.checkpoint(state, "transaction-start")
    try:
        outcome = self._execute_prepare_decision_impl(state, action, **kwargs)
    except BaseException:
        self.checkpoint(state, "transaction-interrupted")
        raise
    self._active_transaction = None
    state.pending_action = None
    self.checkpoint(state, "transaction-completed")
    return outcome


def execute_prepare_decision_impl(
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

    if isinstance(action, CandidateBlockMeshRepairAction):
        return self._execute_candidate_block_mesh_repair(
            state,
            action,
            llm_step=llm_step,
            progress_phase=progress_phase,
            progress_step=progress_step,
            progress_limit=progress_limit,
            native_execution=native_execution,
        )

    if isinstance(action, BlockMeshRepairAction):
        return self._execute_block_mesh_repair(
            state,
            action,
            llm_step=llm_step,
            progress_phase=progress_phase,
            native_execution=native_execution,
        )

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

    if isinstance(action, DesignCaseAction):
        # Stage-1 LLM output contains only Agent-owned CFD decisions. Immutable
        # identity/provenance fields are materialized from controller state so a
        # model cannot fail merely by copying a hash, fact ID, audit binding, target
        # version, or evidence ID incorrectly.
        failures: list[str] = []
        plan = None
        try:
            plan = materialize_engineering_plan(action.plan, state, self.catalog)
        except (DesignSealError, ValueError) as exc:
            failures.append(f"Engineering design could not be sealed: {exc}")
        if plan is not None:
            failures.extend(validate_design(plan, state.intake))
        # v4.1: implementation syntax evidence is DEFERRED to authoring.
        # Design acceptance must not require an explicit source excerpt for every
        # future case file. Raw free-form authoring remains evidence-gated later;
        # typed/structured authoring may instead establish confidence through
        # deterministic serialization and native validation.
        if plan is not None:
            failures.extend(self._validate_observed_provenance(plan, state))
            failures.extend(self._validate_engineering_defaults(plan, state))
        valid = not failures
        if valid:
            assert plan is not None
            self._authoring_task_queue = None
            self._draft_design_plan = plan
            self._draft_authoring_brief = action.authoring_brief
            self._mark_evidence_gaps_satisfied("prepare")
            event = self._event(
                llm_step,
                action.type,
                True,
                "Engineering design accepted; controller sealed immutable metadata and case authoring will run in a separate compact turn.",
            )
        else:
            event = self._event(
                llm_step,
                action.type,
                False,
                "Engineering design rejected by deterministic provider/semantic validation.",
                "\n".join(failures),
            )
        state.engineering_events.append(event)
        self._emit_engineering_event(
            progress_phase, event, step=progress_step, limit=progress_limit, state=state
        )
        return False

    if isinstance(action, CaseAuthoringAction) and not isinstance(action, ExecuteCasePlanAction):
        plan = self._draft_design_plan
        if plan is None:
            event = self._event(
                llm_step, action.type, False, "No Python-held staged EngineeringPlan exists for case authoring."
            )
            state.engineering_events.append(event)
            self._emit_engineering_event(
                progress_phase, event, step=progress_step, limit=progress_limit, state=state
            )
            return False
        data = action.model_dump(mode="python")
        data["type"] = "execute_case_plan"
        data["plan"] = plan.model_dump(mode="python")
        # The frozen plan is the sole authority for the solve-required manifest.
        # Never let a redundant model echo block staged authoring.
        data["required_case_files"] = list(plan.required_case_files)
        try:
            if self._authoring_task_queue is not None:
                from copy import deepcopy
                candidate_queue = deepcopy(self._authoring_task_queue)
                execution = accept_task(candidate_queue, action, plan)
                self._authoring_task_queue = candidate_queue
                if execution is None:
                    state.engineering_events.append(self._event(llm_step, action.type, True,
                        "Authoring partition validated and checkpointed; no case mutation/native execution yet."))
                    self.checkpoint(state, "authoring-partition-accepted")
                    return False
            else:
                if action.task_id is not None or action.defer_native:
                    raise ValueError("No controller-issued authoring task is pending.")
                execution = ExecuteCasePlanAction.model_validate(data)
        except ValueError as exc:
            event = self._event(
                llm_step, action.type, False, f"Staged case authoring contract rejected: {exc}"
            )
            state.engineering_events.append(event)
            self._emit_engineering_event(
                progress_phase, event, step=progress_step, limit=progress_limit, state=state
            )
            return False
        self._draft_design_plan = None
        self._authoring_task_queue = None
        self._draft_authoring_brief = ""
        return self._execute_case_plan(
            state,
            execution,
            llm_step=llm_step,
            progress_phase=progress_phase,
            progress_step=progress_step,
            progress_limit=progress_limit,
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

    # v4.8.2: a native diagnostic that already enumerates supported alternatives,
    # together with implicated case files, is sufficient for one direct repair
    # attempt. The controller blocks reference/capability retrieval for that first
    # attempt without selecting any alternative itself. A recorded repair at the
    # current validation generation, or a new native failure generation, reopens
    # normal retrieval policy.
    if (
        isinstance(action, (SearchReferencesAction, ReadReferenceAction, SearchCapabilitiesAction, GatherEvidenceAction))
        and repair_episode_requires_direct_attempt(state.repair_episode)
    ):
        event = self._event(
            llm_step,
            action.type,
            False,
            "Reference retrieval deferred: the current native diagnostic already enumerates supported alternatives and implicated files are available; attempt one direct repair first.",
            validation_status="inconclusive",
            failure_category="case",
        )
        state.engineering_events.append(event)
        self._emit_engineering_event(
            progress_phase, event, step=progress_step, limit=progress_limit, state=state
        )
        return False

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
    if progress_phase.startswith("revision") and isinstance(
        action,
        (
            WriteCaseFileAction,
            PatchCaseFileAction,
            DeleteCaseFileAction,
            RunMeshCommandAction,
            RunNativeOpenFOAMAction,
        ),
    ):
        self._begin_confirmed_revision_mutation(state)
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
    if not event.success:
        return self._route_failed_event(state, event, default_terminal=terminal)
    return terminal


def execute_prepare_sequence(
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
                validation_status="fail", failure_category="infra",
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
        if progress_phase.startswith("revision") and isinstance(
            member,
            (
                WriteCaseFileAction,
                PatchCaseFileAction,
                DeleteCaseFileAction,
                RunMeshCommandAction,
                RunNativeOpenFOAMAction,
            ),
        ):
            self._begin_confirmed_revision_mutation(state)
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
            return self._route_failed_event(state, event, default_terminal=terminal)
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


def dispatch_prepare(
    self,
    state: CFDState,
    action,
    *,
    step: int,
    native_execution: bool,
) -> tuple[EngineeringEvent, bool]:
    if isinstance(action, FinishPreviewAction):
        validation = self.safety.validate_plan(action.plan, state.intake)  # type: ignore[arg-type]
        state.semantic_assurance_warnings = list(validation.warnings)
        validation.failures.extend(self._validate_observed_provenance(action.plan, state))
        validation.failures.extend(self._validate_engineering_defaults(action.plan, state))
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
                    for warning in presolve.warnings:
                        tagged = f"Pre-solve advisory: {warning}"
                        if tagged not in state.semantic_assurance_warnings:
                            state.semantic_assurance_warnings.append(tagged)
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
            self._clear_unresolved_failure(state)
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
                state.revision_decision_complete = False
                state.revision_target_case_files = []
                state.pending_revision_plan = None
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
            assurance_output = "\n".join(state.semantic_assurance_warnings)
            assurance_count = len(state.semantic_assurance_warnings)
            return (
                self._event(
                    step,
                    action.type,
                    True,
                    (
                        f"Engineering plan accepted and case sealed with {assurance_count} advisory semantic-assurance gap(s)."
                        if assurance_count
                        else "Engineering plan accepted and case sealed."
                    ),
                    assurance_output,
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
        recoverable, generated_paths = self._recoverable_precommit_authoring_block(state, action)
        if recoverable:
            details = [f"modelReason={action.reason}"]
            if generated_paths:
                details.append("agentOwnedGeneratedArtifacts=" + ", ".join(generated_paths[:12]))
            return (
                self._event(
                    step,
                    action.type,
                    False,
                    "Pre-commit authoring strategy is infeasible; escalating to Agent strategy revision instead of terminally blocking.",
                    "\n".join(details),
                    failure_signature="authoring_feasibility:precommit_strategy",
                    failure_scope="strategy",
                    failure_category="case",
                ),
                False,
            )
        if action.block_kind == "engineering_choice_missing":
            missing = ", ".join(action.missing_items[:12]) or "delegated engineering details"
            return (
                self._event(
                    step,
                    action.type,
                    False,
                    "Block rejected by delegated-assumption policy: choose the missing engineering details as engineering_defaults.",
                    f"delegatedMissing={missing}",
                    failure_signature="assumption_policy:delegated_engineering_choice",
                    failure_scope="pipeline",
                ),
                False,
            )
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

