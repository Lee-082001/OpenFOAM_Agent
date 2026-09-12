from __future__ import annotations



from openfoam_agent.progress import ProgressEvent
from openfoam_agent.schemas.engineering import (
    EngineeringBudgetExtension,
    EngineeringPlan,
    RepairCasePlanAction,
    RevisionDecisionAction,
)
from openfoam_agent.tools.workspace import WorkspaceSafetyError
from openfoam_agent.workflow.state import CFDState
from openfoam_agent.workflow.states import State

# v4.8 lifecycle phase controller: long-running prepare/revision loops live outside the facade.


def prepare(self, state: CFDState, *, native_execution: bool = True) -> CFDState:
    state.assert_confirmed_intake()
    from openfoam_agent.tools.assets import ingest_request_assets
    ingest_request_assets(state, self.workspace)
    self.bind_checkpoint(state)
    resumed = self._resume_pending
    self._resume_pending = False
    if not resumed:
        self._evidence_gap_ledger["prepare"] = {}
        self._draft_design_plan = None
        self._authoring_task_queue = None
        self._draft_authoring_brief = ""
        self._retrieval_cycles["prepare"] = 0
        self._evidence_retrieval_disabled.pop("prepare", None)
        state.engineering_next_step = 1
    if native_execution and not self._checkmesh_preflight(state, phase="preflight"):
        return state
    if not resumed:
        state.engineering_round_start_index = len(state.engineering_events)
    self.checkpoint(state, "confirmed-intake")
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
    if resumed and state.engineering_budget_extensions:
        current_limit = max(current_limit, max(x.new_limit for x in state.engineering_budget_extensions))
    step = state.engineering_next_step
    while True:
        while step <= current_limit:
            state.engineering_next_step = step + 1
            self.checkpoint(state, "before-model-turn")
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
            self.checkpoint(state, "design-or-authoring-turn-completed")
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

    for feedback in state.human_feedback:
        if feedback.feedback_id in proposal.feedback_ids:
            feedback.status = "revision_in_progress"
    state.revision_decision_complete = False
    state.revision_target_case_files = []
    state.pending_revision_plan = None
    state.engineering_round_start_index = len(state.engineering_events)

    # Do not archive/remove prior outputs before the first controller-validated
    # case delta exists. Revision reasoning/read-only turns need the baseline
    # runtime evidence, and a context/LLM failure before mutation must leave the
    # prior solved case/results intact. Solve approval is revoked immediately;
    # numerical/post-processing state is cleared atomically at mutation start.
    state.solve_approved = False
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
            try:
                turn = self._generate_turn(
                    state,
                    step=global_step,
                    local_step=local_step,
                    current_step_limit=current_limit,
                    phase="human_revision",
                    native_execution=native_execution,
                )
                if isinstance(turn.action, RevisionDecisionAction):
                    terminal = self._execute_revision_decision(state, turn.action, step=global_step)
                else:
                    if state.revision_decision_complete and isinstance(turn.action, RepairCasePlanAction) and (
                        turn.action.plan_patch is not None or turn.action.updated_plan is not None
                    ):
                        raise WorkspaceSafetyError(
                            "Revision authoring phase cannot modify EngineeringPlan; plan changes belong to decide_revision."
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
            except BaseException:
                self._restore_unmutated_revision_ready(state, proposal)
                raise
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


def begin_confirmed_revision_mutation(self, state: CFDState) -> None:
    """Archive baseline outputs only when a validated revision delta will commit."""
    proposal = state.active_revision_proposal
    if proposal is None:
        return
    if state.pending_revision_archive_path is None:
        revision_id = f"rev-{len(state.revision_history) + 1:04d}"
        state.pending_revision_archive_path = self.workspace.archive_revision_outputs(revision_id)
        if state.engineering_plan is not None and state.case_seal is not None:
            self.safety.verify_seal(state.engineering_plan, state.case_seal)

    # Prior numerical evidence remains in the revision archive/audit history, but
    # must not be presented as evidence for the newly mutated, unsolved case.
    state.simulation = None
    state.runtime_report = None
    state.simulation_attempts = 0
    state.last_runtime_log_excerpt = None
    state.postprocessing_events = []
    state.force_coefficient_analysis = None
    state.postprocessing_report = None


def restore_unmutated_revision_ready(self, state: CFDState, proposal) -> None:
    """Keep an approved proposal retryable when revision planning failed pre-mutation."""
    if state.pending_revision_archive_path is not None:
        return
    if state.engineering_plan is None or state.case_seal is None:
        return
    if state.engineering_plan.digest() != proposal.baseline_plan_sha256:
        return
    if self.workspace.manifest_digest() != proposal.baseline_manifest_sha256:
        return
    for feedback in state.human_feedback:
        if feedback.feedback_id in proposal.feedback_ids and feedback.status == "revision_in_progress":
            feedback.status = "revision_proposed"
    state.revision_decision_complete = False
    state.revision_target_case_files = []
    state.pending_revision_plan = None
    state.transition(
        State.REVISION_READY,
        f"Revision planning failed before case mutation; proposal {proposal.proposal_id} remains pending and retryable.",
    )

