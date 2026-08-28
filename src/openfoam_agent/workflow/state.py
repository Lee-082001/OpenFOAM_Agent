from __future__ import annotations

from pydantic import BaseModel, Field

from openfoam_agent.schemas.engineering import (
    CaseSeal,
    EngineeringBudgetExtension,
    EngineeringEvent,
    EngineeringPlan,
    MeshEvidence,
)
from openfoam_agent.schemas.intake import CFDIntakeSpec
from openfoam_agent.schemas.request import UserRequest
from openfoam_agent.schemas.simulation import RuntimeReport, SimulationResult
from openfoam_agent.schemas.feedback import HumanFeedback, RevisionProposal, RevisionRecord
from openfoam_agent.schemas.postprocessing import (
    ForceCoefficientAnalysis,
    PostProcessingEvent,
    PostProcessingReport,
)
from .states import State


class CFDState(BaseModel):
    run_id: str
    user_request: UserRequest

    intake: CFDIntakeSpec | None = None
    intake_confirmed: bool = False
    intake_digest: str | None = None

    engineering_plan: EngineeringPlan | None = None
    engineering_events: list[EngineeringEvent] = Field(default_factory=list)
    engineering_budget_extensions: list[EngineeringBudgetExtension] = Field(default_factory=list)
    engineering_round_start_index: int = Field(default=0, ge=0)
    case_seal: CaseSeal | None = None
    case_dir: str | None = None
    mesh_evidence: MeshEvidence | None = None

    solve_approved: bool = False
    simulation: SimulationResult | None = None
    runtime_report: RuntimeReport | None = None
    simulation_attempts: int = 0
    last_runtime_log_excerpt: str | None = None

    postprocessing_events: list[PostProcessingEvent] = Field(default_factory=list)
    force_coefficient_analysis: ForceCoefficientAnalysis | None = None
    postprocessing_report: PostProcessingReport | None = None

    human_feedback: list[HumanFeedback] = Field(default_factory=list)
    revision_proposals: list[RevisionProposal] = Field(default_factory=list)
    active_revision_proposal: RevisionProposal | None = None
    revision_history: list[RevisionRecord] = Field(default_factory=list)
    pending_revision_archive_path: str | None = None

    current_state: State = State.INIT
    history: list[dict[str, str]] = Field(default_factory=list)

    def transition(self, new_state: State, note: str = "") -> None:
        self.history.append(
            {"from": self.current_state.value, "to": new_state.value, "note": note}
        )
        self.current_state = new_state

    def confirm_intake(self) -> None:
        if self.intake is None:
            raise ValueError("CFDIntakeSpec is required before confirmation.")
        if self.intake.status != "ready_for_review":
            raise ValueError("A blocked CFD intake cannot be confirmed.")
        self.intake_confirmed = True
        self.intake_digest = self.intake.digest()

    def assert_confirmed_intake(self) -> None:
        if self.intake is None or not self.intake_confirmed or not self.intake_digest:
            raise ValueError("A confirmed CFDIntakeSpec is required downstream.")
        if self.intake.digest() != self.intake_digest:
            raise ValueError("Confirmed CFDIntakeSpec changed after confirmation.")


    def reject_revision(self) -> None:
        if self.current_state != State.REVISION_READY or self.active_revision_proposal is None:
            raise ValueError("Revision rejection requires REVISION_READY with an active proposal.")
        linked = [
            feedback
            for feedback in self.human_feedback
            if feedback.feedback_id in self.active_revision_proposal.feedback_ids
        ]
        target = (
            State.RESULT_REVIEW_REQUIRED
            if any(item.submitted_state == State.RESULT_REVIEW_REQUIRED.value for item in linked)
            else (State.SOLVE_READY if any(item.submitted_state == State.SOLVE_READY.value for item in linked) else State.MESH_READY)
        )
        proposal_id = self.active_revision_proposal.proposal_id
        for feedback in linked:
            feedback.status = "proposal_rejected"
        self.active_revision_proposal = None
        self.transition(target, f"User rejected revision proposal {proposal_id}; sealed case remains unchanged.")

    def accept_result(self) -> None:
        if self.current_state != State.RESULT_REVIEW_REQUIRED:
            raise ValueError("Result acceptance requires RESULT_REVIEW_REQUIRED.")
        for feedback in self.human_feedback:
            if feedback.status in {"awaiting_review", "revision_proposed", "unresolved"}:
                feedback.status = "resolved"
        self.active_revision_proposal = None
        self.transition(State.COMPLETE, "User accepted the reviewed CFD result.")

    def mark_feedback_awaiting_review(self) -> None:
        for feedback in self.human_feedback:
            if feedback.status == "awaiting_rerun":
                feedback.status = "awaiting_review"

    def approve_solve(self) -> None:
        if self.current_state not in {State.MESH_READY, State.SOLVE_READY}:
            raise ValueError("Solve approval requires a mesh-ready or solve-ready sealed case.")
        if self.engineering_plan is None or self.case_seal is None:
            raise ValueError("Solve approval requires an engineering plan and case seal.")
        self.solve_approved = True
        self.transition(State.SIMULATION, "User approved solver execution for the sealed case.")
