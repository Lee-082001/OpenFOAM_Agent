from __future__ import annotations

import hashlib
import json

from openfoam_agent.llm.prompts import FEEDBACK_REVIEW_SYSTEM_PROMPT
from openfoam_agent.llm.protocol import StructuredLLM
from openfoam_agent.progress import NullProgressReporter, ProgressEvent, ProgressReporter
from openfoam_agent.schemas.feedback import FeedbackAssessment, HumanFeedback, RevisionProposal
from openfoam_agent.workflow.state import CFDState
from openfoam_agent.workflow.states import State


class CFDFeedbackReviewAgent:
    """Turn human review observations into an auditable, non-executing revision proposal."""

    def __init__(
        self,
        llm: StructuredLLM,
        *,
        progress: ProgressReporter | None = None,
    ) -> None:
        self.llm = llm
        self.progress = progress or NullProgressReporter()

    def review(self, state: CFDState, statement: str) -> CFDState:
        text = statement.strip()
        if not text:
            raise ValueError("Human feedback must not be blank.")
        if state.current_state not in {State.MESH_READY, State.RESULT_REVIEW_REQUIRED}:
            raise ValueError("Human feedback is accepted only at MESH_READY or RESULT_REVIEW_REQUIRED.")
        if state.engineering_plan is None or state.case_seal is None:
            raise ValueError("Human feedback requires an existing sealed engineering case.")

        scope = "mesh" if state.current_state == State.MESH_READY else "result"
        feedback = HumanFeedback(
            feedback_id=f"hf-{len(state.human_feedback) + 1:04d}",
            run_id=state.run_id,
            scope=scope,
            statement=text,
            submitted_state=state.current_state.value,
            evidence_snapshot_sha256=self._evidence_snapshot_digest(state),
        )
        origin_state = state.current_state
        state.human_feedback.append(feedback)
        state.transition(State.FEEDBACK_RECEIVED, f"Human feedback {feedback.feedback_id} recorded for review.")
        self.progress.emit(
            ProgressEvent(
                phase="feedback-review",
                message=f"human feedback 진단 시작: {feedback.feedback_id}",
                status="start",
                metrics={"scope": scope},
            )
        )

        payload = {
            "feedback": feedback.model_dump(mode="json"),
            "feedback_history": [item.model_dump(mode="json") for item in state.human_feedback],
            "revision_history": [item.model_dump(mode="json") for item in state.revision_history[-5:]],
            "confirmed_intake": state.intake.model_dump(mode="json") if state.intake else None,
            "engineering_plan": state.engineering_plan.model_dump(mode="json"),
            "mesh_evidence": state.mesh_evidence.model_dump(mode="json") if state.mesh_evidence else None,
            "runtime_report": state.runtime_report.model_dump(mode="json") if state.runtime_report else None,
            "postprocessing_report": (
                state.postprocessing_report.model_dump(mode="json")
                if state.postprocessing_report
                else None
            ),
            "case_manifest": [
                {"path": item.path, "sha256": item.sha256, "size_bytes": item.size_bytes}
                for item in state.case_seal.files
            ],
            "instructions": {
                "current_case_is_immutable_until_user_confirms_revision": True,
                "confirmed_user_facts_must_not_be_silently_changed": True,
                "diagnoses_are_hypotheses_until checked_by_tools": True,
            },
        }
        try:
            assessment = self.llm.generate(
                FeedbackAssessment,
                "Assess this human CFD review feedback and propose the next revision route:\n"
                + json.dumps(payload, ensure_ascii=False, indent=2),
                system_prompt=FEEDBACK_REVIEW_SYSTEM_PROMPT,
            )

            proposal = RevisionProposal(
                proposal_id=f"rp-{len(state.revision_proposals) + 1:04d}",
                feedback_ids=[feedback.feedback_id],
                diagnosis_summary=assessment.diagnosis_summary,
                hypotheses=assessment.hypotheses,
                proposed_changes=assessment.proposed_changes,
                expected_cost=assessment.expected_cost,
                requires_case_revision=assessment.requires_case_revision,
                requires_intake_revision=assessment.requires_intake_revision,
                intake_revision_reason=assessment.intake_revision_reason,
                review_limitations=assessment.review_limitations,
                baseline_plan_sha256=state.engineering_plan.digest(),
                baseline_manifest_sha256=state.case_seal.manifest_sha256,
            )
        except Exception:
            self.progress.emit(
                ProgressEvent(
                    phase="feedback-review",
                    message=f"feedback 자동 진단 실패: {feedback.feedback_id}",
                    status="failure",
                )
            )
            # The human observation is durable provenance, but a cloud/model/schema
            # failure must not strand the workflow in an intermediate state.  Return
            # to the exact review gate from which feedback was submitted so the user
            # can retry, accept the existing result, or start a new run.
            feedback.status = "unresolved"
            state.active_revision_proposal = None
            state.transition(
                origin_state,
                f"Feedback {feedback.feedback_id} was recorded, but automated review failed; the sealed case remains unchanged.",
            )
            raise
        state.revision_proposals.append(proposal)
        self.progress.emit(
            ProgressEvent(
                phase="feedback-review",
                message=f"feedback assessment 완료: {proposal.proposal_id}",
                status="success",
                metrics={
                    "caseRevision": proposal.requires_case_revision,
                    "intakeRevision": proposal.requires_intake_revision,
                    "expectedCost": proposal.expected_cost,
                },
            )
        )
        if proposal.requires_intake_revision:
            state.active_revision_proposal = proposal
            feedback.status = "revision_proposed"
            state.transition(
                State.ENGINEERING_REVIEW_REQUIRED,
                "Human feedback changes confirmed user requirements; revise/reconfirm intake before case engineering.",
            )
        elif proposal.requires_case_revision:
            state.active_revision_proposal = proposal
            feedback.status = "revision_proposed"
            state.transition(
                State.REVISION_READY,
                f"Revision proposal {proposal.proposal_id} is ready; /confirm is required before modifying the sealed case.",
            )
        else:
            # A review may legitimately conclude that the observation does not yet
            # justify mutating the sealed case.  Preserve the assessment as
            # provenance and return to the original human gate rather than forcing
            # a pointless engineering revision.
            state.active_revision_proposal = None
            feedback.status = (
                "awaiting_review" if origin_state == State.RESULT_REVIEW_REQUIRED else "unresolved"
            )
            state.transition(
                origin_state,
                f"Feedback assessment {proposal.proposal_id} found no justified case revision; the sealed case remains unchanged.",
            )
        return state

    @staticmethod
    def _evidence_snapshot_digest(state: CFDState) -> str:
        payload = {
            "plan": state.engineering_plan.model_dump(mode="json") if state.engineering_plan else None,
            "seal": state.case_seal.model_dump(mode="json") if state.case_seal else None,
            "mesh": state.mesh_evidence.model_dump(mode="json") if state.mesh_evidence else None,
            "runtime": state.runtime_report.model_dump(mode="json") if state.runtime_report else None,
            "postprocess": (
                state.postprocessing_report.model_dump(mode="json")
                if state.postprocessing_report
                else None
            ),
        }
        raw = json.dumps(payload, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()
