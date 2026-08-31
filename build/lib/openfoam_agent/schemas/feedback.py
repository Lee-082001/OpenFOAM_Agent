from __future__ import annotations

import hashlib
import json
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


class _FeedbackModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class HumanFeedback(_FeedbackModel):
    feedback_id: str = Field(pattern=r"^hf-[0-9]{4}$")
    run_id: str = Field(min_length=1, max_length=100)
    scope: Literal["mesh", "result", "engineering"]
    statement: str = Field(min_length=1, max_length=8000)
    submitted_state: str = Field(min_length=1, max_length=80)
    status: Literal[
        "unresolved",
        "revision_proposed",
        "proposal_rejected",
        "revision_in_progress",
        "awaiting_rerun",
        "awaiting_review",
        "resolved",
    ] = "unresolved"
    evidence_snapshot_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class FeedbackHypothesis(_FeedbackModel):
    hypothesis: str = Field(min_length=1, max_length=1200)
    rationale: str = Field(min_length=1, max_length=2000)
    evidence_to_check: list[str] = Field(default_factory=list, max_length=30)


class ProposedRevisionChange(_FeedbackModel):
    area: str = Field(min_length=1, max_length=100)
    change: str = Field(min_length=1, max_length=1600)
    rationale: str = Field(min_length=1, max_length=2000)


class FeedbackAssessment(_FeedbackModel):
    diagnosis_summary: str = Field(min_length=1, max_length=4000)
    hypotheses: list[FeedbackHypothesis] = Field(default_factory=list, max_length=30)
    proposed_changes: list[ProposedRevisionChange] = Field(default_factory=list, max_length=40)
    expected_cost: Literal["lower", "similar", "moderate_increase", "large_increase", "unknown"]
    requires_case_revision: bool
    requires_intake_revision: bool = False
    intake_revision_reason: str = Field(default="", max_length=2000)
    review_limitations: list[str] = Field(default_factory=list, max_length=30)

    @model_validator(mode="after")
    def validate_revision_route(self) -> Self:
        if self.requires_intake_revision and not self.intake_revision_reason.strip():
            raise ValueError("An intake revision route requires an explicit reason.")
        if self.requires_intake_revision and self.requires_case_revision:
            raise ValueError("A confirmed user-fact change must return to intake before case revision.")
        return self


class RevisionProposal(_FeedbackModel):
    proposal_id: str = Field(pattern=r"^rp-[0-9]{4}$")
    feedback_ids: list[str] = Field(min_length=1, max_length=30)
    diagnosis_summary: str = Field(min_length=1, max_length=4000)
    hypotheses: list[FeedbackHypothesis] = Field(default_factory=list, max_length=30)
    proposed_changes: list[ProposedRevisionChange] = Field(default_factory=list, max_length=40)
    expected_cost: Literal["lower", "similar", "moderate_increase", "large_increase", "unknown"]
    requires_case_revision: bool
    requires_intake_revision: bool = False
    intake_revision_reason: str = Field(default="", max_length=2000)
    review_limitations: list[str] = Field(default_factory=list, max_length=30)
    baseline_plan_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    baseline_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    def digest(self) -> str:
        payload = json.dumps(
            self.model_dump(mode="json"),
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class RevisionFileChange(_FeedbackModel):
    path: str = Field(min_length=1, max_length=500)
    change: Literal["added", "removed", "modified"]
    before_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    after_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")


class RevisionRecord(_FeedbackModel):
    revision_id: str = Field(pattern=r"^rev-[0-9]{4}$")
    proposal_id: str = Field(pattern=r"^rp-[0-9]{4}$")
    feedback_ids: list[str] = Field(min_length=1, max_length=30)
    before_plan_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    after_plan_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    before_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    after_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    archive_path: str | None = Field(default=None, max_length=500)
    file_changes: list[RevisionFileChange] = Field(default_factory=list, max_length=500)
