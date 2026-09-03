from __future__ import annotations

import hashlib
import json
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


FactSource = Literal["user", "derived"]
FactCategory = Literal[
    "context",
    "classification",
    "objective",
    "domain",
    "geometry",
    "scale",
    "material",
    "property",
    "physics",
    "temporal",
    "motion",
    "boundary",
    "output",
    "fidelity",
    "assumption",
]


class IntakeFact(BaseModel):
    """One auditable value in the user-facing CFD definition."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=r"^[a-z][a-z0-9_.-]*$")
    category: FactCategory
    label: str = Field(min_length=1)
    value: str = Field(min_length=1)
    unit: str | None = None
    source: FactSource
    evidence: str | None = None
    reason: str | None = None
    depends_on: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_provenance(self) -> Self:
        if self.source == "user" and not self.evidence:
            raise ValueError("user fact requires evidence.")
        if self.source == "derived" and not self.reason:
            raise ValueError("Derived fact requires a reason.")
        if self.source == "user" and self.depends_on:
            raise ValueError("User facts cannot declare inferred dependencies.")
        return self


class BlockingUnknown(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=r"^[a-z][a-z0-9_.-]*$")
    question: str = Field(min_length=1)
    impact: str = Field(min_length=1)
    suggested_default: str | None = None


class CFDIntakeSpec(BaseModel):
    """Solver-independent CFD definition produced before technical analysis."""

    model_config = ConfigDict(extra="forbid")

    semantic_contract_version: Literal["1", "2"] = "1"
    title: str = Field(min_length=1)
    facts: list[IntakeFact] = Field(default_factory=list)
    blocking_unknowns: list[BlockingUnknown] = Field(default_factory=list, max_length=3)
    status: Literal["needs_user_input", "ready_for_review"]

    @model_validator(mode="after")
    def validate_definition(self) -> Self:
        ids = [fact.id for fact in self.facts]
        if len(ids) != len(set(ids)):
            raise ValueError("CFDIntakeSpec contains duplicate fact IDs.")
        unknown_ids = [item.id for item in self.blocking_unknowns]
        if len(unknown_ids) != len(set(unknown_ids)):
            raise ValueError("CFDIntakeSpec contains duplicate blocking unknown IDs.")
        if self.status == "ready_for_review" and self.blocking_unknowns:
            raise ValueError("A review-ready intake cannot contain blocking unknowns.")
        if self.status == "needs_user_input" and not self.blocking_unknowns:
            raise ValueError("A blocked intake requires at least one blocking unknown.")
        fact_ids = set(ids)
        for fact in self.facts:
            missing = set(fact.depends_on) - fact_ids
            if missing:
                raise ValueError(
                    f"Fact '{fact.id}' depends on unknown facts: {sorted(missing)}"
                )
        return self

    def fact(self, fact_id: str) -> IntakeFact | None:
        return next((fact for fact in self.facts if fact.id == fact_id), None)

    def digest(self) -> str:
        data = self.model_dump(mode="json")
        # v2.15 adds an explicit semantic-contract version.  Preserve the exact
        # pre-v2.15 digest for legacy/default v1 intakes so existing confirmed
        # states remain rehydratable after upgrade.
        if self.semantic_contract_version == "1":
            data.pop("semantic_contract_version", None)
        payload = json.dumps(
            data,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()
