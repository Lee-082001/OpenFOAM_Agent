from __future__ import annotations

import hashlib
import json
import re
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


class _EngineeringModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class EngineeringDecision(_EngineeringModel):
    area: str = Field(min_length=1, max_length=80)
    choice: str = Field(min_length=1, max_length=500)
    rationale: str = Field(min_length=1, max_length=2000)


def canonical_engineering_evidence_id(kind: str, reference: str) -> str:
    """Return a stable opaque ID for evidence observed by deterministic tools."""

    if kind not in {"capability", "openfoam_reference"}:
        raise ValueError(f"Unsupported engineering evidence kind: {kind}")
    digest = hashlib.sha256(f"{kind}\0{reference}".encode("utf-8")).hexdigest()[:20]
    prefix = "cap" if kind == "capability" else "ref"
    return f"ev_{prefix}_{digest}"


class EngineeringEvidence(_EngineeringModel):
    """An LLM-selected pointer to evidence that Python already observed.

    The model is intentionally not allowed to restate the evidence kind/reference.
    It may only select an opaque canonical ID from ``available_evidence`` supplied
    in the engineering prompt.
    """

    evidence_id: str = Field(pattern=r"^ev_(?:cap|ref)_[0-9a-f]{20}$")
    note: str = Field(default="", max_length=1000)


class ObservedEngineeringEvidence(_EngineeringModel):
    """Deterministically issued evidence record attached to successful tool events."""

    evidence_id: str = Field(pattern=r"^ev_(?:cap|ref)_[0-9a-f]{20}$")
    kind: Literal["capability", "openfoam_reference"]
    reference: str = Field(min_length=1, max_length=1000)
    summary: str = Field(min_length=1, max_length=1200)


class EngineeringPlan(_EngineeringModel):
    """Agent-owned CFD engineering decisions.

    The schema intentionally records decisions without encoding OpenFOAM implementation
    policy.  Python validates provenance, safety and runtime evidence; it does not
    infer a mesh method, boundary condition, solver or numerical scheme from these
    fields.
    """

    schema_version: Literal["2.0"] = "2.0"
    case_name: str = Field(pattern=r"^[A-Za-z][A-Za-z0-9_.-]*$", max_length=80)
    solver: str = Field(pattern=r"^[A-Za-z][A-Za-z0-9_]*$", max_length=120)
    solver_provider_id: str = Field(min_length=1, max_length=200)
    openfoam_distribution: Literal["foundation"] = "foundation"
    openfoam_version: str = Field(pattern=r"^(?:13|14)$")
    problem_interpretation: str = Field(min_length=1, max_length=4000)
    temporal_behavior: Literal["steady", "transient", "custom"]
    motion_kind: Literal[
        "static",
        "rigid_body",
        "prescribed_deformation",
        "free_body",
        "two_way_fsi",
        "custom",
    ]
    mesh_motion_requirement: Literal[
        "static", "moving", "deforming", "topology_change", "custom"
    ]
    mesh_strategy: str = Field(min_length=1, max_length=1000)
    decisions: list[EngineeringDecision] = Field(default_factory=list, max_length=80)
    assumptions: list[str] = Field(default_factory=list, max_length=80)
    confirmed_fact_ids: list[str] = Field(default_factory=list, max_length=200)
    evidence: list[EngineeringEvidence] = Field(default_factory=list, max_length=120)
    required_case_files: list[str] = Field(default_factory=list, max_length=80)
    postprocess_strategy: list[str] = Field(default_factory=list, max_length=40)
    confirmed_intake_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_unique_audit_fields(self) -> Self:
        if len(self.confirmed_fact_ids) != len(set(self.confirmed_fact_ids)):
            raise ValueError("Engineering plan contains duplicate confirmed fact IDs.")
        if len(self.required_case_files) != len(set(self.required_case_files)):
            raise ValueError("Engineering plan contains duplicate required case files.")
        for path in self.required_case_files:
            if not re.fullmatch(r"(?:0|constant|system)/[A-Za-z0-9_.\/-]+", path) or ".." in path:
                raise ValueError(f"Unsafe required case file path: {path}")
        evidence_ids = [item.evidence_id for item in self.evidence]
        if len(evidence_ids) != len(set(evidence_ids)):
            raise ValueError("Engineering plan contains duplicate evidence IDs.")
        return self

    def digest(self) -> str:
        payload = json.dumps(
            self.model_dump(mode="json"),
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class InspectEnvironmentAction(_EngineeringModel):
    type: Literal["inspect_environment"]
    rationale: str = Field(min_length=1, max_length=1500)


class SearchCapabilitiesAction(_EngineeringModel):
    type: Literal["search_capabilities"]
    query: str = Field(min_length=1, max_length=500)
    rationale: str = Field(min_length=1, max_length=1500)


class SearchReferencesAction(_EngineeringModel):
    type: Literal["search_references"]
    query: str = Field(min_length=1, max_length=500)
    scope: Literal["all", "tutorials", "source", "etc"] = "all"
    rationale: str = Field(min_length=1, max_length=1500)


class ReadReferenceAction(_EngineeringModel):
    type: Literal["read_reference"]
    reference: str = Field(min_length=1, max_length=1000)
    start_line: int = Field(default=1, ge=1, le=1_000_000)
    line_count: int = Field(default=160, ge=1, le=400)
    rationale: str = Field(min_length=1, max_length=1500)


class ListCaseFilesAction(_EngineeringModel):
    type: Literal["list_case_files"]
    rationale: str = Field(min_length=1, max_length=1500)


class ReadCaseFileAction(_EngineeringModel):
    type: Literal["read_case_file"]
    path: str = Field(min_length=1, max_length=240)
    rationale: str = Field(min_length=1, max_length=1500)


class WriteCaseFileAction(_EngineeringModel):
    type: Literal["write_case_file"]
    path: str = Field(min_length=1, max_length=240)
    content: str = Field(min_length=1, max_length=1_000_000)
    rationale: str = Field(min_length=1, max_length=1500)


class DeleteCaseFileAction(_EngineeringModel):
    type: Literal["delete_case_file"]
    path: str = Field(min_length=1, max_length=240)
    rationale: str = Field(min_length=1, max_length=1500)


class ValidateDictionaryAction(_EngineeringModel):
    type: Literal["validate_dictionary"]
    path: str = Field(min_length=1, max_length=240)
    rationale: str = Field(min_length=1, max_length=1500)


class SurfaceCheckAction(_EngineeringModel):
    type: Literal["surface_check"]
    path: str = Field(min_length=1, max_length=240)
    rationale: str = Field(min_length=1, max_length=1500)


class RunMeshCommandAction(_EngineeringModel):
    type: Literal["run_mesh_command"]
    command: Literal[
        "blockMesh",
        "surfaceFeatureExtract",
        "snappyHexMesh",
        "createPatch",
        "checkMesh",
    ]
    rationale: str = Field(min_length=1, max_length=1500)


class FinishPreviewAction(_EngineeringModel):
    type: Literal["finish_preview"]
    plan: EngineeringPlan
    rationale: str = Field(min_length=1, max_length=1500)


class RetrySolverAction(_EngineeringModel):
    type: Literal["retry_solver"]
    plan: EngineeringPlan
    rationale: str = Field(min_length=1, max_length=1500)


class BlockAction(_EngineeringModel):
    type: Literal["block"]
    reason: str = Field(min_length=1, max_length=4000)
    needs_user_input: bool = False
    rationale: str = Field(min_length=1, max_length=1500)


# Keep this as a plain Union rather than a Pydantic discriminated union.
# Pydantic emits `oneOf` + `discriminator` for discriminated unions, while
# OpenAI Structured Outputs supports nested `anyOf` but rejects `oneOf`.
# Each branch still has a unique Literal["type"], so Pydantic validation is
# unambiguous without the discriminator keyword.
EngineeringAction = (
    InspectEnvironmentAction
    | SearchCapabilitiesAction
    | SearchReferencesAction
    | ReadReferenceAction
    | ListCaseFilesAction
    | ReadCaseFileAction
    | WriteCaseFileAction
    | DeleteCaseFileAction
    | ValidateDictionaryAction
    | SurfaceCheckAction
    | RunMeshCommandAction
    | FinishPreviewAction
    | RetrySolverAction
    | BlockAction
)


class EngineeringTurn(_EngineeringModel):
    action: EngineeringAction


class EngineeringEvent(_EngineeringModel):
    step: int = Field(ge=1)
    action_type: str = Field(min_length=1, max_length=80)
    success: bool
    summary: str = Field(min_length=1, max_length=4000)
    output_excerpt: str = Field(default="", max_length=12000)
    artifact_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    native_command_executed: bool = False
    mesh_command_executed: bool = False
    observed_evidence: list[ObservedEngineeringEvidence] = Field(default_factory=list, max_length=24)

    @model_validator(mode="after")
    def validate_resource_markers(self) -> Self:
        if self.mesh_command_executed and not self.native_command_executed:
            raise ValueError("A mesh command event must also be a native command event.")
        return self


class EngineeringBudgetExtension(_EngineeringModel):
    boundary_step: int = Field(ge=1)
    previous_limit: int = Field(ge=1)
    new_limit: int = Field(ge=1)
    reason: str = Field(min_length=1, max_length=1000)

    @model_validator(mode="after")
    def validate_extension(self) -> Self:
        if self.new_limit <= self.previous_limit:
            raise ValueError("Budget extension must increase the current step limit.")
        if self.boundary_step != self.previous_limit:
            raise ValueError("Budget extension boundary must match the previous step limit.")
        return self


class CaseFileSeal(_EngineeringModel):
    path: str = Field(min_length=1, max_length=240)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    size_bytes: int = Field(ge=0)
    origin: Literal["agent", "native"] = "agent"


class CaseSeal(_EngineeringModel):
    plan_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    files: list[CaseFileSeal] = Field(default_factory=list)


class MeshEvidence(_EngineeringModel):
    command_succeeded: bool
    mesh_ok: bool
    cell_count: int | None = Field(default=None, ge=0)
    max_non_orthogonality: float | None = None
    max_skewness: float | None = None
    negative_volume_cells: int | None = Field(default=None, ge=0)
    raw_log_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    warnings: list[str] = Field(default_factory=list)

    @property
    def passed(self) -> bool:
        return bool(
            self.command_succeeded
            and self.mesh_ok
            and self.cell_count is not None
            and (self.negative_volume_cells in {None, 0})
        )
