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
    rationale: str = Field(min_length=1, max_length=500)


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
    note: str = Field(default="", max_length=300)


class ObservedEngineeringEvidence(_EngineeringModel):
    """Deterministically issued evidence record attached to successful tool events."""

    evidence_id: str = Field(pattern=r"^ev_(?:cap|ref)_[0-9a-f]{20}$")
    kind: Literal["capability", "openfoam_reference"]
    reference: str = Field(min_length=1, max_length=1000)
    summary: str = Field(min_length=1, max_length=1200)




_BINDABLE_PLAN_FIELDS = Literal[
    "problem_interpretation",
    "temporal_behavior",
    "motion_kind",
    "mesh_motion_requirement",
    "mesh_strategy",
    "decisions",
    "assumptions",
    "required_case_files",
    "postprocess_strategy",
]


class ConfirmedFactBinding(_EngineeringModel):
    """Audit mapping from a confirmed fact to its claimed implementation.

    The wire format deliberately separates case-file paths from plan fields so the
    model does not need to memorize a string-prefix mini-protocol such as
    ``case:...``/``plan:...``.  A small legacy adapter still accepts that older
    representation when loading persisted v2.10.1 state.
    """

    fact_id: str = Field(pattern=r"^[a-z][a-z0-9_.-]*$")
    case_files: list[str] = Field(default_factory=list, max_length=12)
    plan_fields: list[_BINDABLE_PLAN_FIELDS] = Field(default_factory=list, max_length=12)
    explanation: str = Field(min_length=1, max_length=300)

    @model_validator(mode="before")
    @classmethod
    def migrate_legacy_refs(cls, value):
        if not isinstance(value, dict) or "implementation_refs" not in value:
            return value
        migrated = dict(value)
        refs = migrated.pop("implementation_refs") or []
        case_files = list(migrated.get("case_files") or [])
        plan_fields = list(migrated.get("plan_fields") or [])
        for ref in refs:
            if not isinstance(ref, str):
                continue
            if ref.startswith("case:"):
                case_files.append(ref[5:])
            elif ref.startswith("plan:"):
                plan_fields.append(ref[5:])
        migrated["case_files"] = case_files
        migrated["plan_fields"] = plan_fields
        return migrated

    @model_validator(mode="after")
    def validate_refs(self) -> Self:
        if not self.case_files and not self.plan_fields:
            raise ValueError("Confirmed fact binding requires at least one case_files or plan_fields entry.")
        if len(self.case_files) != len(set(self.case_files)):
            raise ValueError("Confirmed fact binding contains duplicate case file refs.")
        if len(self.plan_fields) != len(set(self.plan_fields)):
            raise ValueError("Confirmed fact binding contains duplicate plan field refs.")
        for path in self.case_files:
            if not re.fullmatch(r"(?:0|constant|system)/[A-Za-z0-9_.\/-]+", path) or ".." in path:
                raise ValueError(f"Unsafe case implementation ref: {path}")
        return self

    @property
    def implementation_refs(self) -> list[str]:
        """Compatibility/audit projection used by older internal callers."""

        return [*(f"case:{path}" for path in self.case_files), *(f"plan:{field}" for field in self.plan_fields)]


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
    confirmed_fact_bindings: list[ConfirmedFactBinding] = Field(default_factory=list, max_length=200)
    evidence: list[EngineeringEvidence] = Field(default_factory=list, max_length=120)
    required_case_files: list[str] = Field(default_factory=list, max_length=80)
    postprocess_strategy: list[str] = Field(default_factory=list, max_length=40)
    confirmed_intake_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_unique_audit_fields(self) -> Self:
        if len(self.confirmed_fact_ids) != len(set(self.confirmed_fact_ids)):
            raise ValueError("Engineering plan contains duplicate confirmed fact IDs.")
        binding_ids = [item.fact_id for item in self.confirmed_fact_bindings]
        if len(binding_ids) != len(set(binding_ids)):
            raise ValueError("Engineering plan contains duplicate confirmed fact bindings.")
        if set(binding_ids) != set(self.confirmed_fact_ids):
            raise ValueError("Engineering plan confirmed fact bindings must exactly cover confirmed_fact_ids.")
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
    rationale: str = Field(default="", max_length=200)


class SearchCapabilitiesAction(_EngineeringModel):
    type: Literal["search_capabilities"]
    query: str = Field(min_length=1, max_length=500)
    rationale: str = Field(default="", max_length=200)


class SearchReferencesAction(_EngineeringModel):
    type: Literal["search_references"]
    query: str = Field(min_length=1, max_length=500)
    scope: Literal["all", "tutorials", "source", "etc"] = "all"
    rationale: str = Field(default="", max_length=200)


class ReadReferenceAction(_EngineeringModel):
    type: Literal["read_reference"]
    reference: str = Field(min_length=1, max_length=1000)
    start_line: int = Field(default=1, ge=1, le=1_000_000)
    line_count: int = Field(default=160, ge=1, le=400)
    rationale: str = Field(default="", max_length=200)


class ListCaseFilesAction(_EngineeringModel):
    type: Literal["list_case_files"]
    rationale: str = Field(default="", max_length=200)


class ReadCaseFileAction(_EngineeringModel):
    type: Literal["read_case_file"]
    path: str = Field(min_length=1, max_length=240)
    rationale: str = Field(default="", max_length=200)


class WriteCaseFileAction(_EngineeringModel):
    type: Literal["write_case_file"]
    path: str = Field(min_length=1, max_length=240)
    content: str = Field(min_length=1, max_length=1_000_000)
    rationale: str = Field(default="", max_length=200)


class CaseBundleFile(_EngineeringModel):
    """One agent-authored OpenFOAM case file in a high-level execution plan."""

    path: str = Field(min_length=1, max_length=240)
    content: str = Field(min_length=1, max_length=1_000_000)

    @model_validator(mode="after")
    def validate_path(self) -> Self:
        if not re.fullmatch(r"(?:0|constant|system)/[A-Za-z0-9_.\/-]+", self.path) or ".." in self.path:
            raise ValueError(f"Unsafe case bundle path: {self.path}")
        return self


class FoamDictionaryEntry(_EngineeringModel):
    """One deterministic OpenFOAM dictionary assignment.

    ``path`` expresses dictionary nesting while ``value`` carries only the Agent-owned
    OpenFOAM value expression. Python owns braces and semicolons.
    """

    path: str = Field(
        min_length=1,
        max_length=300,
        pattern=r"^[A-Za-z_][A-Za-z0-9_:+-]*(?:\.[A-Za-z_][A-Za-z0-9_:+-]*)*$",
    )
    value: str = Field(min_length=1, max_length=20_000)


class TypedFoamDictionaryFile(_EngineeringModel):
    """Compact dictionary representation serialized by deterministic Python."""

    path: str = Field(min_length=1, max_length=240)
    entries: list[FoamDictionaryEntry] = Field(min_length=1, max_length=300)

    @model_validator(mode="after")
    def validate_path_and_entries(self) -> Self:
        if not re.fullmatch(r"(?:0|constant|system|postprocessConfig)/[A-Za-z0-9_.\/-]+", self.path) or ".." in self.path:
            raise ValueError(f"Unsafe typed dictionary path: {self.path}")
        keys = [item.path for item in self.entries]
        if len(keys) != len(set(keys)):
            raise ValueError("Typed dictionary contains duplicate entry paths.")
        return self


class CaseFilePatch(_EngineeringModel):
    """Exact deterministic patch for one already-observed case file."""

    path: str = Field(min_length=1, max_length=240)
    old: str = Field(min_length=1, max_length=80_000)
    new: str = Field(max_length=80_000)

    @model_validator(mode="after")
    def validate_path(self) -> Self:
        if not re.fullmatch(r"(?:0|constant|system)/[A-Za-z0-9_.\/-]+", self.path) or ".." in self.path:
            raise ValueError(f"Unsafe patch path: {self.path}")
        return self


class PatchCaseFileAction(_EngineeringModel):
    type: Literal["patch_case_file"]
    patch: CaseFilePatch


class DeleteCaseFileAction(_EngineeringModel):
    type: Literal["delete_case_file"]
    path: str = Field(min_length=1, max_length=240)
    rationale: str = Field(default="", max_length=200)


class ValidateDictionaryAction(_EngineeringModel):
    type: Literal["validate_dictionary"]
    path: str = Field(min_length=1, max_length=240)
    rationale: str = Field(default="", max_length=200)


class SurfaceCheckAction(_EngineeringModel):
    type: Literal["surface_check"]
    path: str = Field(min_length=1, max_length=240)
    rationale: str = Field(default="", max_length=200)


class RunMeshCommandAction(_EngineeringModel):
    type: Literal["run_mesh_command"]
    command: Literal[
        "blockMesh",
        "surfaceFeatureExtract",
        "snappyHexMesh",
        "createPatch",
        "checkMesh",
    ]
    rationale: str = Field(default="", max_length=200)


class ValidatePreSolveAction(_EngineeringModel):
    """Run deterministic solver-input completeness checks for Agent-declared files.

    This deliberately carries only the solver-required file declaration needed by
    the deterministic gate.  It does not encode solver-to-field policy in Python.
    """

    type: Literal["validate_pre_solve"]
    required_case_files: list[str] = Field(min_length=1, max_length=80)
    rationale: str = Field(default="", max_length=200)

    @model_validator(mode="after")
    def validate_required_case_files(self) -> Self:
        if len(self.required_case_files) != len(set(self.required_case_files)):
            raise ValueError("validate_pre_solve contains duplicate required case files.")
        for path in self.required_case_files:
            if not re.fullmatch(r"(?:0|constant|system)/[A-Za-z0-9_.\/-]+", path) or ".." in path:
                raise ValueError(f"Unsafe required case file path: {path}")
        return self


class FinishPreviewAction(_EngineeringModel):
    type: Literal["finish_preview"]
    plan: EngineeringPlan
    rationale: str = Field(default="", max_length=200)


class RetrySolverAction(_EngineeringModel):
    type: Literal["retry_solver"]
    plan: EngineeringPlan
    rationale: str = Field(default="", max_length=200)


class BlockAction(_EngineeringModel):
    type: Literal["block"]
    reason: str = Field(min_length=1, max_length=4000)
    needs_user_input: bool = False
    rationale: str = Field(default="", max_length=200)


EngineeringSequenceMemberAction = (
    WriteCaseFileAction
    | DeleteCaseFileAction
    | ValidateDictionaryAction
    | SurfaceCheckAction
    | RunMeshCommandAction
    | ValidatePreSolveAction
    | FinishPreviewAction
    | RetrySolverAction
)


class ExecuteCasePlanAction(_EngineeringModel):
    """High-level case construction + deterministic validation/execution plan.

    One LLM turn may author the complete case bundle and the predictable native
    pipeline. Python still executes every file write and native validation through
    the existing sandbox, budgets, safety gates and stop-on-failure semantics.
    On success the supplied EngineeringPlan is finalized and sealed in the same
    LLM turn; on the first failure execution stops and the native evidence is
    returned to the next LLM turn for repair.
    """

    type: Literal["execute_case_plan"]
    goal: str = Field(min_length=1, max_length=1000)
    files: list[CaseBundleFile] = Field(default_factory=list, max_length=40)
    typed_dictionaries: list[TypedFoamDictionaryFile] = Field(default_factory=list, max_length=40)
    validate_dictionaries: list[str] = Field(default_factory=list, max_length=40)
    surface_checks: list[str] = Field(default_factory=list, max_length=16)
    mesh_commands: list[Literal[
        "blockMesh",
        "surfaceFeatureExtract",
        "snappyHexMesh",
        "createPatch",
        "checkMesh",
    ]] = Field(min_length=1, max_length=8)
    required_case_files: list[str] = Field(min_length=1, max_length=80)
    plan: EngineeringPlan
    rationale: str = Field(default="", max_length=200)

    @model_validator(mode="after")
    def validate_execution_plan(self) -> Self:
        if not self.files and not self.typed_dictionaries:
            raise ValueError("execute_case_plan requires at least one raw or typed case file.")
        paths = [item.path for item in self.files] + [item.path for item in self.typed_dictionaries]
        if len(paths) != len(set(paths)):
            raise ValueError("execute_case_plan contains duplicate file paths.")

        for collection_name, paths_to_check in (
            ("validate_dictionaries", self.validate_dictionaries),
            ("surface_checks", self.surface_checks),
            ("required_case_files", self.required_case_files),
        ):
            if len(paths_to_check) != len(set(paths_to_check)):
                raise ValueError(f"execute_case_plan contains duplicate {collection_name} paths.")
            for path in paths_to_check:
                if not re.fullmatch(r"(?:0|constant|system)/[A-Za-z0-9_.\/-]+", path) or ".." in path:
                    raise ValueError(f"Unsafe {collection_name} path: {path}")

        if self.mesh_commands[-1] != "checkMesh":
            raise ValueError("execute_case_plan mesh_commands must end with checkMesh.")
        if self.mesh_commands.count("checkMesh") != 1:
            raise ValueError("execute_case_plan must contain exactly one checkMesh command.")
        if set(self.required_case_files) != set(self.plan.required_case_files):
            raise ValueError(
                "execute_case_plan required_case_files must exactly match plan.required_case_files."
            )
        return self


class RepairCasePlanAction(_EngineeringModel):
    """Delta-only repair plan. Existing plan and unchanged files remain Python state."""

    type: Literal["repair_case_plan"]
    diagnosis: str = Field(min_length=1, max_length=800)
    patches: list[CaseFilePatch] = Field(default_factory=list, max_length=20)
    replacement_files: list[CaseBundleFile] = Field(default_factory=list, max_length=12)
    typed_dictionaries: list[TypedFoamDictionaryFile] = Field(default_factory=list, max_length=12)
    validate_dictionaries: list[str] = Field(default_factory=list, max_length=24)
    surface_checks: list[str] = Field(default_factory=list, max_length=12)
    mesh_commands: list[Literal[
        "blockMesh", "surfaceFeatureExtract", "snappyHexMesh", "createPatch", "checkMesh"
    ]] = Field(default_factory=list, max_length=8)
    validate_pre_solve: bool = True
    retry_solver: bool = False
    updated_plan: EngineeringPlan | None = None

    @model_validator(mode="after")
    def validate_repair(self) -> Self:
        # A repair may be artifact-changing or metadata-only.  The latter is
        # required when deterministic execution already succeeded but the
        # EngineeringPlan itself is inconsistent with observed capability/case
        # evidence (for example, a stale or placeholder solver name).
        if not (
            self.patches
            or self.replacement_files
            or self.typed_dictionaries
            or self.updated_plan is not None
        ):
            raise ValueError(
                "repair_case_plan requires at least one changed file/patch or an updated_plan."
            )
        paths = [x.path for x in self.patches] + [x.path for x in self.replacement_files] + [x.path for x in self.typed_dictionaries]
        if len(paths) != len(set(paths)):
            raise ValueError("repair_case_plan may change each file at most once per turn.")
        if self.mesh_commands and self.mesh_commands.count("checkMesh") > 1:
            raise ValueError("repair_case_plan may run checkMesh at most once.")
        return self



class CandidateCasePlanRepairAction(_EngineeringModel):
    """Delta repair for an in-memory, not-yet-committed execute_case_plan candidate."""

    type: Literal["repair_candidate_case_plan"]
    diagnosis: str = Field(min_length=1, max_length=800)
    patches: list[CaseFilePatch] = Field(default_factory=list, max_length=12)
    replacement_files: list[CaseBundleFile] = Field(default_factory=list, max_length=12)
    typed_dictionaries: list[TypedFoamDictionaryFile] = Field(default_factory=list, max_length=12)
    drop_paths: list[str] = Field(default_factory=list, max_length=12)

    @model_validator(mode="after")
    def validate_candidate_repair(self) -> Self:
        if not (
            self.patches
            or self.replacement_files
            or self.typed_dictionaries
            or self.drop_paths
        ):
            raise ValueError(
                "repair_candidate_case_plan requires at least one candidate file change."
            )
        paths = (
            [item.path for item in self.patches]
            + [item.path for item in self.replacement_files]
            + [item.path for item in self.typed_dictionaries]
            + list(self.drop_paths)
        )
        if len(paths) != len(set(paths)):
            raise ValueError("repair_candidate_case_plan may change each path at most once per turn.")
        for path in self.drop_paths:
            if not re.fullmatch(r"(?:0|constant|system)/[A-Za-z0-9_.\/-]+", path) or ".." in path:
                raise ValueError(f"Unsafe candidate drop path: {path}")
        return self

# Phase-specific compact contracts. Agent identity remains one CFDEngineeringAgent; only
# permissions/schema vary by phase so repeated calls do not carry the giant all-phase union.
PrepareAction = SearchCapabilitiesAction | SearchReferencesAction | ReadReferenceAction | ReadCaseFileAction | ExecuteCasePlanAction | BlockAction
class PrepareTurn(_EngineeringModel):
    action: PrepareAction

# A case-plan authoring failure happens before any candidate file is committed.
# At that point reference/tool exploration is usually counterproductive: the model
# already has the engineering decision and a deterministic serialization/safety
# diagnostic. Force the next turn to either resubmit one complete corrected plan
# or block, preventing long search/repair thrash against an intentionally empty case.
CasePlanRetryAction = CandidateCasePlanRepairAction | BlockAction
class CasePlanRetryTurn(_EngineeringModel):
    action: CasePlanRetryAction

RepairAction = SearchReferencesAction | ReadReferenceAction | ReadCaseFileAction | RepairCasePlanAction | BlockAction
class RepairTurn(_EngineeringModel):
    action: RepairAction

RevisionAction = SearchReferencesAction | ReadReferenceAction | ReadCaseFileAction | RepairCasePlanAction | BlockAction
class RevisionTurn(_EngineeringModel):
    action: RevisionAction

FinalizationAction = FinishPreviewAction | BlockAction
class FinalizationTurn(_EngineeringModel):
    action: FinalizationAction

RuntimeRepairAction = SearchReferencesAction | ReadReferenceAction | ReadCaseFileAction | RepairCasePlanAction | BlockAction
class RuntimeRepairTurn(_EngineeringModel):
    action: RuntimeRepairAction


class EngineeringSequenceAction(_EngineeringModel):
    """A short ordered engineering intention executed without intermediate LLM calls."""

    type: Literal["sequence"]
    goal: str = Field(min_length=1, max_length=1000)
    actions: list[EngineeringSequenceMemberAction] = Field(min_length=2, max_length=6)
    rationale: str = Field(default="", max_length=200)

    @model_validator(mode="after")
    def validate_sequence_shape(self) -> Self:
        terminal_types = {"finish_preview", "retry_solver"}
        for index, action in enumerate(self.actions):
            if action.type in terminal_types and index != len(self.actions) - 1:
                raise ValueError(
                    f"{action.type} may appear only as the final action in an engineering sequence."
                )

        # Prevent the exact repair-thrashing pattern observed in long runs: the same
        # file may not be rewritten repeatedly without an intervening deterministic
        # validator/native execution.  The Agent may still rewrite after validation.
        last_write_index: dict[str, int] = {}
        validation_since_write: dict[str, bool] = {}
        for index, action in enumerate(self.actions):
            if isinstance(action, WriteCaseFileAction):
                previous = last_write_index.get(action.path)
                if previous is not None and not validation_since_write.get(action.path, False):
                    raise ValueError(
                        f"Sequence rewrites {action.path} without an intervening validation/native action."
                    )
                last_write_index[action.path] = index
                validation_since_write[action.path] = False
                continue
            if isinstance(action, ValidateDictionaryAction):
                validation_since_write[action.path] = True
                continue
            if isinstance(action, SurfaceCheckAction):
                validation_since_write[action.path] = True
                continue
            if isinstance(action, (RunMeshCommandAction, ValidatePreSolveAction)):
                for path in list(validation_since_write):
                    validation_since_write[path] = True
        return self


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
    | PatchCaseFileAction
    | DeleteCaseFileAction
    | ValidateDictionaryAction
    | SurfaceCheckAction
    | RunMeshCommandAction
    | ValidatePreSolveAction
    | FinishPreviewAction
    | RetrySolverAction
    | BlockAction
    | EngineeringSequenceAction
    | ExecuteCasePlanAction
    | RepairCasePlanAction
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
    sequence_id: str | None = Field(default=None, max_length=120)
    sequence_goal: str | None = Field(default=None, max_length=1000)
    sequence_index: int | None = Field(default=None, ge=1, le=64)
    sequence_length: int | None = Field(default=None, ge=2, le=128)

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
