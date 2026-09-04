from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic.json_schema import SkipJsonSchema


class _EngineeringModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


ENGINEERING_EVENT_OBSERVED_EVIDENCE_LIMIT = 24


def _soft_text(value: object, *, limit: int) -> str:
    """Normalize non-authoritative protocol text without changing CFD semantics."""

    text = str(value or "").strip()
    return text[:limit]


def _soft_query_list(value: object, *, limit: int, max_items: int) -> list[str]:
    """Normalize retrieval queries; malformed query prose must not kill a run."""

    if value is None:
        items: list[object] = []
    elif isinstance(value, str):
        items = [value]
    elif isinstance(value, (list, tuple)):
        items = list(value)
    else:
        items = [value]
    result: list[str] = []
    seen: set[str] = set()
    for item in items:
        text = _soft_text(item, limit=limit)
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text)
        if len(result) >= max_items:
            break
    return result


def _normalize_gap_id(value: object, *, fallback_text: str = "") -> str:
    """Normalize an opaque evidence-gap identifier; IDs carry no CFD meaning."""

    text = str(value or "").strip().upper()
    match = re.fullmatch(r"G([0-9]{1,4})", text)
    if match:
        return f"G{int(match.group(1)):02d}"
    digits = "".join(ch for ch in text if ch.isdigit())[:4]
    if digits:
        return f"G{int(digits):02d}"
    digest = int(hashlib.sha256(fallback_text.encode("utf-8")).hexdigest()[:6], 16) % 10000
    return f"G{digest:04d}"


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
    """Deterministically issued compact evidence descriptor."""

    evidence_id: str = Field(pattern=r"^ev_(?:cap|ref)_[0-9a-f]{20}$")
    kind: Literal["capability", "openfoam_reference"]
    reference: str = Field(min_length=1, max_length=1000)
    summary: str = Field(min_length=1, max_length=1200)


class EngineeringEvidenceRecord(_EngineeringModel):
    """Durable structured payload produced by deterministic evidence retrieval.

    Large capability/reference payloads live here instead of in EngineeringEvent.output_excerpt.
    Events remain compact progress/audit records while the context compiler selectively projects
    these structured records back to the LLM.
    """

    record_id: str = Field(pattern=r"^evrec_[0-9a-f]{20}$")
    phase: str = Field(min_length=1, max_length=80)
    step: int = Field(ge=1)
    action_type: str = Field(min_length=1, max_length=80)
    payload: Any
    observed_evidence: list[ObservedEngineeringEvidence] = Field(default_factory=list)


class EngineeringDefaultAssumption(_EngineeringModel):
    """Agent-selected value for a detail the user intentionally delegated.

    This is not user evidence.  The explicit provenance marker prevents representative
    engineering choices from being confused with confirmed intake facts.
    """

    parameter: str = Field(min_length=1, max_length=160)
    value: str = Field(min_length=1, max_length=300)
    unit: str = Field(default="", max_length=80)
    basis: Literal[
        "representative",
        "common_practice",
        "simplified_geometry",
        "dimensionless_normalization",
        "material_reference",
        "conservative",
        "other",
    ] = "representative"
    rationale: str = Field(min_length=1, max_length=600)
    source: Literal["engineering_default"] = "engineering_default"
    evidence_ids: list[str] = Field(default_factory=list, max_length=8)

    @model_validator(mode="after")
    def validate_evidence_ids(self) -> Self:
        for evidence_id in self.evidence_ids:
            if not re.fullmatch(r"ev_(?:cap|ref)_[0-9a-f]{20}", evidence_id):
                raise ValueError(f"Invalid engineering-default evidence ID: {evidence_id}")
        if len(self.evidence_ids) != len(set(self.evidence_ids)):
            raise ValueError("Engineering default contains duplicate evidence IDs.")
        return self


_BINDABLE_PLAN_FIELDS = Literal[
    "problem_interpretation",
    "temporal_behavior",
    "motion_kind",
    "mesh_motion_requirement",
    "mesh_strategy",
    "decisions",
    "assumptions",
    "engineering_defaults",
    "required_case_files",
    "postprocess_strategy",
]


class CaseContentAssertion(_EngineeringModel):
    """Compact structural evidence pointer; ``contains`` is legacy v2.15 input."""

    path: str = Field(min_length=1, max_length=240)
    entry_path: str | None = Field(default=None, max_length=300)
    expected_value: str = Field(default="", max_length=160)
    anchor: str = Field(default="", max_length=160)
    contains: SkipJsonSchema[list[str]] = Field(default_factory=list, max_length=8)

    @model_validator(mode="before")
    @classmethod
    def normalize_pointer(cls, value: Any):
        if not isinstance(value, dict):
            return value
        normalized = dict(value)
        normalized["entry_path"] = _soft_text(normalized.get("entry_path"), limit=300) or None
        normalized["expected_value"] = _soft_text(normalized.get("expected_value"), limit=160)
        normalized["anchor"] = _soft_text(normalized.get("anchor"), limit=160)
        raw = normalized.get("contains")
        if isinstance(raw, str):
            raw = [raw]
        if not isinstance(raw, (list, tuple)):
            raw = []
        snippets: list[str] = []
        for item in raw:
            text = str(item or "").strip()[:500]
            if text and text not in snippets:
                snippets.append(text)
            if len(snippets) >= 8:
                break
        normalized["contains"] = snippets
        return normalized

    @model_validator(mode="after")
    def validate_assertion(self) -> Self:
        if not re.fullmatch(r"(?:0|constant|system)/[A-Za-z0-9_.\/-]+", self.path) or ".." in self.path:
            raise ValueError(f"Unsafe semantic assertion path: {self.path}")
        return self


class NumericEvidenceTerm(_EngineeringModel):
    """Compact numeric artifact locator; excerpt/value_token are legacy v2.15 input."""

    path: str = Field(min_length=1, max_length=240)
    entry_path: str | None = Field(default=None, max_length=300)
    anchor: str = Field(default="", max_length=160)
    number_index: int = Field(default=0, ge=0, le=31)
    occurrence: int = Field(default=0, ge=0, le=31)
    multiplier: float = 1.0
    excerpt: SkipJsonSchema[str] = Field(default="", max_length=500)
    value_token: SkipJsonSchema[str] = Field(default="", max_length=80)

    @model_validator(mode="before")
    @classmethod
    def normalize_protocol_text(cls, value: Any):
        if not isinstance(value, dict):
            return value
        normalized = dict(value)
        normalized["entry_path"] = _soft_text(normalized.get("entry_path"), limit=300) or None
        normalized["anchor"] = _soft_text(normalized.get("anchor"), limit=160)
        normalized["excerpt"] = str(normalized.get("excerpt") or "").strip()[:500]
        normalized["value_token"] = str(normalized.get("value_token") or "").strip()[:80]
        return normalized

    @model_validator(mode="after")
    def validate_term(self) -> Self:
        if not re.fullmatch(r"(?:0|constant|system)/[A-Za-z0-9_.\/-]+", self.path) or ".." in self.path:
            raise ValueError(f"Unsafe numeric semantic evidence path: {self.path}")
        return self


class NumericRelationAssertion(_EngineeringModel):
    """Generic numerator-product / denominator-product semantic relation."""

    numerator: list[NumericEvidenceTerm] = Field(default_factory=list, max_length=8)
    denominator: list[NumericEvidenceTerm] = Field(default_factory=list, max_length=8)
    relative_tolerance: float = 1e-6


class ConfirmedFactBinding(_EngineeringModel):
    """Audit mapping from a confirmed fact to its claimed implementation.

    The wire format deliberately separates case-file paths from plan fields so the
    model does not need to memorize a string-prefix mini-protocol such as
    ``case:...``/``plan:...``.  A small legacy adapter still accepts that older
    representation when loading persisted v2.10.1 state.
    """

    fact_id: str = Field(pattern=r"^[a-z][a-z0-9_.-]*$")
    case_files: SkipJsonSchema[list[str]] = Field(default_factory=list, max_length=12)
    plan_fields: list[_BINDABLE_PLAN_FIELDS] = Field(default_factory=list, max_length=12)
    case_assertions: list[CaseContentAssertion] = Field(default_factory=list, max_length=12)
    numeric_relation: NumericRelationAssertion | None = None
    explanation: SkipJsonSchema[str] = Field(default="", max_length=160)

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
        evidence_paths = {item.path for item in self.case_assertions}
        if self.numeric_relation is not None:
            evidence_paths.update(
                item.path for item in [*self.numeric_relation.numerator, *self.numeric_relation.denominator]
            )
        if not self.case_files and not self.plan_fields and not evidence_paths:
            raise ValueError("Confirmed fact binding requires a plan field, case file, or semantic evidence pointer.")
        if len(self.case_files) != len(set(self.case_files)):
            raise ValueError("Confirmed fact binding contains duplicate case file refs.")
        if len(self.plan_fields) != len(set(self.plan_fields)):
            raise ValueError("Confirmed fact binding contains duplicate plan field refs.")
        for path in self.case_files:
            if not re.fullmatch(r"(?:0|constant|system)/[A-Za-z0-9_.\/-]+", path) or ".." in path:
                raise ValueError(f"Unsafe case implementation ref: {path}")
        assertion_keys = [
            (
                item.path,
                item.entry_path,
                item.expected_value,
                item.anchor,
                tuple(item.contains),
            )
            for item in self.case_assertions
        ]
        if len(assertion_keys) != len(set(assertion_keys)):
            raise ValueError("Confirmed fact binding contains duplicate semantic assertions.")
        relation_paths = {
            item.path
            for item in (
                [*self.numeric_relation.numerator, *self.numeric_relation.denominator]
                if self.numeric_relation is not None
                else []
            )
        }
        return self

    @property
    def implementation_refs(self) -> list[str]:
        """Compatibility/audit projection used by older internal callers."""

        return [*(f"case:{path}" for path in self.case_files), *(f"plan:{field}" for field in self.plan_fields)]


class RegionSolverAssignment(_EngineeringModel):
    region: str = Field(pattern=r"^[A-Za-z][A-Za-z0-9_.-]*$", max_length=120)
    solver_module: str = Field(pattern=r"^[A-Za-z][A-Za-z0-9_]*$", max_length=120)
    provider_id: str = Field(min_length=1, max_length=240)


class OpenFOAMExecutionSpec(_EngineeringModel):
    """Agent-selected native execution topology for Foundation v13/v14.

    Python does not choose the driver or modules. It verifies that the selected
    executable is present in the sourced trusted installation and that the case
    declares the matching solver/regionSolvers semantics.
    """

    driver: str = Field(pattern=r"^[A-Za-z][A-Za-z0-9_.+-]*$", max_length=160)
    driver_provider_id: str = Field(min_length=1, max_length=240)
    solver_module: str | None = Field(default=None, pattern=r"^[A-Za-z][A-Za-z0-9_]*$", max_length=120)
    solver_provider_id: str | None = Field(default=None, max_length=240)
    regions: list[RegionSolverAssignment] = Field(default_factory=list, max_length=64)
    arguments: list[str] = Field(default_factory=list, max_length=24)

    @model_validator(mode="after")
    def validate_execution_topology(self) -> Self:
        if self.driver == "foamRun":
            if not self.solver_module or not self.solver_provider_id:
                raise ValueError("foamRun execution requires solver_module and solver_provider_id.")
            if self.regions:
                raise ValueError("foamRun execution cannot declare region solver assignments.")
        elif self.driver == "foamMultiRun":
            if self.solver_module is not None or self.solver_provider_id is not None:
                raise ValueError("foamMultiRun uses region solver assignments, not one solver_module.")
            if not self.regions:
                raise ValueError("foamMultiRun execution requires at least one region solver assignment.")
        elif self.regions or self.solver_module is not None or self.solver_provider_id is not None:
            raise ValueError("Direct solver applications cannot declare modular solver fields.")
        region_names = [item.region for item in self.regions]
        if len(region_names) != len(set(region_names)):
            raise ValueError("Execution spec contains duplicate region names.")
        for arg in self.arguments:
            if not arg or len(arg) > 1000 or "\x00" in arg or "\n" in arg or "\r" in arg:
                raise ValueError("Execution arguments must be bounded single-line strings.")
        return self


class NativeOpenFOAMCommand(_EngineeringModel):
    """One discovered OpenFOAM application invocation inside the case workspace."""

    command: str = Field(pattern=r"^[A-Za-z][A-Za-z0-9_.+-]*$", max_length=160)
    arguments: list[str] = Field(default_factory=list, max_length=24)
    role: Literal["preprocess", "mesh", "mesh_validation", "initialization", "utility"] = "utility"
    rationale: str = Field(default="", max_length=240)

    @model_validator(mode="after")
    def validate_arguments(self) -> Self:
        for arg in self.arguments:
            if not arg or len(arg) > 1000 or "\x00" in arg or "\n" in arg or "\r" in arg:
                raise ValueError("Native command arguments must be bounded single-line strings.")
            if ".." in re.split(r"[/\\]+", arg):
                raise ValueError("Native command arguments cannot contain parent traversal.")
        return self


class EngineeringPlan(_EngineeringModel):
    """Agent-owned CFD engineering decisions.

    The schema intentionally records decisions without encoding OpenFOAM implementation
    policy.  Python validates provenance, safety and runtime evidence; it does not
    infer a mesh method, boundary condition, solver or numerical scheme from these
    fields.
    """

    schema_version: Literal["2.0"] = "2.0"
    case_name: str = Field(pattern=r"^[A-Za-z][A-Za-z0-9_.-]*$", max_length=80)
    solver: str = Field(pattern=r"^[A-Za-z][A-Za-z0-9_.+-]*$", max_length=160)
    solver_provider_id: str = Field(min_length=1, max_length=240)
    execution: OpenFOAMExecutionSpec | None = None
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
    engineering_defaults: list[EngineeringDefaultAssumption] = Field(default_factory=list, max_length=80)
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
        default_parameters = [item.parameter.casefold() for item in self.engineering_defaults]
        if len(default_parameters) != len(set(default_parameters)):
            raise ValueError("Engineering plan contains duplicate engineering-default parameters.")
        if self.execution is not None:
            if self.execution.driver == "foamRun":
                if self.solver != self.execution.solver_module:
                    raise ValueError("EngineeringPlan.solver must mirror the foamRun solver_module.")
                if self.solver_provider_id != self.execution.solver_provider_id:
                    raise ValueError("EngineeringPlan.solver_provider_id must mirror the foamRun solver provider.")
            elif self.execution.driver == "foamMultiRun":
                if self.solver != "foamMultiRun":
                    raise ValueError("EngineeringPlan.solver must be foamMultiRun for multi-region execution.")
                if self.solver_provider_id != self.execution.driver_provider_id:
                    raise ValueError("EngineeringPlan.solver_provider_id must mirror the foamMultiRun driver provider.")
            else:
                if self.solver != self.execution.driver:
                    raise ValueError("EngineeringPlan.solver must mirror a direct solver application driver.")
                if self.solver_provider_id != self.execution.driver_provider_id:
                    raise ValueError("EngineeringPlan.solver_provider_id must mirror the direct driver provider.")
        return self

    def digest(self) -> str:
        data = self.model_dump(mode="json")
        # Empty v2.15 semantic assertion fields must not invalidate case seals
        # created by older releases.  Non-empty assertions remain part of the
        # digest and are therefore revision-bound like every other plan field.
        for binding in data.get("confirmed_fact_bindings", []):
            if not binding.get("case_assertions"):
                binding.pop("case_assertions", None)
            else:
                for assertion in binding["case_assertions"]:
                    for key in ("entry_path", "expected_value", "anchor"):
                        if assertion.get(key) in {None, ""}:
                            assertion.pop(key, None)
            relation = binding.get("numeric_relation")
            if relation is None:
                binding.pop("numeric_relation", None)
            else:
                for term in [*relation.get("numerator", []), *relation.get("denominator", [])]:
                    for key, default in (("entry_path", None), ("anchor", ""), ("number_index", 0), ("occurrence", 0)):
                        if term.get(key) == default:
                            term.pop(key, None)
            if binding.get("explanation") == "":
                binding.pop("explanation", None)
        payload = json.dumps(
            data,
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


class EvidenceGapRequest(_EngineeringModel):
    """One explicit tool/version evidence gap for deterministic batch retrieval.

    Query strings and opaque gap IDs are protocol metadata, not CFD decisions. They are
    therefore normalized deterministically so harmless formatting mistakes cannot abort
    the engineering workflow. A follow-up search must use a new gap ID and may identify
    the previously retrieved gap via ``refines_gap_id``.
    """

    gap_id: str = Field(pattern=r"^G[0-9]{2,4}$")
    refines_gap_id: str | None = Field(default=None, pattern=r"^G[0-9]{2,4}$")
    missing_evidence: str = Field(min_length=1, max_length=400)
    why_required: str = Field(min_length=1, max_length=400)
    capability_queries: list[str] = Field(default_factory=list, max_length=2)
    reference_queries: list[str] = Field(default_factory=list, max_length=3)
    reference_scope: Literal["all", "tutorials", "source", "etc"] = "all"
    read_top_reference_matches: int = Field(default=1, ge=0, le=2)

    @model_validator(mode="before")
    @classmethod
    def normalize_protocol_fields(cls, value: Any):
        if not isinstance(value, dict):
            return value
        normalized = dict(value)
        missing = _soft_text(normalized.get("missing_evidence"), limit=400)
        why = _soft_text(normalized.get("why_required"), limit=400)
        if not missing:
            missing = "Exact tool/version evidence needed for the proposed engineering action."
        if not why:
            why = "The Agent declared this external evidence necessary before proceeding."
        normalized["missing_evidence"] = missing
        normalized["why_required"] = why
        normalized["gap_id"] = _normalize_gap_id(
            normalized.get("gap_id"), fallback_text=missing
        )
        parent = normalized.get("refines_gap_id")
        if parent not in {None, ""}:
            normalized["refines_gap_id"] = _normalize_gap_id(parent, fallback_text=str(parent))
        else:
            normalized["refines_gap_id"] = None
        normalized["capability_queries"] = _soft_query_list(
            normalized.get("capability_queries"), limit=500, max_items=2
        )
        normalized["reference_queries"] = _soft_query_list(
            normalized.get("reference_queries"), limit=500, max_items=3
        )
        # Empty/overlong query prose is a retrieval-protocol issue, not a reason to
        # fail the CFD run. Fall back to the Agent's own missing-evidence statement.
        if not normalized["capability_queries"] and not normalized["reference_queries"]:
            normalized["reference_queries"] = [missing[:500]]
        if normalized.get("reference_scope") not in {"all", "tutorials", "source", "etc"}:
            normalized["reference_scope"] = "all"
        try:
            read_top = int(normalized.get("read_top_reference_matches", 1))
        except (TypeError, ValueError):
            read_top = 1
        normalized["read_top_reference_matches"] = max(0, min(2, read_top))
        return normalized

    # Gap identity/refinement is protocol metadata, not CFD semantics.  Do not
    # reject harmless ID mistakes here: the authoritative EvidenceGapLedger in
    # CFDEngineeringAgent deterministically reissues colliding/self-refining IDs.


class GatherEvidenceAction(_EngineeringModel):
    """Batch retrieval for explicit unresolved evidence gaps.

    This replaces free-form prepare search loops. Python performs bounded capability/
    reference retrieval, records novelty per gap, and can refuse repeated stagnant gaps.
    """

    type: Literal["gather_evidence"]
    gaps: list[EvidenceGapRequest] = Field(min_length=1, max_length=4)
    rationale: str = Field(default="", max_length=200)

    # Duplicate/colliding IDs are intentionally preserved until the Agent's
    # EvidenceGapLedger sees the current phase history.  Only that ledger can
    # distinguish an exact duplicate from a legitimate refinement that needs a
    # freshly issued opaque ID.


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
    """Compact OpenFOAM file representation serialized by deterministic Python.

    ``foam_class`` is transport metadata, not a CFD engineering choice. Ordinary
    system/constant dictionaries default to ``dictionary``. Initial fields normally
    infer their class from ``internalField``; set ``foam_class`` only when that shape
    is not statically unambiguous. Python always derives ``object`` and ``location``
    from ``path`` and owns the complete ``FoamFile`` header.
    """

    path: str = Field(min_length=1, max_length=240)
    foam_class: str | None = Field(
        default=None,
        min_length=1,
        max_length=80,
        pattern=r"^[A-Za-z][A-Za-z0-9_]*$",
    )
    entries: list[FoamDictionaryEntry] = Field(min_length=1, max_length=300)

    @model_validator(mode="after")
    def validate_path_and_entries(self) -> Self:
        if not re.fullmatch(r"(?:0|constant|system|postprocessConfig)/[A-Za-z0-9_.\/-]+", self.path) or ".." in self.path:
            raise ValueError(f"Unsafe typed dictionary path: {self.path}")
        keys = [item.path for item in self.entries]
        if len(keys) != len(set(keys)):
            raise ValueError("Typed dictionary contains duplicate entry paths.")
        return self


class BlockMeshVertex(_EngineeringModel):
    coordinates: tuple[float, float, float]


class BlockMeshBlock(_EngineeringModel):
    vertices: tuple[int, int, int, int, int, int, int, int]
    cells: tuple[int, int, int] = Field()
    grading: str = Field(default="simpleGrading (1 1 1)", min_length=1, max_length=1000)

    @model_validator(mode="after")
    def validate_block(self) -> Self:
        if any(index < 0 for index in self.vertices):
            raise ValueError("blockMesh vertex indices must be non-negative.")
        if any(count < 1 for count in self.cells):
            raise ValueError("blockMesh cell counts must be positive.")
        return self


class BlockMeshEdge(_EngineeringModel):
    kind: Literal["arc", "line", "spline", "polyLine"]
    start: int = Field(ge=0)
    end: int = Field(ge=0)
    definition: str = Field(default="", max_length=4000)


class BlockMeshBoundaryPatch(_EngineeringModel):
    name: str = Field(pattern=r"^[A-Za-z_][A-Za-z0-9_.:+-]*$", max_length=120)
    type: str = Field(min_length=1, max_length=120)
    faces: list[tuple[int, int, int, int]] = Field(min_length=1, max_length=400)

    @model_validator(mode="after")
    def validate_faces(self) -> Self:
        if any(index < 0 for face in self.faces for index in face):
            raise ValueError("blockMesh face vertex indices must be non-negative.")
        return self


class TypedBlockMeshFile(_EngineeringModel):
    """Structured blockMeshDict DSL rendered by deterministic Python.

    Python owns OpenFOAM list/dictionary punctuation for vertices/blocks/edges/boundary.
    The Agent still owns the geometry, topology, resolution, grading and patch types.
    """

    path: Literal["system/blockMeshDict"] = "system/blockMeshDict"
    scale: float = Field(default=1.0, gt=0)
    scale_keyword: Literal["scale", "convertToMeters"] = "scale"
    vertices: list[BlockMeshVertex] = Field(min_length=8, max_length=2000)
    blocks: list[BlockMeshBlock] = Field(min_length=1, max_length=500)
    edges: list[BlockMeshEdge] = Field(default_factory=list, max_length=1000)
    boundary: list[BlockMeshBoundaryPatch] = Field(min_length=1, max_length=200)
    merge_patch_pairs: list[tuple[str, str]] = Field(default_factory=list, max_length=100)

    @model_validator(mode="after")
    def validate_topology_refs(self) -> Self:
        vertex_count = len(self.vertices)
        refs = [index for block in self.blocks for index in block.vertices]
        refs.extend(index for edge in self.edges for index in (edge.start, edge.end))
        refs.extend(index for patch in self.boundary for face in patch.faces for index in face)
        if refs and max(refs) >= vertex_count:
            raise ValueError("blockMesh topology references a vertex index outside vertices[].")
        names = [patch.name for patch in self.boundary]
        if len(names) != len(set(names)):
            raise ValueError("blockMesh boundary contains duplicate patch names.")
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


class RunNativeOpenFOAMAction(_EngineeringModel):
    type: Literal["run_openfoam_command"]
    invocation: NativeOpenFOAMCommand


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
    block_kind: Literal[
        "physical_objective_unknown",
        "routing_physics_unknown",
        "engineering_choice_missing",
        "tool_version_unsupported",
        "environment_unavailable",
        "safety_or_integrity",
        "other",
    ] = "other"
    missing_items: list[str] = Field(default_factory=list, max_length=24)
    needs_user_input: bool = False
    rationale: str = Field(default="", max_length=200)


EngineeringSequenceMemberAction = (
    WriteCaseFileAction
    | DeleteCaseFileAction
    | ValidateDictionaryAction
    | SurfaceCheckAction
    | RunMeshCommandAction
    | RunNativeOpenFOAMAction
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
    block_mesh: TypedBlockMeshFile | None = None
    validate_dictionaries: list[str] = Field(default_factory=list, max_length=40)
    surface_checks: list[str] = Field(default_factory=list, max_length=16)
    mesh_commands: list[str] = Field(default_factory=list, max_length=12)
    native_pipeline: list[NativeOpenFOAMCommand] = Field(default_factory=list, max_length=20)
    required_case_files: list[str] = Field(min_length=1, max_length=80)
    plan: EngineeringPlan
    rationale: str = Field(default="", max_length=200)

    @model_validator(mode="after")
    def validate_execution_plan(self) -> Self:
        if not self.files and not self.typed_dictionaries and self.block_mesh is None:
            raise ValueError("execute_case_plan requires at least one raw, typed, or blockMesh case file.")
        paths = [item.path for item in self.files] + [item.path for item in self.typed_dictionaries]
        if self.block_mesh is not None:
            paths.append(self.block_mesh.path)
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

        if any(item.path == "system/blockMeshDict" for item in self.typed_dictionaries):
            raise ValueError("Use block_mesh for system/blockMeshDict; generic typed dictionaries cannot represent blockMesh list syntax safely.")
        for command in self.mesh_commands:
            if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_.+-]*", command):
                raise ValueError(f"Unsafe mesh command identifier: {command}")
        if self.native_pipeline and self.mesh_commands:
            raise ValueError("Use either native_pipeline or legacy mesh_commands, not both.")
        if not self.native_pipeline and not self.mesh_commands:
            raise ValueError("execute_case_plan requires a native validation pipeline.")
        pipeline_names = [item.command for item in self.native_pipeline] if self.native_pipeline else list(self.mesh_commands)
        if pipeline_names.count("checkMesh") != 1:
            raise ValueError("execute_case_plan requires exactly one checkMesh validation command.")
        if pipeline_names[-1] != "checkMesh":
            raise ValueError("execute_case_plan native pipeline must end with checkMesh.")
        if set(self.required_case_files) != set(self.plan.required_case_files):
            raise ValueError(
                "execute_case_plan required_case_files must exactly match plan.required_case_files."
            )
        return self


class ExactCaseFileEdit(_EngineeringModel):
    old: str = Field(min_length=1, max_length=80_000)
    new: str = Field(max_length=80_000)


class CaseFilePatchGroup(_EngineeringModel):
    """Ordered exact edits applied sequentially to one case file."""

    path: str = Field(min_length=1, max_length=240)
    edits: list[ExactCaseFileEdit] = Field(min_length=1, max_length=12)

    @model_validator(mode="after")
    def validate_path(self) -> Self:
        if not re.fullmatch(r"(?:0|constant|system)/[A-Za-z0-9_.\/-]+", self.path) or ".." in self.path:
            raise ValueError(f"Unsafe grouped patch path: {self.path}")
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
    mesh_commands: list[str] = Field(default_factory=list, max_length=12)
    native_pipeline: list[NativeOpenFOAMCommand] = Field(default_factory=list, max_length=16)
    validate_pre_solve: bool = True
    retry_solver: bool = False
    updated_plan: EngineeringPlan | None = None

    @model_validator(mode="after")
    def validate_repair(self) -> Self:
        # A repair may be artifact-changing or metadata-only.  The latter is
        # required when deterministic execution already succeeded but the
        # EngineeringPlan itself is inconsistent with observed capability/case
        # evidence (for example, a stale or placeholder solver name).
        # A true no-op is handled as a controlled unsuccessful engineering action by
        # the executor. It is not a schema error: protocol-shape mistakes must not
        # terminate the whole run before the Agent can correct them.
        patch_paths = [x.path for x in self.patches]
        replacement_paths = [x.path for x in self.replacement_files]
        typed_paths = [x.path for x in self.typed_dictionaries]
        if len(replacement_paths) != len(set(replacement_paths)) or len(typed_paths) != len(set(typed_paths)):
            raise ValueError("repair_case_plan contains duplicate replacement file paths.")
        non_patch_paths = replacement_paths + typed_paths
        if len(non_patch_paths) != len(set(non_patch_paths)):
            raise ValueError("repair_case_plan may replace a file in only one representation per turn.")
        if set(patch_paths) & set(non_patch_paths):
            raise ValueError("repair_case_plan cannot patch and replace the same file in one turn.")
        if any(item.path == "system/blockMeshDict" for item in self.typed_dictionaries):
            raise ValueError("Use block_mesh for system/blockMeshDict repairs.")
        if self.native_pipeline and self.mesh_commands:
            raise ValueError("repair_case_plan must use native_pipeline or mesh_commands, not both.")
        commands = [item.command for item in self.native_pipeline] if self.native_pipeline else list(self.mesh_commands)
        if commands.count("checkMesh") > 1:
            raise ValueError("repair_case_plan may run checkMesh at most once.")
        for command in commands:
            if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_.+-]*", command):
                raise ValueError(f"Unsafe repair command identifier: {command}")
        return self


class RuntimeCaseRepairAction(_EngineeringModel):
    """Runtime-only delta repair with multiple ordered edits per file.

    The user-approved solver/EngineeringPlan is not carried by this contract, which
    keeps the runtime schema small and prevents metadata churn during automatic retry.
    """

    type: Literal["repair_runtime_case"]
    diagnosis: str = Field(min_length=1, max_length=800)
    file_patches: list[CaseFilePatchGroup] = Field(default_factory=list, max_length=8)
    replacement_files: list[CaseBundleFile] = Field(default_factory=list, max_length=8)
    typed_dictionaries: list[TypedFoamDictionaryFile] = Field(default_factory=list, max_length=8)
    validate_dictionaries: list[str] = Field(default_factory=list, max_length=16)
    surface_checks: list[str] = Field(default_factory=list, max_length=8)
    mesh_commands: list[str] = Field(default_factory=list, max_length=10)
    native_pipeline: list[NativeOpenFOAMCommand] = Field(default_factory=list, max_length=12)
    validate_pre_solve: bool = True
    retry_solver: bool = True

    @model_validator(mode="after")
    def validate_runtime_repair(self) -> Self:
        # Empty deltas are handled by the runtime executor as a controlled non-retry.
        # Keep semantic/safety conflicts below as hard schema constraints.
        paths = (
            [item.path for item in self.file_patches]
            + [item.path for item in self.replacement_files]
            + [item.path for item in self.typed_dictionaries]
        )
        if len(paths) != len(set(paths)):
            raise ValueError("repair_runtime_case may represent each file in only one repair mode per turn.")
        if self.native_pipeline and self.mesh_commands:
            raise ValueError("repair_runtime_case must use native_pipeline or mesh_commands, not both.")
        commands = [item.command for item in self.native_pipeline] if self.native_pipeline else list(self.mesh_commands)
        if commands.count("checkMesh") > 1:
            raise ValueError("repair_runtime_case may run checkMesh at most once.")
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
        # Empty candidate deltas are handled as a controlled failed action so the
        # next turn can correct the protocol without mutating the workspace.
        patch_paths = [item.path for item in self.patches]
        replacement_paths = [item.path for item in self.replacement_files]
        typed_paths = [item.path for item in self.typed_dictionaries]
        drop_paths = list(self.drop_paths)
        if len(replacement_paths) != len(set(replacement_paths)) or len(typed_paths) != len(set(typed_paths)) or len(drop_paths) != len(set(drop_paths)):
            raise ValueError("repair_candidate_case_plan contains duplicate replacement/drop paths.")
        exclusive_paths = replacement_paths + typed_paths + drop_paths
        if len(exclusive_paths) != len(set(exclusive_paths)):
            raise ValueError("repair_candidate_case_plan may replace/drop a path in only one mode per turn.")
        if set(patch_paths) & set(exclusive_paths):
            raise ValueError("repair_candidate_case_plan cannot patch and replace/drop the same path in one turn.")
        if any(item.path == "system/blockMeshDict" for item in self.typed_dictionaries):
            raise ValueError("Use block_mesh for system/blockMeshDict candidate repairs.")
        for path in self.drop_paths:
            if not re.fullmatch(r"(?:0|constant|system)/[A-Za-z0-9_.\/-]+", path) or ".." in path:
                raise ValueError(f"Unsafe candidate drop path: {path}")
        return self

class CandidateBlockMeshRepairAction(_EngineeringModel):
    """Replace only the structured blockMesh candidate after pre-commit topology rejection."""

    type: Literal["repair_candidate_block_mesh"]
    diagnosis: str = Field(min_length=1, max_length=800)
    block_mesh: TypedBlockMeshFile


class BlockMeshRepairAction(_EngineeringModel):
    """Local semantic blockMesh repair after a native blockMesh failure.

    The existing EngineeringPlan and all non-mesh files remain Python state. The
    deterministic executor writes this complete structured replacement, validates it,
    reruns blockMesh/checkMesh, then re-runs pre-solve completeness.
    """

    type: Literal["repair_block_mesh"]
    diagnosis: str = Field(min_length=1, max_length=800)
    block_mesh: TypedBlockMeshFile


class StrategyRevisionAction(_EngineeringModel):
    """Delta strategy replacement after a meshing/tool contract is invalidated.

    This is intentionally separate from local repair: the Agent may replace/drop the
    failed meshing artifacts and choose a different mesh command pipeline while the
    confirmed intake remains immutable.
    """

    type: Literal["revise_mesh_strategy"]
    diagnosis: str = Field(min_length=1, max_length=1200)
    patches: list[CaseFilePatch] = Field(default_factory=list, max_length=20)
    replacement_files: list[CaseBundleFile] = Field(default_factory=list, max_length=16)
    typed_dictionaries: list[TypedFoamDictionaryFile] = Field(default_factory=list, max_length=16)
    block_mesh: TypedBlockMeshFile | None = None
    drop_paths: list[str] = Field(default_factory=list, max_length=16)
    validate_dictionaries: list[str] = Field(default_factory=list, max_length=24)
    surface_checks: list[str] = Field(default_factory=list, max_length=12)
    mesh_commands: list[str] = Field(default_factory=list, max_length=12)
    native_pipeline: list[NativeOpenFOAMCommand] = Field(default_factory=list, max_length=20)
    validate_pre_solve: bool = True
    updated_plan: EngineeringPlan | None = None

    @model_validator(mode="after")
    def validate_strategy_revision(self) -> Self:
        paths = [item.path for item in self.replacement_files] + [item.path for item in self.typed_dictionaries] + list(self.drop_paths)
        if self.block_mesh is not None:
            paths.append(self.block_mesh.path)
        if len(paths) != len(set(paths)):
            raise ValueError("revise_mesh_strategy may replace/drop a path in only one mode per turn.")
        if set(item.path for item in self.patches) & set(paths):
            raise ValueError("revise_mesh_strategy cannot patch and replace/drop the same file in one turn.")
        for path in self.drop_paths:
            if not re.fullmatch(r"(?:0|constant|system)/[A-Za-z0-9_.\/-]+", path) or ".." in path:
                raise ValueError(f"Unsafe strategy drop path: {path}")
        if self.native_pipeline and self.mesh_commands:
            raise ValueError("revise_mesh_strategy must use native_pipeline or mesh_commands, not both.")
        commands = [item.command for item in self.native_pipeline] if self.native_pipeline else list(self.mesh_commands)
        if not commands or commands[-1] != "checkMesh" or commands.count("checkMesh") != 1:
            raise ValueError("revise_mesh_strategy native pipeline must end with exactly one checkMesh.")
        if any(item.path == "system/blockMeshDict" for item in self.typed_dictionaries):
            raise ValueError("Use block_mesh for system/blockMeshDict in strategy revisions.")
        return self


def _route_action_payload(value: Any, routes: dict[str, type[_EngineeringModel]]):
    """Validate only the union branch named by ``action.type``.

    JSON transport schemas remain plain ``anyOf`` unions for Codex/Claude compatibility,
    but Python validation does not need to fan a bad payload through every unrelated
    action model.  This keeps diagnostics local to the intended protocol branch.
    """

    if not isinstance(value, dict):
        return value
    raw_action = value.get("action")
    if not isinstance(raw_action, dict):
        return value
    action_type = str(raw_action.get("type", "")).strip()
    model = routes.get(action_type)
    if model is None:
        return value
    normalized = dict(value)
    normalized["action"] = model.model_validate(raw_action)
    return normalized


# Phase-specific compact contracts. Agent identity remains one CFDEngineeringAgent; only
# permissions/schema vary by phase so repeated calls do not carry the giant all-phase union.
PrepareAction = GatherEvidenceAction | ReadCaseFileAction | ExecuteCasePlanAction | BlockAction
class PrepareTurn(_EngineeringModel):
    action: PrepareAction

    @model_validator(mode="before")
    @classmethod
    def route_action(cls, value: Any):
        return _route_action_payload(
            value,
            {
                "gather_evidence": GatherEvidenceAction,
                "read_case_file": ReadCaseFileAction,
                "execute_case_plan": ExecuteCasePlanAction,
                "block": BlockAction,
            },
        )

PrepareDecisionOnlyAction = ExecuteCasePlanAction | BlockAction
class PrepareDecisionOnlyTurn(_EngineeringModel):
    action: PrepareDecisionOnlyAction

    @model_validator(mode="before")
    @classmethod
    def route_action(cls, value: Any):
        return _route_action_payload(
            value,
            {"execute_case_plan": ExecuteCasePlanAction, "block": BlockAction},
        )

# A case-plan authoring failure happens before any candidate file is committed.
# At that point reference/tool exploration is usually counterproductive: the model
# already has the engineering decision and a deterministic serialization/safety
# diagnostic. Force the next turn to either resubmit one complete corrected plan
# or block, preventing long search/repair thrash against an intentionally empty case.
CasePlanRetryAction = CandidateCasePlanRepairAction | BlockAction
class CasePlanRetryTurn(_EngineeringModel):
    action: CasePlanRetryAction

    @model_validator(mode="before")
    @classmethod
    def route_action(cls, value: Any):
        return _route_action_payload(
            value,
            {"repair_candidate_case_plan": CandidateCasePlanRepairAction, "block": BlockAction},
        )

CandidateBlockMeshRepairTurnAction = CandidateBlockMeshRepairAction | BlockAction
class CandidateBlockMeshRepairTurn(_EngineeringModel):
    action: CandidateBlockMeshRepairTurnAction

    @model_validator(mode="before")
    @classmethod
    def route_action(cls, value: Any):
        return _route_action_payload(
            value,
            {"repair_candidate_block_mesh": CandidateBlockMeshRepairAction, "block": BlockAction},
        )

BlockMeshRepairTurnAction = BlockMeshRepairAction | BlockAction
class BlockMeshRepairTurn(_EngineeringModel):
    action: BlockMeshRepairTurnAction

    @model_validator(mode="before")
    @classmethod
    def route_action(cls, value: Any):
        return _route_action_payload(
            value,
            {"repair_block_mesh": BlockMeshRepairAction, "block": BlockAction},
        )

RepairAction = SearchReferencesAction | ReadReferenceAction | ReadCaseFileAction | RepairCasePlanAction | BlockAction
class RepairTurn(_EngineeringModel):
    action: RepairAction

    @model_validator(mode="before")
    @classmethod
    def route_action(cls, value: Any):
        return _route_action_payload(
            value,
            {
                "search_references": SearchReferencesAction,
                "read_reference": ReadReferenceAction,
                "read_case_file": ReadCaseFileAction,
                "repair_case_plan": RepairCasePlanAction,
                "block": BlockAction,
            },
        )

StrategyRevisionTurnAction = StrategyRevisionAction | BlockAction
class StrategyRevisionTurn(_EngineeringModel):
    action: StrategyRevisionTurnAction

    @model_validator(mode="before")
    @classmethod
    def route_action(cls, value: Any):
        return _route_action_payload(
            value,
            {"revise_mesh_strategy": StrategyRevisionAction, "block": BlockAction},
        )

RevisionAction = SearchReferencesAction | ReadReferenceAction | ReadCaseFileAction | RepairCasePlanAction | BlockAction
class RevisionTurn(_EngineeringModel):
    action: RevisionAction

    @model_validator(mode="before")
    @classmethod
    def route_action(cls, value: Any):
        return _route_action_payload(
            value,
            {
                "search_references": SearchReferencesAction,
                "read_reference": ReadReferenceAction,
                "read_case_file": ReadCaseFileAction,
                "repair_case_plan": RepairCasePlanAction,
                "block": BlockAction,
            },
        )

FinalizationAction = FinishPreviewAction | BlockAction
class FinalizationTurn(_EngineeringModel):
    action: FinalizationAction

    @model_validator(mode="before")
    @classmethod
    def route_action(cls, value: Any):
        return _route_action_payload(
            value,
            {"finish_preview": FinishPreviewAction, "block": BlockAction},
        )

RuntimeRepairAction = GatherEvidenceAction | RuntimeCaseRepairAction | BlockAction
class RuntimeRepairTurn(_EngineeringModel):
    action: RuntimeRepairAction

    @model_validator(mode="before")
    @classmethod
    def route_action(cls, value: Any):
        return _route_action_payload(
            value,
            {
                "gather_evidence": GatherEvidenceAction,
                "repair_runtime_case": RuntimeCaseRepairAction,
                "block": BlockAction,
            },
        )


class EngineeringSequenceAction(_EngineeringModel):
    """A short ordered engineering intention executed without intermediate LLM calls."""

    type: Literal["sequence"]
    goal: str = Field(min_length=1, max_length=1000)
    actions: list[EngineeringSequenceMemberAction] = Field(min_length=2, max_length=6)
    rationale: str = Field(default="", max_length=200)

    @model_validator(mode="before")
    @classmethod
    def route_member_actions(cls, value: Any):
        if not isinstance(value, dict) or not isinstance(value.get("actions"), list):
            return value
        routes: dict[str, type[_EngineeringModel]] = {
            "write_case_file": WriteCaseFileAction,
            "delete_case_file": DeleteCaseFileAction,
            "validate_dictionary": ValidateDictionaryAction,
            "surface_check": SurfaceCheckAction,
            "run_mesh_command": RunMeshCommandAction,
            "run_openfoam_command": RunNativeOpenFOAMAction,
            "validate_pre_solve": ValidatePreSolveAction,
            "finish_preview": FinishPreviewAction,
            "retry_solver": RetrySolverAction,
        }
        normalized = dict(value)
        routed: list[object] = []
        for item in value["actions"]:
            if isinstance(item, dict):
                model = routes.get(str(item.get("type", "")).strip())
                if model is not None:
                    item = model.model_validate(item)
            routed.append(item)
        normalized["actions"] = routed
        return normalized

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
    | RunNativeOpenFOAMAction
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

    @model_validator(mode="before")
    @classmethod
    def route_action(cls, value: Any):
        return _route_action_payload(
            value,
            {
                "inspect_environment": InspectEnvironmentAction,
                "search_capabilities": SearchCapabilitiesAction,
                "search_references": SearchReferencesAction,
                "read_reference": ReadReferenceAction,
                "list_case_files": ListCaseFilesAction,
                "read_case_file": ReadCaseFileAction,
                "write_case_file": WriteCaseFileAction,
                "patch_case_file": PatchCaseFileAction,
                "delete_case_file": DeleteCaseFileAction,
                "validate_dictionary": ValidateDictionaryAction,
                "surface_check": SurfaceCheckAction,
                "run_mesh_command": RunMeshCommandAction,
                "run_openfoam_command": RunNativeOpenFOAMAction,
                "validate_pre_solve": ValidatePreSolveAction,
                "finish_preview": FinishPreviewAction,
                "retry_solver": RetrySolverAction,
                "block": BlockAction,
                "sequence": EngineeringSequenceAction,
                "execute_case_plan": ExecuteCasePlanAction,
                "repair_case_plan": RepairCasePlanAction,
            },
        )


class EngineeringEvent(_EngineeringModel):
    @model_validator(mode="before")
    @classmethod
    def normalize_bounded_projection(cls, value: object) -> object:
        """Keep progress/audit projection limits from becoming workflow-fatal.

        EngineeringEvent is not the durable evidence store.  When a caller supplies a
        larger observed-evidence set, retain a bounded projection here; the complete set
        belongs in EngineeringEvidenceRecord and is referenced by payload_ref.
        """

        if not isinstance(value, dict):
            return value
        observed = value.get("observed_evidence")
        if isinstance(observed, (list, tuple)) and len(observed) > ENGINEERING_EVENT_OBSERVED_EVIDENCE_LIMIT:
            normalized = dict(value)
            normalized["observed_evidence"] = list(observed)[:ENGINEERING_EVENT_OBSERVED_EVIDENCE_LIMIT]
            return normalized
        return value

    step: int = Field(ge=1)
    action_type: str = Field(min_length=1, max_length=80)
    success: bool
    summary: str = Field(min_length=1, max_length=4000)
    output_excerpt: str = Field(default="", max_length=12000)
    payload_ref: str | None = Field(default=None, pattern=r"^evrec_[0-9a-f]{20}$")
    artifact_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    native_command_executed: bool = False
    mesh_command_executed: bool = False
    failure_signature: str | None = Field(default=None, max_length=160)
    failure_scope: Literal["local", "pipeline", "strategy"] | None = None
    observed_evidence: list[ObservedEngineeringEvidence] = Field(default_factory=list, max_length=ENGINEERING_EVENT_OBSERVED_EVIDENCE_LIMIT)
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
