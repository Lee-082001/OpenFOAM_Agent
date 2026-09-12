from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator
from pydantic.json_schema import SkipJsonSchema
from openfoam_agent.contracts.models import ConservationCheck
from openfoam_agent.contracts.models import (ImplementationEvidenceBinding, ParallelExecution, RegionCaseLayout, RegionInterface, CompletionContract, QuantityOfInterest)


class _EngineeringModel(BaseModel):
    model_config = ConfigDict(extra="ignore")


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
    risk_level: Literal["low", "medium", "high", "critical"] = "medium"
    evidence_policy: Literal["mandatory", "deferred", "advisory"] = "advisory"
    verification_stage: Literal[
        "design", "authoring", "pre_validation", "pre_execution", "runtime", "result_review"
    ] = "pre_validation"


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


class EngineeringDesignDefault(_EngineeringModel):
    """Agent-owned ordinary design choice without controller/evidence metadata.

    Canonical evidence IDs are controller/audit state, not CFD design content. The
    staged design contract therefore carries only the value selected by the Agent;
    the sealed EngineeringPlan receives provenance metadata deterministically later.
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
    parallel: ParallelExecution = Field(default_factory=ParallelExecution)

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




class RepairEpisode(_EngineeringModel):
    """Controller-owned continuity record for one multi-turn native/case repair.

    Read/search support turns may add observations, but only a newly observed failing
    native/deterministic validation may replace current_failure.  This prevents the
    model from losing the original/current OpenFOAM diagnostic mid-repair.
    """

    episode_id: str = Field(pattern=r"^repair-[0-9]{4,}$")
    root_failure: dict[str, Any]
    current_failure: dict[str, Any]
    current_implicated_files: list[str] = Field(default_factory=list, max_length=24)
    previous_repairs: list[dict[str, Any]] = Field(default_factory=list, max_length=24)
    supporting_observations: list[dict[str, Any]] = Field(default_factory=list, max_length=24)
    validation_generation: int = Field(default=1, ge=1)

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
    region_layouts: list[RegionCaseLayout] = Field(default_factory=list)
    interfaces: list[RegionInterface] = Field(default_factory=list)
    completion: CompletionContract | None = None
    quantities_of_interest: list[QuantityOfInterest] = Field(default_factory=list)
    conservation_checks: list[ConservationCheck] = Field(default_factory=list)
    implementation_evidence_bindings: list[ImplementationEvidenceBinding] = Field(default_factory=list)
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
    plan_conflicts: SkipJsonSchema[list[str]] = Field(default_factory=list)
    confirmed_fact_ids: list[str] = Field(default_factory=list, max_length=200)
    confirmed_fact_bindings: list[ConfirmedFactBinding] = Field(default_factory=list, max_length=200)
    evidence: list[EngineeringEvidence] = Field(default_factory=list, max_length=120)
    required_case_files: list[str] = Field(default_factory=list, max_length=80)
    postprocess_strategy: list[str] = Field(default_factory=list, max_length=40)
    confirmed_intake_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="before")
    @classmethod
    def normalize_progressive_plan(cls, value: Any):
        """Normalize redundant/optional planning metadata instead of rejecting a CFD design.

        Confirmed-fact identity and unsafe paths remain strict later. Duplicate audit metadata,
        redundant solver mirrors and optional result-analysis intent are controller concerns, not
        reasons to discard an otherwise usable engineering plan.
        """
        if not isinstance(value, dict):
            return value
        data = dict(value)
        for key in ("confirmed_fact_ids", "required_case_files"):
            raw = data.get(key) or []
            data[key] = list(dict.fromkeys(str(x) for x in raw if str(x).strip()))
        # Binding file references are redundant plan metadata. Promote safe references
        # into required_case_files rather than rejecting the plan for an omission.
        required = list(data.get("required_case_files") or [])
        for binding in data.get("confirmed_fact_bindings") or []:
            if not isinstance(binding, dict):
                continue
            refs = list(binding.get("case_files") or [])
            refs += [a.get("path") for a in (binding.get("case_assertions") or []) if isinstance(a, dict)]
            rel = binding.get("numeric_relation") or {}
            if isinstance(rel, dict):
                for side in ("numerator", "denominator"):
                    refs += [t.get("path") for t in (rel.get(side) or []) if isinstance(t, dict)]
            for ref in refs:
                text = str(ref or "")
                if re.fullmatch(r"(?:0|constant|system)/[A-Za-z0-9_.\/-]+", text) and ".." not in text and text not in required:
                    required.append(text)
        data["required_case_files"] = required

        conflicts = [str(x) for x in (data.get("plan_conflicts") or []) if str(x).strip()]
        for key, ident in (("confirmed_fact_bindings", "fact_id"), ("evidence", "evidence_id"), ("engineering_defaults", "parameter")):
            raw = data.get(key) or []
            out, seen = [], {}
            for item in raw:
                if not isinstance(item, dict):
                    out.append(item); continue
                token = str(item.get(ident, "")).casefold() if ident == "parameter" else str(item.get(ident, ""))
                if token and token in seen:
                    prior = seen[token]
                    if json.dumps(prior, sort_keys=True, ensure_ascii=True, default=str) != json.dumps(item, sort_keys=True, ensure_ascii=True, default=str):
                        marker = f"conflicting-{key}:{token}"
                        if marker not in conflicts:
                            conflicts.append(marker)
                        out.append(item)
                    continue
                if token:
                    seen[token] = item
                out.append(item)
            data[key] = out
        data["plan_conflicts"] = conflicts[:40]
        execution = data.get("execution")
        if isinstance(execution, dict):
            driver = execution.get("driver")
            if driver == "foamRun":
                if execution.get("solver_module"):
                    data["solver"] = execution["solver_module"]
                if execution.get("solver_provider_id"):
                    data["solver_provider_id"] = execution["solver_provider_id"]
            elif driver == "foamMultiRun":
                data["solver"] = "foamMultiRun"
                if execution.get("driver_provider_id"):
                    data["solver_provider_id"] = execution["driver_provider_id"]
            elif driver:
                data["solver"] = driver
                if execution.get("driver_provider_id"):
                    data["solver_provider_id"] = execution["driver_provider_id"]
        return data

    @model_validator(mode="after")
    def validate_unique_audit_fields(self) -> Self:
        binding_ids = [item.fact_id for item in self.confirmed_fact_bindings]
        if set(binding_ids) != set(self.confirmed_fact_ids):
            raise ValueError("Engineering plan confirmed fact bindings must exactly cover confirmed_fact_ids.")
        for path in self.required_case_files:
            if not re.fullmatch(r"(?:0|constant|system)/[A-Za-z0-9_.\/-]+", path) or ".." in path:
                raise ValueError(f"Unsafe required case file path: {path}")
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


class EngineeringPlanPatch(_EngineeringModel):
    """Delta-only update for an existing sealed EngineeringPlan.

    Human-feedback revision and strategy revision should not make the model repeat
    immutable confirmed-fact/audit metadata just to change a solver, temporal mode,
    mesh strategy or engineering default.  Only fields that are legitimately owned
    by the Engineering Agent are patchable here; Python merges them onto the sealed
    baseline and re-validates the resulting complete EngineeringPlan.
    """

    solver: str | None = Field(default=None, pattern=r"^[A-Za-z][A-Za-z0-9_.+-]*$", max_length=160)
    solver_provider_id: str | None = Field(default=None, min_length=1, max_length=240)
    execution: OpenFOAMExecutionSpec | None = None
    region_layouts: list[RegionCaseLayout] | None = None
    interfaces: list[RegionInterface] | None = None
    completion: CompletionContract | None = None
    quantities_of_interest: list[QuantityOfInterest] | None = None
    conservation_checks: list[ConservationCheck] | None = None
    problem_interpretation: str | None = Field(default=None, min_length=1, max_length=4000)
    temporal_behavior: Literal["steady", "transient", "custom"] | None = None
    motion_kind: Literal[
        "static", "rigid_body", "prescribed_deformation", "free_body", "two_way_fsi", "custom"
    ] | None = None
    mesh_motion_requirement: Literal[
        "static", "moving", "deforming", "topology_change", "custom"
    ] | None = None
    mesh_strategy: str | None = Field(default=None, min_length=1, max_length=1000)
    decisions: list[EngineeringDecision] | None = Field(default=None, max_length=80)
    assumptions: list[str] | None = Field(default=None, max_length=80)
    engineering_defaults: list[EngineeringDefaultAssumption] | None = Field(default=None, max_length=80)
    # Fact identity remains frozen, but implementation bindings may legitimately
    # change when a mesh/case strategy changes the files that realize a confirmed
    # semantic fact. EngineeringPlan validation still requires exact fact-ID closure.
    confirmed_fact_bindings: list[ConfirmedFactBinding] | None = Field(default=None, max_length=200)
    required_case_files: list[str] | None = Field(default=None, max_length=80)
    postprocess_strategy: list[str] | None = Field(default=None, max_length=40)

    def apply(self, baseline: EngineeringPlan) -> EngineeringPlan:
        data = baseline.model_dump(mode="python")
        updates = self.model_dump(mode="python", exclude_unset=True)
        non_nullable = {
            "solver", "solver_provider_id", "region_layouts", "interfaces",
            "quantities_of_interest", "conservation_checks", "problem_interpretation",
            "temporal_behavior", "motion_kind", "mesh_motion_requirement", "mesh_strategy",
            "decisions", "assumptions", "engineering_defaults", "confirmed_fact_bindings",
            "required_case_files", "postprocess_strategy",
        }
        for key, value in updates.items():
            if key in non_nullable and value is None:
                raise ValueError(f"EngineeringPlanPatch cannot clear required/list field: {key}")
            data[key] = value
        return EngineeringPlan.model_validate(data)


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
    scope: Literal["all", "tutorials", "source", "etc", "modules"] = "all"
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
    reference_scope: Literal["all", "tutorials", "source", "etc", "modules"] = "all"
    read_top_reference_matches: int = Field(default=1, ge=0, le=2)
    target_case_files: list[str] = Field(default_factory=list, max_length=80)
    evidence_policy: Literal["mandatory", "deferred", "advisory"] = "deferred"
    risk_level: Literal["low", "medium", "high", "critical"] = "medium"
    verification_stage: Literal[
        "design", "authoring", "pre_validation", "pre_execution", "runtime", "result_review"
    ] = "authoring"

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
        if normalized.get("reference_scope") not in {"all", "tutorials", "source", "etc", "modules"}:
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
    target_case_files: list[str] = Field(default_factory=list, max_length=80)
    type: Literal["read_reference"]
    reference: str = Field(min_length=1, max_length=1000)
    start_line: int = Field(default=1, ge=1, le=1_000_000)
    line_count: int = Field(default=160, ge=1, le=400)
    rationale: str = Field(default="", max_length=200)

    @model_validator(mode="after")
    def reject_native_source_path_as_reference(self) -> Self:
        text = self.reference.strip()
        if (
            text.startswith(('/', '\\'))
            or re.match(r"^[A-Za-z]:[\\/]", text)
            or '<OPENFOAM_ROOT>' in text
            or '<LOCAL_PATH:' in text
        ):
            raise ValueError(
                "read_reference requires an indexed reference ID returned by search_references; "
                "native diagnostic source/template paths are provenance, not reference IDs."
            )
        return self


class ListCaseFilesAction(_EngineeringModel):
    type: Literal["list_case_files"]
    rationale: str = Field(default="", max_length=200)


class ReadCaseFileAction(_EngineeringModel):
    type: Literal["read_case_file"]
    path: str = Field(min_length=1, max_length=240)
    rationale: str = Field(default="", max_length=200)


class WriteCaseFileAction(_EngineeringModel):
    type: Literal["write_case_file"]
    evidence_ids: list[str] = Field(default_factory=list)
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


def _canonical_typed_value_for_duplicate_check(value: object) -> str:
    """Compare duplicate typed values without changing the authored value.

    The deterministic serializer owns the trailing semicolon, so ``x`` and ``x;``
    are equivalent authoring echoes. Internal whitespace is intentionally preserved:
    quoted/string-like OpenFOAM expressions may make it semantically meaningful.
    """

    text = str(value or "").strip()
    if text.endswith(";"):
        text = text[:-1].rstrip()
    return text


def _raw_model_mapping(value: object) -> dict[str, object] | None:
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, BaseModel):
        return value.model_dump(mode="python")
    return None


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
    # Controller-owned semantic conflict record. It is intentionally absent from the
    # LLM JSON schema: structured-output validation should normalize harmless echoes
    # rather than forcing an expensive model retry. A real conflicting duplicate is
    # carried to deterministic candidate validation/repair instead.
    entry_conflicts: SkipJsonSchema[list[str]] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def normalize_duplicate_entries(cls, value: Any):
        if not isinstance(value, dict):
            return value
        data = dict(value)
        raw_entries = data.get("entries") or []
        normalized: list[object] = []
        seen: dict[str, tuple[str, str]] = {}
        conflicts: list[str] = [str(x) for x in (data.get("entry_conflicts") or []) if str(x).strip()]
        for raw in raw_entries:
            mapping = _raw_model_mapping(raw)
            if mapping is None:
                normalized.append(raw)
                continue
            path = str(mapping.get("path") or "").strip()
            authored_value = str(mapping.get("value") or "")
            canonical_value = _canonical_typed_value_for_duplicate_check(authored_value)
            prior = seen.get(path)
            if prior is None:
                seen[path] = (canonical_value, authored_value)
                normalized.append(raw)
                continue
            prior_canonical, _ = prior
            if canonical_value == prior_canonical:
                # Harmless LLM echo: keep the first authored representation.
                continue
            if path and path not in conflicts:
                conflicts.append(path)
            # Keep the first value only so the object remains serializable for a
            # compact retained-candidate repair. Python never silently chooses the
            # later conflicting CFD value.
        data["entries"] = normalized
        data["entry_conflicts"] = conflicts[:40]
        return data

    @model_validator(mode="after")
    def validate_path_and_entries(self) -> Self:
        if not re.fullmatch(r"(?:0|constant|system|postprocessConfig)/[A-Za-z0-9_.\/-]+", self.path) or ".." in self.path:
            raise ValueError(f"Unsafe typed dictionary path: {self.path}")
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
    evidence_ids: list[str] = Field(default_factory=list)
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
        "authoring_strategy_infeasible",
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


class EngineeringDesign(_EngineeringModel):
    """LLM-facing CFD design, deliberately excluding controller-owned audit state.

    The Agent owns physical/numerical/geometry/execution choices. Python owns only
    immutable identity and provenance mechanics: intake digest/fact closure, audit
    bindings, canonical evidence IDs, Foundation-version sealing, and redundant solver
    mirrors. Keeping those concerns out of the LLM response removes copy/retry failure
    modes without moving CFD engineering decisions into Python.
    """

    case_name: str = Field(pattern=r"^[A-Za-z][A-Za-z0-9_.-]*$", max_length=80)
    # Compatibility fallback for legacy/direct plans. New staged designs should
    # prefer execution; when execution is present the controller derives these mirrors.
    solver: str | None = Field(default=None, pattern=r"^[A-Za-z][A-Za-z0-9_.+-]*$", max_length=160)
    solver_provider_id: str | None = Field(default=None, min_length=1, max_length=240)
    execution: OpenFOAMExecutionSpec | None = None
    region_layouts: list[RegionCaseLayout] = Field(default_factory=list)
    interfaces: list[RegionInterface] = Field(default_factory=list)
    completion: CompletionContract | None = None
    quantities_of_interest: list[QuantityOfInterest] = Field(default_factory=list)
    conservation_checks: list[ConservationCheck] = Field(default_factory=list)
    problem_interpretation: str = Field(min_length=1, max_length=4000)
    temporal_behavior: Literal["steady", "transient", "custom"]
    motion_kind: Literal[
        "static", "rigid_body", "prescribed_deformation", "free_body", "two_way_fsi", "custom"
    ]
    mesh_motion_requirement: Literal[
        "static", "moving", "deforming", "topology_change", "custom"
    ]
    mesh_strategy: str = Field(min_length=1, max_length=1000)
    decisions: list[EngineeringDecision] = Field(default_factory=list, max_length=80)
    assumptions: list[str] = Field(default_factory=list, max_length=80)
    engineering_defaults: list[EngineeringDesignDefault] = Field(default_factory=list, max_length=80)
    required_case_files: list[str] = Field(default_factory=list, max_length=80)
    postprocess_strategy: list[str] = Field(default_factory=list, max_length=40)

    @model_validator(mode="before")
    @classmethod
    def normalize_design_lists(cls, value: Any):
        if not isinstance(value, dict):
            return value
        data = dict(value)
        raw = data.get("required_case_files") or []
        data["required_case_files"] = list(
            dict.fromkeys(str(item) for item in raw if str(item).strip())
        )
        return data

    @model_validator(mode="after")
    def validate_design_contract(self) -> Self:
        if self.execution is None and (not self.solver or not self.solver_provider_id):
            raise ValueError(
                "EngineeringDesign requires execution or the legacy solver/solver_provider_id pair."
            )
        for path in self.required_case_files:
            if not re.fullmatch(r"(?:0|constant|system)/[A-Za-z0-9_.\/-]+", path) or ".." in path:
                raise ValueError(f"Unsafe required case file path: {path}")
        return self


class DesignCaseAction(_EngineeringModel):
    """Stage-1 Agent-owned CFD design without controller/audit or file-authoring payload."""

    type: Literal["design_case"]
    plan: EngineeringDesign
    authoring_brief: str = Field(default="", max_length=1200)

    @model_validator(mode="before")
    @classmethod
    def strip_controller_owned_plan_fields(cls, value: Any):
        """Project persisted/full plans into the smaller staged-design wire view."""
        if not isinstance(value, dict):
            return value
        data = dict(value)
        raw = data.get("plan")
        if hasattr(raw, "model_dump"):
            raw = raw.model_dump(mode="python")
        if isinstance(raw, dict):
            allowed = set(EngineeringDesign.model_fields)
            data["plan"] = {key: item for key, item in raw.items() if key in allowed}
        return data


class CaseAuthoringAction(_EngineeringModel):
    """Stage-2 case bundle authored against a Python-held EngineeringPlan."""

    type: Literal["author_case"]
    task_id: str | None = None
    defer_native: bool = False
    goal: str = Field(min_length=1, max_length=1000)
    files: list[CaseBundleFile] = Field(default_factory=list, max_length=80)
    typed_dictionaries: list[TypedFoamDictionaryFile] = Field(default_factory=list, max_length=80)
    block_mesh: TypedBlockMeshFile | None = None
    validate_dictionaries: list[str] = Field(default_factory=list, max_length=40)
    surface_checks: list[str] = Field(default_factory=list, max_length=16)
    mesh_commands: list[str] = Field(default_factory=list, max_length=12)
    native_pipeline: list[NativeOpenFOAMCommand] = Field(default_factory=list, max_length=20)
    # Compatibility mirror only. The authoritative manifest lives on the frozen
    # EngineeringPlan and is injected by the controller when the authoring response
    # is assembled. Keeping this optional prevents a harmless model echo mismatch
    # from rejecting an otherwise valid case bundle.
    required_case_files: list[str] = Field(default_factory=list, max_length=80)
    rationale: str = Field(default="", max_length=200)
    authoring_conflicts: SkipJsonSchema[list[str]] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def normalize_authoring_mirrors(cls, value: Any):
        if not isinstance(value, dict):
            return value
        data = dict(value)
        authoring_conflicts = [str(x) for x in (data.get("authoring_conflicts") or []) if str(x).strip()]

        # Normalize repeated raw files. Exact repeats are harmless; conflicting
        # content is retained as a controller-visible semantic conflict rather than
        # rejected at the structured-output/Pydantic boundary.
        raw_files: list[object] = []
        raw_by_path: dict[str, str] = {}
        for raw in data.get("files") or []:
            mapping = _raw_model_mapping(raw)
            if mapping is None:
                raw_files.append(raw)
                continue
            path = str(mapping.get("path") or "").strip()
            content = str(mapping.get("content") or "")
            if path not in raw_by_path:
                raw_by_path[path] = content
                raw_files.append(raw)
            elif raw_by_path[path] != content and path and f"raw-file:{path}" not in authoring_conflicts:
                authoring_conflicts.append(f"raw-file:{path}")
        data["files"] = raw_files

        # Merge repeated typed dictionary *files* by path. Their leaf entries are
        # normalized by TypedFoamDictionaryFile itself, including conflict capture.
        typed_by_path: dict[str, dict[str, object]] = {}
        typed_order: list[str] = []
        passthrough_typed: list[object] = []
        for raw in data.get("typed_dictionaries") or []:
            mapping = _raw_model_mapping(raw)
            if mapping is None:
                passthrough_typed.append(raw)
                continue
            path = str(mapping.get("path") or "").strip()
            if not path or path not in typed_by_path:
                typed_by_path[path] = mapping
                typed_order.append(path)
                continue
            prior = typed_by_path[path]
            prior_class = prior.get("foam_class")
            next_class = mapping.get("foam_class")
            if prior_class and next_class and prior_class != next_class:
                token = f"typed-class:{path}"
                if token not in authoring_conflicts:
                    authoring_conflicts.append(token)
            elif not prior_class and next_class:
                prior["foam_class"] = next_class
            prior["entries"] = list(prior.get("entries") or []) + list(mapping.get("entries") or [])
            prior["entry_conflicts"] = list(prior.get("entry_conflicts") or []) + list(mapping.get("entry_conflicts") or [])
        data["typed_dictionaries"] = [typed_by_path[path] for path in typed_order] + passthrough_typed

        # Raw and typed representations may not both own one case path. Preserve the
        # first representations for compact repair, but mark the ambiguity explicitly.
        typed_paths = {str((_raw_model_mapping(item) or {}).get("path") or "").strip() for item in data["typed_dictionaries"]}
        mixed_paths = set(raw_by_path).intersection(typed_paths)
        for path in mixed_paths:
            token = f"mixed-representation:{path}"
            if path and token not in authoring_conflicts:
                authoring_conflicts.append(token)
        if mixed_paths:
            # Keep one representation so nested Pydantic validation can succeed; the
            # controller-visible conflict marker forces retained-candidate repair before write.
            data["typed_dictionaries"] = [
                item for item in data["typed_dictionaries"]
                if str((_raw_model_mapping(item) or {}).get("path") or "").strip() not in mixed_paths
            ]
        data["authoring_conflicts"] = authoring_conflicts[:40]

        # These lists are execution hints/compatibility mirrors, not independent
        # sources of truth. Deduplicate harmless repeats instead of forcing the LLM
        # through another structured-output retry.
        for key in ("validate_dictionaries", "surface_checks", "mesh_commands", "required_case_files"):
            raw = data.get(key) or []
            data[key] = list(dict.fromkeys(str(x) for x in raw if str(x).strip()))
        # v4.5: foamDictionary is an advisory probe, not part of the mandatory native
        # authoring pipeline. Static header/semantic checks plus the actual OpenFOAM
        # mesh/solver consumers are stronger and avoid N redundant subprocesses.
        data["mesh_commands"] = [x for x in data.get("mesh_commands", []) if x != "foamDictionary"]
        raw_pipeline = data.get("native_pipeline") or []
        data["native_pipeline"] = [
            item for item in raw_pipeline
            if (item.get("command") if isinstance(item, dict) else getattr(item, "command", None)) != "foamDictionary"
        ]
        # Intermediate partition tasks are controller-owned and may never run native
        # commands. If the model redundantly emits them, dropping them is safer and
        # more useful than rejecting the entire task.
        if data.get("defer_native"):
            data["native_pipeline"] = []
            data["mesh_commands"] = []
        # Prefer the richer native_pipeline representation if both legacy and current
        # forms are emitted. They describe the same controller stage.
        elif data.get("native_pipeline") and data.get("mesh_commands"):
            data["mesh_commands"] = []

        # Exact duplicate native invocations are harmless model repetition. Collapse
        # them before semantic validation, and keep mesh validation at the end so a
        # later mesh-mutating utility cannot accidentally run after checkMesh.
        pipeline = data.get("native_pipeline") or []
        if pipeline:
            unique, seen = [], set()
            for item in pipeline:
                if isinstance(item, dict):
                    token = (
                        str(item.get("command", "")),
                        tuple(str(x) for x in (item.get("arguments") or [])),
                        str(item.get("role", "utility")),
                    )
                else:
                    token = repr(item)
                if token in seen:
                    continue
                seen.add(token); unique.append(item)
            checks = [item for item in unique if (item.get("command") if isinstance(item, dict) else getattr(item, "command", None)) == "checkMesh"]
            non_checks = [item for item in unique if (item.get("command") if isinstance(item, dict) else getattr(item, "command", None)) != "checkMesh"]
            data["native_pipeline"] = non_checks + checks
        elif data.get("mesh_commands") and "checkMesh" in data["mesh_commands"]:
            data["mesh_commands"] = [x for x in data["mesh_commands"] if x != "checkMesh"] + ["checkMesh"]
        return data

    @model_validator(mode="after")
    def validate_case_authoring(self) -> Self:
        if not self.files and not self.typed_dictionaries and self.block_mesh is None:
            raise ValueError("author_case requires at least one raw, typed, or blockMesh case file.")
        paths = [item.path for item in self.files] + [item.path for item in self.typed_dictionaries]
        if self.block_mesh is not None:
            paths.append(self.block_mesh.path)
        if len(paths) != len(set(paths)):
            raise ValueError("author_case contains duplicate file paths.")

        for collection_name, paths_to_check in (
            ("validate_dictionaries", self.validate_dictionaries),
            ("surface_checks", self.surface_checks),
            ("required_case_files", self.required_case_files),
        ):
            if len(paths_to_check) != len(set(paths_to_check)):
                raise ValueError(f"author_case contains duplicate {collection_name} paths.")
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
        if self.defer_native:
            if self.type == "execute_case_plan" or not self.task_id:
                raise ValueError("Deferred native work requires a controller-issued authoring task.")
            if self.native_pipeline or self.mesh_commands:
                raise ValueError("Intermediate authoring tasks cannot request native execution.")
            return self

        # v4.6: native/validation lists are strategy hints, not execution authority.
        # The controller compiles the final CaseBuildGraph from the frozen required
        # manifest plus the artifacts actually authored. Therefore an otherwise valid
        # authoring response does not need to echo checkMesh or any validation list.
        # If hints are supplied, keep only structural safety checks here; executable
        # ordering, prerequisites and final checkMesh ownership are deterministic.
        if self.native_pipeline:
            regions = []
            for command in self.native_pipeline:
                if command.command != "checkMesh":
                    continue
                args = command.arguments
                region = args[args.index("-region")+1] if "-region" in args and args.index("-region")+1 < len(args) else ""
                if region in regions:
                    raise ValueError("author_case contains duplicate checkMesh validation for a region.")
                regions.append(region)
        return self


class ExecuteCasePlanAction(CaseAuthoringAction):
    """Controller-bound executable case bundle.

    ``required_case_files`` is intentionally not an independent contract. The frozen
    EngineeringPlan is the single source of truth; model-authored mirrors are
    normalized away before validation. Actual authored-file coverage and pre-solve
    completeness are checked deterministically later.
    """

    type: Literal["execute_case_plan"]
    plan: EngineeringPlan

    @model_validator(mode="before")
    @classmethod
    def bind_required_manifest_to_plan(cls, value: Any):
        if not isinstance(value, dict):
            return value
        data = dict(value)
        plan = data.get("plan")
        if isinstance(plan, EngineeringPlan):
            canonical = list(plan.required_case_files)
        elif isinstance(plan, dict):
            canonical = list(plan.get("required_case_files") or [])
        else:
            canonical = None
        if canonical is not None:
            data["required_case_files"] = canonical
        return data

    @model_validator(mode="after")
    def validate_execution_plan(self) -> Self:
        # Defensive assertion only; the before-validator above establishes the
        # controller-owned canonical manifest.
        if self.required_case_files != self.plan.required_case_files:
            raise ValueError("Controller-owned required case manifest could not be normalized.")
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




def _normalize_delta_authoring_payload(value: Any) -> Any:
    """Normalize harmless repeated repair metadata before Pydantic semantics.

    Exact duplicates are collapsed. Conflicting representations are intentionally
    retained so the controller-owned CaseDeltaGraph, not Structured Output retry,
    can report the semantic conflict without discarding the candidate.
    """
    if not isinstance(value, dict):
        return value
    data = dict(value)
    for key in ("validate_dictionaries", "surface_checks", "mesh_commands", "drop_paths"):
        data[key] = list(dict.fromkeys(str(x) for x in (data.get(key) or []) if str(x).strip()))
    for key in ("replacement_files", "typed_dictionaries", "file_patches", "patches"):
        out, seen = [], set()
        for item in data.get(key) or []:
            mapping = _raw_model_mapping(item)
            token = json.dumps(mapping, sort_keys=True, ensure_ascii=True, default=str) if mapping is not None else repr(item)
            if token in seen:
                continue
            seen.add(token); out.append(item)
        data[key] = out
    pipeline, seen = [], set()
    for item in data.get("native_pipeline") or []:
        mapping = _raw_model_mapping(item)
        token = json.dumps(mapping, sort_keys=True, ensure_ascii=True, default=str) if mapping is not None else repr(item)
        if token in seen:
            continue
        seen.add(token); pipeline.append(item)
    data["native_pipeline"] = [
        item for item in pipeline
        if ((_raw_model_mapping(item) or {}).get("command") if _raw_model_mapping(item) is not None else getattr(item, "command", None)) != "foamDictionary"
    ]
    data["mesh_commands"] = [x for x in data.get("mesh_commands", []) if x != "foamDictionary"]
    if data.get("native_pipeline") and data.get("mesh_commands"):
        data["mesh_commands"] = []
    return data


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
    plan_patch: EngineeringPlanPatch | None = None
    updated_plan: EngineeringPlan | None = None

    @model_validator(mode="before")
    @classmethod
    def normalize_repair_native_probes(cls, value: Any):
        if not isinstance(value, dict):
            return value
        return _normalize_delta_authoring_payload(value)

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
        if self.plan_patch is not None and self.updated_plan is not None:
            raise ValueError("repair_case_plan must use plan_patch or updated_plan, not both.")
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

    @model_validator(mode="before")
    @classmethod
    def normalize_runtime_native_probes(cls, value: Any):
        if not isinstance(value, dict):
            return value
        return _normalize_delta_authoring_payload(value)

    @model_validator(mode="after")
    def validate_runtime_repair(self) -> Self:
        # Empty deltas are handled by the runtime executor as a controlled non-retry.
        # Keep semantic/safety conflicts below as hard schema constraints.
        paths = (
            [item.path for item in self.file_patches]
            + [item.path for item in self.replacement_files]
            + [item.path for item in self.typed_dictionaries]
        )
        if self.native_pipeline and self.mesh_commands:
            raise ValueError("repair_runtime_case must use native_pipeline or mesh_commands, not both.")
        commands = [item.command for item in self.native_pipeline] if self.native_pipeline else list(self.mesh_commands)
        if commands.count("checkMesh") > 1:
            raise ValueError("repair_runtime_case may run checkMesh at most once.")
        return self



class CandidateCasePlanRepairAction(_EngineeringModel):
    """Delta repair for an in-memory, not-yet-committed execute_case_plan candidate."""

    @model_validator(mode="before")
    @classmethod
    def normalize_candidate_delta(cls, value: Any):
        return _normalize_delta_authoring_payload(value)

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

    @model_validator(mode="before")
    @classmethod
    def normalize_strategy_delta(cls, value: Any):
        return _normalize_delta_authoring_payload(value)

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
    plan_patch: EngineeringPlanPatch | None = None
    updated_plan: EngineeringPlan | None = None

    @model_validator(mode="after")
    def validate_strategy_revision(self) -> Self:
        paths = [item.path for item in self.replacement_files] + [item.path for item in self.typed_dictionaries] + list(self.drop_paths)
        if self.block_mesh is not None:
            paths.append(self.block_mesh.path)
        for path in self.drop_paths:
            if not re.fullmatch(r"(?:0|constant|system)/[A-Za-z0-9_.\/-]+", path) or ".." in path:
                raise ValueError(f"Unsafe strategy drop path: {path}")
        commands = [item.command for item in self.native_pipeline] if self.native_pipeline else list(self.mesh_commands)
        for command in commands:
            if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_.+-]*", command):
                raise ValueError(f"Unsafe strategy command identifier: {command}")
        if any(item.path == "system/blockMeshDict" for item in self.typed_dictionaries):
            raise ValueError("Use block_mesh for system/blockMeshDict in strategy revisions.")
        if self.plan_patch is not None and self.updated_plan is not None:
            raise ValueError("revise_mesh_strategy must use plan_patch or updated_plan, not both.")
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
    try:
        normalized["action"] = model.model_validate(raw_action)
    except ValidationError as exc:
        # Progress-first structured-output recovery: planning-time result-analysis
        # details are optional and often cannot be concrete before runtime. If *all*
        # validation failures are confined to those sections, discard only those
        # malformed optional sections and preserve the CFD design. Never apply this
        # fallback to confirmed facts, execution/provider fields, paths, or authoring.
        if action_type != "design_case" or not isinstance(raw_action.get("plan"), dict):
            raise
        optional = {"quantities_of_interest", "conservation_checks", "completion", "implementation_evidence_bindings"}
        roots = set()
        for error in exc.errors():
            loc = tuple(error.get("loc") or ())
            # model_validate(raw_action) reports plan.<field>...
            if len(loc) >= 2 and loc[0] == "plan":
                roots.add(str(loc[1]))
            elif loc:
                roots.add(str(loc[0]))
        if not roots or not roots.issubset(optional):
            raise
        retry_action = dict(raw_action)
        retry_plan = dict(retry_action["plan"])
        for field in roots:
            retry_plan[field] = None if field == "completion" else []
        retry_action["plan"] = retry_plan
        normalized["action"] = model.model_validate(retry_action)
    return normalized


# Phase-specific compact contracts. Agent identity remains one CFDEngineeringAgent; only
# permissions/schema vary by phase so repeated calls do not carry the giant all-phase union.
# v3.6 staged preparation keeps the large file-authoring DSL out of the design/evidence turn.
PrepareDesignAction = GatherEvidenceAction | ReadCaseFileAction | DesignCaseAction | BlockAction
class PrepareDesignTurn(_EngineeringModel):
    action: PrepareDesignAction

    @model_validator(mode="before")
    @classmethod
    def route_action(cls, value: Any):
        return _route_action_payload(
            value,
            {
                "gather_evidence": GatherEvidenceAction,
                "read_case_file": ReadCaseFileAction,
                "design_case": DesignCaseAction,
                "block": BlockAction,
            },
        )


PrepareDecisionDesignAction = DesignCaseAction | BlockAction
class PrepareDecisionDesignTurn(_EngineeringModel):
    action: PrepareDecisionDesignAction

    @model_validator(mode="before")
    @classmethod
    def route_action(cls, value: Any):
        return _route_action_payload(
            value,
            {"design_case": DesignCaseAction, "block": BlockAction},
        )


CaseAuthoringTurnAction = CaseAuthoringAction | BlockAction
class CaseAuthoringTurn(_EngineeringModel):
    action: CaseAuthoringTurnAction

    @model_validator(mode="before")
    @classmethod
    def route_action(cls, value: Any):
        return _route_action_payload(
            value,
            {"author_case": CaseAuthoringAction, "block": BlockAction},
        )


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

class RevisionDecisionAction(_EngineeringModel):
    """Plan-only decision made before any human-revision case mutation.

    The model decides which Agent-owned engineering fields and case artifacts need
    revision. Python applies the optional plan patch to the sealed baseline, while
    the following revision-authoring turn owns the actual file delta.
    """

    type: Literal["decide_revision"]
    diagnosis: str = Field(min_length=1, max_length=3000)
    plan_patch: EngineeringPlanPatch | None = None
    target_case_files: list[str] = Field(default_factory=list, max_length=40)
    rationale: str = Field(default="", max_length=1500)

    @model_validator(mode="after")
    def validate_targets(self) -> Self:
        normalized: list[str] = []
        for path in self.target_case_files:
            text = str(path).strip()
            if not re.fullmatch(r"(?:0|constant|system)/[A-Za-z0-9_.\/-]+", text) or ".." in text:
                raise ValueError(f"Unsafe revision target case path: {text}")
            if text not in normalized:
                normalized.append(text)
        self.target_case_files = normalized
        return self


RevisionDecisionTurnAction = RevisionDecisionAction | BlockAction
class RevisionDecisionTurn(_EngineeringModel):
    action: RevisionDecisionTurnAction

    @model_validator(mode="before")
    @classmethod
    def route_action(cls, value: Any):
        return _route_action_payload(
            value,
            {"decide_revision": RevisionDecisionAction, "block": BlockAction},
        )


RevisionAuthoringAction = SearchReferencesAction | ReadReferenceAction | ReadCaseFileAction | RepairCasePlanAction | BlockAction
class RevisionAuthoringTurn(_EngineeringModel):
    action: RevisionAuthoringAction

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
        normalized = dict(value)
        observed = normalized.get("observed_evidence")
        if isinstance(observed, (list, tuple)) and len(observed) > ENGINEERING_EVENT_OBSERVED_EVIDENCE_LIMIT:
            normalized["observed_evidence"] = list(observed)[:ENGINEERING_EVENT_OBSERVED_EVIDENCE_LIMIT]
        if "validation_status" not in normalized:
            normalized["validation_status"] = "pass" if bool(normalized.get("success")) else "fail"
        return normalized

    step: int = Field(ge=1)
    action_type: str = Field(min_length=1, max_length=80)
    success: bool
    validation_status: Literal["pass", "fail", "inconclusive"] = "pass"
    failure_category: Literal["case", "tool", "infra", "security", "user_contract"] | None = None
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
        if self.validation_status == "fail" and self.success:
            raise ValueError("A failed validation cannot be marked workflow-successful.")
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
    origin: Literal["agent", "native", "user_asset"] = "agent"


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
