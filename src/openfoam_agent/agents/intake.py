from __future__ import annotations

import hashlib
import json
import math
import re

from openfoam_agent.llm.prompts import INTAKE_SYSTEM_PROMPT
from openfoam_agent.llm.protocol import StructuredLLM
from openfoam_agent.schemas.intake import CFDIntakeSpec, RequirementHistory, UserEvidenceLocator
from openfoam_agent.contracts.requirements import active_requirement_texts
from openfoam_agent.contracts.quantities import si_value, UNITS
from openfoam_agent.schemas.request import UserRequest
from openfoam_agent.workflow.state import CFDState
from openfoam_agent.workflow.states import State


def build_intake_prompt(request: UserRequest) -> str:
    payload = {
        "user_evidence": {
            "conversation_turns": [request.prompt, *request.conversation_turns],
            "geometry_file_names": [_file_name(path) for path in request.geometry_files],
            "additional_file_names": [_file_name(path) for path in request.additional_files],
        },
        "policy": {
            "interaction_mode": request.interaction_mode,
            "exploratory_completion_authorized": request.exploratory_completion_authorized,
        },
    }
    return (
        "Create a solver-independent CFD intake definition from this delimited JSON. "
        "Only values under user_evidence may support source=user facts. The policy "
        "object is workflow authorization and must not become a CFD fact. Preserve active requirements "
        "chronologically, not superseded/cancelled historical values. Emit typed quantities with SI dimensions "
        "and region/patch/material targets when known. List materially different interpretations as ambiguities; "
        "easy mode does not authorize choosing a different physical problem:\n"
        + json.dumps(payload, ensure_ascii=False, indent=2)
    )


def _file_name(path: str) -> str:
    return path.replace("\\", "/").rsplit("/", maxsplit=1)[-1]


def _user_evidence_sources(request: UserRequest) -> list[tuple[str, int, str]]:
    sources: list[tuple[str, int, str]] = []
    for index, value in enumerate([request.prompt, *request.conversation_turns]):
        sources.append(("conversation_turn", index, value))
    for index, value in enumerate(request.geometry_files):
        sources.append(("geometry_file_name", index, _file_name(value)))
    for index, value in enumerate(request.additional_files):
        sources.append(("additional_file_name", index, _file_name(value)))
    return sources


def _issue_user_evidence_locator(
    evidence: str, sources: list[tuple[str, int, str]]
) -> UserEvidenceLocator:
    candidates: list[UserEvidenceLocator] = []
    for source_kind, source_index, source_text in sources:
        start = 0
        while True:
            position = source_text.find(evidence, start)
            if position < 0:
                break
            candidates.append(
                UserEvidenceLocator(
                    source_kind=source_kind,
                    source_index=source_index,
                    source_sha256=hashlib.sha256(source_text.encode("utf-8")).hexdigest(),
                    start_char=position,
                    end_char=position + len(evidence),
                )
            )
            start = position + 1
    if not candidates:
        raise ValueError("source=user evidence is not an exact contiguous user-provided span.")
    if len(candidates) != 1:
        raise ValueError(
            "source=user evidence is ambiguous across user inputs; use a longer exact span so provenance resolves uniquely."
        )
    return candidates[0]


def validate_intake_provenance(spec: CFDIntakeSpec, request: UserRequest) -> None:
    _normalize_review_critical_source_attribution(spec)
    user_sources = _user_evidence_sources(request)
    user_texts = [item[2] for item in user_sources]
    for fact in spec.facts:
        if fact.source == "user":
            assert fact.evidence is not None
            locator = _issue_user_evidence_locator(fact.evidence, user_sources)
            if fact.evidence_locator is not None and fact.evidence_locator != locator:
                raise ValueError(
                    f"User fact '{fact.id}' supplied an evidence locator that does not match the immutable user input."
                )
            fact.evidence_locator = locator
    active_texts, history, assignments = active_requirement_texts([request.prompt, *request.conversation_turns])
    spec.requirement_history = [RequirementHistory.model_validate(row) for row in history]
    for record in history:
        if record["status"] == "cancelled" and spec.fact(record["target"]) is not None:
            raise ValueError(f"Cancelled requirement is still active: {record['target']}")
    for key, assignment in assignments.items():
        fact = spec.fact(key)
        if fact is None and key == "operating.reynolds_number":
            candidates = [f for f in spec.facts if f.id.endswith(".reynolds_number")]
            if len(candidates) == 1:
                fact = candidates[0]
        if fact is None or fact.source != "user":
            raise ValueError(f"Active explicit requirement has no matching user fact: {key}")
        if fact is not None and fact.source == "user":
            values = _finite_numbers(fact.value)
            if values and not any(math.isclose(v, float(assignment["value"]), rel_tol=1e-12, abs_tol=1e-12) for _,v in values):
                # Unit normalization is handled by an explicit typed quantity, not
                # by letting arbitrary historical numbers satisfy preservation.
                if fact.quantity is None or not _verified_unit_source(fact, assignment["evidence"]):
                    raise ValueError(f"Active fact {key} does not match its latest explicit assignment.")
    for fact in spec.facts:
        if fact.quantity is not None:
            si_value(fact.quantity)
    for ambiguity in spec.ambiguities:
        if ambiguity.selected is not None and not any(ambiguity.user_evidence in text for text in user_texts):
            raise ValueError("Ambiguity selection evidence was not supplied by the user.")
    # Asset names are identifiers, never physical numeric constraints.
    supplied_numbers = _finite_numbers("\n".join(active_texts))
    represented_numbers = _finite_numbers(
        "\n".join(
            " ".join(part for part in (fact.value, fact.unit or "") if part)
            for fact in spec.facts
            if fact.category != "context" and fact.source == "user"
        )
    )
    for fact in spec.facts:
        if fact.source == "user" and fact.quantity is not None:
            converted = _verified_unit_source(fact, fact.evidence or "")
            if converted:
                represented_numbers.extend(converted)
    missing_numbers = [
        token
        for token, value in supplied_numbers
        if not any(
            math.isclose(value, candidate, rel_tol=1e-12, abs_tol=1e-12)
            for _, candidate in represented_numbers
        )
    ]
    if missing_numbers:
        raise ValueError(
            "User-supplied numeric values are missing from normalized CFD facts: "
            + ", ".join(dict.fromkeys(missing_numbers))
        )

    if spec.status == "ready_for_review":
        ids = {fact.id for fact in spec.facts}
        required = {"request.summary", "classification.problem_type"}
        missing = required - ids
        if missing:
            raise ValueError(f"Review-ready intake is missing facts: {sorted(missing)}")
        if not any(fact.category == "objective" for fact in spec.facts):
            raise ValueError("Review-ready intake requires an objective fact.")
        classification = spec.fact("classification.problem_type")
        assert classification is not None
        if classification.value not in {
            "internal_flow",
            "external_flow",
            "heat_transfer",
            "multiphase",
            "species_transport",
            "custom",
        }:
            raise ValueError("classification.problem_type has an unsupported value.")



def _verified_unit_source(fact, evidence):
    """Only credit a source number after an explicit, dimension-checked conversion.

    This is a scalar literal conversion, not inference of region/patch semantics.
    Multiple quantities in one evidence span require separate facts.
    """
    quantity = fact.quantity
    if quantity is None:
        return []
    units = "|".join(re.escape(unit) for unit in sorted(UNITS, key=len, reverse=True))
    pattern = r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)\s*(" + units + r")(?![A-Za-z0-9/])"
    pairs = re.findall(pattern, evidence)
    if len(pairs) != 1:
        return []
    value, unit = pairs[0]
    scale, offset, dimensions = UNITS[unit]
    if dimensions != quantity.dimensions:
        return []
    if not math.isclose(float(value)*scale+offset, si_value(quantity), rel_tol=1e-10, abs_tol=1e-12):
        return []
    if fact.unit is not None and fact.unit != quantity.unit:
        raise ValueError("Normalized fact unit disagrees with its typed quantity.")
    if not any(math.isclose(v,quantity.value,rel_tol=1e-10,abs_tol=1e-12) for _,v in _finite_numbers(fact.value)):
        raise ValueError("Normalized fact value disagrees with its typed quantity.")
    return [(value,float(value))]

_CLASSIFICATION_DIRECT_CUES: dict[str, tuple[str, ...]] = {
    "internal_flow": (
        "internal_flow",
        "internal flow",
        "internal-flow",
        "내부유동",
        "내부 유동",
        "channel flow",
        "pipe flow",
        "duct flow",
    ),
    "external_flow": (
        "external_flow",
        "external flow",
        "external-flow",
        "외부유동",
        "외부 유동",
        "freestream",
        "free stream",
    ),
    "heat_transfer": ("heat_transfer", "heat transfer", "열전달", "열 전달"),
    "multiphase": ("multiphase", "multi-phase", "다상", "다상유동", "다상 유동"),
    "species_transport": (
        "species_transport",
        "species transport",
        "species",
        "종수송",
        "종 수송",
    ),
    "custom": ("custom", "사용자 정의"),
}

_TEMPORAL_DIRECT_CUES = (
    "steady",
    "transient",
    "unsteady",
    "time-dependent",
    "time dependent",
    "정상상태",
    "정상 상태",
    "비정상",
    "시간의존",
    "시간 의존",
)


def _normalize_review_critical_source_attribution(spec: CFDIntakeSpec) -> None:
    """Demote unsupported ``source=user`` claims for high-impact interpretations.

    This is provenance enforcement, not CFD decision-making.  In particular a user
    saying that a cylinder is *inside a rectangular computational domain* is not
    deterministic evidence that the physical problem is an internal/channel flow.
    Likewise, requesting vortex shedding does not mean the user literally supplied
    the word "transient".  Such interpretations may still be correct, but they must
    be presented to the human as ``derived`` before /confirm freezes them.
    """

    to_demote: list[IntakeFact] = []
    for fact in spec.facts:
        if fact.source != "user" or not fact.evidence:
            continue
        evidence = fact.evidence.casefold()
        supported = True
        if fact.id == "classification.problem_type" or fact.category == "classification":
            cues = _CLASSIFICATION_DIRECT_CUES.get(fact.value.casefold(), ())
            supported = bool(cues) and any(cue.casefold() in evidence for cue in cues)
        elif fact.id == "temporal.behavior" or fact.category == "temporal":
            supported = any(cue.casefold() in evidence for cue in _TEMPORAL_DIRECT_CUES)
        if not supported:
            to_demote.append(fact)

    for fact in to_demote:
        fact.source = "derived"
        fact.evidence = None
        fact.evidence_locator = None
        fact.reason = (
            "The value is a routing/physics interpretation inferred from the request; "
            "the user did not explicitly state this normalized classification."
        )
        fact.depends_on = ["request.summary"] if spec.fact("request.summary") is not None and fact.id != "request.summary" else []


def _finite_numbers(text: str) -> list[tuple[str, float]]:
    matches = re.findall(
        r"(?<![A-Za-z0-9_.])[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
        r"(?![A-Za-z0-9_.]|차)",
        text,
    )
    result: list[tuple[str, float]] = []
    for token in matches:
        value = float(token)
        if math.isfinite(value):
            result.append((token, value))
    return result


def confirmed_intake_definition(state: CFDState) -> dict[str, object]:
    state.assert_confirmed_intake()
    assert state.intake is not None
    facts = []
    for fact in state.intake.facts:
        if fact.category == "context":
            continue
        facts.append(fact.model_dump(mode="json", exclude={"evidence", "evidence_locator"}))
    # The semantic projection omits conversational context, but derived facts
    # must not acquire dangling provenance dependencies. Carry only referenced
    # frozen context facts separately; raw conversation turns stay excluded.
    by_id = {fact.id: fact for fact in state.intake.facts}
    selected = {fact["id"] for fact in facts}
    todo = [dependency for fact in facts for dependency in fact.get("depends_on", [])]
    dependencies = set()
    while todo:
        dependency = todo.pop()
        if dependency in selected or dependency in dependencies:
            continue
        if dependency not in by_id:
            raise ValueError("Frozen intake has an unknown provenance dependency: " + dependency)
        dependencies.add(dependency)
        todo.extend(by_id[dependency].depends_on)
    return {
        "semantic_contract_version": state.intake.semantic_contract_version,
        "title": state.intake.title,
        "facts": facts,
        "provenance_dependencies": [fact.model_dump(mode="json", exclude={"evidence", "evidence_locator"})
            for fact in state.intake.facts if fact.id in dependencies],
        "status": state.intake.status,
    }


def _uses_extended_intake_provenance_repair(llm: StructuredLLM) -> bool:
    return hasattr(llm, "intake_validation_repair_attempts")


def _legacy_intake_validation_repair_prompt(
    *, base_prompt: str, error: ValueError
) -> str:
    """Preserve the pre-v2.7.3 repair prompt for adapters without the local hint."""

    return (
        base_prompt
        + "\n\nThe previous draft failed deterministic intake validation. "
        "Regenerate the full CFDIntakeSpec once. Preserve user numeric values "
        "and exact evidence. Validation error: "
        + json.dumps(str(error), ensure_ascii=False)
    )


def _intake_validation_repair_budget(llm: StructuredLLM) -> int:
    """Return semantic intake-repair attempts advertised by an LLM adapter.

    Cloud/OpenAI adapters intentionally keep the historical single deterministic
    intake retry because they do not expose this capability hint. Local adapters
    may opt into a slightly larger budget without making IntakeAgent provider-aware.
    """

    value = getattr(llm, "intake_validation_repair_attempts", 1)
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        return 1
    return min(value, 3)


def _build_intake_validation_repair_prompt(
    *,
    base_prompt: str,
    request: UserRequest,
    previous: CFDIntakeSpec,
    error: ValueError,
    attempt: int,
    max_attempts: int,
) -> str:
    """Build a provenance-aware deterministic repair prompt.

    The important distinction for small/local models is that a normalized or
    multi-turn synthesis is *not* verbatim user evidence. We therefore show the
    exact user turns again and make the source=user/derived boundary explicit.
    """

    exact_turns = [request.prompt, *request.conversation_turns]
    repair_context = {
        "repair_attempt": attempt,
        "repair_attempts_allowed": max_attempts,
        "validation_error": str(error),
        "exact_user_turns": [
            {"turn": index, "text": text} for index, text in enumerate(exact_turns, 1)
        ],
        "exact_geometry_file_names": [
            _file_name(path) for path in request.geometry_files
        ],
        "exact_additional_file_names": [
            _file_name(path) for path in request.additional_files
        ],
        "previous_invalid_spec": previous.model_dump(mode="json"),
    }
    rules = (
        "PROVENANCE REPAIR RULES:\n"
        "1. For source=user, evidence MUST be one contiguous verbatim substring "
        "copied from exactly one exact_user_turn or supplied file name. Do not "
        "translate, normalize, paraphrase, concatenate multiple turns, or invent evidence.\n"
        "2. Keep direct user facts such as geometry, Reynolds number, objective, "
        "material, and explicitly stated conditions as source=user with short exact evidence.\n"
        "3. If a fact synthesizes or summarizes information from multiple user turns, "
        "use source=derived, provide a reason, and set depends_on to the direct fact IDs "
        "that support it. request.summary commonly needs source=derived after follow-up turns.\n"
        "4. classification.problem_type and temporal.behavior are review-critical. Mark them "
        "source=user only when the exact user evidence explicitly states the normalized class/time behavior; "
        "otherwise use source=derived. A rectangular computational domain around an obstacle does not by itself mean internal_flow.\n"
        "5. Do not delete explicit user numbers or other supported facts merely to pass "
        "validation. Preserve later-turn overrides.\n"
        "6. Regenerate the COMPLETE CFDIntakeSpec, not only the offending fact."
    )
    return (
        base_prompt
        + "\n\nThe previous draft failed deterministic intake validation. "
        "Repair the full CFDIntakeSpec using the exact evidence below.\n\n"
        + rules
        + "\n\nDETERMINISTIC REPAIR CONTEXT:\n"
        + json.dumps(repair_context, ensure_ascii=False, indent=2)
    )


class IntakeAgent:
    def __init__(self, llm: StructuredLLM):
        self.llm = llm

    def run(self, state: CFDState) -> CFDState:
        prompt = build_intake_prompt(state.user_request)
        max_repairs = _intake_validation_repair_budget(self.llm)
        result: CFDIntakeSpec | None = None
        validation_error: ValueError | None = None

        for attempt in range(max_repairs + 1):
            current_prompt = prompt
            if attempt > 0:
                assert result is not None
                assert validation_error is not None
                if _uses_extended_intake_provenance_repair(self.llm):
                    current_prompt = _build_intake_validation_repair_prompt(
                        base_prompt=prompt,
                        request=state.user_request,
                        previous=result,
                        error=validation_error,
                        attempt=attempt,
                        max_attempts=max_repairs,
                    )
                else:
                    current_prompt = _legacy_intake_validation_repair_prompt(
                        base_prompt=prompt, error=validation_error
                    )
            result = self.llm.generate(
                CFDIntakeSpec, current_prompt, system_prompt=INTAKE_SYSTEM_PROMPT
            )
            try:
                validate_intake_provenance(result, state.user_request)
                # IntakeAgent-issued definitions opt into the v2 semantic-fidelity
                # contract. Persisted/directly constructed v1 specs remain loadable
                # for backward compatibility, but new interactive/cloud runs get
                # assertion-backed downstream verification after /confirm.
                if result.semantic_contract_version != "2":
                    result = result.model_copy(update={"semantic_contract_version": "2"})
                validation_error = None
                break
            except ValueError as exc:
                validation_error = exc
                if attempt >= max_repairs:
                    raise

        assert result is not None
        state.intake = result
        state.intake_confirmed = False
        state.intake_digest = None
        if result.status == "needs_user_input":
            questions = "; ".join(item.question for item in result.blocking_unknowns)
            state.transition(
                State.NEEDS_CLARIFICATION,
                f"CFD intake needs user input: {questions}",
            )
        else:
            state.transition(
                State.INTAKE_REVIEW_REQUIRED,
                "CFD intake draft is ready; user confirmation is required.",
            )
        return state
