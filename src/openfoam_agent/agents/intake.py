from __future__ import annotations

import json
import math
import re

from openfoam_agent.llm.prompts import INTAKE_SYSTEM_PROMPT
from openfoam_agent.llm.protocol import StructuredLLM
from openfoam_agent.schemas.intake import CFDIntakeSpec
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
        "object is workflow authorization and must not become a CFD fact:\n"
        + json.dumps(payload, ensure_ascii=False, indent=2)
    )


def _file_name(path: str) -> str:
    return path.replace("\\", "/").rsplit("/", maxsplit=1)[-1]


def _user_evidence(request: UserRequest) -> list[str]:
    return [
        request.prompt,
        *request.conversation_turns,
        *(_file_name(path) for path in request.geometry_files),
        *(_file_name(path) for path in request.additional_files),
    ]


def validate_intake_provenance(spec: CFDIntakeSpec, request: UserRequest) -> None:
    _normalize_review_critical_source_attribution(spec)
    user_evidence = _user_evidence(request)
    for fact in spec.facts:
        if fact.source == "user":
            assert fact.evidence is not None
            if not any(
                fact.evidence.casefold() in evidence.casefold()
                for evidence in user_evidence
            ):
                raise ValueError(
                    f"User fact '{fact.id}' has evidence not found in user-provided input."
                )
    supplied_numbers = _finite_numbers("\n".join(user_evidence))
    represented_numbers = _finite_numbers(
        "\n".join(
            " ".join(part for part in (fact.value, fact.unit or "") if part)
            for fact in spec.facts
            if fact.category != "context" and fact.source == "user"
        )
    )
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

    remaining_direct_ids = [
        fact.id
        for fact in spec.facts
        if fact.source == "user"
        and fact.category != "context"
        and fact not in to_demote
    ]
    for fact in to_demote:
        fact.source = "derived"
        fact.evidence = None
        fact.reason = (
            "The value is a routing/physics interpretation inferred from the request; "
            "the user did not explicitly state this normalized classification."
        )
        fact.depends_on = [item for item in remaining_direct_ids if item != fact.id][:20]


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
        facts.append(fact.model_dump(mode="json", exclude={"evidence"}))
    return {
        "semantic_contract_version": state.intake.semantic_contract_version,
        "title": state.intake.title,
        "facts": facts,
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
