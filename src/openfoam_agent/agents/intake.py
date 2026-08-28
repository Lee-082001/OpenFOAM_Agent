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
        "title": state.intake.title,
        "facts": facts,
        "status": state.intake.status,
    }


class IntakeAgent:
    def __init__(self, llm: StructuredLLM):
        self.llm = llm

    def run(self, state: CFDState) -> CFDState:
        prompt = build_intake_prompt(state.user_request)
        try:
            result = self.llm.generate(
                CFDIntakeSpec, prompt, system_prompt=INTAKE_SYSTEM_PROMPT
            )
            validate_intake_provenance(result, state.user_request)
        except ValueError as exc:
            repair_prompt = (
                prompt
                + "\n\nThe previous draft failed deterministic intake validation. "
                "Regenerate the full CFDIntakeSpec once. Preserve user numeric values "
                "and exact evidence. Validation error: "
                + json.dumps(str(exc), ensure_ascii=False)
            )
            result = self.llm.generate(
                CFDIntakeSpec,
                repair_prompt,
                system_prompt=INTAKE_SYSTEM_PROMPT,
            )
            validate_intake_provenance(result, state.user_request)
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
