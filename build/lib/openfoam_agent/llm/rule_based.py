from __future__ import annotations

import json
import re
from typing import TypeVar

from pydantic import BaseModel

from openfoam_agent.schemas.intake import BlockingUnknown, CFDIntakeSpec, IntakeFact


T = TypeVar("T", bound=BaseModel)

_CFD_TERMS = (
    "flow", "fluid", "cfd", "pipe", "channel", "vortex", "wake", "obstacle",
    "mesh", "reynolds", "species", "transport", "diffusion", "temperature",
    "heat", "multiphase", "turbulence", "유동", "유체", "파이프", "와류", "후류",
    "장애물", "격자", "레이놀즈", "수송", "확산", "온도", "열전달", "다상", "난류",
)
_NUMBER = re.compile(
    r"(?<![A-Za-z0-9_.])[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?(?![A-Za-z0-9_.]|차)"
)


class RuleBasedLLM:
    """Offline intake regression baseline only.

    v2 intentionally has no rule-based engineering/case fallback. Confirmed CFD
    problems must be handed to a real CFDEngineeringAgent backend.
    """

    def generate(
        self,
        schema: type[T],
        prompt: str,
        *,
        system_prompt: str | None = None,
    ) -> T:
        del system_prompt
        if schema is CFDIntakeSpec:
            return _intake_from_prompt(prompt)  # type: ignore[return-value]
        raise NotImplementedError(
            "RuleBasedLLM is intake-only in v2; autonomous engineering has no "
            "deterministic case-template fallback."
        )


def _intake_from_prompt(prompt: str) -> CFDIntakeSpec:
    start = prompt.find("{")
    try:
        payload = json.loads(prompt[start:]) if start >= 0 else {}
    except json.JSONDecodeError:
        payload = {}
    evidence = payload.get("user_evidence", {}) if isinstance(payload, dict) else {}
    turns = evidence.get("conversation_turns", []) if isinstance(evidence, dict) else []
    if not isinstance(turns, list) or not all(isinstance(item, str) for item in turns):
        turns = []
    text = "\n".join(turns).strip()
    lowered = text.casefold()
    if not text or not any(term in lowered for term in _CFD_TERMS):
        return CFDIntakeSpec(
            title="CFD request needs clarification",
            blocking_unknowns=[
                BlockingUnknown(
                    id="objective.cfd_problem",
                    question="어떤 CFD 현상과 목표를 해석하려는지 설명해 주세요.",
                    impact="The physical CFD problem cannot yet be identified.",
                )
            ],
            status="needs_user_input",
        )

    problem_type = _problem_type(lowered)
    facts: list[IntakeFact] = [
        IntakeFact(
            id="request.summary",
            category="context",
            label="Normalized CFD request",
            value=text,
            source="derived",
            reason="Merged chronological user turns.",
        ),
        IntakeFact(
            id="classification.problem_type",
            category="classification",
            label="Problem class",
            value=problem_type,
            source="derived",
            reason="Classified the stated physical CFD problem.",
            depends_on=["request.summary"],
        ),
        IntakeFact(
            id="objective.primary",
            category="objective",
            label="Primary objective",
            value="Evaluate the requested CFD behavior and outputs.",
            source="derived",
            reason="Normalized the user's requested CFD outcome.",
            depends_on=["request.summary"],
        ),
    ]

    temporal = "transient" if any(
        token in lowered for token in ("transient", "비정상", "vortex shedding", "진동", "oscillat", "moving")
    ) else "steady" if any(token in lowered for token in ("steady", "정상상태")) else "unspecified"
    facts.append(
        IntakeFact(
            id="temporal.behavior",
            category="temporal",
            label="Time behavior",
            value=temporal,
            source="derived",
            reason="Normalized explicit time-dependent language without choosing numerics.",
            depends_on=["request.summary"],
        )
    )

    for turn in reversed(turns):
        dim = re.search(r"(?:^|\s)([23])\s*(?:d|차원)", turn.casefold())
        if dim:
            facts.append(
                IntakeFact(
                    id="geometry.dimension",
                    category="geometry",
                    label="Analysis dimension",
                    value=f"{dim.group(1)}D",
                    source="user",
                    evidence=turn,
                )
            )
            break

    reynolds_pattern = re.compile(
        r"(?:\bre\b|reynolds(?:\s*number)?|레이놀즈(?:\s*수)?)\s*(?:=|:|는|은|약)?\s*"
        r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)",
        re.IGNORECASE,
    )
    for turn in reversed(turns):
        match = reynolds_pattern.search(turn)
        if match:
            facts.append(
                IntakeFact(
                    id="operating.reynolds_number",
                    category="scale",
                    label="Reynolds number",
                    value=match.group(1),
                    source="user",
                    evidence=turn,
                )
            )
            break

    if any(token in lowered for token in ("moving", "oscillat", "진동", "움직", "변형", "deform")):
        evidence_turn = next(
            turn for turn in turns
            if any(token in turn.casefold() for token in ("moving", "oscillat", "진동", "움직", "변형", "deform"))
        )
        facts.append(
            IntakeFact(
                id="motion.primary",
                category="motion",
                label="Requested motion",
                value=evidence_turn,
                source="user",
                evidence=evidence_turn,
            )
        )

    represented = {token for fact in facts for token in _NUMBER.findall(fact.value)}
    numeric_index = 0
    for turn in turns:
        for token in _NUMBER.findall(turn):
            if token in represented:
                continue
            numeric_index += 1
            facts.append(
                IntakeFact(
                    id=f"scale.user_numeric_{numeric_index}",
                    category="scale",
                    label=f"Explicit user numeric value {numeric_index}",
                    value=token,
                    source="user",
                    evidence=turn,
                )
            )
            represented.add(token)

    facts = _apply_explicit_set_facts(facts, turns)
    return CFDIntakeSpec(
        title={
            "internal_flow": "Internal-flow CFD study",
            "external_flow": "External-flow CFD study",
            "heat_transfer": "Heat-transfer CFD study",
            "multiphase": "Multiphase CFD study",
            "species_transport": "Species-transport CFD study",
            "custom": "Custom CFD study",
        }[problem_type],
        facts=facts,
        status="ready_for_review",
    )


def _problem_type(text: str) -> str:
    if any(token in text for token in ("multiphase", "two-phase", "다상")):
        return "multiphase"
    if any(token in text for token in ("species", "scalar transport", "농도", "수송")):
        return "species_transport"
    if any(token in text for token in ("heat transfer", "열전달", "temperature", "온도")) and not any(
        token in text for token in ("flow", "유동", "fluid", "유체")
    ):
        return "heat_transfer"
    if any(token in text for token in ("pipe", "channel", "파이프", "채널", "관 내부")):
        return "internal_flow"
    if any(token in text for token in ("external flow", "vortex", "wake", "obstacle", "외부 유동", "와류", "후류", "장애물")):
        return "external_flow"
    return "custom"


def _apply_explicit_set_facts(facts: list[IntakeFact], turns: list[str]) -> list[IntakeFact]:
    categories = {
        "classification", "objective", "domain", "geometry", "scale", "material",
        "property", "physics", "temporal", "motion", "boundary", "output", "fidelity", "assumption",
    }
    pattern = re.compile(r"^Set (?P<id>[a-z][a-z0-9_.-]*) to (?P<value>.+) as a user-provided value\.$")
    by_id = {fact.id: fact for fact in facts}
    for turn in turns:
        matched = pattern.fullmatch(turn)
        if not matched:
            continue
        fact_id = matched.group("id")
        category = fact_id.split(".", 1)[0]
        if category not in categories:
            continue
        by_id[fact_id] = IntakeFact(
            id=fact_id,
            category=category,  # type: ignore[arg-type]
            label=fact_id.replace(".", " ").replace("_", " ").title(),
            value=matched.group("value"),
            source="user",
            evidence=turn,
        )
    return list(by_id.values())
