from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from openfoam_agent.agents.intake import IntakeAgent
from openfoam_agent.llm.ollama_client import OllamaLLM
from openfoam_agent.schemas.intake import CFDIntakeSpec
from openfoam_agent.schemas.request import UserRequest
from openfoam_agent.workflow.state import CFDState
from openfoam_agent.workflow.states import State


def _spec(*, summary_source: str, summary_evidence: str | None) -> CFDIntakeSpec:
    summary = {
        "id": "request.summary",
        "category": "objective",
        "label": "Request summary",
        "value": "Vortex shedding flow visualization around a circular cylinder at Re=1000",
        "source": summary_source,
        "evidence": summary_evidence,
        "reason": None,
        "depends_on": [],
    }
    if summary_source == "derived":
        summary["evidence"] = None
        summary["reason"] = "Combines the user's geometry, Reynolds number, and objective across turns."
        summary["depends_on"] = [
            "geometry.type",
            "physics.reynolds_number",
            "objective.primary",
        ]

    return CFDIntakeSpec.model_validate(
        {
            "title": "Vortex Shedding Simulation",
            "facts": [
                summary,
                {
                    "id": "classification.problem_type",
                    "category": "classification",
                    "label": "Problem type",
                    "value": "external_flow",
                    "source": "derived",
                    "reason": "Flow around a circular cylinder is an external-flow configuration.",
                    "depends_on": ["geometry.type"],
                },
                {
                    "id": "geometry.type",
                    "category": "geometry",
                    "label": "Obstacle geometry",
                    "value": "circular cylinder",
                    "source": "user",
                    "evidence": "circular cylinder",
                },
                {
                    "id": "physics.reynolds_number",
                    "category": "physics",
                    "label": "Reynolds number",
                    "value": "1000",
                    "unit": "dimensionless",
                    "source": "user",
                    "evidence": "1000",
                },
                {
                    "id": "objective.primary",
                    "category": "objective",
                    "label": "Primary objective",
                    "value": "flow visualization",
                    "source": "user",
                    "evidence": "flow visulization",
                },
            ],
            "blocking_unknowns": [],
            "status": "ready_for_review",
        }
    )


@dataclass
class _ScriptedLocalLLM:
    outputs: list[CFDIntakeSpec]
    intake_validation_repair_attempts: int = 2
    prompts: list[str] = field(default_factory=list)

    def generate(self, schema, prompt, *, system_prompt=None):
        assert schema is CFDIntakeSpec
        self.prompts.append(prompt)
        return self.outputs[len(self.prompts) - 1]


@dataclass
class _ScriptedCloudLikeLLM:
    outputs: list[CFDIntakeSpec]
    prompts: list[str] = field(default_factory=list)

    def generate(self, schema, prompt, *, system_prompt=None):
        assert schema is CFDIntakeSpec
        self.prompts.append(prompt)
        return self.outputs[len(self.prompts) - 1]


def _request() -> UserRequest:
    return UserRequest(
        prompt="Vortex shedding 시뮬레이션 좀 ㄱㄱ",
        conversation_turns=[
            "flow visulization용이고, reynolds넘버는 1000정도로 해 circular cylinder로"
        ],
        interaction_mode="easy",
        exploratory_completion_authorized=True,
    )


def test_ollama_advertises_two_semantic_intake_validation_repairs():
    # Construction with a fake client avoids any network/SDK behavior; this test only
    # pins the local-adapter capability hint consumed by IntakeAgent.
    llm = OllamaLLM(client=object())
    assert llm.intake_validation_repair_attempts == 2


def test_local_intake_repairs_multi_turn_paraphrased_user_evidence_until_derived_summary():
    invalid = _spec(
        summary_source="user",
        summary_evidence="Vortex shedding flow visualization around a circular cylinder at Re=1000",
    )
    valid = _spec(summary_source="derived", summary_evidence=None)
    llm = _ScriptedLocalLLM(outputs=[invalid, invalid, valid])
    state = CFDState(run_id="run-local-provenance", user_request=_request())

    result = IntakeAgent(llm).run(state)

    assert result.current_state == State.INTAKE_REVIEW_REQUIRED
    assert result.intake is not None
    assert result.intake.fact("request.summary").source == "derived"
    assert len(llm.prompts) == 3

    first_repair = llm.prompts[1]
    assert "PROVENANCE REPAIR RULES" in first_repair
    assert "source=user" in first_repair
    assert "contiguous verbatim substring" in first_repair
    assert "request.summary commonly needs source=derived" in first_repair
    assert '"turn": 1' in first_repair
    assert "Vortex shedding 시뮬레이션 좀 ㄱㄱ" in first_repair
    assert '"turn": 2' in first_repair
    assert "flow visulization용이고" in first_repair
    assert "previous_invalid_spec" in first_repair
    assert "User fact 'request.summary' has evidence not found" in first_repair


def test_openai_like_adapter_keeps_historical_single_semantic_retry():
    invalid = _spec(
        summary_source="user",
        summary_evidence="Vortex shedding flow visualization around a circular cylinder at Re=1000",
    )
    llm = _ScriptedCloudLikeLLM(outputs=[invalid, invalid, invalid])
    state = CFDState(run_id="run-cloud-provenance", user_request=_request())

    with pytest.raises(ValueError, match="request.summary"):
        IntakeAgent(llm).run(state)

    # No local capability hint -> one historical retry (two total generations),
    # with the exact pre-v2.7.3 generic repair wording rather than local guidance.
    assert len(llm.prompts) == 2
    assert "Regenerate the full CFDIntakeSpec once" in llm.prompts[1]
    assert "PROVENANCE REPAIR RULES" not in llm.prompts[1]


def test_local_provenance_repair_does_not_weaken_verbatim_gate():
    invalid = _spec(
        summary_source="user",
        summary_evidence="Vortex shedding flow visualization around a circular cylinder at Re=1000",
    )
    llm = _ScriptedLocalLLM(outputs=[invalid, invalid, invalid])
    state = CFDState(run_id="run-local-still-blocked", user_request=_request())

    with pytest.raises(ValueError, match="evidence not found in user-provided input"):
        IntakeAgent(llm).run(state)

    assert len(llm.prompts) == 3
