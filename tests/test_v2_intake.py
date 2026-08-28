from __future__ import annotations

import pytest

from openfoam_agent.agents.intake import build_intake_prompt, validate_intake_provenance
from openfoam_agent.conversation import ConversationSession, InteractionMode, exploratory_authorization
from openfoam_agent.llm.rule_based import RuleBasedLLM
from openfoam_agent.schemas.engineering import EngineeringTurn
from openfoam_agent.schemas.intake import CFDIntakeSpec
from openfoam_agent.schemas.request import UserRequest


def test_exploratory_authorization_is_policy_not_cfd_default():
    session = ConversationSession(mode=InteractionMode.GUIDED)
    session.add_turn("사각형 장애물 주위 vortex shedding Re=1000 나머지는 탐색용으로 정해줘")
    request = session.to_request()
    assert request.exploratory_completion_authorized is True
    assert not hasattr(request, "defaults")
    assert not hasattr(request, "assumptions")


def test_explicit_rejection_overrides_exploratory_default():
    assert exploratory_authorization("알아서 정하지 말고 물어봐") is False


def test_rule_based_backend_is_intake_only():
    with pytest.raises(NotImplementedError, match="intake-only"):
        RuleBasedLLM().generate(EngineeringTurn, "anything")


def test_rule_based_intake_preserves_user_reynolds_number():
    request = UserRequest(
        prompt="사각형 장애물 주위 vortex shedding Re=1000 나머지는 탐색용으로 정해줘",
        exploratory_completion_authorized=True,
    )
    spec = RuleBasedLLM().generate(CFDIntakeSpec, build_intake_prompt(request))
    validate_intake_provenance(spec, request)
    re_fact = spec.fact("operating.reynolds_number")
    assert re_fact is not None
    assert re_fact.value == "1000"
    assert re_fact.source == "user"
    assert all(fact.source in {"user", "derived"} for fact in spec.facts)


def test_intake_schema_has_no_default_fact_source():
    schema = CFDIntakeSpec.model_json_schema()
    assert '"default"' not in str(schema)
