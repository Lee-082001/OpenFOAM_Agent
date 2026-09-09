"""v4.3.1 progress-first geometry ownership regressions."""

from conftest import FakeOpenFOAMTools, ScriptedLLM, make_state
from openfoam_agent.engineering import CFDEngineeringAgent
from openfoam_agent.llm.prompts.engineering import (
    CASE_AUTHORING_SYSTEM_PROMPT,
    PREPARE_DESIGN_SYSTEM_PROMPT,
)
from openfoam_agent.schemas.engineering import BlockAction
from openfoam_agent.workflow.states import State


def _agent(tmp_path, graph_path):
    return CFDEngineeringAgent(
        ScriptedLLM([]), workspace=tmp_path, capability_db=graph_path, tools=FakeOpenFOAMTools()
    )


def test_v431_generic_geometry_without_asset_is_agent_owned_and_self_contained(tmp_path, graph_path):
    state = make_state()
    state.assets = []
    policy = _agent(tmp_path, graph_path)._geometry_authoring_policy(state)
    assert policy["user_assets_present"] is False
    assert policy["agent_generated_geometry_authorized"] is True
    assert policy["representative_geometry_defaults_authorized"] is True
    assert any("blockMesh" in item for item in policy["preferred_self_contained_methods"])
    assert any("not fabrication" in item for item in policy["rules"])


def test_v431_geometry_policy_preserves_user_assets_without_forbidding_supplemental_agent_geometry(tmp_path, graph_path):
    state = make_state()
    state.assets = [{"case_path": "constant/triSurface/user.stl", "sha256": "a" * 64}]
    policy = _agent(tmp_path, graph_path)._geometry_authoring_policy(state)
    assert policy["user_assets_present"] is True
    assert policy["user_assets"][0]["case_path"].endswith("user.stl")
    assert any("Do not overwrite" in item for item in policy["rules"])


def test_v431_engineering_choice_block_is_rejected_even_without_legacy_exploratory_flag(tmp_path, graph_path):
    agent = _agent(tmp_path, graph_path)
    state = make_state()
    state.user_request = state.user_request.model_copy(update={"exploratory_completion_authorized": False})
    state.transition(State.ENGINEERING, "test")
    event, terminal = agent._dispatch_prepare(
        state,
        BlockAction(
            type="block",
            reason="A representative pipe radius is missing.",
            block_kind="engineering_choice_missing",
            missing_items=["pipe radius"],
            needs_user_input=True,
        ),
        step=1,
        native_execution=False,
    )
    assert terminal is False
    assert not event.success
    assert state.current_state == State.ENGINEERING
    assert "engineering_defaults" in event.summary


def test_v431_authoring_prompt_explicitly_allows_case_local_surface_generation():
    lower = CASE_AUTHORING_SYSTEM_PROMPT.lower()
    assert "missing cad/stl is not a reason to block" in lower
    assert "constant/trisurface" in lower
    assert "not fabrication" in lower


def test_v431_design_prompt_requires_self_contained_geometry_strategy():
    lower = PREPARE_DESIGN_SYSTEM_PROMPT.lower()
    assert "self-contained geometry strategy" in lower
    assert "do not design a case that later blocks" in lower
