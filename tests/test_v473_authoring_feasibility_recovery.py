from __future__ import annotations

from conftest import FakeOpenFOAMTools, ScriptedLLM, make_plan, make_state
from openfoam_agent.engineering import CFDEngineeringAgent, EngineeringPolicy
from openfoam_agent.llm.prompts.engineering import CASE_AUTHORING_SYSTEM_PROMPT
from openfoam_agent.schemas.engineering import (
    BlockAction,
    EngineeringPlanPatch,
    StrategyRevisionAction,
    StrategyRevisionTurn,
    CaseAuthoringTurn,
)
from openfoam_agent.workflow.states import State


def _agent(tmp_path, graph_path):
    return CFDEngineeringAgent(
        ScriptedLLM([]), workspace=tmp_path, capability_db=graph_path, tools=FakeOpenFOAMTools(),
        policy=EngineeringPolicy(compact_phase_schemas=True, staged_case_authoring=True),
    )


def _draft_surface_plan(state):
    plan = make_plan(state.intake)
    plan.mesh_strategy = "Agent-generated STL plus snappyHexMesh for a representative branch junction."
    plan.required_case_files = [
        "constant/triSurface/generatedBranch.stl",
        "system/snappyHexMeshDict",
    ]
    return plan


def test_authoring_contract_has_explicit_strategy_infeasible_escape_hatch():
    action = BlockAction(
        type="block",
        reason="The generated surface cannot be completed within this bounded authoring turn.",
        block_kind="authoring_strategy_infeasible",
        needs_user_input=False,
    )
    assert action.block_kind == "authoring_strategy_infeasible"
    assert "authoring_strategy_infeasible" in CASE_AUTHORING_SYSTEM_PROMPT


def test_agent_owned_incomplete_surface_escalates_to_strategy_not_terminal_block(tmp_path, graph_path):
    agent = _agent(tmp_path, graph_path)
    state = make_state(); state.transition(State.ENGINEERING, "test")
    agent._draft_design_plan = _draft_surface_plan(state)
    event, terminal = agent._dispatch_prepare(
        state,
        BlockAction(
            type="block",
            reason="The required surface artifact is incomplete; I cannot truthfully return a validation-ready case bundle.",
            block_kind="other",
            needs_user_input=False,
        ),
        step=2,
        native_execution=False,
    )
    assert terminal is False
    assert event.success is False
    assert event.failure_scope == "strategy"
    assert event.failure_signature == "authoring_feasibility:precommit_strategy"
    assert state.current_state == State.ENGINEERING
    state.engineering_events.append(event)
    schema, _, phase = agent._phase_contract(state, "prepare")
    assert schema is StrategyRevisionTurn
    assert phase == "strategy_revision"


def test_explicit_precommit_strategy_failure_recovers_even_without_surface_path(tmp_path, graph_path):
    agent = _agent(tmp_path, graph_path)
    state = make_state(); state.transition(State.ENGINEERING, "test")
    plan = make_plan(state.intake); plan.required_case_files = ["system/blockMeshDict"]
    agent._draft_design_plan = plan
    event, terminal = agent._dispatch_prepare(
        state,
        BlockAction(
            type="block",
            reason="The frozen implementation cannot be authored within the bounded contract.",
            block_kind="authoring_strategy_infeasible",
            needs_user_input=False,
        ),
        step=2,
        native_execution=False,
    )
    assert not terminal and not event.success
    assert event.failure_scope == "strategy"


def test_user_owned_surface_missing_is_not_reclassified_as_agent_owned_strategy_failure(tmp_path, graph_path):
    agent = _agent(tmp_path, graph_path)
    state = make_state(); state.transition(State.ENGINEERING, "test")
    plan = _draft_surface_plan(state)
    state.assets = [{"case_path": "constant/triSurface/generatedBranch.stl"}]
    agent._draft_design_plan = plan
    event, terminal = agent._dispatch_prepare(
        state,
        BlockAction(
            type="block",
            reason="The required user surface artifact is unavailable.",
            block_kind="other",
            needs_user_input=False,
        ),
        step=2,
        native_execution=False,
    )
    assert terminal is True
    assert event.success is True
    assert state.current_state == State.ENGINEERING_BLOCKED


def test_precommit_strategy_revision_updates_draft_plan_without_workspace_mutation(tmp_path, graph_path):
    agent = _agent(tmp_path, graph_path)
    state = make_state(); state.transition(State.ENGINEERING, "test")
    original = _draft_surface_plan(state)
    agent._draft_design_plan = original
    failure, _ = agent._dispatch_prepare(
        state,
        BlockAction(
            type="block",
            reason="Required Agent-owned STL is incomplete.",
            block_kind="authoring_strategy_infeasible",
            needs_user_input=False,
        ),
        step=2,
        native_execution=False,
    )
    state.engineering_events.append(failure)

    revision = StrategyRevisionAction(
        type="revise_mesh_strategy",
        diagnosis="Replace infeasible generated-surface authoring with self-contained blockMesh.",
        plan_patch=EngineeringPlanPatch(
            mesh_strategy="Self-contained representative multi-block branch junction using typed blockMesh.",
            required_case_files=["system/blockMeshDict"],
        ),
        # These are deliberately irrelevant in precommit mode and must not mutate the workspace.
        mesh_commands=["blockMesh"],
    )
    terminal = agent._execute_strategy_revision(
        state,
        revision,
        llm_step=3,
        progress_phase="engineering",
        native_execution=False,
    )
    assert terminal is False
    assert agent._draft_design_plan is not None
    assert agent._draft_design_plan.required_case_files == ["system/blockMeshDict"]
    assert "blockMesh" in agent._draft_design_plan.mesh_strategy
    assert agent.workspace.execution_file_seals() == []
    assert state.engineering_events[-1].success
    assert state.engineering_events[-1].action_type == "revise_mesh_strategy"

    schema, _, phase = agent._phase_contract(state, "prepare")
    assert schema is CaseAuthoringTurn
    assert phase == "author_case"


def test_precommit_strategy_revision_requires_an_actual_plan_change(tmp_path, graph_path):
    agent = _agent(tmp_path, graph_path)
    state = make_state(); state.transition(State.ENGINEERING, "test")
    plan = _draft_surface_plan(state); agent._draft_design_plan = plan
    action = StrategyRevisionAction(
        type="revise_mesh_strategy",
        diagnosis="No-op",
        plan_patch=EngineeringPlanPatch(mesh_strategy=plan.mesh_strategy),
    )
    terminal = agent._execute_strategy_revision(
        state, action, llm_step=3, progress_phase="engineering", native_execution=False
    )
    assert terminal is False
    assert not state.engineering_events[-1].success
    assert "did not change" in state.engineering_events[-1].output_excerpt


def test_successful_strategy_revision_resolves_prior_strategy_trigger(tmp_path, graph_path):
    agent = _agent(tmp_path, graph_path)
    state = make_state(); state.transition(State.ENGINEERING, "test")
    failure = agent._event(
        2, "block", False, "infeasible", failure_signature="authoring_feasibility:x",
        failure_scope="strategy", failure_category="case",
    )
    success = agent._event(3, "revise_mesh_strategy", True, "revised")
    state.engineering_events.extend([failure, success])
    assert agent._strategy_revision_required(state) is False


def test_interactive_command_normalization_removes_invisible_and_ansi_noise():
    from openfoam_agent.cli import _normalize_interactive_command
    assert _normalize_interactive_command("\u200b/confirm\ufeff") == "/confirm"
    assert _normalize_interactive_command("\x1b[2C/confirm\x1b[0m") == "/confirm"


def test_precommit_strategy_revision_uses_compact_dedicated_context(tmp_path, graph_path):
    class CaptureLLM:
        def __init__(self):
            self.prompts = []
            self.schemas = []
        def generate(self, schema, prompt, *, system_prompt=None):
            self.schemas.append(schema); self.prompts.append(prompt)
            return StrategyRevisionTurn(
                action=BlockAction(
                    type="block",
                    reason="test stop",
                    block_kind="other",
                    needs_user_input=True,
                )
            )

    llm = CaptureLLM()
    agent = CFDEngineeringAgent(
        llm, workspace=tmp_path, capability_db=graph_path, tools=FakeOpenFOAMTools(),
        policy=EngineeringPolicy(
            compact_phase_schemas=True,
            staged_case_authoring=True,
            bounded_evidence_context=True,
            max_model_prompt_chars=18_000,
        ),
    )
    state = make_state(); state.transition(State.ENGINEERING, "test")
    agent._draft_design_plan = _draft_surface_plan(state)
    failure = agent._event(
        2, "block", False, "surface incomplete",
        failure_signature="authoring_feasibility:precommit_strategy",
        failure_scope="strategy", failure_category="case",
    )
    state.engineering_events.append(failure)
    agent._generate_turn(
        state, step=3, local_step=3, current_step_limit=12,
        phase="prepare", native_execution=False,
    )
    assert llm.schemas[-1] is StrategyRevisionTurn
    prompt = llm.prompts[-1]
    assert "precommit_strategy_revision_v1" in prompt
    assert '"precommit":true' in prompt.replace(" ", "")
    assert "agent_owned_generated_geometry" in prompt
    assert "capability_graph_hint" not in prompt
    assert len(prompt) <= 18_000


def test_plan_patch_can_update_implementation_bindings_when_required_files_change():
    state = make_state(); plan = make_plan(state.intake)
    first = plan.confirmed_fact_bindings[0].model_copy(
        update={"case_files": ["constant/triSurface/generatedBranch.stl"]}
    )
    plan.confirmed_fact_bindings[0] = first
    plan.required_case_files = ["constant/triSurface/generatedBranch.stl"]
    updated = [
        item.model_copy(update={"case_files": ["system/blockMeshDict"]})
        if item.fact_id == first.fact_id else item
        for item in plan.confirmed_fact_bindings
    ]
    patched = EngineeringPlanPatch(
        required_case_files=["system/blockMeshDict"],
        confirmed_fact_bindings=updated,
        mesh_strategy="self-contained blockMesh",
    ).apply(plan)
    assert patched.required_case_files == ["system/blockMeshDict"]
    assert patched.confirmed_fact_bindings[0].case_files == ["system/blockMeshDict"]
