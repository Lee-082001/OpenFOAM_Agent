from __future__ import annotations

import ast
from pathlib import Path

from conftest import FakeOpenFOAMTools, make_plan, make_state
from openfoam_agent.engineering import CFDEngineeringAgent, EngineeringPolicy
from openfoam_agent.engineering.revision_context import project_strategy_plan
from openfoam_agent.schemas.engineering import (
    BlockAction,
    EngineeringDefaultAssumption,
    StrategyRevisionTurn,
)
from openfoam_agent.workflow.states import State


def _default_with_unit() -> EngineeringDefaultAssumption:
    return EngineeringDefaultAssumption(
        parameter="inlet_velocity",
        value="1.0",
        unit="m/s",
        rationale="Representative inlet speed selected by the engineering Agent.",
    )


def _surface_plan(state):
    plan = make_plan(state.intake)
    plan.mesh_strategy = "Agent-generated STL plus snappyHexMesh for a representative branch junction."
    plan.required_case_files = [
        "constant/triSurface/bifurcation.stl",
        "system/snappyHexMeshDict",
    ]
    plan.engineering_defaults = [_default_with_unit()]
    return plan


def test_v503_strategy_plan_projection_accepts_normal_engineering_units():
    state = make_state()
    plan = _surface_plan(state)

    projected = project_strategy_plan(plan)

    assert projected is not None
    assert projected["engineering_defaults"][0]["unit"] == "m/s"


def test_v503_precommit_strategy_revision_context_survives_engineering_default_units(
    tmp_path, graph_path
):
    class CaptureLLM:
        def __init__(self):
            self.prompts = []
            self.schemas = []

        def generate(self, schema, prompt, *, system_prompt=None):
            self.schemas.append(schema)
            self.prompts.append(prompt)
            return StrategyRevisionTurn(
                action=BlockAction(
                    type="block",
                    reason="test stop after strategy-revision context compilation",
                    block_kind="other",
                    needs_user_input=True,
                )
            )

    llm = CaptureLLM()
    agent = CFDEngineeringAgent(
        llm,
        workspace=tmp_path,
        capability_db=graph_path,
        tools=FakeOpenFOAMTools(),
        policy=EngineeringPolicy(
            compact_phase_schemas=True,
            staged_case_authoring=True,
            bounded_evidence_context=True,
            max_model_prompt_chars=18_000,
        ),
    )
    state = make_state()
    state.transition(State.ENGINEERING, "test")
    agent._draft_design_plan = _surface_plan(state)
    state.engineering_events.append(
        agent._event(
            2,
            "block",
            False,
            "The bounded authoring turn cannot completely represent bifurcation.stl.",
            failure_signature="authoring_feasibility:precommit_strategy",
            failure_scope="strategy",
            failure_category="case",
        )
    )

    turn = agent._generate_turn(
        state,
        step=3,
        local_step=3,
        current_step_limit=12,
        phase="prepare",
        native_execution=False,
    )

    assert isinstance(turn, StrategyRevisionTurn)
    assert llm.schemas[-1] is StrategyRevisionTurn
    prompt = llm.prompts[-1]
    assert "precommit_strategy_revision_v1" in prompt
    assert '"unit":"m/s"' in prompt.replace(" ", "")
    assert len(prompt) <= 18_000


def test_v503_no_literal_compact_text_limit_violates_helper_contract():
    root = Path(__file__).resolve().parents[1] / "src" / "openfoam_agent"
    violations: list[str] = []
    for path in root.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or len(node.args) < 2:
                continue
            if not isinstance(node.func, ast.Name) or node.func.id != "compact_text":
                continue
            limit = node.args[1]
            if isinstance(limit, ast.Constant) and isinstance(limit.value, int) and limit.value < 64:
                violations.append(f"{path.relative_to(root.parent.parent)}:{node.lineno}={limit.value}")

    assert violations == []
