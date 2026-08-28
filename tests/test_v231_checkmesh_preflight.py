from __future__ import annotations

from openfoam_agent.engineering import CFDEngineeringAgent
from openfoam_agent.schemas.engineering import BlockAction
from openfoam_agent.workflow.states import State

from conftest import FakeOpenFOAMTools, ScriptedLLM, make_plan, make_state


def test_native_prepare_blocks_before_llm_when_checkmesh_is_unavailable(tmp_path, graph_path):
    state = make_state()
    llm = ScriptedLLM([])
    agent = CFDEngineeringAgent(
        llm,
        workspace=tmp_path,
        capability_db=graph_path,
        tools=FakeOpenFOAMTools(check_mesh_available=False),
    )

    agent.prepare(state, native_execution=True)

    assert state.current_state == State.ENGINEERING_BLOCKED
    assert state.engineering_events == []
    assert llm.prompts == []
    assert "checkMesh" in state.history[-1]["note"]


def test_dry_run_skips_native_checkmesh_preflight(tmp_path, graph_path):
    state = make_state()
    llm = ScriptedLLM(
        [
            BlockAction(
                type="block",
                reason="Stop after proving the dry-run reached the engineering LLM.",
                needs_user_input=False,
                rationale="No native preflight should run in dry-run mode.",
            )
        ]
    )
    agent = CFDEngineeringAgent(
        llm,
        workspace=tmp_path,
        capability_db=graph_path,
        tools=FakeOpenFOAMTools(check_mesh_available=False),
    )

    agent.prepare(state, native_execution=False)

    assert len(llm.prompts) == 1
    assert len(state.engineering_events) == 1
    assert state.engineering_events[0].action_type == "block"


def test_real_openfoam_tools_preflight_reports_missing_checkmesh():
    from openfoam_agent.tools.openfoam import OpenFOAMTools
    from openfoam_agent.tools.safe_runner import SafeRunner

    tools = OpenFOAMTools(SafeRunner(base_env={}, trusted_executable_roots=[]))
    status = tools.check_mesh_preflight()

    assert status["name"] == "checkMesh"
    assert status["available"] is False
    assert status["trusted"] is False
