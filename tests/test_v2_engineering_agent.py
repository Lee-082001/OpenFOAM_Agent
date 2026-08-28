from __future__ import annotations

from openfoam_agent.engineering import CFDEngineeringAgent, EngineeringPolicy
from openfoam_agent.schemas.engineering import (
    FinishPreviewAction,
    RunMeshCommandAction,
    SearchCapabilitiesAction,
    WriteCaseFileAction,
)
from openfoam_agent.workflow.states import State

from conftest import (
    FakeOpenFOAMTools,
    ScriptedLLM,
    control_dict,
    make_plan,
    make_state,
    mesh_ok_log,
    tool_result,
)


def test_agent_can_author_non_template_case_preview(tmp_path, graph_path):
    state = make_state()
    plan = make_plan(state.intake)
    llm = ScriptedLLM([
        SearchCapabilitiesAction(type="search_capabilities", query="incompressibleFluid", rationale="Observe solver capability evidence."),
        WriteCaseFileAction(
            type="write_case_file",
            path="system/controlDict",
            content=control_dict(),
            rationale="Create runtime control chosen by the engineering agent.",
        ),
        FinishPreviewAction(
            type="finish_preview",
            plan=plan,
            rationale="Static safety gate can now review the agent-authored preview.",
        ),
    ])
    agent = CFDEngineeringAgent(
        llm,
        workspace=tmp_path,
        capability_db=graph_path,
        tools=FakeOpenFOAMTools(),
    )
    agent.prepare(state, native_execution=False)
    assert state.current_state == State.CASE_PREVIEW_READY
    assert state.engineering_plan == plan
    assert state.case_seal is not None
    assert agent.workspace.list_authored() == ["system/controlDict"]


def test_failed_mesh_command_is_observation_then_agent_repairs(tmp_path, graph_path):
    state = make_state()
    plan = make_plan(state.intake)
    llm = ScriptedLLM([
        SearchCapabilitiesAction(type="search_capabilities", query="incompressibleFluid", rationale="Observe solver capability evidence."),
        WriteCaseFileAction(
            type="write_case_file",
            path="system/controlDict",
            content=control_dict(),
            rationale="Write control dictionary.",
        ),
        WriteCaseFileAction(
            type="write_case_file",
            path="system/blockMeshDict",
            content="vertices ();\n",
            rationale="First exploratory mesh attempt.",
        ),
        RunMeshCommandAction(type="run_mesh_command", command="blockMesh", rationale="Try mesh."),
        WriteCaseFileAction(
            type="write_case_file",
            path="system/blockMeshDict",
            content="vertices (); // repaired after real log\n",
            rationale="Repair mesh dictionary after observing the failure.",
        ),
        RunMeshCommandAction(type="run_mesh_command", command="blockMesh", rationale="Retry mesh."),
        RunMeshCommandAction(type="run_mesh_command", command="checkMesh", rationale="Validate mesh."),
        FinishPreviewAction(type="finish_preview", plan=plan, rationale="Seal passing case."),
    ])
    tools = FakeOpenFOAMTools(
        mesh_results={
            "blockMesh": [
                tool_result("blockMesh", success=False, stderr="FOAM FATAL ERROR: bad vertex\n"),
                tool_result("blockMesh", success=True, stdout="blockMesh complete\n"),
            ],
            "checkMesh": [tool_result("checkMesh", success=True, stdout=mesh_ok_log())],
        }
    )
    agent = CFDEngineeringAgent(
        llm,
        workspace=tmp_path,
        capability_db=graph_path,
        tools=tools,
        policy=EngineeringPolicy(max_agent_steps=11),
    )
    agent.prepare(state, native_execution=True)
    assert state.current_state == State.MESH_READY
    assert tools.mesh_calls == ["blockMesh", "blockMesh", "checkMesh"]
    failed = [event for event in state.engineering_events if not event.success]
    assert any("blockMesh" in event.summary for event in failed)
    assert any("FOAM FATAL ERROR" in prompt for prompt in llm.prompts[3:])


def test_solver_input_change_preserves_previous_checkmesh_evidence(tmp_path, graph_path):
    state = make_state()
    plan = make_plan(state.intake)
    llm = ScriptedLLM([
        SearchCapabilitiesAction(type="search_capabilities", query="incompressibleFluid", rationale="Observe solver capability evidence."),
        WriteCaseFileAction(type="write_case_file", path="system/controlDict", content=control_dict(), rationale="control"),
        RunMeshCommandAction(type="run_mesh_command", command="checkMesh", rationale="check"),
        WriteCaseFileAction(type="write_case_file", path="system/fvSchemes", content="ddtSchemes {};\n", rationale="change solver input"),
        FinishPreviewAction(type="finish_preview", plan=plan, rationale="finish with still-current mesh evidence"),
    ])
    tools = FakeOpenFOAMTools(
        mesh_results={"checkMesh": [tool_result("checkMesh", success=True, stdout=mesh_ok_log())]}
    )
    agent = CFDEngineeringAgent(
        llm,
        workspace=tmp_path,
        capability_db=graph_path,
        tools=tools,
        policy=EngineeringPolicy(max_agent_steps=5),
    )
    agent.prepare(state, native_execution=True)
    assert state.current_state == State.MESH_READY
    assert state.mesh_evidence is not None and state.mesh_evidence.passed
    assert tools.mesh_calls == ["checkMesh"]


def test_mesh_input_change_invalidates_previous_checkmesh_evidence(tmp_path, graph_path):
    state = make_state()
    plan = make_plan(state.intake)
    llm = ScriptedLLM([
        SearchCapabilitiesAction(type="search_capabilities", query="incompressibleFluid", rationale="Observe solver capability evidence."),
        WriteCaseFileAction(type="write_case_file", path="system/controlDict", content=control_dict(), rationale="control"),
        RunMeshCommandAction(type="run_mesh_command", command="checkMesh", rationale="check"),
        WriteCaseFileAction(
            type="write_case_file",
            path="system/blockMeshDict",
            content="FoamFile { version 2.0; format ascii; class dictionary; object blockMeshDict; }\n",
            rationale="change mesh input",
        ),
        FinishPreviewAction(type="finish_preview", plan=plan, rationale="try stale evidence"),
    ])
    tools = FakeOpenFOAMTools(
        mesh_results={"checkMesh": [tool_result("checkMesh", success=True, stdout=mesh_ok_log())]}
    )
    agent = CFDEngineeringAgent(
        llm,
        workspace=tmp_path,
        capability_db=graph_path,
        tools=tools,
        policy=EngineeringPolicy(max_agent_steps=5),
    )
    agent.prepare(state, native_execution=True)
    assert state.current_state == State.ENGINEERING_BLOCKED
    assert state.mesh_evidence is None
    assert any(
        event.action_type == "finish_preview" and not event.success
        for event in state.engineering_events
    )


def test_generic_cell_budget_blocks_oversized_mesh(tmp_path, graph_path):
    state = make_state()
    plan = make_plan(state.intake)
    llm = ScriptedLLM([
        SearchCapabilitiesAction(type="search_capabilities", query="incompressibleFluid", rationale="Observe solver capability evidence."),
        WriteCaseFileAction(type="write_case_file", path="system/controlDict", content=control_dict(), rationale="control"),
        RunMeshCommandAction(type="run_mesh_command", command="checkMesh", rationale="check"),
        FinishPreviewAction(type="finish_preview", plan=plan, rationale="finish"),
    ])
    tools = FakeOpenFOAMTools(
        mesh_results={"checkMesh": [tool_result("checkMesh", success=True, stdout=mesh_ok_log(cells=2001))]}
    )
    agent = CFDEngineeringAgent(
        llm,
        workspace=tmp_path,
        capability_db=graph_path,
        tools=tools,
        policy=EngineeringPolicy(max_agent_steps=4, max_mesh_cells=2000),
    )
    agent.prepare(state, native_execution=True)
    assert state.current_state == State.ENGINEERING_BLOCKED
    assert any("exceeds bounded policy limit" in event.output_excerpt for event in state.engineering_events)


def test_passing_checkmesh_on_last_tool_step_gets_bounded_finalization_window(tmp_path, graph_path):
    """Regression: real runs could hit step 40 on checkMesh and never get to finish_preview."""
    state = make_state()
    plan = make_plan(state.intake)
    llm = ScriptedLLM([
        SearchCapabilitiesAction(
            type="search_capabilities",
            query="incompressibleFluid",
            rationale="Observe solver capability evidence.",
        ),
        WriteCaseFileAction(
            type="write_case_file",
            path="system/controlDict",
            content=control_dict(),
            rationale="Write the case runtime control.",
        ),
        RunMeshCommandAction(
            type="run_mesh_command",
            command="checkMesh",
            rationale="Validate the current mesh on the final ordinary tool step.",
        ),
        FinishPreviewAction(
            type="finish_preview",
            plan=plan,
            rationale="Use the bounded finalization-only turn to seal the validated case.",
        ),
    ])
    tools = FakeOpenFOAMTools(
        mesh_results={
            "checkMesh": [tool_result("checkMesh", success=True, stdout=mesh_ok_log(cells=17520))]
        }
    )
    agent = CFDEngineeringAgent(
        llm,
        workspace=tmp_path,
        capability_db=graph_path,
        tools=tools,
        policy=EngineeringPolicy(max_agent_steps=3, max_finalization_steps=2),
    )

    agent.prepare(state, native_execution=True)

    assert state.current_state == State.MESH_READY
    assert state.mesh_evidence is not None and state.mesh_evidence.passed
    assert state.engineering_events[-1].step == 4
    assert state.engineering_events[-1].action_type == "finish_preview"
    assert '"finalization_only": true' in llm.prompts[-1]
    assert '"ready_for_finalization": true' in llm.prompts[-1]


def test_long_run_keeps_compact_capability_provenance_visible_to_model(tmp_path, graph_path):
    state = make_state()
    plan = make_plan(state.intake)
    llm = ScriptedLLM([
        SearchCapabilitiesAction(
            type="search_capabilities",
            query="incompressibleFluid",
            rationale="Observe solver capability evidence early.",
        ),
        WriteCaseFileAction(
            type="write_case_file",
            path="system/controlDict",
            content=control_dict(),
            rationale="Write control.",
        ),
        RunMeshCommandAction(type="run_mesh_command", command="checkMesh", rationale="Validate."),
        FinishPreviewAction(type="finish_preview", plan=plan, rationale="Finalize."),
    ])
    tools = FakeOpenFOAMTools(
        mesh_results={"checkMesh": [tool_result("checkMesh", success=True, stdout=mesh_ok_log())]}
    )
    agent = CFDEngineeringAgent(
        llm,
        workspace=tmp_path,
        capability_db=graph_path,
        tools=tools,
        policy=EngineeringPolicy(max_agent_steps=4, observation_history=1),
    )

    agent.prepare(state, native_execution=True)

    assert state.current_state == State.MESH_READY
    # The direct capability-search event has already fallen outside recent_observations,
    # but its provider ID remains available in compact cumulative provenance.
    assert '"observed_capability_provider_ids"' in llm.prompts[-1]
    assert 'solver.incompressibleFluid' in llm.prompts[-1]


def test_exact_step_40_checkmesh_can_finalize_on_step_41(tmp_path, graph_path):
    state = make_state()
    plan = make_plan(state.intake)
    actions = [
        SearchCapabilitiesAction(
            type="search_capabilities",
            query="incompressibleFluid",
            rationale=f"Capability evidence pass {i}.",
        )
        for i in range(1, 39)
    ]
    actions.extend([
        WriteCaseFileAction(
            type="write_case_file",
            path="system/controlDict",
            content=control_dict(),
            rationale="Write controlDict on step 39.",
        ),
        RunMeshCommandAction(
            type="run_mesh_command",
            command="checkMesh",
            rationale="Passing checkMesh on ordinary step 40.",
        ),
        FinishPreviewAction(
            type="finish_preview",
            plan=plan,
            rationale="Finalize on bounded step 41.",
        ),
    ])
    llm = ScriptedLLM(actions)
    tools = FakeOpenFOAMTools(
        mesh_results={
            "checkMesh": [tool_result("checkMesh", success=True, stdout=mesh_ok_log(cells=17520))]
        }
    )
    agent = CFDEngineeringAgent(
        llm,
        workspace=tmp_path,
        capability_db=graph_path,
        tools=tools,
        policy=EngineeringPolicy(max_agent_steps=40, max_finalization_steps=4),
    )

    agent.prepare(state, native_execution=True)

    assert state.current_state == State.MESH_READY
    assert state.engineering_events[-2].step == 40
    assert state.engineering_events[-2].summary.startswith("checkMesh returned status 0")
    assert state.engineering_events[-1].step == 41
    assert state.engineering_events[-1].action_type == "finish_preview"
