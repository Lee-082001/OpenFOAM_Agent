from __future__ import annotations

from openfoam_agent.engineering import CFDEngineeringAgent, EngineeringPolicy
from openfoam_agent.runtime import RuntimeOrchestrator
from openfoam_agent.schemas.common import ToolResult
from openfoam_agent.schemas.engineering import (
    BlockAction,
    FinishPreviewAction,
    RetrySolverAction,
    RunMeshCommandAction,
    SearchCapabilitiesAction,
    WriteCaseFileAction,
)
from openfoam_agent.schemas.simulation import RuntimePolicy
from openfoam_agent.workflow.states import State

from conftest import FakeOpenFOAMTools, ScriptedLLM, control_dict, make_plan, make_state, mesh_ok_log, tool_result


def _prepared_agent(tmp_path, graph_path, tools, extra_actions):
    state = make_state()
    plan = make_plan(state.intake)
    llm = ScriptedLLM([
        SearchCapabilitiesAction(type="search_capabilities", query="incompressibleFluid", rationale="Observe solver capability evidence."),
        WriteCaseFileAction(type="write_case_file", path="system/controlDict", content=control_dict(), rationale="control"),
        RunMeshCommandAction(type="run_mesh_command", command="checkMesh", rationale="mesh evidence"),
        FinishPreviewAction(type="finish_preview", plan=plan, rationale="seal"),
        *extra_actions,
    ])
    agent = CFDEngineeringAgent(
        llm,
        workspace=tmp_path,
        capability_db=graph_path,
        tools=tools,
        policy=EngineeringPolicy(max_agent_steps=12),
    )
    agent.prepare(state, native_execution=True)
    assert state.current_state == State.MESH_READY
    return state, plan, llm, agent


def test_runtime_failure_log_returns_to_agent_and_retries(tmp_path, graph_path):
    fail_log = "Time = 0.1\n--> FOAM FATAL ERROR: boundary dictionary mismatch\n"
    ok_log = "Time = 0.2\nCourant Number mean: 0.1 max: 0.2\nEnd\n"
    tools = FakeOpenFOAMTools(
        mesh_results={
            "checkMesh": [
                tool_result("checkMesh", success=True, stdout=mesh_ok_log()),
                tool_result("checkMesh", success=True, stdout=mesh_ok_log()),
            ]
        },
        foam_runs=[
            ToolResult(success=False, command=["foamRun"], return_code=1, stderr=fail_log),
            ToolResult(success=True, command=["foamRun"], return_code=0, stdout=ok_log),
        ],
    )
    state_seed = make_state()
    plan = make_plan(state_seed.intake)
    state, plan, llm, agent = _prepared_agent(
        tmp_path,
        graph_path,
        tools,
        [
            WriteCaseFileAction(
                type="write_case_file",
                path="system/fvSolution",
                content="solvers {};\n",
                rationale="Repair dictionary based on the real foamRun failure.",
            ),
            RunMeshCommandAction(type="run_mesh_command", command="checkMesh", rationale="Refresh evidence after edit."),
            RetrySolverAction(type="retry_solver", plan=plan, rationale="Retry the same approved solver."),
        ],
    )
    state.approve_solve()
    runtime = RuntimeOrchestrator(tools, agent, RuntimePolicy(max_attempts=2, solver_timeout_seconds=30))
    runtime.run(state)
    assert state.current_state == State.EXECUTION_DONE
    assert state.runtime_report is not None and state.runtime_report.success
    assert len(state.runtime_report.attempts) == 2
    assert tools.foam_run_solvers == ["incompressibleFluid", "incompressibleFluid"]
    repair_prompts = [prompt for prompt in llm.prompts if '"phase": "runtime_repair"' in prompt]
    assert repair_prompts
    assert "FOAM FATAL ERROR: boundary dictionary mismatch" in repair_prompts[0]


def test_runtime_repair_cannot_switch_solver_without_user_review(tmp_path, graph_path):
    tools = FakeOpenFOAMTools(
        mesh_results={"checkMesh": [tool_result("checkMesh", success=True, stdout=mesh_ok_log())]}
    )
    state, plan, _, agent = _prepared_agent(tmp_path, graph_path, tools, [])
    changed = make_plan(state.intake, solver="fluid")
    agent.llm = ScriptedLLM([
        RetrySolverAction(type="retry_solver", plan=changed, rationale="Try another solver."),
        BlockAction(type="block", reason="A solver change requires user review.", needs_user_input=True, rationale="Do not bypass approval."),
    ])
    outcome = agent.repair_runtime(
        state,
        runtime_log="--> FOAM FATAL ERROR: model mismatch\n",
        attempt=1,
        native_execution=True,
    )
    assert not outcome.retry
    assert state.current_state == State.ENGINEERING_REVIEW_REQUIRED
    assert any(
        "user-approved solver" in event.output_excerpt
        for event in state.engineering_events
        if event.action_type == "retry_solver"
    )


def test_solver_execution_requires_explicit_approval(tmp_path, graph_path):
    tools = FakeOpenFOAMTools(
        mesh_results={"checkMesh": [tool_result("checkMesh", success=True, stdout=mesh_ok_log())]}
    )
    state, _, _, agent = _prepared_agent(tmp_path, graph_path, tools, [])
    runtime = RuntimeOrchestrator(tools, agent, RuntimePolicy(max_attempts=1, solver_timeout_seconds=30))
    runtime.run(state)
    assert state.current_state == State.FAILED
    assert "approval is missing" in state.history[-1]["note"]


def test_runtime_openfoam13_seconds_suffix_finishes_on_first_attempt(tmp_path, graph_path):
    ok_log = (
        "Courant Number mean: 0.02095074 max: 0.034800772\n"
        "Time = 20s\n"
        "smoothSolver:  Solving for Ux, Initial residual = 1.9344907e-05, "
        "Final residual = 2.2917407e-10, No Iterations 2\n"
        "GAMG:  Solving for p, Initial residual = 5.7748509e-05, "
        "Final residual = 5.337679e-06, No Iterations 2\n"
        "time step continuity errors : sum local = 4.397203e-13, "
        "global = 8.5185882e-15, cumulative = 1.2487946e-10\n"
        "End\n"
    )
    tools = FakeOpenFOAMTools(
        mesh_results={"checkMesh": [tool_result("checkMesh", success=True, stdout=mesh_ok_log())]},
        foam_runs=[
            ToolResult(success=True, command=["foamRun"], return_code=0, stdout=ok_log),
        ],
    )
    state, _, llm, agent = _prepared_agent(tmp_path, graph_path, tools, [])
    prompts_before_solve = len(llm.prompts)
    state.approve_solve()
    runtime = RuntimeOrchestrator(tools, agent, RuntimePolicy(max_attempts=9, solver_timeout_seconds=30))
    runtime.run(state)

    assert state.current_state == State.EXECUTION_DONE
    assert state.runtime_report is not None and state.runtime_report.success
    assert len(state.runtime_report.attempts) == 1
    assert state.runtime_report.final_result.last_time == 20.0
    assert state.runtime_report.final_result.courant_max == 0.034800772
    assert tools.foam_run_solvers == ["incompressibleFluid"]
    assert len(llm.prompts) == prompts_before_solve
