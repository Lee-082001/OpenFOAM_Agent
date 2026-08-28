from __future__ import annotations

from openfoam_agent.engineering import CFDEngineeringAgent, EngineeringPolicy
from openfoam_agent.schemas.engineering import (
    BlockAction,
    FinishPreviewAction,
    RunMeshCommandAction,
    SearchCapabilitiesAction,
    ValidateDictionaryAction,
    WriteCaseFileAction,
)
from openfoam_agent.schemas.simulation import RuntimePolicy
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


def test_default_budgets_are_production_sized():
    engineering = EngineeringPolicy()
    runtime = RuntimePolicy()

    assert engineering.max_agent_steps == 120
    assert engineering.hard_max_agent_steps == 200
    assert engineering.step_extension == 20
    assert engineering.max_finalization_steps == 8
    assert engineering.max_native_commands == 40
    assert engineering.max_mesh_repair_cycles == 10
    assert engineering.max_runtime_repair_steps == 60
    assert runtime.max_attempts == 9
    assert runtime.max_repair_cycles == 8


def test_progress_at_soft_boundary_extends_engineering_window(tmp_path, graph_path):
    state = make_state()
    plan = make_plan(state.intake)
    llm = ScriptedLLM([
        SearchCapabilitiesAction(
            type="search_capabilities",
            query="incompressibleFluid",
            rationale="Observe solver capability.",
        ),
        WriteCaseFileAction(
            type="write_case_file",
            path="system/controlDict",
            content=control_dict(),
            rationale="Write control.",
        ),
        SearchCapabilitiesAction(
            type="search_capabilities",
            query="incompressibleFluid",
            rationale="Repeat an existing observation.",
        ),
        SearchCapabilitiesAction(
            type="search_capabilities",
            query="incompressibleFluid",
            rationale="Repeat an existing observation again.",
        ),
        WriteCaseFileAction(
            type="write_case_file",
            path="system/fvSchemes",
            content="ddtSchemes { default Euler; }\n",
            rationale="New artifact at the soft boundary proves work progressed.",
        ),
        RunMeshCommandAction(
            type="run_mesh_command",
            command="checkMesh",
            rationale="Validate after the progress-aware extension.",
        ),
        FinishPreviewAction(
            type="finish_preview",
            plan=plan,
            rationale="Finish inside the extended window.",
        ),
    ])
    tools = FakeOpenFOAMTools(
        mesh_results={
            "checkMesh": [tool_result("checkMesh", success=True, stdout=mesh_ok_log())]
        }
    )
    agent = CFDEngineeringAgent(
        llm,
        workspace=tmp_path,
        capability_db=graph_path,
        tools=tools,
        policy=EngineeringPolicy(
            max_agent_steps=5,
            hard_max_agent_steps=7,
            step_extension=2,
            progress_window=2,
        ),
    )

    agent.prepare(state, native_execution=True)

    assert state.current_state == State.MESH_READY
    assert len(state.engineering_events) == 7
    assert len(state.engineering_budget_extensions) == 1
    assert state.engineering_budget_extensions[0].previous_limit == 5
    assert state.engineering_budget_extensions[0].new_limit == 7
    assert '"current_engineering_step_limit": 7' in llm.prompts[5]
    assert '"hard_engineering_step_limit": 7' in llm.prompts[5]


def test_repeated_action_result_loop_does_not_receive_extension(tmp_path, graph_path):
    state = make_state()
    repeated = [
        SearchCapabilitiesAction(
            type="search_capabilities",
            query="incompressibleFluid",
            rationale=f"Repeated query {index}.",
        )
        for index in range(5)
    ]
    llm = ScriptedLLM(repeated)
    agent = CFDEngineeringAgent(
        llm,
        workspace=tmp_path,
        capability_db=graph_path,
        tools=FakeOpenFOAMTools(),
        policy=EngineeringPolicy(
            max_agent_steps=5,
            hard_max_agent_steps=7,
            step_extension=2,
            progress_window=2,
        ),
    )

    agent.prepare(state, native_execution=True)

    assert state.current_state == State.ENGINEERING_BLOCKED
    assert len(state.engineering_events) == 5
    assert "without new deterministic progress evidence" in state.history[-1]["note"]


def test_progress_never_extends_past_hard_cap(tmp_path, graph_path):
    state = make_state()
    actions = [
        SearchCapabilitiesAction(
            type="search_capabilities",
            query="incompressibleFluid",
            rationale="Capability provenance.",
        ),
        *[
            WriteCaseFileAction(
                type="write_case_file",
                path=f"system/progress{index}",
                content=f"value {index};\n",
                rationale=f"Create a genuinely new artifact {index}.",
            )
            for index in range(2, 8)
        ],
    ]
    llm = ScriptedLLM(actions)
    agent = CFDEngineeringAgent(
        llm,
        workspace=tmp_path,
        capability_db=graph_path,
        tools=FakeOpenFOAMTools(),
        policy=EngineeringPolicy(
            max_agent_steps=5,
            hard_max_agent_steps=7,
            step_extension=2,
            progress_window=2,
        ),
    )

    agent.prepare(state, native_execution=True)

    assert state.current_state == State.ENGINEERING_BLOCKED
    assert len(state.engineering_events) == 7
    assert "hard step budget exhausted (7)" in state.history[-1]["note"]


def test_native_command_budget_is_independent_from_agent_turn_budget(tmp_path, graph_path):
    state = make_state()
    llm = ScriptedLLM([
        SearchCapabilitiesAction(
            type="search_capabilities",
            query="incompressibleFluid",
            rationale="Observe solver capability.",
        ),
        WriteCaseFileAction(
            type="write_case_file",
            path="system/controlDict",
            content=control_dict(),
            rationale="Write control.",
        ),
        ValidateDictionaryAction(
            type="validate_dictionary",
            path="system/controlDict",
            rationale="Use the only permitted native command.",
        ),
        RunMeshCommandAction(
            type="run_mesh_command",
            command="checkMesh",
            rationale="This must be blocked before native execution.",
        ),
        BlockAction(
            type="block",
            reason="Native command budget is intentionally exhausted in this test.",
            needs_user_input=False,
            rationale="Stop safely.",
        ),
    ])
    tools = FakeOpenFOAMTools()
    agent = CFDEngineeringAgent(
        llm,
        workspace=tmp_path,
        capability_db=graph_path,
        tools=tools,
        policy=EngineeringPolicy(
            max_agent_steps=5,
            hard_max_agent_steps=5,
            max_native_commands=1,
        ),
    )

    agent.prepare(state, native_execution=True)

    assert state.current_state == State.ENGINEERING_BLOCKED
    assert len(tools.dictionary_calls) == 1
    assert tools.mesh_calls == []
    blocked = state.engineering_events[3]
    assert not blocked.success
    assert "Native OpenFOAM command budget exhausted (1)" in blocked.summary
    assert sum(event.native_command_executed for event in state.engineering_events) == 1


def test_mesh_repair_cycle_budget_blocks_eleventh_style_repair_group(tmp_path, graph_path):
    state = make_state()
    llm = ScriptedLLM([
        SearchCapabilitiesAction(
            type="search_capabilities",
            query="incompressibleFluid",
            rationale="Observe solver capability.",
        ),
        WriteCaseFileAction(
            type="write_case_file",
            path="system/controlDict",
            content=control_dict(),
            rationale="Write control.",
        ),
        WriteCaseFileAction(
            type="write_case_file",
            path="system/blockMeshDict",
            content="generation 0;\n",
            rationale="Initial mesh input.",
        ),
        RunMeshCommandAction(
            type="run_mesh_command",
            command="blockMesh",
            rationale="First failed mesh attempt.",
        ),
        WriteCaseFileAction(
            type="write_case_file",
            path="system/blockMeshDict",
            content="generation 1;\n",
            rationale="Allowed repair cycle one.",
        ),
        RunMeshCommandAction(
            type="run_mesh_command",
            command="blockMesh",
            rationale="Mesh still fails after repair cycle one.",
        ),
        WriteCaseFileAction(
            type="write_case_file",
            path="system/blockMeshDict",
            content="generation 2;\n",
            rationale="This second repair cycle must be blocked.",
        ),
        BlockAction(
            type="block",
            reason="Mesh repair budget exhausted in test.",
            needs_user_input=False,
            rationale="Stop safely.",
        ),
    ])
    tools = FakeOpenFOAMTools(
        mesh_results={
            "blockMesh": [
                tool_result("blockMesh", success=False, stderr="mesh fail 1\n"),
                tool_result("blockMesh", success=False, stderr="mesh fail 2\n"),
            ]
        }
    )
    agent = CFDEngineeringAgent(
        llm,
        workspace=tmp_path,
        capability_db=graph_path,
        tools=tools,
        policy=EngineeringPolicy(
            max_agent_steps=8,
            hard_max_agent_steps=8,
            max_mesh_repair_cycles=1,
        ),
    )

    agent.prepare(state, native_execution=True)

    assert state.current_state == State.ENGINEERING_BLOCKED
    assert agent.workspace.read_text("system/blockMeshDict") == "generation 1;\n"
    assert "Mesh repair cycle budget exhausted (1)" in state.engineering_events[6].summary
    assert agent._mesh_repair_cycle_count(state) == 1


def test_runtime_default_allows_eight_repair_cycles(tmp_path, graph_path):
    from openfoam_agent.runtime import RuntimeOrchestrator
    from openfoam_agent.schemas.common import ToolResult
    from openfoam_agent.schemas.engineering import RetrySolverAction

    state = make_state()
    plan = make_plan(state.intake)
    prepare_actions = [
        SearchCapabilitiesAction(
            type="search_capabilities",
            query="incompressibleFluid",
            rationale="Observe solver capability.",
        ),
        WriteCaseFileAction(
            type="write_case_file",
            path="system/controlDict",
            content=control_dict(),
            rationale="Write control.",
        ),
        RunMeshCommandAction(
            type="run_mesh_command",
            command="checkMesh",
            rationale="Establish current mesh evidence.",
        ),
        FinishPreviewAction(
            type="finish_preview",
            plan=plan,
            rationale="Seal case before runtime.",
        ),
    ]
    repair_actions = [
        RetrySolverAction(
            type="retry_solver",
            plan=plan,
            rationale=f"Retry unchanged approved solver after failure {index}.",
        )
        for index in range(1, 9)
    ]
    llm = ScriptedLLM([*prepare_actions, *repair_actions])
    failed_runs = [
        ToolResult(
            success=False,
            command=["foamRun"],
            return_code=1,
            stderr=f"Time = {index / 10}\n--> FOAM FATAL ERROR: scripted failure {index}\n",
        )
        for index in range(1, 9)
    ]
    final_run = ToolResult(
        success=True,
        command=["foamRun"],
        return_code=0,
        stdout="Time = 0.9\nCourant Number mean: 0.1 max: 0.2\nEnd\n",
    )
    tools = FakeOpenFOAMTools(
        mesh_results={
            "checkMesh": [tool_result("checkMesh", success=True, stdout=mesh_ok_log())]
        },
        foam_runs=[*failed_runs, final_run],
    )
    agent = CFDEngineeringAgent(
        llm,
        workspace=tmp_path,
        capability_db=graph_path,
        tools=tools,
        policy=EngineeringPolicy(max_agent_steps=10, hard_max_agent_steps=10),
    )
    agent.prepare(state, native_execution=True)
    assert state.current_state == State.MESH_READY

    state.approve_solve()
    runtime = RuntimeOrchestrator(tools, agent, RuntimePolicy(solver_timeout_seconds=30))
    runtime.run(state)

    assert state.current_state == State.EXECUTION_DONE
    assert state.runtime_report is not None
    assert len(state.runtime_report.attempts) == 9
    assert sum(attempt.repair_requested for attempt in state.runtime_report.attempts) == 8
    assert len(tools.foam_run_solvers) == 9


def test_cli_budget_flags_build_the_same_policy_defaults(graph_path):
    from openfoam_agent.cli import _policies_from_args, _validate_args, build_parser

    parser = build_parser()
    args = parser.parse_args(["test prompt", "--capability-db", str(graph_path)])
    assert _validate_args(args, parser) == "test prompt"
    engineering, runtime, postprocessing = _policies_from_args(args)

    assert engineering.max_agent_steps == 120
    assert engineering.hard_max_agent_steps == 200
    assert engineering.max_finalization_steps == 8
    assert engineering.max_native_commands == 40
    assert engineering.max_mesh_repair_cycles == 10
    assert runtime.max_repair_cycles == 8
    assert postprocessing.max_steps == 40
    assert postprocessing.max_native_commands == 8


def test_cli_budget_flags_are_configurable(graph_path):
    from openfoam_agent.cli import _policies_from_args, _validate_args, build_parser

    parser = build_parser()
    args = parser.parse_args([
        "test prompt",
        "--capability-db",
        str(graph_path),
        "--engineering-steps",
        "100",
        "--engineering-hard-cap",
        "180",
        "--engineering-extension",
        "10",
        "--finalization-steps",
        "6",
        "--native-command-budget",
        "30",
        "--mesh-repair-cycles",
        "7",
        "--runtime-repair-cycles",
        "5",
        "--runtime-repair-steps",
        "45",
        "--postprocess-steps",
        "30",
        "--postprocess-native-budget",
        "6",
    ])
    assert _validate_args(args, parser) == "test prompt"
    engineering, runtime, postprocessing = _policies_from_args(args)

    assert engineering.max_agent_steps == 100
    assert engineering.hard_max_agent_steps == 180
    assert engineering.step_extension == 10
    assert engineering.max_finalization_steps == 6
    assert engineering.max_native_commands == 30
    assert engineering.max_mesh_repair_cycles == 7
    assert engineering.max_runtime_repair_steps == 45
    assert runtime.max_repair_cycles == 5
    assert postprocessing.max_steps == 30
    assert postprocessing.max_native_commands == 6


def test_checkmesh_return_code_zero_with_failed_evidence_counts_as_mesh_failure(tmp_path, graph_path):
    state = make_state()
    llm = ScriptedLLM([
        SearchCapabilitiesAction(
            type="search_capabilities",
            query="incompressibleFluid",
            rationale="Observe capability.",
        ),
        WriteCaseFileAction(
            type="write_case_file",
            path="system/controlDict",
            content=control_dict(),
            rationale="Write control.",
        ),
        RunMeshCommandAction(
            type="run_mesh_command",
            command="checkMesh",
            rationale="checkMesh process exits zero but mesh evidence is not OK.",
        ),
        BlockAction(
            type="block",
            reason="Stop after evidence failure test.",
            needs_user_input=False,
            rationale="Stop safely.",
        ),
    ])
    # ToolResult.success=True means return code 0. Absence of "Mesh OK." means
    # parsed mesh evidence must still reject the result.
    tools = FakeOpenFOAMTools(
        mesh_results={
            "checkMesh": [
                tool_result(
                    "checkMesh",
                    success=True,
                    stdout=(
                        "cells: 1000\n"
                        "Mesh non-orthogonality Max: 25 average: 5\n"
                        "Max skewness = 2\n"
                        "cells with negative volume: 0\n"
                        "Failed 1 mesh checks.\n"
                    ),
                )
            ]
        }
    )
    agent = CFDEngineeringAgent(
        llm,
        workspace=tmp_path,
        capability_db=graph_path,
        tools=tools,
        policy=EngineeringPolicy(max_agent_steps=4, hard_max_agent_steps=4),
    )

    agent.prepare(state, native_execution=True)

    mesh_event = state.engineering_events[2]
    assert mesh_event.mesh_command_executed
    assert mesh_event.native_command_executed
    assert not mesh_event.success
    assert state.mesh_evidence is not None and not state.mesh_evidence.passed
    _, failure_pending, _ = agent._mesh_repair_status(state)
    assert failure_pending


def test_resource_accounting_survives_engineering_agent_rehydration(tmp_path, graph_path):
    state = make_state()
    plan = make_plan(state.intake)
    first_llm = ScriptedLLM([
        SearchCapabilitiesAction(
            type="search_capabilities",
            query="incompressibleFluid",
            rationale="Observe capability.",
        ),
        WriteCaseFileAction(
            type="write_case_file",
            path="system/controlDict",
            content=control_dict(),
            rationale="Write control.",
        ),
        RunMeshCommandAction(
            type="run_mesh_command",
            command="checkMesh",
            rationale="Consume one native command.",
        ),
        FinishPreviewAction(
            type="finish_preview",
            plan=plan,
            rationale="Seal case.",
        ),
    ])
    first_tools = FakeOpenFOAMTools(
        mesh_results={
            "checkMesh": [tool_result("checkMesh", success=True, stdout=mesh_ok_log())]
        }
    )
    policy = EngineeringPolicy(
        max_agent_steps=4,
        hard_max_agent_steps=4,
        max_native_commands=1,
    )
    first = CFDEngineeringAgent(
        first_llm,
        workspace=tmp_path,
        capability_db=graph_path,
        tools=first_tools,
        policy=policy,
    )
    first.prepare(state, native_execution=True)
    assert state.current_state == State.MESH_READY
    assert first._native_command_count(state) == 1

    second_tools = FakeOpenFOAMTools()
    second = CFDEngineeringAgent(
        ScriptedLLM([]),
        workspace=tmp_path,
        capability_db=graph_path,
        tools=second_tools,
        policy=policy,
    )
    second.workspace.adopt_seal(state.case_seal)

    assert second._native_command_count(state) == 1
    blocked = second._dispatch_tool_action(
        RunMeshCommandAction(
            type="run_mesh_command",
            command="checkMesh",
            rationale="A fresh Agent object must honor the previous native-command count.",
        ),
        step=len(state.engineering_events) + 1,
        native_execution=True,
        phase="runtime_repair",
        state=state,
    )
    assert not blocked.success
    assert "Native OpenFOAM command budget exhausted (1)" in blocked.summary
    assert second_tools.mesh_calls == []


def test_default_soft_boundary_120_extends_to_140_on_new_artifact(tmp_path, graph_path):
    state = make_state()
    actions = [
        SearchCapabilitiesAction(
            type="search_capabilities",
            query="incompressibleFluid",
            rationale=f"Repeated capability observation {index}.",
        )
        for index in range(1, 120)
    ]
    actions.extend([
        WriteCaseFileAction(
            type="write_case_file",
            path="system/controlDict",
            content=control_dict(),
            rationale="A genuinely new artifact appears exactly at the default soft boundary.",
        ),
        BlockAction(
            type="block",
            reason="Stop after proving the 120->140 extension.",
            needs_user_input=False,
            rationale="Test terminal.",
        ),
    ])
    llm = ScriptedLLM(actions)
    agent = CFDEngineeringAgent(
        llm,
        workspace=tmp_path,
        capability_db=graph_path,
        tools=FakeOpenFOAMTools(),
    )

    agent.prepare(state, native_execution=True)

    assert len(state.engineering_budget_extensions) == 1
    extension = state.engineering_budget_extensions[0]
    assert extension.boundary_step == 120
    assert extension.previous_limit == 120
    assert extension.new_limit == 140
    assert len(state.engineering_events) == 121


def test_default_soft_boundary_120_rejects_stagnant_loop(tmp_path, graph_path):
    state = make_state()
    llm = ScriptedLLM([
        SearchCapabilitiesAction(
            type="search_capabilities",
            query="incompressibleFluid",
            rationale=f"Repeated capability observation {index}.",
        )
        for index in range(1, 121)
    ])
    agent = CFDEngineeringAgent(
        llm,
        workspace=tmp_path,
        capability_db=graph_path,
        tools=FakeOpenFOAMTools(),
    )

    agent.prepare(state, native_execution=True)

    assert state.current_state == State.ENGINEERING_BLOCKED
    assert len(state.engineering_events) == 120
    assert state.engineering_budget_extensions == []
    assert "without new deterministic progress evidence" in state.history[-1]["note"]
