from __future__ import annotations

from io import StringIO

from openfoam_agent.engineering import CFDEngineeringAgent, EngineeringPolicy
from openfoam_agent.postprocessing import CFDPostProcessingAgent, PostProcessingPolicy
from openfoam_agent.progress import CLIProgressReporter
from openfoam_agent.runtime import RuntimeOrchestrator
from openfoam_agent.schemas.common import ToolResult
from openfoam_agent.schemas.engineering import (
    BlockAction,
    FinishPreviewAction,
    RunMeshCommandAction,
    SearchCapabilitiesAction,
    SurfaceCheckAction,
    ValidateDictionaryAction,
    WriteCaseFileAction,
)
from openfoam_agent.schemas.postprocessing import (
    BlockPostProcessingAction,
    RunFoamPostProcessAction,
    WritePostProcessConfigAction,
)
from openfoam_agent.schemas.simulation import RuntimePolicy, RuntimeReport, SimulationAttempt
from openfoam_agent.tools.diagnostics import diagnose_openfoam_failure
from openfoam_agent.tools.parsers import parse_runtime_log
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


SNAPPY_FATAL = """Create time\n\n--> FOAM FATAL IO ERROR:\nkeyword locationInMesh is undefined in dictionary system/snappyHexMeshDict/castellatedMeshControls\nfile: /tmp/private/case/system/snappyHexMeshDict at line 81.\n\nFOAM exiting\n"""
RUNTIME_FATAL = """Time = 0.1\n--> FOAM FATAL ERROR:\nCannot find patch field entry for cylinder\nfile: /tmp/private/case/0/U at line 42.\n\nFOAM aborting\n"""
POST_FATAL = """--> FOAM FATAL IO ERROR:\nUnknown function type forceCoeffz\nfile: /tmp/private/case/postprocessConfig/wakeDict at line 18.\n\nFOAM exiting\n"""


def _prepare_mesh_ready(tmp_path, graph_path, tools, *, extra_actions=(), progress=None):
    state = make_state()
    plan = make_plan(state.intake)
    llm = ScriptedLLM(
        [
            SearchCapabilitiesAction(
                type="search_capabilities",
                query="incompressibleFluid",
                rationale="Observe solver evidence.",
            ),
            WriteCaseFileAction(
                type="write_case_file",
                path="system/controlDict",
                content=control_dict(),
                rationale="Write runtime dictionary.",
            ),
            RunMeshCommandAction(
                type="run_mesh_command",
                command="checkMesh",
                rationale="Establish mesh evidence.",
            ),
            FinishPreviewAction(type="finish_preview", plan=plan, rationale="Seal case."),
            *extra_actions,
        ]
    )
    agent = CFDEngineeringAgent(
        llm,
        workspace=tmp_path,
        capability_db=graph_path,
        tools=tools,
        policy=EngineeringPolicy(max_agent_steps=12, hard_max_agent_steps=12),
        progress=progress,
    )
    agent.prepare(state, native_execution=True)
    assert state.current_state == State.MESH_READY
    return state, plan, llm, agent


def test_native_failure_diagnostic_classifies_command_and_fatal_kind():
    result = ToolResult(
        success=False,
        command=["snappyHexMesh", "-overwrite"],
        return_code=1,
        stderr=SNAPPY_FATAL,
    )
    diagnostic = diagnose_openfoam_failure(result)

    assert diagnostic.command == "snappyHexMesh"
    assert diagnostic.return_code == 1
    assert diagnostic.kind == "foam_fatal_io_error"
    assert "locationInMesh" in diagnostic.excerpt
    assert "diagnosticKind: foam_fatal_io_error" in diagnostic.render()


def test_snappy_failure_reaches_user_and_next_engineering_turn(tmp_path, graph_path):
    state = make_state()
    llm = ScriptedLLM(
        [
            RunMeshCommandAction(
                type="run_mesh_command",
                command="snappyHexMesh",
                rationale="Snap cylinder surface.",
            ),
            BlockAction(
                type="block",
                reason="Stop after observing snappy diagnostic.",
                needs_user_input=False,
                rationale="Test terminal action.",
            ),
        ]
    )
    tools = FakeOpenFOAMTools(
        mesh_results={
            "snappyHexMesh": [
                tool_result("snappyHexMesh", success=False, stderr=SNAPPY_FATAL)
            ]
        }
    )
    stream = StringIO()
    agent = CFDEngineeringAgent(
        llm,
        workspace=tmp_path,
        capability_db=graph_path,
        tools=tools,
        policy=EngineeringPolicy(max_agent_steps=3, hard_max_agent_steps=3),
        progress=CLIProgressReporter("normal", stream=stream),
    )

    agent.prepare(state, native_execution=True)

    failed = state.engineering_events[0]
    assert not failed.success
    assert "snappyHexMesh returned status 1; native diagnostic captured." == failed.summary
    assert "diagnosticKind: foam_fatal_io_error" in failed.output_excerpt
    assert "locationInMesh" in failed.output_excerpt
    assert "locationInMesh" in llm.prompts[1]

    progress = stream.getvalue()
    assert "snappyHexMesh returned status 1; native diagnostic captured." in progress
    assert "FOAM FATAL IO ERROR" in progress
    assert "locationInMesh" in progress
    assert "/tmp/private" not in progress

    raw = (agent.workspace.log_dir / "001.snappyHexMesh.log").read_text(encoding="utf-8")
    assert "/tmp/private/case/system/snappyHexMeshDict" in raw


class _FailingInspectionTools(FakeOpenFOAMTools):
    def foam_dictionary_validate(self, file_path, cwd=None):
        del cwd
        self.dictionary_calls.append(str(file_path))
        return ToolResult(
            success=False,
            command=["foamDictionary"],
            return_code=1,
            stderr="--> FOAM FATAL IO ERROR:\nbad dictionary token\nfile: /tmp/private/system/testDict at line 4.\n",
        )

    def surface_check(self, geometry_path, cwd=None):
        del cwd, geometry_path
        return ToolResult(
            success=False,
            command=["surfaceCheck"],
            return_code=1,
            stderr="--> FOAM FATAL ERROR:\nSurface is not closed\nfile: /tmp/private/cylinder.stl\n",
        )


def test_dictionary_and_surface_failures_use_same_native_diagnostic_path(tmp_path, graph_path):
    state = make_state()
    llm = ScriptedLLM(
        [
            WriteCaseFileAction(
                type="write_case_file",
                path="system/testDict",
                content="FoamFile { format ascii; class dictionary; object testDict; }\nvalue 1;\n",
                rationale="Create dictionary.",
            ),
            ValidateDictionaryAction(
                type="validate_dictionary",
                path="system/testDict",
                rationale="Validate dictionary.",
            ),
            WriteCaseFileAction(
                type="write_case_file",
                path="constant/triSurface/cylinder.stl",
                content="solid cylinder\nendsolid cylinder\n",
                rationale="Create geometry placeholder.",
            ),
            SurfaceCheckAction(
                type="surface_check",
                path="constant/triSurface/cylinder.stl",
                rationale="Validate surface.",
            ),
            BlockAction(
                type="block",
                reason="Stop after native inspection failures.",
                needs_user_input=False,
                rationale="Test terminal action.",
            ),
        ]
    )
    stream = StringIO()
    tools = _FailingInspectionTools()
    agent = CFDEngineeringAgent(
        llm,
        workspace=tmp_path,
        capability_db=graph_path,
        tools=tools,
        policy=EngineeringPolicy(max_agent_steps=6, hard_max_agent_steps=6),
        progress=CLIProgressReporter("normal", stream=stream),
    )

    agent.prepare(state, native_execution=True)

    dictionary_event = next(event for event in state.engineering_events if event.action_type == "validate_dictionary")
    surface_event = next(event for event in state.engineering_events if event.action_type == "surface_check")
    assert "native diagnostic captured" in dictionary_event.summary
    assert "bad dictionary token" in dictionary_event.output_excerpt
    assert "native diagnostic captured" in surface_event.summary
    assert "Surface is not closed" in surface_event.output_excerpt
    assert (agent.workspace.log_dir / "002.foamDictionary.log").exists()
    assert (agent.workspace.log_dir / "004.surfaceCheck.log").exists()
    assert "/tmp/private" not in stream.getvalue()


def test_foamrun_failure_progress_and_repair_prompt_use_native_diagnostic(tmp_path, graph_path):
    tools = FakeOpenFOAMTools(
        mesh_results={"checkMesh": [tool_result("checkMesh", success=True, stdout=mesh_ok_log())]},
        foam_runs=[ToolResult(success=False, command=["foamRun"], return_code=1, stderr=RUNTIME_FATAL)],
    )
    stream = StringIO()
    state, _, llm, agent = _prepare_mesh_ready(
        tmp_path,
        graph_path,
        tools,
        extra_actions=[
            BlockAction(
                type="block",
                reason="Runtime diagnostic observed; stop test repair.",
                needs_user_input=False,
                rationale="Test terminal action.",
            )
        ],
        progress=CLIProgressReporter("normal", stream=stream),
    )
    state.approve_solve()
    runtime = RuntimeOrchestrator(
        tools,
        agent,
        RuntimePolicy(max_attempts=2, solver_timeout_seconds=30),
        progress=CLIProgressReporter("normal", stream=stream),
    )

    runtime.run(state)

    repair_prompts = [prompt for prompt in llm.prompts if '"phase": "runtime_repair"' in prompt]
    assert repair_prompts
    assert "diagnosticKind: foam_fatal_error" in repair_prompts[0]
    assert "Cannot find patch field entry for cylinder" in repair_prompts[0]
    progress = stream.getvalue()
    assert "foamRun attempt 1 실패; native diagnostic captured" in progress
    assert "Cannot find patch field entry for cylinder" in progress
    assert "/tmp/private" not in progress
    raw = (agent.workspace.log_dir / "foamRun.attempt-001.log").read_text(encoding="utf-8")
    assert "/tmp/private/case/0/U" in raw


def test_foampostprocess_failure_reaches_user_and_next_postprocess_turn(tmp_path, graph_path):
    tools = FakeOpenFOAMTools(
        mesh_results={"checkMesh": [tool_result("checkMesh", success=True, stdout=mesh_ok_log())]},
        postprocess_runs=[
            ToolResult(success=False, command=["foamPostProcess"], return_code=1, stderr=POST_FATAL)
        ],
    )
    state, _, _, _ = _prepare_mesh_ready(tmp_path, graph_path, tools)
    runtime_result = parse_runtime_log("Time = 1s\nEnd\n", return_code=0)
    state.runtime_report = RuntimeReport(
        success=True,
        attempts=[SimulationAttempt(attempt=1, result=runtime_result)],
        final_result=runtime_result,
    )
    state.simulation = runtime_result
    state.current_state = State.EXECUTION_DONE

    config = "FoamFile { format ascii; class dictionary; object wakeDict; }\nfunctions {}\n"
    post_llm = ScriptedLLM(
        [
            WritePostProcessConfigAction(
                type="write_postprocess_config",
                path="postprocessConfig/wakeDict",
                content=config,
                rationale="Write postprocess dictionary.",
            ),
            RunFoamPostProcessAction(
                type="run_foam_postprocess",
                dictionary_path="postprocessConfig/wakeDict",
                time_selection="latest",
                use_solver_context=True,
                rationale="Execute postprocessing.",
            ),
            BlockPostProcessingAction(
                type="block_postprocessing",
                reason="Stop after observing native diagnostic.",
                rationale="Test terminal action.",
            ),
        ]
    )
    stream = StringIO()
    post = CFDPostProcessingAgent(
        post_llm,
        workspace=tmp_path,
        tools=tools,
        policy=PostProcessingPolicy(max_steps=4, max_native_commands=2),
        progress=CLIProgressReporter("normal", stream=stream),
    )

    post.run(state)

    failed = next(event for event in state.postprocessing_events if event.action_type == "run_foam_postprocess")
    assert "native diagnostic captured" in failed.summary
    assert "Unknown function type forceCoeffz" in failed.output_excerpt
    assert len(post_llm.prompts) >= 3
    assert "Unknown function type forceCoeffz" in post_llm.prompts[2]
    progress = stream.getvalue()
    assert "foamPostProcess returned status 1; native diagnostic captured." in progress
    assert "Unknown function type forceCoeffz" in progress
    assert "/tmp/private" not in progress
    raw = (post.workspace.log_dir / "postprocess.002.foamPostProcess.log").read_text(encoding="utf-8")
    assert "/tmp/private/case/postprocessConfig/wakeDict" in raw
