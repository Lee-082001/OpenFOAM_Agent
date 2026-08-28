from __future__ import annotations

import math

import pytest

from openfoam_agent.engineering import CFDEngineeringAgent, EngineeringPolicy
from openfoam_agent.llm.openai_client import validate_structured_output_schema
from openfoam_agent.postprocessing.analysis import analyze_force_coefficients
from openfoam_agent.postprocessing import PostProcessingPolicy
from openfoam_agent.schemas.common import ToolResult
from openfoam_agent.schemas.engineering import (
    FinishPreviewAction,
    RunMeshCommandAction,
    SearchCapabilitiesAction,
    WriteCaseFileAction,
)
from openfoam_agent.schemas.postprocessing import (
    AnalyzeForceCoefficientsAction,
    FinishPostProcessingAction,
    PostProcessingTurn,
    RunFoamPostProcessAction,
    WritePostProcessConfigAction,
)
from openfoam_agent.tools.workspace import CaseWorkspace, WorkspaceSafetyError
from openfoam_agent.workflow.engine import CFDWorkflow
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


def _force_coeff_text(*, frequency: float = 0.2, end_time: float = 50.0, dt: float = 0.1) -> str:
    lines = [
        "# Force coefficients",
        "# Time Cm Cd Cl Cl(f) Cl(r)",
    ]
    count = int(round(end_time / dt)) + 1
    for index in range(count):
        time = index * dt
        phase = 2.0 * math.pi * frequency * time
        cl = 0.8 * math.sin(phase)
        cd = 1.5 + 0.05 * math.cos(2.0 * phase)
        lines.append(f"{time:.8f} 0 {cd:.12f} {cl:.12f} {cl/2:.12f} {cl/2:.12f}")
    return "\n".join(lines) + "\n"


def _postprocess_dict() -> str:
    return """FoamFile
{
    format ascii;
    class dictionary;
    object postProcessDict;
}
functions
{
    forceCoeffsObstacle
    {
        type forceCoeffs;
        libs (\"libforces.so\");
        patches (obstacle);
        liftDir (0 1 0);
        dragDir (1 0 0);
        pitchAxis (0 0 1);
        magUInf 1;
        lRef 1;
        Aref 1;
    }
}
"""


def test_postprocessing_turn_is_strict_structured_output_compatible():
    validate_structured_output_schema(PostProcessingTurn)
    schema = PostProcessingTurn.model_json_schema()
    action_schema = schema["properties"]["action"]
    assert "anyOf" in action_schema
    assert "oneOf" not in action_schema
    assert "discriminator" not in action_schema


def test_force_coefficients_analysis_estimates_frequency_and_strouhal():
    analysis = analyze_force_coefficients(
        _force_coeff_text(frequency=0.2),
        _postprocess_dict(),
        source_path="postProcessing/forceCoeffsObstacle/0/coefficient.dat",
        dictionary_path="postprocessConfig/wakeAnalysisDict",
        discard_fraction=0.2,
    )
    assert analysis.samples_total == 501
    assert analysis.samples_used > 350
    assert analysis.mean_cd == pytest.approx(1.5, abs=5e-3)
    assert analysis.rms_cl == pytest.approx(0.8 / math.sqrt(2), rel=0.03)
    assert analysis.shedding_frequency == pytest.approx(0.2, rel=0.01)
    assert analysis.reference_velocity == 1.0
    assert analysis.reference_length == 1.0
    assert analysis.strouhal_number == pytest.approx(0.2, rel=0.01)
    assert analysis.periods_observed >= 6


def test_postprocess_config_does_not_mutate_solve_input_seal(tmp_path):
    state = make_state()
    plan = make_plan(state.intake)
    workspace = CaseWorkspace(tmp_path)
    workspace.write_text("system/controlDict", control_dict())
    seal = workspace.seal(plan)
    before = workspace.manifest_digest()

    workspace.write_postprocess_config("postprocessConfig/wakeAnalysisDict", _postprocess_dict())

    assert workspace.manifest_digest() == before
    workspace.verify_seal(seal, plan)
    with pytest.raises(WorkspaceSafetyError):
        workspace.write_postprocess_config("system/analysisDict", _postprocess_dict())


def test_postprocess_config_hash_binding_detects_out_of_band_change(tmp_path, graph_path):
    state = make_state()
    plan = make_plan(state.intake)
    llm = ScriptedLLM(
        [
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
                rationale="Verify mesh.",
            ),
            FinishPreviewAction(type="finish_preview", plan=plan, rationale="Seal."),
        ]
    )
    tools = FakeOpenFOAMTools(
        mesh_results={
            "checkMesh": [tool_result("checkMesh", success=True, stdout=mesh_ok_log())]
        }
    )
    prep_agent = CFDEngineeringAgent(
        llm,
        workspace=tmp_path,
        capability_db=graph_path,
        tools=tools,
        policy=EngineeringPolicy(max_agent_steps=12),
    )
    prep_agent.prepare(state, native_execution=True)
    assert state.current_state == State.MESH_READY

    from openfoam_agent.postprocessing.agent import CFDPostProcessingAgent
    from openfoam_agent.schemas.postprocessing import PostProcessingReport
    from openfoam_agent.schemas.simulation import RuntimeReport, SimulationAttempt
    from openfoam_agent.tools.parsers import parse_runtime_log

    runtime_result = parse_runtime_log("Time = 1s\nEnd\n", return_code=0)
    state.runtime_report = RuntimeReport(
        success=True,
        attempts=[SimulationAttempt(attempt=1, result=runtime_result)],
        final_result=runtime_result,
    )
    state.simulation = runtime_result
    state.current_state = State.EXECUTION_DONE

    post_llm = ScriptedLLM(
        [
            WritePostProcessConfigAction(
                type="write_postprocess_config",
                path="postprocessConfig/wakeAnalysisDict",
                content=_postprocess_dict(),
                rationale="Write config.",
            ),
            RunFoamPostProcessAction(
                type="run_foam_postprocess",
                dictionary_path="postprocessConfig/wakeAnalysisDict",
                time_selection="all",
                use_solver_context=True,
                rationale="Execute config.",
            ),
            FinishPostProcessingAction(
                type="finish_postprocessing",
                summary="Finish after tamper test.",
                limitations=[],
                rationale="Finish.",
            ),
        ]
    )
    post = CFDPostProcessingAgent(
        post_llm,
        workspace=tmp_path,
        tools=tools,
        policy=PostProcessingPolicy(max_steps=3, max_native_commands=2),
    )

    original_dispatch = post._dispatch
    calls = {"count": 0}

    def tampering_dispatch(state_arg, action, *, step):
        event, terminal = original_dispatch(state_arg, action, step=step)
        calls["count"] += 1
        if calls["count"] == 1:
            path = tmp_path / "case" / "postprocessConfig" / "wakeAnalysisDict"
            path.write_text(_postprocess_dict().replace("lRef 1;", "lRef 2;"), encoding="utf-8")
        return event, terminal

    post._dispatch = tampering_dispatch  # type: ignore[method-assign]
    post.run(state)

    assert state.current_state == State.RESULT_REVIEW_REQUIRED
    assert state.postprocessing_report is not None
    assert isinstance(state.postprocessing_report, PostProcessingReport)
    run_events = [
        event for event in state.postprocessing_events if event.action_type == "run_foam_postprocess"
    ]
    assert run_events and not run_events[0].success
    assert "changed after the agent-authored hash" in run_events[0].summary
    assert not tools.postprocess_calls


def test_runtime_success_automatically_continues_into_postprocessing(tmp_path, graph_path):
    state = make_state()
    plan = make_plan(state.intake)
    llm = ScriptedLLM(
        [
            SearchCapabilitiesAction(
                type="search_capabilities",
                query="incompressibleFluid",
                rationale="Observe solver capability evidence.",
            ),
            WriteCaseFileAction(
                type="write_case_file",
                path="system/controlDict",
                content=control_dict(),
                rationale="Write solver control.",
            ),
            RunMeshCommandAction(
                type="run_mesh_command",
                command="checkMesh",
                rationale="Verify current mesh.",
            ),
            FinishPreviewAction(
                type="finish_preview",
                plan=plan,
                rationale="Seal the validated case.",
            ),
            WritePostProcessConfigAction(
                type="write_postprocess_config",
                path="postprocessConfig/wakeAnalysisDict",
                content=_postprocess_dict(),
                rationale="Configure force coefficients without modifying solve inputs.",
            ),
            RunFoamPostProcessAction(
                type="run_foam_postprocess",
                dictionary_path="postprocessConfig/wakeAnalysisDict",
                time_selection="all",
                use_solver_context=True,
                rationale="Evaluate post-processing functions over all saved times.",
            ),
            AnalyzeForceCoefficientsAction(
                type="analyze_force_coefficients",
                coefficient_path="postProcessing/forceCoeffsObstacle/0/coefficient.dat",
                dictionary_path="postprocessConfig/wakeAnalysisDict",
                discard_fraction=0.2,
                rationale="Compute force statistics and shedding frequency from real output.",
            ),
            FinishPostProcessingAction(
                type="finish_postprocessing",
                summary="Wake post-processing evidence collected from native outputs.",
                limitations=["Mesh/time-step independence has not been established."],
                rationale="Finish with observed post-processing evidence.",
            ),
        ]
    )
    tools = FakeOpenFOAMTools(
        mesh_results={
            "checkMesh": [tool_result("checkMesh", success=True, stdout=mesh_ok_log())]
        },
        foam_runs=[
            ToolResult(
                success=True,
                command=["foamRun"],
                return_code=0,
                stdout="Time = 20s\nCourant Number mean: 0.01 max: 0.03\nEnd\n",
            )
        ],
        postprocess_runs=[
            ToolResult(
                success=True,
                command=["foamPostProcess"],
                return_code=0,
                stdout="Time = 20s\nEnd\n",
            )
        ],
    )

    prep_agent = CFDEngineeringAgent(
        llm,
        workspace=tmp_path,
        capability_db=graph_path,
        tools=tools,
        policy=EngineeringPolicy(max_agent_steps=12),
    )
    prep_agent.prepare(state, native_execution=True)
    assert state.current_state == State.MESH_READY

    case_dir = tmp_path / "case"
    coeff_path = case_dir / "postProcessing" / "forceCoeffsObstacle" / "0" / "coefficient.dat"
    coeff_path.parent.mkdir(parents=True, exist_ok=True)
    coeff_path.write_text(_force_coeff_text(), encoding="utf-8")
    vort_path = case_dir / "20" / "vorticity"
    vort_path.parent.mkdir(parents=True, exist_ok=True)
    vort_path.write_text("FoamFile {}\ninternalField uniform (0 0 0);\n", encoding="utf-8")

    state.approve_solve()
    workflow = CFDWorkflow(
        llm=llm,
        capability_db=graph_path,
        workspace=tmp_path,
        openfoam_tools=tools,
        engineering_policy=EngineeringPolicy(max_agent_steps=12),
        postprocessing_policy=PostProcessingPolicy(max_steps=10, max_native_commands=3),
        native_execution=True,
    )
    workflow.run(state)

    assert state.current_state == State.RESULT_REVIEW_REQUIRED
    assert state.runtime_report is not None and state.runtime_report.success
    assert len(state.runtime_report.attempts) == 1
    assert state.postprocessing_report is not None and state.postprocessing_report.success
    assert state.postprocessing_report.force_analysis is not None
    assert state.postprocessing_report.force_analysis.shedding_frequency == pytest.approx(0.2, rel=0.01)
    assert state.postprocessing_report.force_analysis.strouhal_number == pytest.approx(0.2, rel=0.01)
    artifact_paths = {item.path for item in state.postprocessing_report.artifacts}
    assert "20/vorticity" in artifact_paths
    assert "postProcessing/forceCoeffsObstacle/0/coefficient.dat" in artifact_paths
    assert len(tools.postprocess_calls) == 1
    assert tools.postprocess_calls[0]["solver"] == "incompressibleFluid"
    assert any(item["to"] == State.POSTPROCESSING.value for item in state.history)


def test_postprocessing_llm_failure_does_not_erase_successful_solver_result(tmp_path, graph_path):
    state = make_state()
    plan = make_plan(state.intake)
    prep_llm = ScriptedLLM(
        [
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
                rationale="Verify mesh.",
            ),
            FinishPreviewAction(type="finish_preview", plan=plan, rationale="Seal."),
        ]
    )
    tools = FakeOpenFOAMTools(
        mesh_results={
            "checkMesh": [tool_result("checkMesh", success=True, stdout=mesh_ok_log())]
        },
        foam_runs=[
            ToolResult(
                success=True,
                command=["foamRun"],
                return_code=0,
                stdout="Time = 1s\nEnd\n",
            )
        ],
    )
    prep_agent = CFDEngineeringAgent(
        prep_llm,
        workspace=tmp_path,
        capability_db=graph_path,
        tools=tools,
        policy=EngineeringPolicy(max_agent_steps=12),
    )
    prep_agent.prepare(state, native_execution=True)
    state.approve_solve()

    class FailingPostLLM:
        def generate(self, *args, **kwargs):
            del args, kwargs
            raise RuntimeError("synthetic post-processing model outage")

    workflow = CFDWorkflow(
        llm=FailingPostLLM(),
        capability_db=graph_path,
        workspace=tmp_path,
        openfoam_tools=tools,
        postprocessing_policy=PostProcessingPolicy(max_steps=4, max_native_commands=2),
        native_execution=True,
    )
    workflow.run(state)

    assert state.current_state == State.RESULT_REVIEW_REQUIRED
    assert state.runtime_report is not None and state.runtime_report.success
    assert state.postprocessing_report is not None
    assert not state.postprocessing_report.success
    assert any("synthetic post-processing model outage" in item for item in state.postprocessing_report.limitations)


def test_postprocess_config_reuses_content_safety_and_result_reads_are_read_only(tmp_path):
    workspace = CaseWorkspace(tmp_path)
    with pytest.raises(WorkspaceSafetyError):
        workspace.write_postprocess_config(
            "postprocessConfig/unsafeDict",
            "functions { x { type coded; #codeStream {}; } }\n",
        )
    with pytest.raises(WorkspaceSafetyError):
        workspace.resolve_result_path("system/controlDict")
    with pytest.raises(WorkspaceSafetyError):
        workspace.resolve_result_path("postprocessConfig/wakeAnalysisDict")


def test_result_text_read_is_bounded(tmp_path):
    workspace = CaseWorkspace(tmp_path)
    path = workspace.case_dir / "20" / "vorticity"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("x" * 100_000, encoding="utf-8")

    text = workspace.read_result_text("20/vorticity", max_chars=32)

    assert text.startswith("x" * 32)
    assert text.endswith("... [truncated]")
    assert len(text) < 100
