from __future__ import annotations

from io import StringIO

from openfoam_agent.engineering import CFDEngineeringAgent, EngineeringPolicy
from openfoam_agent.progress import (
    CLIProgressReporter,
    ProgressEvent,
    ProgressImportance,
    SolverProgressTracker,
)
from openfoam_agent.schemas.engineering import (
    FinishPreviewAction,
    ReadCaseFileAction,
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


def test_cli_progress_normal_filters_verbose_events():
    stream = StringIO()
    reporter = CLIProgressReporter("normal", stream=stream)
    reporter.emit(ProgressEvent(phase="engineering", message="major event"))
    reporter.emit(
        ProgressEvent(
            phase="engineering",
            message="verbose event",
            importance=ProgressImportance.VERBOSE,
        )
    )
    text = stream.getvalue()
    assert "major event" in text
    assert "verbose event" not in text


def test_cli_progress_quiet_emits_nothing():
    stream = StringIO()
    reporter = CLIProgressReporter("quiet", stream=stream)
    reporter.emit(ProgressEvent(phase="engineering", message="hidden"))
    assert stream.getvalue() == ""


def test_verbose_progress_includes_verbose_events():
    stream = StringIO()
    reporter = CLIProgressReporter("verbose", stream=stream)
    reporter.emit(
        ProgressEvent(
            phase="engineering",
            message="read case file",
            importance=ProgressImportance.VERBOSE,
            step=7,
            limit=120,
        )
    )
    text = stream.getvalue()
    assert "[ENGINEERING 07/120]" in text
    assert "read case file" in text


def test_confirm_engineering_emits_live_actions_and_checkmesh_metrics(tmp_path, graph_path):
    state = make_state()
    plan = make_plan(state.intake)
    llm = ScriptedLLM(
        [
            SearchCapabilitiesAction(
                type="search_capabilities",
                query="incompressibleFluid transient",
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
                rationale="Verify mesh.",
            ),
            FinishPreviewAction(
                type="finish_preview",
                plan=plan,
                rationale="Seal case.",
            ),
        ]
    )
    tools = FakeOpenFOAMTools(
        mesh_results={
            "checkMesh": [tool_result("checkMesh", success=True, stdout=mesh_ok_log(cells=3766))]
        }
    )
    stream = StringIO()
    reporter = CLIProgressReporter("normal", stream=stream)
    agent = CFDEngineeringAgent(
        llm,
        workspace=tmp_path,
        capability_db=graph_path,
        tools=tools,
        policy=EngineeringPolicy(max_agent_steps=10, hard_max_agent_steps=10),
        progress=reporter,
    )

    agent.prepare(state, native_execution=True)

    assert state.current_state == State.MESH_READY
    text = stream.getvalue()
    assert "autonomous engineering 시작" in text
    assert "capability graph 조회" in text
    assert "case 파일 작성: system/controlDict" in text
    assert "mesh command 실행: checkMesh" in text
    assert "cells=3766" in text
    assert "Engineering plan accepted and case sealed." in text
    # Model rationale must never be surfaced as progress text.
    assert "Observe solver capability" not in text
    assert "Seal case" not in text


def test_normal_progress_hides_read_case_file_action(tmp_path, graph_path):
    state = make_state()
    plan = make_plan(state.intake)
    llm = ScriptedLLM(
        [
            SearchCapabilitiesAction(
                type="search_capabilities", query="incompressibleFluid", rationale="Observe."
            ),
            WriteCaseFileAction(
                type="write_case_file",
                path="system/controlDict",
                content=control_dict(),
                rationale="Write.",
            ),
            ReadCaseFileAction(
                type="read_case_file",
                path="system/controlDict",
                rationale="Inspect.",
            ),
            RunMeshCommandAction(
                type="run_mesh_command", command="checkMesh", rationale="Verify."
            ),
            FinishPreviewAction(type="finish_preview", plan=plan, rationale="Finish."),
        ]
    )
    tools = FakeOpenFOAMTools(
        mesh_results={"checkMesh": [tool_result("checkMesh", success=True, stdout=mesh_ok_log())]}
    )
    stream = StringIO()
    agent = CFDEngineeringAgent(
        llm,
        workspace=tmp_path,
        capability_db=graph_path,
        tools=tools,
        policy=EngineeringPolicy(max_agent_steps=10, hard_max_agent_steps=10),
        progress=CLIProgressReporter("normal", stream=stream),
    )
    agent.prepare(state, native_execution=True)
    assert "case 파일 읽기" not in stream.getvalue()


def test_solver_progress_parses_of13_time_suffix_and_courant():
    stream = StringIO()
    reporter = CLIProgressReporter("normal", stream=stream)
    tracker = SolverProgressTracker(
        reporter,
        attempt=1,
        attempt_limit=9,
        normal_interval_seconds=0.0,
    )
    tracker.feed("Courant Number mean: 0.02095074 max: 0.034800772\n")
    tracker.feed("Time = 20s\n")
    text = stream.getvalue()
    assert "foamRun 진행" in text
    assert "Time=20" in text
    assert "CoMax=0.0348008" in text
    assert "attempt=1/9" in text


def test_solver_progress_residuals_are_verbose_only():
    line = (
        "smoothSolver: Solving for Ux, Initial residual = 1.9e-05, "
        "Final residual = 2.2e-10, No Iterations 2\n"
    )
    normal_stream = StringIO()
    SolverProgressTracker(
        CLIProgressReporter("normal", stream=normal_stream),
        attempt=1,
        attempt_limit=2,
    ).feed(line)
    assert normal_stream.getvalue() == ""

    verbose_stream = StringIO()
    SolverProgressTracker(
        CLIProgressReporter("verbose", stream=verbose_stream),
        attempt=1,
        attempt_limit=2,
    ).feed(line)
    assert "linear solve: Ux" in verbose_stream.getvalue()


def test_safe_runner_progress_callback_cannot_abort_native_process(tmp_path):
    from openfoam_agent.tools.safe_runner import SafeRunner

    install = tmp_path / "OpenFOAM-13"
    bin_dir = install / "bin"
    bin_dir.mkdir(parents=True)
    executable = bin_dir / "checkMesh"
    executable.write_text("#!/bin/sh\necho 'Mesh OK'\n", encoding="utf-8")
    executable.chmod(0o755)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    runner = SafeRunner(
        workspace_root=workspace,
        trusted_executable_roots=[install],
        base_env={
            "PATH": str(bin_dir),
            "WM_PROJECT_DIR": str(install),
            "HOME": str(tmp_path),
        },
    )

    def broken_callback(line: str) -> None:
        del line
        raise RuntimeError("renderer failure must be observational only")

    result = runner.run(["checkMesh"], cwd=workspace, output_callback=broken_callback)
    assert result.success
    assert "Mesh OK" in result.stdout


def test_runtime_orchestrator_emits_of13_live_progress(tmp_path, graph_path):
    from openfoam_agent.runtime import RuntimeOrchestrator
    from openfoam_agent.schemas.common import ToolResult
    from openfoam_agent.schemas.simulation import RuntimePolicy

    state = make_state()
    plan = make_plan(state.intake)
    llm = ScriptedLLM(
        [
            SearchCapabilitiesAction(
                type="search_capabilities", query="incompressibleFluid", rationale="Observe."
            ),
            WriteCaseFileAction(
                type="write_case_file",
                path="system/controlDict",
                content=control_dict(),
                rationale="Write.",
            ),
            RunMeshCommandAction(
                type="run_mesh_command", command="checkMesh", rationale="Verify."
            ),
            FinishPreviewAction(type="finish_preview", plan=plan, rationale="Seal."),
        ]
    )
    tools = FakeOpenFOAMTools(
        mesh_results={"checkMesh": [tool_result("checkMesh", success=True, stdout=mesh_ok_log())]},
        foam_runs=[
            ToolResult(
                success=True,
                command=["foamRun"],
                return_code=0,
                stdout=(
                    "Courant Number mean: 0.02095074 max: 0.034800772\n"
                    "Time = 20s\n"
                    "End\n"
                ),
            )
        ],
    )
    stream = StringIO()
    reporter = CLIProgressReporter("normal", stream=stream)
    agent = CFDEngineeringAgent(
        llm,
        workspace=tmp_path,
        capability_db=graph_path,
        tools=tools,
        policy=EngineeringPolicy(max_agent_steps=10, hard_max_agent_steps=10),
        progress=reporter,
    )
    agent.prepare(state, native_execution=True)
    state.approve_solve()
    RuntimeOrchestrator(
        tools,
        agent,
        RuntimePolicy(max_attempts=2, solver_timeout_seconds=30),
        progress=reporter,
    ).run(state)

    assert state.current_state == State.EXECUTION_DONE
    text = stream.getvalue()
    assert "foamRun attempt 1/2" in text
    assert "Time=20" in text
    assert "maxCo=0.0348008" in text
    assert "foamRun attempt 1 완료" in text


def test_postprocessing_emits_live_progress(tmp_path, graph_path):
    from openfoam_agent.postprocessing import CFDPostProcessingAgent, PostProcessingPolicy
    from openfoam_agent.schemas.common import ToolResult
    from openfoam_agent.schemas.postprocessing import (
        FinishPostProcessingAction,
        RunFoamPostProcessAction,
        WritePostProcessConfigAction,
    )
    from openfoam_agent.schemas.simulation import RuntimeReport, SimulationAttempt
    from openfoam_agent.tools.parsers import parse_runtime_log

    state = make_state()
    plan = make_plan(state.intake)
    prep_llm = ScriptedLLM(
        [
            SearchCapabilitiesAction(
                type="search_capabilities", query="incompressibleFluid", rationale="Observe."
            ),
            WriteCaseFileAction(
                type="write_case_file",
                path="system/controlDict",
                content=control_dict(),
                rationale="Write.",
            ),
            RunMeshCommandAction(
                type="run_mesh_command", command="checkMesh", rationale="Verify."
            ),
            FinishPreviewAction(type="finish_preview", plan=plan, rationale="Seal."),
        ]
    )
    tools = FakeOpenFOAMTools(
        mesh_results={"checkMesh": [tool_result("checkMesh", success=True, stdout=mesh_ok_log())]},
        postprocess_runs=[
            ToolResult(
                success=True,
                command=["foamPostProcess"],
                return_code=0,
                stdout="Time = 20s\nEnd\n",
            )
        ],
    )
    prep = CFDEngineeringAgent(
        prep_llm,
        workspace=tmp_path,
        capability_db=graph_path,
        tools=tools,
        policy=EngineeringPolicy(max_agent_steps=10, hard_max_agent_steps=10),
    )
    prep.prepare(state, native_execution=True)
    result = parse_runtime_log("Time = 20s\nEnd\n", return_code=0)
    state.runtime_report = RuntimeReport(
        success=True,
        attempts=[SimulationAttempt(attempt=1, result=result)],
        final_result=result,
    )
    state.simulation = result
    state.current_state = State.EXECUTION_DONE

    config = """FoamFile\n{\n    format ascii;\n    class dictionary;\n    object postProcessDict;\n}\nfunctions {}\n"""
    post_llm = ScriptedLLM(
        [
            WritePostProcessConfigAction(
                type="write_postprocess_config",
                path="postprocessConfig/basicDict",
                content=config,
                rationale="Write.",
            ),
            RunFoamPostProcessAction(
                type="run_foam_postprocess",
                dictionary_path="postprocessConfig/basicDict",
                time_selection="latest",
                use_solver_context=True,
                rationale="Run.",
            ),
            FinishPostProcessingAction(
                type="finish_postprocessing",
                summary="Postprocess review ready.",
                limitations=["No quantitative coefficient analysis requested."],
                scientific_confidence="low",
                review_reasons=["Human review remains required."],
                recommended_human_checks=["Inspect fields."],
                rationale="Finish.",
            ),
        ]
    )
    stream = StringIO()
    post = CFDPostProcessingAgent(
        post_llm,
        workspace=tmp_path,
        tools=tools,
        policy=PostProcessingPolicy(max_steps=5, max_native_commands=2),
        progress=CLIProgressReporter("normal", stream=stream),
    )
    post.run(state)

    assert state.current_state == State.RESULT_REVIEW_REQUIRED
    text = stream.getvalue()
    assert "자동 post-processing 시작" in text
    assert "post-processing config 작성: postprocessConfig/basicDict" in text
    assert "foamPostProcess 실행: postprocessConfig/basicDict" in text
    assert "post-processing 결과 최종화" in text


def test_feedback_review_emits_progress_without_exposing_model_rationale(tmp_path, graph_path):
    from openfoam_agent.review import CFDFeedbackReviewAgent
    from openfoam_agent.schemas.feedback import FeedbackAssessment

    state = make_state()
    plan = make_plan(state.intake)
    prep_llm = ScriptedLLM(
        [
            SearchCapabilitiesAction(
                type="search_capabilities", query="incompressibleFluid", rationale="Observe."
            ),
            WriteCaseFileAction(
                type="write_case_file",
                path="system/controlDict",
                content=control_dict(),
                rationale="Write.",
            ),
            RunMeshCommandAction(
                type="run_mesh_command", command="checkMesh", rationale="Verify."
            ),
            FinishPreviewAction(type="finish_preview", plan=plan, rationale="Seal."),
        ]
    )
    tools = FakeOpenFOAMTools(
        mesh_results={"checkMesh": [tool_result("checkMesh", success=True, stdout=mesh_ok_log())]}
    )
    CFDEngineeringAgent(
        prep_llm,
        workspace=tmp_path,
        capability_db=graph_path,
        tools=tools,
        policy=EngineeringPolicy(max_agent_steps=10, hard_max_agent_steps=10),
    ).prepare(state, native_execution=True)

    assessment = FeedbackAssessment(
        diagnosis_summary="The wake observation warrants a bounded case revision.",
        hypotheses=[
            {
                "hypothesis": "Wake resolution may be insufficient.",
                "rationale": "This is a hypothesis, not a proven cause.",
                "evidence_to_check": ["mesh", "Cl"],
            }
        ],
        proposed_changes=[
            {
                "area": "wake mesh",
                "change": "Inspect and refine if evidence supports it.",
                "rationale": "Address the human observation.",
            }
        ],
        expected_cost="moderate_increase",
        requires_case_revision=True,
        requires_intake_revision=False,
        intake_revision_reason="",
        review_limitations=["Diagnosis remains provisional."],
    )
    stream = StringIO()
    review = CFDFeedbackReviewAgent(
        ScriptedLLM([assessment]),
        progress=CLIProgressReporter("normal", stream=stream),
    )
    review.review(state, "후류가 너무 대칭적으로 보인다")

    assert state.current_state == State.REVISION_READY
    text = stream.getvalue()
    assert "human feedback 진단 시작" in text
    assert "feedback assessment 완료" in text
    assert "moderate_increase" in text
    assert "This is a hypothesis" not in text
