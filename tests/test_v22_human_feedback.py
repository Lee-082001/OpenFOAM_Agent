from __future__ import annotations

from pathlib import Path

import pytest

from openfoam_agent.engineering import CFDEngineeringAgent, EngineeringPolicy
from openfoam_agent.llm.openai_client import validate_structured_output_schema
from openfoam_agent.review import CFDFeedbackReviewAgent
from openfoam_agent.schemas.common import ToolResult
from openfoam_agent.schemas.engineering import (
    BlockAction,
    FinishPreviewAction,
    ReadCaseFileAction,
    RunMeshCommandAction,
    SearchCapabilitiesAction,
    WriteCaseFileAction,
)
from openfoam_agent.schemas.feedback import FeedbackAssessment
from openfoam_agent.schemas.postprocessing import PostProcessingReport
from openfoam_agent.schemas.simulation import RuntimeReport, SimulationAttempt
from openfoam_agent.tools.parsers import parse_runtime_log
from openfoam_agent.tools.workspace import WorkspaceSafetyError
from openfoam_agent.workflow.state import CFDState
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


def _assessment(*, intake_revision: bool = False, case_revision: bool = True) -> FeedbackAssessment:
    return FeedbackAssessment(
        diagnosis_summary="The human-observed wake behavior should be investigated before accepting the result.",
        hypotheses=[
            {
                "hypothesis": "The wake resolution or integration duration may be insufficient.",
                "rationale": "A numerically completed run does not prove resolved periodic shedding.",
                "evidence_to_check": ["near-wake mesh", "Cl time history", "saved periods"],
            }
        ],
        proposed_changes=(
            []
            if intake_revision
            else [
                {
                    "area": "wake resolution",
                    "change": "Refine the near wake and extend the transient observation window if evidence supports it.",
                    "rationale": "Resolve the reported abnormal shedding behavior with better spatial/temporal evidence.",
                }
            ]
        ),
        expected_cost="moderate_increase",
        requires_case_revision=case_revision and not intake_revision,
        requires_intake_revision=intake_revision,
        intake_revision_reason=(
            "The feedback explicitly changes the confirmed Reynolds number."
            if intake_revision
            else ""
        ),
        review_limitations=["Root cause is a hypothesis until revision tools inspect the case."],
    )


def _prepare_mesh_ready(tmp_path: Path, graph_path: Path, extra_actions=()):
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
                rationale="Write initial control.",
            ),
            RunMeshCommandAction(
                type="run_mesh_command",
                command="checkMesh",
                rationale="Verify initial mesh.",
            ),
            FinishPreviewAction(type="finish_preview", plan=plan, rationale="Seal initial case."),
            *extra_actions,
        ]
    )
    tools = FakeOpenFOAMTools(
        mesh_results={
            "checkMesh": [
                tool_result("checkMesh", success=True, stdout=mesh_ok_log()),
                tool_result("checkMesh", success=True, stdout=mesh_ok_log(cells=2400)),
            ]
        }
    )
    agent = CFDEngineeringAgent(
        llm,
        workspace=tmp_path,
        capability_db=graph_path,
        tools=tools,
        policy=EngineeringPolicy(max_agent_steps=20, hard_max_agent_steps=20),
    )
    agent.prepare(state, native_execution=True)
    assert state.current_state == State.MESH_READY
    return state, plan, llm, tools, agent


def _mark_result_review(state: CFDState) -> None:
    result = parse_runtime_log(
        "Time = 20s\nCourant Number mean: 0.01 max: 0.03\nEnd\n",
        return_code=0,
    )
    state.simulation = result
    state.runtime_report = RuntimeReport(
        success=True,
        attempts=[SimulationAttempt(attempt=1, result=result)],
        final_result=result,
    )
    state.postprocessing_report = PostProcessingReport(
        success=True,
        summary="Post-processing evidence exists but human review remains required.",
        scientific_confidence="low",
        review_reasons=["The observed wake was reported as suspicious by the human reviewer."],
        recommended_human_checks=["Inspect vorticity animation and Cl periodicity."],
    )
    state.current_state = State.RESULT_REVIEW_REQUIRED


def test_feedback_assessment_schema_is_strict_output_compatible():
    validate_structured_output_schema(FeedbackAssessment)
    schema = FeedbackAssessment.model_json_schema()
    assert schema.get("type") == "object"
    assert "oneOf" not in str(schema)


def test_result_feedback_creates_proposal_without_mutating_sealed_case(tmp_path, graph_path):
    state, _, _, _, _ = _prepare_mesh_ready(tmp_path, graph_path)
    _mark_result_review(state)
    before_manifest = state.case_seal.manifest_sha256
    before_bytes = (tmp_path / "case" / "system" / "controlDict").read_bytes()

    review = CFDFeedbackReviewAgent(ScriptedLLM([_assessment()]))
    review.review(state, "후류가 너무 대칭이고 vortex shedding이 이상해 보여")

    assert state.current_state == State.REVISION_READY
    assert state.case_seal.manifest_sha256 == before_manifest
    assert (tmp_path / "case" / "system" / "controlDict").read_bytes() == before_bytes
    assert len(state.human_feedback) == 1
    assert state.human_feedback[0].status == "revision_proposed"
    assert state.active_revision_proposal is not None
    assert state.active_revision_proposal.baseline_manifest_sha256 == before_manifest
    assert state.active_revision_proposal.requires_case_revision


def test_confirmed_revision_changes_case_records_diff_and_requires_new_solve(tmp_path, graph_path):
    revised_plan_holder = {}
    # Build the initial state first so the revised plan can reuse the confirmed intake digest.
    state = make_state()
    initial_plan = make_plan(state.intake)
    revised_plan = make_plan(state.intake)
    revised_plan.mesh_strategy = "Human-feedback revision: refined near wake with extended transient observation."
    revised_plan.assumptions = [
        "The revision keeps the confirmed Reynolds number and increases wake resolution/observation duration."
    ]
    revised_plan_holder["plan"] = revised_plan

    llm = ScriptedLLM(
        [
            SearchCapabilitiesAction(type="search_capabilities", query="incompressibleFluid", rationale="Observe capability."),
            WriteCaseFileAction(type="write_case_file", path="system/controlDict", content=control_dict(), rationale="Write initial control."),
            RunMeshCommandAction(type="run_mesh_command", command="checkMesh", rationale="Verify initial mesh."),
            FinishPreviewAction(type="finish_preview", plan=initial_plan, rationale="Seal initial case."),
            _assessment(),
            ReadCaseFileAction(type="read_case_file", path="system/controlDict", rationale="Inspect the current transient duration before revising."),
            WriteCaseFileAction(
                type="write_case_file",
                path="system/controlDict",
                content=control_dict().replace("endTime 10;", "endTime 40;"),
                rationale="Extend the transient observation window based on human feedback.",
            ),
            RunMeshCommandAction(type="run_mesh_command", command="checkMesh", rationale="Re-establish mesh evidence after case-input revision."),
            FinishPreviewAction(type="finish_preview", plan=revised_plan, rationale="Seal the revised case."),
        ]
    )
    tools = FakeOpenFOAMTools(
        mesh_results={
            "checkMesh": [
                tool_result("checkMesh", success=True, stdout=mesh_ok_log()),
                tool_result("checkMesh", success=True, stdout=mesh_ok_log(cells=2400)),
            ]
        }
    )
    agent = CFDEngineeringAgent(
        llm,
        workspace=tmp_path,
        capability_db=graph_path,
        tools=tools,
        policy=EngineeringPolicy(max_agent_steps=20, hard_max_agent_steps=20),
    )
    agent.prepare(state, native_execution=True)
    _mark_result_review(state)
    old_runtime = state.runtime_report
    old_manifest = state.case_seal.manifest_sha256
    old_vorticity = tmp_path / "case" / "20" / "vorticity"
    old_vorticity.parent.mkdir(parents=True, exist_ok=True)
    old_vorticity.write_text("old-run-vorticity\n", encoding="utf-8")
    old_coeff = tmp_path / "case" / "postProcessing" / "forceCoeffs" / "0" / "coefficient.dat"
    old_coeff.parent.mkdir(parents=True, exist_ok=True)
    old_coeff.write_text("# old run\n", encoding="utf-8")

    CFDFeedbackReviewAgent(llm).review(state, "와류가 거의 안 보이고 계산 시간이 너무 짧아 보인다")
    assert state.current_state == State.REVISION_READY
    agent.revise_from_feedback(state, native_execution=True)

    assert state.current_state == State.MESH_READY
    assert not state.solve_approved
    assert state.runtime_report is None
    assert state.postprocessing_report is None
    assert old_runtime is not None
    assert state.case_seal.manifest_sha256 != old_manifest
    assert len(state.revision_history) == 1
    record = state.revision_history[0]
    assert record.before_manifest_sha256 == old_manifest
    assert record.after_manifest_sha256 == state.case_seal.manifest_sha256
    assert any(
        item.path == "system/controlDict" and item.change == "modified"
        for item in record.file_changes
    )
    assert state.human_feedback[0].status == "awaiting_rerun"
    assert state.active_revision_proposal is None
    assert record.archive_path == "revision-history/rev-0001"
    assert not old_vorticity.exists()
    assert not old_coeff.exists()
    assert (tmp_path / record.archive_path / "baseline_inputs" / "system" / "controlDict").is_file()
    assert (tmp_path / record.archive_path / "case_outputs" / "20" / "vorticity").is_file()
    assert (tmp_path / record.archive_path / "case_outputs" / "postProcessing" / "forceCoeffs" / "0" / "coefficient.dat").is_file()
    # Revision gets a fresh resource accounting round even though prior events are retained.
    assert state.engineering_round_start_index > 0
    current_round = state.engineering_events[state.engineering_round_start_index :]
    assert [event.action_type for event in current_round] == [
        "read_case_file",
        "write_case_file",
        "run_mesh_command",
        "finish_preview",
    ]


def test_revision_proposal_is_hash_bound_to_case_before_any_edit(tmp_path, graph_path):
    state, _, _, _, agent = _prepare_mesh_ready(tmp_path, graph_path)
    _mark_result_review(state)
    CFDFeedbackReviewAgent(ScriptedLLM([_assessment()])).review(state, "mesh가 이상해 보여")
    path = tmp_path / "case" / "system" / "controlDict"
    path.write_text(path.read_text() + "// tampered\n", encoding="utf-8")

    with pytest.raises(WorkspaceSafetyError):
        agent.revise_from_feedback(state, native_execution=True)

    assert not state.revision_history


def test_required_case_revision_cannot_finish_with_unchanged_manifest(tmp_path, graph_path):
    state = make_state()
    plan = make_plan(state.intake)
    llm = ScriptedLLM(
        [
            SearchCapabilitiesAction(type="search_capabilities", query="incompressibleFluid", rationale="Observe capability."),
            WriteCaseFileAction(type="write_case_file", path="system/controlDict", content=control_dict(), rationale="Write control."),
            RunMeshCommandAction(type="run_mesh_command", command="checkMesh", rationale="Verify mesh."),
            FinishPreviewAction(type="finish_preview", plan=plan, rationale="Seal."),
            _assessment(),
            FinishPreviewAction(type="finish_preview", plan=plan, rationale="Incorrectly claim revision without changing inputs."),
            BlockAction(type="block", reason="No justified revision was applied.", needs_user_input=False, rationale="Stop rather than fake progress."),
        ]
    )
    tools = FakeOpenFOAMTools(mesh_results={"checkMesh": [tool_result("checkMesh", success=True, stdout=mesh_ok_log())]})
    agent = CFDEngineeringAgent(
        llm,
        workspace=tmp_path,
        capability_db=graph_path,
        tools=tools,
        policy=EngineeringPolicy(max_agent_steps=4, hard_max_agent_steps=4),
    )
    agent.prepare(state, native_execution=True)
    _mark_result_review(state)
    CFDFeedbackReviewAgent(llm).review(state, "결과가 이상하다")
    agent.revise_from_feedback(state, native_execution=True)

    assert state.current_state == State.ENGINEERING_BLOCKED
    assert not state.revision_history
    rejected = [event for event in state.engineering_events if event.action_type == "finish_preview" and not event.success]
    assert rejected
    assert "manifest is unchanged" in rejected[-1].output_excerpt


def test_feedback_that_changes_confirmed_fact_routes_back_to_intake_review(tmp_path, graph_path):
    state, _, _, _, _ = _prepare_mesh_ready(tmp_path, graph_path)
    _mark_result_review(state)
    review = CFDFeedbackReviewAgent(
        ScriptedLLM([_assessment(intake_revision=True, case_revision=False)])
    )
    review.review(state, "Reynolds 수를 500으로 바꿔서 다시 해줘")

    assert state.current_state == State.ENGINEERING_REVIEW_REQUIRED
    assert state.active_revision_proposal is not None
    assert state.active_revision_proposal.requires_intake_revision
    assert state.human_feedback[0].status == "revision_proposed"


def test_only_human_acceptance_transitions_reviewed_result_to_complete(tmp_path, graph_path):
    state, _, _, _, _ = _prepare_mesh_ready(tmp_path, graph_path)
    _mark_result_review(state)
    state.accept_result()
    assert state.current_state == State.COMPLETE

    with pytest.raises(ValueError):
        state.accept_result()


def test_user_can_reject_revision_without_changing_case(tmp_path, graph_path):
    state, _, _, _, _ = _prepare_mesh_ready(tmp_path, graph_path)
    _mark_result_review(state)
    before = state.case_seal.manifest_sha256
    CFDFeedbackReviewAgent(ScriptedLLM([_assessment()])).review(state, "후류 결과가 이상하다")
    assert state.current_state == State.REVISION_READY
    state.reject_revision()
    assert state.current_state == State.RESULT_REVIEW_REQUIRED
    assert state.active_revision_proposal is None
    assert state.case_seal.manifest_sha256 == before
    assert state.human_feedback[-1].status == "proposal_rejected"
    assert len(state.revision_proposals) == 1


def test_review_model_failure_returns_to_original_gate_without_mutating_case(tmp_path, graph_path):
    state, _, _, _, _ = _prepare_mesh_ready(tmp_path, graph_path)
    _mark_result_review(state)
    before_manifest = state.case_seal.manifest_sha256

    review = CFDFeedbackReviewAgent(ScriptedLLM([]))
    with pytest.raises(AssertionError, match="ScriptedLLM exhausted"):
        review.review(state, "와류가 이상해 보이는데 원인을 확인해줘")

    assert state.current_state == State.RESULT_REVIEW_REQUIRED
    assert state.case_seal.manifest_sha256 == before_manifest
    assert state.active_revision_proposal is None
    assert state.human_feedback[-1].status == "unresolved"
    assert "automated review failed" in state.history[-1]["note"]


def test_feedback_can_conclude_no_case_revision_without_forcing_mutation(tmp_path, graph_path):
    state, _, _, _, _ = _prepare_mesh_ready(tmp_path, graph_path)
    _mark_result_review(state)
    assessment = _assessment(case_revision=False)
    assessment.proposed_changes = []

    CFDFeedbackReviewAgent(ScriptedLLM([assessment])).review(
        state, "현재 결과가 이상한지 설명만 먼저 해줘"
    )

    assert state.current_state == State.RESULT_REVIEW_REQUIRED
    assert state.active_revision_proposal is None
    assert state.human_feedback[-1].status == "awaiting_review"
    assert len(state.revision_proposals) == 1
    assert not state.revision_proposals[-1].requires_case_revision
