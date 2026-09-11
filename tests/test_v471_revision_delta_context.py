from __future__ import annotations

import pytest

from conftest import FakeOpenFOAMTools, control_dict, make_plan, make_state
from test_v210_token_optimization import FlexibleScriptedLLM

from openfoam_agent.engineering import CFDEngineeringAgent, EngineeringPolicy
from openfoam_agent.llm.context import ContextBudgetError
from openfoam_agent.schemas.engineering import (
    BlockAction,
    CaseBundleFile,
    EngineeringDecision,
    EngineeringDefaultAssumption,
    EngineeringPlanPatch,
    RepairCasePlanAction,
    RevisionDecisionAction,
    RevisionDecisionTurn,
    SearchCapabilitiesAction,
)
from openfoam_agent.schemas.feedback import HumanFeedback, RevisionProposal
from openfoam_agent.workflow.states import State


def _proposal(state, plan, manifest_sha: str) -> RevisionProposal:
    return RevisionProposal(
        proposal_id="rp-0001",
        feedback_ids=["hf-0001"],
        diagnosis_summary="The reviewed result should be revised toward an immediate bifurcation with time-dependent junction structures.",
        hypotheses=[],
        proposed_changes=[
            {
                "area": "flow regime",
                "change": "Reassess the delegated inlet flow and use a transient treatment when justified.",
                "rationale": "The user wants time-dependent structures near the immediate split.",
            },
            {
                "area": "mesh",
                "change": "Refine the splitter and branch entrances.",
                "rationale": "Resolve the junction shear layers without changing confirmed geometry facts.",
            },
        ],
        expected_cost="moderate_increase",
        requires_case_revision=True,
        baseline_plan_sha256=plan.digest(),
        baseline_manifest_sha256=manifest_sha,
    )


def _revision_state(agent, *, bloated: bool = False):
    state = make_state()
    plan = make_plan(state.intake)
    agent.workspace.write_text("system/controlDict", control_dict())
    if bloated:
        plan.decisions = [
            EngineeringDecision(
                area=f"decision-{i}",
                choice="transient junction modeling " + ("x" * 300),
                rationale="engineering rationale " + ("y" * 300),
            )
            for i in range(24)
        ]
        plan.assumptions = [("baseline assumption " + ("z" * 450)) for _ in range(24)]
        plan.engineering_defaults = [
            EngineeringDefaultAssumption(
                parameter=f"parameter_{i}",
                value=str(i + 1),
                unit="1",
                rationale="representative revision default " + ("r" * 450),
            )
            for i in range(24)
        ]
    capability_event = agent._dispatch_tool_action(
        SearchCapabilitiesAction(
            type="search_capabilities",
            query="incompressibleFluid",
            rationale="Observe the baseline solver capability before revision.",
        ),
        step=1,
        native_execution=False,
        phase="prepare",
        state=state,
    )
    state.engineering_events.append(capability_event)
    state.engineering_plan = plan
    state.case_seal = agent.workspace.seal(plan)
    state.case_dir = str(agent.workspace.case_dir)
    state.human_feedback = [
        HumanFeedback(
            feedback_id="hf-0001",
            run_id=state.run_id,
            scope="result",
            statement="나는 분기가 바로 두 갈래로 나뉘면서 생기는 시간의존 구조를 보고 싶어",
            submitted_state=State.RESULT_REVIEW_REQUIRED.value,
            evidence_snapshot_sha256="a" * 64,
            status="revision_proposed",
        )
    ]
    state.active_revision_proposal = _proposal(state, plan, state.case_seal.manifest_sha256)
    state.current_state = State.REVISION_READY
    return state, plan


def test_revision_delta_context_fits_cli_18k_even_with_bloated_baseline(tmp_path, graph_path):
    llm = FlexibleScriptedLLM([
        BlockAction(type="block", reason="test stop", block_kind="engineering_choice_missing")
    ])
    agent = CFDEngineeringAgent(
        llm,
        workspace=tmp_path,
        capability_db=graph_path,
        tools=FakeOpenFOAMTools(),
        policy=EngineeringPolicy(
            max_model_prompt_chars=18_000,
            compact_phase_schemas=True,
            state_delta_context=True,
            bounded_evidence_context=True,
            staged_case_authoring=True,
        ),
    )
    state, _ = _revision_state(agent, bloated=True)
    turn = agent._generate_turn(state, step=1, phase="human_revision", native_execution=False)
    assert isinstance(turn, RevisionDecisionTurn)
    assert len(llm.prompts[0]) <= 18_000
    assert '"state_mode": "human_revision_decision_v2"' in llm.prompts[0]
    assert '"baseline_plan_core"' in llm.prompts[0]
    assert '"capability_graph_hint"' not in llm.prompts[0]
    assert '"cumulative_provenance"' not in llm.prompts[0]
    assert "plan_patch" in llm.prompts[0]


def test_plan_patch_changes_agent_owned_fields_and_preserves_confirmed_audit():
    state = make_state(); plan = make_plan(state.intake)
    before_bindings = plan.confirmed_fact_bindings
    patch = EngineeringPlanPatch(
        temporal_behavior="steady",
        mesh_strategy="Revised immediate bifurcation mesh.",
        engineering_defaults=[
            EngineeringDefaultAssumption(
                parameter="inlet_velocity", value="0.5", unit="m/s",
                rationale="User-confirmed revision delegates a stronger representative inlet.",
            )
        ],
    )
    revised = patch.apply(plan)
    assert revised.temporal_behavior == "steady"
    assert revised.mesh_strategy == "Revised immediate bifurcation mesh."
    assert revised.confirmed_intake_sha256 == plan.confirmed_intake_sha256
    assert revised.confirmed_fact_ids == plan.confirmed_fact_ids
    assert revised.confirmed_fact_bindings == before_bindings


def test_repair_case_plan_accepts_delta_plan_patch_without_full_updated_plan():
    action = RepairCasePlanAction.model_validate({
        "type": "repair_case_plan",
        "diagnosis": "Change temporal treatment only.",
        "plan_patch": {"temporal_behavior": "steady"},
    })
    assert action.plan_patch is not None
    assert action.updated_plan is None


def test_confirmed_revision_uses_plan_patch_and_archives_only_when_delta_commits(tmp_path, graph_path):
    revised_control = control_dict().replace("endTime 10;", "endTime 40;")
    llm = FlexibleScriptedLLM([
        RevisionDecisionAction(
            type="decide_revision",
            diagnosis="Apply the confirmed revision as a plan decision first.",
            plan_patch=EngineeringPlanPatch(
                temporal_behavior="steady",
                mesh_strategy="Immediate bifurcation revision with locally refined junction.",
            ),
            target_case_files=["system/controlDict"],
        ),
        RepairCasePlanAction(
            type="repair_case_plan",
            diagnosis="Author only the accepted revision file delta.",
            replacement_files=[CaseBundleFile(path="system/controlDict", content=revised_control)],
            validate_pre_solve=False,
        ),
    ])
    agent = CFDEngineeringAgent(
        llm,
        workspace=tmp_path,
        capability_db=graph_path,
        tools=FakeOpenFOAMTools(),
        policy=EngineeringPolicy(
            max_agent_steps=2,
            hard_max_agent_steps=2,
            max_model_prompt_chars=18_000,
            compact_phase_schemas=True,
            state_delta_context=True,
            bounded_evidence_context=True,
            staged_case_authoring=True,
        ),
    )
    state, _ = _revision_state(agent)
    old_output = agent.workspace.case_dir / "10" / "U"
    old_output.parent.mkdir(parents=True, exist_ok=True)
    old_output.write_text("old result\n", encoding="utf-8")

    agent.revise_from_feedback(state, native_execution=False)

    assert state.current_state == State.CASE_PREVIEW_READY
    assert state.engineering_plan.temporal_behavior == "steady"
    assert state.engineering_plan.mesh_strategy.startswith("Immediate bifurcation")
    assert state.active_revision_proposal is None
    assert state.revision_history[-1].archive_path == "revision-history/rev-0001"
    assert not old_output.exists()
    assert (tmp_path / "revision-history" / "rev-0001" / "case_outputs" / "10" / "U").is_file()


class _FailBeforeResponseLLM:
    store = False
    model = "synthetic"
    def generate(self, schema, prompt, *, system_prompt=None, conversation_key=None, use_previous_response=False, prompt_cache_key=None):
        raise ContextBudgetError("synthetic provider/pre-response failure")


def test_revision_failure_before_valid_delta_keeps_baseline_outputs_and_does_not_archive(tmp_path, graph_path):
    agent = CFDEngineeringAgent(
        _FailBeforeResponseLLM(),
        workspace=tmp_path,
        capability_db=graph_path,
        tools=FakeOpenFOAMTools(),
        policy=EngineeringPolicy(
            max_agent_steps=2,
            hard_max_agent_steps=2,
            max_model_prompt_chars=18_000,
            compact_phase_schemas=True,
            state_delta_context=True,
            bounded_evidence_context=True,
            staged_case_authoring=True,
        ),
    )
    state, _ = _revision_state(agent)
    old_output = agent.workspace.case_dir / "10" / "U"
    old_output.parent.mkdir(parents=True, exist_ok=True)
    old_output.write_text("old result\n", encoding="utf-8")
    old_log = agent.workspace.log_dir / "foamRun.log"
    old_log.write_text("old log\n", encoding="utf-8")

    with pytest.raises(ContextBudgetError):
        agent.revise_from_feedback(state, native_execution=False)

    assert old_output.is_file()
    assert old_log.is_file()
    assert state.pending_revision_archive_path is None
    assert not (tmp_path / "revision-history" / "rev-0001").exists()
    assert state.current_state == State.REVISION_READY
    assert state.human_feedback[0].status == "revision_proposed"
