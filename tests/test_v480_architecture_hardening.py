from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest

from conftest import FakeOpenFOAMTools, make_plan, make_state, fixture_verified_output
from openfoam_agent.cli import _sanitize_user_text
from openfoam_agent.engineering import CFDEngineeringAgent, EngineeringPolicy
from openfoam_agent.engineering.phases.repair_episode import ensure_episode, add_support_observation
from openfoam_agent.llm.context_capsules import project_plan_core
from openfoam_agent.postprocessing.agent import CFDPostProcessingAgent, PostProcessingPolicy
from openfoam_agent.schemas.engineering import (
    BlockAction,
    EngineeringDecision,
    EngineeringDefaultAssumption,
    EngineeringPlanPatch,
    ReadReferenceAction,
    RepairCasePlanAction,
    RevisionDecisionAction,
    RevisionDecisionTurn,
    RevisionAuthoringTurn,
)
from openfoam_agent.schemas.feedback import HumanFeedback, RevisionProposal
from openfoam_agent.schemas.postprocessing import FinishPostProcessingAction
from openfoam_agent.schemas.simulation import RuntimeReport, SimulationAttempt
from openfoam_agent.tools.parsers import parse_runtime_log
from openfoam_agent.contracts.models import CompletionContract
from openfoam_agent.workflow.states import State


def _bloated_plan(state):
    plan = make_plan(state.intake)
    plan.decisions = [
        EngineeringDecision(area=f"area-{i}", choice="x" * 500, rationale="y" * 500)
        for i in range(80)
    ]
    plan.assumptions = ["z" * 1000 for _ in range(80)]
    plan.engineering_defaults = [
        EngineeringDefaultAssumption(
            parameter=f"parameter_{i}", value="v" * 300, unit="1", rationale="r" * 500
        )
        for i in range(80)
    ]
    return plan


def test_unicode_application_boundary_removes_unpaired_surrogates_and_preserves_korean():
    text = "배터리바닥에 \ud800직접 접촉해\udfff"
    cleaned = _sanitize_user_text(text)
    assert cleaned == "배터리바닥에 직접 접촉해"
    cleaned.encode("utf-8")


def test_native_diagnostic_path_cannot_be_used_as_reference_id():
    with pytest.raises(ValueError, match="indexed reference ID"):
        ReadReferenceAction(type="read_reference", reference="/opt/openfoam13/etc/codeTemplates/dynamicCode/solidThermo")
    with pytest.raises(ValueError, match="indexed reference ID"):
        ReadReferenceAction(type="read_reference", reference="<OPENFOAM_ROOT><LOCAL_PATH:solidThermo>")
    assert ReadReferenceAction(type="read_reference", reference="source:thermophysicalModels/foo.C").reference.startswith("source:")


def test_repair_episode_keeps_current_native_failure_across_support_failures():
    state = make_state()
    agent_event = lambda step, action, success, summary: __import__(
        "openfoam_agent.schemas.engineering", fromlist=["EngineeringEvent"]
    ).EngineeringEvent(
        step=step, action_type=action, success=success, summary=summary,
        output_excerpt=summary, validation_status=("pass" if success else "fail"),
        failure_category=(None if success else "case"),
    )
    first = agent_event(1, "validate_pre_solve", False, "Unknown thermo type constIso")
    episode = ensure_episode(state, first, ["constant/solid/physicalProperties"])
    assert "constIso" in json.dumps(episode.current_failure)

    support = agent_event(2, "read_reference", True, "Read source:solidThermo")
    add_support_observation(state, support)
    assert "constIso" in json.dumps(state.repair_episode.current_failure)

    # Even a failed support lookup is not allowed to replace the active native diagnostic.
    failed_support = agent_event(3, "read_reference", False, "Unknown OpenFOAM reference /opt/.../solidThermo")
    add_support_observation(state, failed_support)
    assert "constIso" in json.dumps(state.repair_episode.current_failure)

    second = agent_event(4, "validate_pre_solve", False, "Unknown thermo type hConst; valid: eConst ePower")
    ensure_episode(state, second, ["constant/solid/physicalProperties"])
    assert "hConst" in json.dumps(state.repair_episode.current_failure)
    assert "constIso" in json.dumps(state.repair_episode.root_failure)


class _PostLLM:
    model = "synthetic"
    def __init__(self, *, store: bool):
        self.store = store
        self.prompts = []
        self.calls = []
    def generate(self, schema, prompt, *, system_prompt=None, conversation_key=None, use_previous_response=False, prompt_cache_key=None):
        self.prompts.append(prompt)
        self.calls.append({"conversation_key": conversation_key, "use_previous_response": use_previous_response})
        return schema(action=FinishPostProcessingAction(type="finish_postprocessing", summary="done", limitations=[]))


def _post_state(tmp_path):
    state = make_state(); plan = make_plan(state.intake); state.engineering_plan = plan
    result = fixture_verified_output(parse_runtime_log(
        "Time = 1s\nEnd\n", return_code=0,
        contract=CompletionContract(mode="transient", start_time=0, end_time=1.0),
    ))
    state.runtime_report = RuntimeReport(
        success=True, attempts=[SimulationAttempt(attempt=1, result=result)], final_result=result
    )
    state.simulation = result
    state.current_state = State.EXECUTION_DONE
    return state


def test_postprocess_stateless_backend_never_receives_fake_delta_context(tmp_path):
    state = _post_state(tmp_path); llm = _PostLLM(store=False)
    post = CFDPostProcessingAgent(
        llm, workspace=tmp_path, tools=FakeOpenFOAMTools(),
        policy=PostProcessingPolicy(state_delta_context=True, compact_execution_plan=False),
    )
    post._generate_turn(state, step=1)
    post._generate_turn(state, step=2)
    assert all('"state_mode": "delta_from_previous_response"' not in p for p in llm.prompts)
    assert all(not call["use_previous_response"] for call in llm.calls)
    assert '"confirmed_intake"' in llm.prompts[1]


def test_postprocess_true_stateful_backend_may_use_delta_context(tmp_path):
    state = _post_state(tmp_path); llm = _PostLLM(store=True)
    post = CFDPostProcessingAgent(
        llm, workspace=tmp_path, tools=FakeOpenFOAMTools(),
        policy=PostProcessingPolicy(state_delta_context=True, compact_execution_plan=False),
    )
    post._generate_turn(state, step=1)
    post._generate_turn(state, step=2)
    assert '"state_mode": "delta_from_previous_response"' in llm.prompts[1]
    assert llm.calls[1]["use_previous_response"] is True


def test_schema_valid_bloated_plan_has_bounded_phase_projections():
    state = make_state(); plan = _bloated_plan(state)
    raw = len(json.dumps(plan.model_dump(mode="json"), ensure_ascii=False))
    assert raw > 100_000
    bounds = {"authoring": 26_000, "revision": 14_000, "generic": 18_000, "postprocess": 18_000, "review": 18_000}
    for mode, limit in bounds.items():
        projected = project_plan_core(plan, mode=mode)
        assert len(json.dumps(projected, ensure_ascii=False)) < limit, mode


def test_engineering_facade_is_physically_split_into_phase_controllers():
    import openfoam_agent.engineering.agent as facade
    from openfoam_agent.engineering.phases import context_controller, lifecycle_controller, decision_controller, authoring_controller, repair_controller
    agent_lines = len(Path(inspect.getsourcefile(facade)).read_text().splitlines())
    assert agent_lines < 4500
    for module in (context_controller, lifecycle_controller, decision_controller, authoring_controller, repair_controller):
        assert Path(inspect.getsourcefile(module)).is_file()


def test_revision_contract_is_decision_then_authoring(tmp_path, graph_path):
    class LLM:
        store = False
        model = "synthetic"
        def __init__(self): self.schemas=[]; self.prompts=[]
        def generate(self, schema, prompt, **kwargs):
            self.schemas.append(schema); self.prompts.append(prompt)
            if schema is RevisionDecisionTurn:
                return schema(action=RevisionDecisionAction(
                    type="decide_revision", diagnosis="change runtime duration",
                    plan_patch=EngineeringPlanPatch(mesh_strategy="revised representative mesh"),
                    target_case_files=["system/controlDict"],
                ))
            return schema(action=BlockAction(type="block", reason="stop after contract test", needs_user_input=True))
    llm=LLM(); agent=CFDEngineeringAgent(
        llm, workspace=tmp_path, capability_db=graph_path, tools=FakeOpenFOAMTools(),
        policy=EngineeringPolicy(compact_phase_schemas=True, max_model_prompt_chars=18_000),
    )
    state=make_state(); plan=make_plan(state.intake)
    agent.workspace.write_text("system/controlDict", "FoamFile { version 2.0; format ascii; class dictionary; object controlDict; }\n")
    state.engineering_plan=plan; state.case_seal=agent.workspace.seal(plan); state.case_dir=str(agent.workspace.case_dir)
    state.human_feedback=[HumanFeedback(
        feedback_id="hf-0001", run_id=state.run_id, scope="result", statement="revise", submitted_state=State.RESULT_REVIEW_REQUIRED.value,
        evidence_snapshot_sha256="a"*64, status="revision_proposed",
    )]
    state.active_revision_proposal=RevisionProposal(
        proposal_id="rp-0001", feedback_ids=["hf-0001"], diagnosis_summary="revise",
        hypotheses=[], proposed_changes=[], expected_cost="similar", requires_case_revision=True,
        baseline_plan_sha256=plan.digest(), baseline_manifest_sha256=state.case_seal.manifest_sha256,
    )
    state.current_state=State.ENGINEERING
    first=agent._generate_turn(state, step=1, phase="human_revision", native_execution=False)
    assert isinstance(first, RevisionDecisionTurn)
    agent._execute_revision_decision(state, first.action, step=1)
    second=agent._generate_turn(state, step=2, phase="human_revision", native_execution=False)
    assert isinstance(second, RevisionAuthoringTurn)
    assert '"state_mode": "human_revision_decision_v2"' in llm.prompts[0]
    assert '"state_mode": "human_revision_authoring_v2"' in llm.prompts[1]
