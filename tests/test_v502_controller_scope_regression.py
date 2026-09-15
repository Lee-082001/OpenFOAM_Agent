from __future__ import annotations

from conftest import FakeOpenFOAMTools, ScriptedLLM, make_plan, make_state
from openfoam_agent.engineering.agent import CFDEngineeringAgent
from openfoam_agent.engineering.phases.decision_controller import execute_prepare_decision_impl
from openfoam_agent.schemas.engineering import (
    CaseAuthoringAction,
    CaseBundleFile,
    ValidatePreSolveAction,
)
from openfoam_agent.verification.presolve import PreSolveValidationResult


def test_v502_validate_pre_solve_uses_controller_plan_not_required_files_projection(
    tmp_path, graph_path, monkeypatch
):
    agent = CFDEngineeringAgent(
        ScriptedLLM([]), workspace=tmp_path, capability_db=graph_path, tools=FakeOpenFOAMTools()
    )
    agent.policy.zero_step_consumer_validation = False
    state = make_state()
    plan = make_plan(state.intake)
    plan.required_case_files = ["0/battery/T", "0/heater/T", "system/controlDict"]
    agent._pending_execution_plan = plan

    seen = []

    def validate(full_plan):
        seen.append(full_plan)
        return PreSolveValidationResult(valid=True, checked_files=list(full_plan.required_case_files))

    monkeypatch.setattr(agent.presolve, "validate", validate)
    monkeypatch.setattr(
        agent.presolve,
        "validate_required_case_files",
        lambda *_: (_ for _ in ()).throw(AssertionError("scope-losing legacy projection must not run")),
    )

    action = ValidatePreSolveAction(
        type="validate_pre_solve", required_case_files=list(plan.required_case_files)
    )
    event = agent._dispatch_tool_action(
        action, step=1, native_execution=True, phase="engineering", state=state
    )

    assert event.success
    assert seen == [plan]


def test_v502_validate_pre_solve_rejects_manifest_drift_as_controller_infra_failure(
    tmp_path, graph_path
):
    agent = CFDEngineeringAgent(
        ScriptedLLM([]), workspace=tmp_path, capability_db=graph_path, tools=FakeOpenFOAMTools()
    )
    state = make_state()
    plan = make_plan(state.intake)
    plan.required_case_files = ["0/battery/T", "system/controlDict"]
    agent._pending_execution_plan = plan

    action = ValidatePreSolveAction(
        type="validate_pre_solve", required_case_files=["0/heater/T", "system/controlDict"]
    )
    event = agent._dispatch_tool_action(
        action, step=1, native_execution=True, phase="engineering", state=state
    )

    assert not event.success
    assert event.failure_category == "infra"
    assert "manifest diverged" in event.summary


def test_v502_unpartitioned_authoring_ignores_stray_partition_metadata(
    tmp_path, graph_path, monkeypatch
):
    agent = CFDEngineeringAgent(
        ScriptedLLM([]), workspace=tmp_path, capability_db=graph_path, tools=FakeOpenFOAMTools()
    )
    state = make_state(syntax_evidence=False)
    plan = make_plan(state.intake)
    plan.required_case_files = ["system/controlDict"]
    agent._draft_design_plan = plan
    agent._authoring_task_queue = None

    captured = []

    def execute_case_plan(state_arg, execution, **kwargs):
        captured.append(execution)
        return True

    monkeypatch.setattr(agent, "_execute_case_plan", execute_case_plan)
    action = CaseAuthoringAction(
        type="author_case",
        task_id="hallucinated-unpartitioned-task",
        defer_native=True,
        goal="author frozen manifest",
        files=[CaseBundleFile(path="system/controlDict", content="FoamFile{}")],
    )

    result = execute_prepare_decision_impl(
        agent,
        state,
        action,
        llm_step=2,
        progress_phase="ENGINEERING",
        progress_step=2,
        progress_limit=12,
        native_execution=True,
    )

    assert result is True
    assert len(captured) == 1
    assert captured[0].task_id is None
    assert captured[0].defer_native is False
    assert captured[0].required_case_files == plan.required_case_files
