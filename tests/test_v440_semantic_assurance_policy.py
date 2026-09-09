from __future__ import annotations

from conftest import FakeOpenFOAMTools, ScriptedLLM, control_dict, make_plan
from openfoam_agent.engineering import CFDEngineeringAgent
from openfoam_agent.schemas.engineering import (
    CaseContentAssertion,
    ConfirmedFactBinding,
    EngineeringEvidenceRecord,
    FinishPreviewAction,
    ObservedEngineeringEvidence,
    canonical_engineering_evidence_id,
)
from openfoam_agent.schemas.intake import CFDIntakeSpec, IntakeFact
from openfoam_agent.tools.workspace import CaseWorkspace
from openfoam_agent.verification.safety import DeterministicSafetyGate
from openfoam_agent.verification.semantic_assurance import expectation_for_fact
from openfoam_agent.workflow.state import CFDState
from openfoam_agent.schemas.request import UserRequest
from openfoam_agent.workflow.states import State


def _classification_only_intake() -> CFDIntakeSpec:
    return CFDIntakeSpec(
        semantic_contract_version="2",
        title="Branching internal flow",
        facts=[
            IntakeFact(
                id="classification.problem_type",
                category="classification",
                label="Problem type",
                value="internal_flow",
                source="derived",
                reason="Routing interpretation inferred from fully flooded pipe flow.",
            )
        ],
        status="ready_for_review",
    )


def _write_minimum_case(ws: CaseWorkspace) -> None:
    ws.write_text("system/controlDict", control_dict())


def test_v440_derived_classification_is_provenance_only_not_artifact_mandatory(tmp_path):
    intake = _classification_only_intake()
    fact = intake.fact("classification.problem_type")
    expectation = expectation_for_fact(fact)
    assert expectation.mode == "routing_provenance"
    assert not expectation.machine_assertion_recommended

    plan = make_plan(intake)
    ws = CaseWorkspace(tmp_path)
    _write_minimum_case(ws)
    result = DeterministicSafetyGate(FakeOpenFOAMTools(), ws).validate_plan(plan, intake)
    assert result.valid, result.failures
    assert not any("classification.problem_type" in warning for warning in result.warnings)


def test_v440_user_temporal_fact_without_assertion_is_advisory_not_case_failure(tmp_path):
    intake = CFDIntakeSpec(
        semantic_contract_version="2",
        title="Transient flow",
        facts=[
            IntakeFact(
                id="temporal.behavior",
                category="temporal",
                label="Temporal behavior",
                value="transient",
                source="user",
                evidence="time-dependent flow",
            )
        ],
        status="ready_for_review",
    )
    plan = make_plan(intake)
    ws = CaseWorkspace(tmp_path)
    _write_minimum_case(ws)
    result = DeterministicSafetyGate(FakeOpenFOAMTools(), ws).validate_plan(plan, intake)
    assert result.valid, result.failures
    assert any("temporal.behavior" in warning for warning in result.warnings)


def test_v440_numeric_user_fact_without_relation_is_advisory(tmp_path):
    intake = CFDIntakeSpec(
        semantic_contract_version="2",
        title="Reynolds target",
        facts=[
            IntakeFact(
                id="physics.reynolds_number",
                category="physics",
                label="Reynolds number",
                value="1000",
                unit="dimensionless",
                source="user",
                evidence="Re=1000",
            )
        ],
        status="ready_for_review",
    )
    plan = make_plan(intake)
    ws = CaseWorkspace(tmp_path)
    _write_minimum_case(ws)
    result = DeterministicSafetyGate(FakeOpenFOAMTools(), ws).validate_plan(plan, intake)
    assert result.valid, result.failures
    assert any("numeric user fact" in warning for warning in result.warnings)


def test_v440_claimed_semantic_assertion_still_fails_on_contradiction(tmp_path):
    intake = _classification_only_intake()
    plan = make_plan(intake).model_copy(
        update={
            "confirmed_fact_bindings": [
                ConfirmedFactBinding(
                    fact_id="classification.problem_type",
                    plan_fields=["problem_interpretation"],
                    case_assertions=[
                        CaseContentAssertion(path="system/controlDict", contains=["THIS_TOKEN_DOES_NOT_EXIST"])
                    ],
                )
            ]
        }
    )
    ws = CaseWorkspace(tmp_path)
    _write_minimum_case(ws)
    result = DeterministicSafetyGate(FakeOpenFOAMTools(), ws).validate_plan(plan, intake)
    assert not result.valid
    assert any("THIS_TOKEN_DOES_NOT_EXIST" in failure for failure in result.failures)


def test_v440_missing_confirmed_binding_remains_hard_failure(tmp_path):
    intake = _classification_only_intake()
    plan = make_plan(intake).model_copy(update={"confirmed_fact_bindings": []})
    ws = CaseWorkspace(tmp_path)
    _write_minimum_case(ws)
    result = DeterministicSafetyGate(FakeOpenFOAMTools(), ws).validate_plan(plan, intake)
    assert not result.valid
    assert any("implementation binding mismatch" in failure for failure in result.failures)


def test_v440_finish_preview_seals_case_with_advisory_gap_instead_of_repair(tmp_path, graph_path):
    intake = CFDIntakeSpec(
        semantic_contract_version="2",
        title="Transient flow",
        facts=[
            IntakeFact(
                id="classification.problem_type",
                category="classification",
                label="Problem type",
                value="internal_flow",
                source="derived",
                reason="Routing interpretation.",
            ),
            IntakeFact(
                id="temporal.behavior",
                category="temporal",
                label="Temporal behavior",
                value="transient",
                source="user",
                evidence="transient requested",
            ),
        ],
        status="ready_for_review",
    )
    plan = make_plan(intake)
    state = CFDState(run_id="semantic-assurance", user_request=UserRequest(prompt="transient internal flow"), intake=intake)
    state.current_state = State.ENGINEERING
    agent = CFDEngineeringAgent(
        ScriptedLLM([]), workspace=tmp_path, capability_db=graph_path, tools=FakeOpenFOAMTools()
    )
    provider_ref = "solver.incompressibleFluid"
    state.engineering_evidence_records.append(
        EngineeringEvidenceRecord(
            record_id="evrec_0123456789abcdefabcd",
            phase="prepare",
            step=1,
            action_type="search_capabilities",
            payload=[],
            observed_evidence=[
                ObservedEngineeringEvidence(
                    evidence_id=canonical_engineering_evidence_id("capability", provider_ref),
                    kind="capability",
                    reference=provider_ref,
                    summary="Observed solver capability",
                )
            ],
        )
    )
    _write_minimum_case(agent.workspace)
    event, terminal = agent._dispatch_prepare(
        state,
        FinishPreviewAction(type="finish_preview", plan=plan, rationale=""),
        step=1,
        native_execution=False,
    )
    assert terminal
    assert event.success
    assert state.current_state == State.CASE_PREVIEW_READY
    assert state.primary_failure is None
    assert any("temporal.behavior" in warning for warning in state.semantic_assurance_warnings)
    assert "advisory semantic-assurance" in event.summary
