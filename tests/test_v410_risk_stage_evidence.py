"""v4.1.0 risk/stage-aware evidence regressions; no live LLM/OpenFOAM calls."""
from types import SimpleNamespace

from conftest import FakeOpenFOAMTools, make_plan, make_state
from openfoam_agent.contracts.evidence import authoring_evidence_failures_for_representations
from openfoam_agent.contracts.evidence_policy import POLICY_SUMMARY, provider_is_sufficient
from openfoam_agent.engineering import CFDEngineeringAgent, EngineeringPolicy
from openfoam_agent.engineering.design_context import project_design_capsule
from openfoam_agent.schemas.engineering import DesignCaseAction, EngineeringDecision


def test_v410_decisions_carry_risk_policy_and_stage_without_breaking_old_payloads():
    old = EngineeringDecision(area="mesh", choice="structured", rationale="simple")
    assert old.risk_level == "medium"
    assert old.evidence_policy == "advisory"
    assert old.verification_stage == "pre_validation"
    critical = EngineeringDecision(
        area="execution", choice="foamRun", rationale="installed driver",
        risk_level="critical", evidence_policy="mandatory", verification_stage="pre_execution",
    )
    assert critical.evidence_policy == "mandatory"
    assert set(POLICY_SUMMARY) == {"mandatory", "deferred", "advisory"}


def test_v410_provider_strength_is_stage_aware():
    documented = SimpleNamespace(verified=True, verification_level="documented")
    binary = SimpleNamespace(verified=True, verification_level="binary_present")
    source = SimpleNamespace(verified=True, verification_level="source_discovered")
    assert not provider_is_sufficient(documented, executable=True)
    assert provider_is_sufficient(binary, executable=True)
    assert not provider_is_sufficient(source, executable=True)
    assert provider_is_sufficient(source, executable=False)


def test_v410_design_acceptance_no_longer_requires_file_syntax_evidence(tmp_path, graph_path):
    state = make_state(syntax_evidence=False)
    plan = make_plan(state.intake)
    plan.required_case_files = ["system/controlDict"]
    action = DesignCaseAction(type="design_case", plan=plan, authoring_brief="typed authoring later")
    agent = CFDEngineeringAgent(
        SimpleNamespace(generate=lambda *a, **k: None), workspace=tmp_path,
        capability_db=graph_path, tools=FakeOpenFOAMTools(),
        policy=EngineeringPolicy(staged_case_authoring=True),
    )
    # This test isolates the stage boundary: provider provenance has a separate test.
    agent._validate_observed_provenance = lambda plan, state: []
    terminal = agent._execute_prepare_decision_impl(
        state, action, llm_step=1, progress_phase="engineering", progress_step=1,
        progress_limit=4, native_execution=False,
    )
    assert terminal is False
    assert agent._draft_design_plan is plan
    assert state.engineering_events[-1].success is True


def test_v421_raw_and_typed_authoring_use_deterministic_validation_not_document_gate():
    state = make_state(syntax_evidence=False)
    plan = make_plan(state.intake)
    plan.required_case_files = ["system/controlDict"]
    typed = authoring_evidence_failures_for_representations(
        state, plan, raw_paths=[], typed_paths=["system/controlDict"]
    )
    raw = authoring_evidence_failures_for_representations(
        state, plan, raw_paths=["system/controlDict"], typed_paths=[]
    )
    assert typed == []
    assert raw == []


def test_v410_context_partition_preserves_policy_and_verified_candidates():
    state = make_state()
    payload = {
        "confirmed_intake": state.intake.model_dump(mode="json"),
        "intake_sha256": state.intake_digest,
        "engineering_assumption_policy": {"authorized": True},
        "evidence_policy": {"mode": "risk_stage_aware_v1", "classes": POLICY_SUMMARY},
        "verified_execution_candidates": [
            {"provider_id": "installed.application.foamRun", "verification_level": "binary_present"}
        ],
        "evidence_retrieval_policy": {"available": True},
        "available_evidence": [], "evidence_context": {"total_observed": 0},
        "evidence_gap_status": [], "recent_observations": [], "bindings": {}, "budget": {},
    }
    capsule = project_design_capsule(payload, evidence_limit=0)
    assert capsule["evidence_policy"]["mode"] == "risk_stage_aware_v1"
    assert capsule["verified_execution_candidates"][0]["provider_id"] == "installed.application.foamRun"


def test_v410_cli_suppresses_dimensionless_unit_marker_in_intake_display_source():
    from pathlib import Path
    cli = Path(__file__).resolve().parents[1] / "src/openfoam_agent/cli.py"
    text = cli.read_text()
    assert '{"", "1", "dimensionless", "-"}' in text
