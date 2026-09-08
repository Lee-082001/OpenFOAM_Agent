"""v4.2 progress-first validation regressions; no live LLM/OpenFOAM calls."""
from types import SimpleNamespace

from conftest import FakeOpenFOAMTools, make_plan, make_state
from openfoam_agent.contracts.models import ConservationCheck, QuantityOfInterest
from openfoam_agent.engineering import CFDEngineeringAgent, EngineeringPolicy
from openfoam_agent.schemas.engineering import EngineeringPlan


def test_v420_qoi_may_remain_unresolved_design_intent():
    q = QuantityOfInterest.model_validate({
        "id": "outlet1",
        "quantity": "outlet volume flow",
        "unit": "m3/s",
        "source_path": "postProcessing/flowRate/outlet1.dat",
        "native_field": {
            "quantity_kind": "volume_flow",
            "field": "phi",
            "time_names": ["latestTime"],
            "reduction": "patch_sum",
            "patches": ["outlet1"],
            "dimensions": [0, 3, -1, 0, 0, 0, 0],
        },
        "operation": "balance",
        "other_source_path": "postProcessing/flowRate/inlet.dat",
        "future_metadata": "ignored rather than fatal",
    })
    assert q.resolution_state == "intent"
    assert q.native_field is not None
    assert q.native_field.time_names == ["latestTime"]


def test_v420_conservation_may_use_runtime_placeholder_times_at_design_stage():
    check = ConservationCheck.model_validate({
        "id": "massBalance",
        "kind": "mass",
        "regions": [{
            "region": "", "flux_field": "phi", "storage_mode": "steady", "source_mode": "zero"
        }],
        "time_names": ["latestTime", "end"],
        "relative_tolerance": 0.05,
    })
    assert check.resolution_state == "intent"
    assert check.time_names == ["latestTime", "end"]


def test_v420_resolved_analysis_contracts_remain_strict():
    try:
        QuantityOfInterest.model_validate({
            "id": "bad", "quantity": "q", "resolution_state": "resolved", "unit": "1",
            "operation": "balance", "source_path": ""
        })
    except ValueError as exc:
        assert "Resolved quantity requires" in str(exc)
    else:
        raise AssertionError("resolved quantity must remain strict")


def test_v420_plan_normalizes_redundant_metadata_instead_of_rejecting():
    state = make_state()
    plan = make_plan(state.intake)
    raw = plan.model_dump(mode="python")
    raw["required_case_files"] = ["system/controlDict", "system/controlDict"]
    raw["confirmed_fact_ids"] = raw["confirmed_fact_ids"] + [raw["confirmed_fact_ids"][0]]
    raw["unexpected_planning_hint"] = "ignore me"
    normalized = EngineeringPlan.model_validate(raw)
    assert normalized.required_case_files == ["system/controlDict"]
    assert len(normalized.confirmed_fact_ids) == len(set(normalized.confirmed_fact_ids))


def test_v420_ordinary_engineering_defaults_do_not_require_exploratory_flag(tmp_path, graph_path):
    state = make_state()
    state.user_request.exploratory_completion_authorized = False
    plan = make_plan(state.intake)
    raw = plan.model_dump(mode="python")
    raw["engineering_defaults"] = [{
        "parameter": "inlet velocity", "value": "1", "unit": "m/s",
        "basis": "representative", "rationale": "ordinary missing operating value",
        "source": "engineering_default", "evidence_ids": [],
    }]
    plan = EngineeringPlan.model_validate(raw)
    agent = CFDEngineeringAgent(
        SimpleNamespace(generate=lambda *a, **k: None), workspace=tmp_path,
        capability_db=graph_path, tools=FakeOpenFOAMTools(), policy=EngineeringPolicy(),
    )
    assert agent._validate_engineering_defaults(plan, state) == []


def test_v420_design_does_not_require_same_run_capability_observation_when_catalog_provider_is_sufficient(tmp_path, graph_path):
    state = make_state()
    plan = make_plan(state.intake)
    agent = CFDEngineeringAgent(
        SimpleNamespace(generate=lambda *a, **k: None), workspace=tmp_path,
        capability_db=graph_path, tools=FakeOpenFOAMTools(), policy=EngineeringPolicy(),
    )
    # Fixture provider exists in the deterministic catalog; absence of an extra per-run
    # evidence pointer should not itself be a hard design failure.
    failures = agent._validate_observed_provenance(plan, state)
    assert not any("no successful capability-graph observation" in item for item in failures)


def test_v420_prepare_design_auto_drops_only_malformed_runtime_optional_sections():
    from openfoam_agent.schemas.engineering import PrepareDesignTurn
    state = make_state()
    raw_plan = make_plan(state.intake).model_dump(mode="python")
    raw_plan["quantities_of_interest"] = [{
        "id": "badResolved", "quantity": "flow", "unit": "m3/s",
        "resolution_state": "resolved", "operation": "balance", "source_path": ""
    }]
    turn = PrepareDesignTurn.model_validate({
        "action": {"type": "design_case", "plan": raw_plan, "authoring_brief": "continue"}
    })
    assert turn.action.type == "design_case"
    assert turn.action.plan.quantities_of_interest == []


def test_v420_prepare_design_does_not_auto_drop_hard_execution_errors():
    from pydantic import ValidationError
    from openfoam_agent.schemas.engineering import PrepareDesignTurn
    state = make_state()
    raw_plan = make_plan(state.intake).model_dump(mode="python")
    raw_plan["solver_provider_id"] = ""
    try:
        PrepareDesignTurn.model_validate({
            "action": {"type": "design_case", "plan": raw_plan, "authoring_brief": "bad hard field"}
        })
    except ValidationError:
        pass
    else:
        raise AssertionError("hard execution/provider validation must not be relaxed")
