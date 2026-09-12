from __future__ import annotations

import json
from types import SimpleNamespace

from conftest import make_state
from openfoam_agent.engineering.design_seal import materialize_engineering_plan
from openfoam_agent.schemas.engineering import (
    DesignCaseAction,
    EngineeringDesign,
    EngineeringDesignDefault,
    OpenFOAMExecutionSpec,
    PrepareDesignTurn,
)


class _Catalog:
    def __init__(self):
        self.items = {
            "driver.foamRun": SimpleNamespace(
                id="driver.foamRun", name="foamRun", provider_type="execution_driver", openfoam_version="14"
            ),
            "solver.incompressibleFluid": SimpleNamespace(
                id="solver.incompressibleFluid", name="incompressibleFluid", provider_type="solver_module", openfoam_version="14"
            ),
        }

    def provider(self, provider_id):
        return self.items.get(provider_id)


def _design():
    return EngineeringDesign(
        case_name="squareWake",
        execution=OpenFOAMExecutionSpec(
            driver="foamRun",
            driver_provider_id="driver.foamRun",
            solver_module="incompressibleFluid",
            solver_provider_id="solver.incompressibleFluid",
        ),
        problem_interpretation="Transient external flow around a stationary square obstacle.",
        temporal_behavior="transient",
        motion_kind="static",
        mesh_motion_requirement="static",
        mesh_strategy="Agent-selected exploratory 2D mesh strategy.",
        engineering_defaults=[
            EngineeringDesignDefault(
                parameter="domain extent", value="representative", rationale="Exploratory case"
            )
        ],
        required_case_files=["0/U", "0/p", "system/controlDict"],
    )


def test_prepare_design_schema_excludes_controller_owned_audit_fields():
    schema = json.dumps(PrepareDesignTurn.model_json_schema(), ensure_ascii=False)
    assert "confirmed_intake_sha256" not in schema
    assert "confirmed_fact_ids" not in schema
    assert "confirmed_fact_bindings" not in schema
    assert "EngineeringEvidence" not in schema
    assert "implementation_evidence_bindings" not in schema


def test_controller_seals_frozen_identity_and_provider_version():
    state = make_state()
    plan = materialize_engineering_plan(_design(), state, _Catalog())
    fact_ids = [fact.id for fact in state.intake.facts if fact.category != "context"]
    assert plan.confirmed_intake_sha256 == state.intake_digest
    assert plan.confirmed_fact_ids == fact_ids
    assert [item.fact_id for item in plan.confirmed_fact_bindings] == fact_ids
    assert plan.openfoam_version == "14"
    assert plan.solver == "incompressibleFluid"
    assert plan.solver_provider_id == "solver.incompressibleFluid"
    assert plan.evidence == []
    assert plan.implementation_evidence_bindings == []
    assert all(item.evidence_ids == [] for item in plan.engineering_defaults)


def test_llm_cannot_override_controller_owned_fields_through_design_case():
    payload = _design().model_dump(mode="python")
    payload.update(
        confirmed_intake_sha256="0" * 64,
        confirmed_fact_ids=["fake.fact"],
        confirmed_fact_bindings=[],
        evidence=[{"evidence_id": "ev_cap_" + "0" * 20}],
        openfoam_version="13",
    )
    action = DesignCaseAction.model_validate({"type": "design_case", "plan": payload})
    state = make_state()
    plan = materialize_engineering_plan(action.plan, state, _Catalog())
    assert plan.confirmed_intake_sha256 == state.intake_digest
    assert "fake.fact" not in plan.confirmed_fact_ids
    assert plan.openfoam_version == "14"
    assert plan.evidence == []


def test_legacy_full_plan_input_is_projected_to_design_view():
    state = make_state()
    sealed = materialize_engineering_plan(_design(), state, _Catalog())
    action = DesignCaseAction(type="design_case", plan=sealed)
    assert isinstance(action.plan, EngineeringDesign)
    assert not hasattr(action.plan, "confirmed_intake_sha256")
    assert action.plan.execution.driver == "foamRun"
