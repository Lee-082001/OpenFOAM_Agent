from __future__ import annotations

from openfoam_agent.contracts.models import CompletionContract

from types import SimpleNamespace

import pytest

from conftest import FakeOpenFOAMTools, foam_header, make_state
from openfoam_agent.contracts.models import PhysicalQuantity
from openfoam_agent.engineering.design_seal import materialize_engineering_plan
from openfoam_agent.schemas.engineering import (
    CaseContentAssertion,
    ConfirmedFactBinding,
    EngineeringDesign,
    EngineeringPlan,
    NumericEvidenceTerm,
    NumericRelationAssertion,
    OpenFOAMExecutionSpec,
)
from openfoam_agent.schemas.intake import CFDIntakeSpec, IntakeFact
from openfoam_agent.tools.workspace import CaseWorkspace
from openfoam_agent.verification.safety import DeterministicSafetyGate


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


def _typed_intake() -> CFDIntakeSpec:
    return CFDIntakeSpec(
        semantic_contract_version="2",
        title="Typed inlet velocity",
        facts=[
            IntakeFact(
                id="request.summary",
                category="context",
                label="Request",
                value="Use an inlet speed of 2 m/s.",
                source="derived",
                reason="Normalized request.",
            ),
            IntakeFact(
                id="classification.problem_type",
                category="classification",
                label="Problem",
                value="internal_flow",
                source="derived",
                reason="Routing interpretation.",
                depends_on=["request.summary"],
            ),
            IntakeFact(
                id="boundary.inlet_velocity",
                category="boundary",
                label="Inlet velocity",
                value="2",
                unit="m/s",
                quantity=PhysicalQuantity(
                    quantity="velocity",
                    value=2.0,
                    unit="m/s",
                    dimensions=(0, 1, -1, 0, 0, 0, 0),
                    patch="inlet",
                ),
                source="user",
                evidence="2 m/s",
            ),
        ],
        status="ready_for_review",
    )


def _plan(intake: CFDIntakeSpec, bindings=None) -> EngineeringPlan:
    return EngineeringPlan(
        case_name="typedVelocity",
        solver="incompressibleFluid",
        solver_provider_id="solver.incompressibleFluid",
        openfoam_version="14",
        problem_interpretation="Internal flow with a prescribed inlet velocity.",
        temporal_behavior="steady",
        motion_kind="static",
        mesh_motion_requirement="static",
        mesh_strategy="test mesh",
        decisions=[],
        assumptions=[],
        confirmed_fact_ids=[fact.id for fact in intake.facts if fact.category != "context"],
        confirmed_fact_bindings=list(bindings or []),
        required_case_files=["0/U", "system/controlDict"],
        confirmed_intake_sha256=intake.digest(),
    )


def _workspace(tmp_path) -> CaseWorkspace:
    ws = CaseWorkspace(tmp_path)
    ws.write_text(
        "system/controlDict",
        foam_header("system/controlDict")
        + "solver incompressibleFluid;\nstartFrom startTime;\nstartTime 0;\nendTime 10;\ndeltaT 0.1;\n",
    )
    ws.write_text(
        "0/U",
        foam_header("0/U", "volVectorField")
        + "dimensions [0 1 -1 0 0 0 0];\ninternalField uniform (2 0 0);\nboundaryField { inlet { type fixedValue; value uniform (2 0 0); } }\n",
    )
    return ws


def test_typed_user_quantity_without_machine_binding_is_hard_failure(tmp_path):
    intake = _typed_intake()
    ws = _workspace(tmp_path)
    result = DeterministicSafetyGate(FakeOpenFOAMTools(), ws).validate_plan(_plan(intake), intake)
    assert not result.valid
    assert any("Critical requirement assurance missing for boundary.inlet_velocity" in item for item in result.failures)


def test_typed_user_quantity_numeric_relation_is_verified_against_case(tmp_path):
    intake = _typed_intake()
    ws = _workspace(tmp_path)
    binding = ConfirmedFactBinding(
        fact_id="boundary.inlet_velocity",
        case_files=["0/U"],
        numeric_relation=NumericRelationAssertion(
            numerator=[NumericEvidenceTerm(path="0/U", entry_path="boundaryField.inlet.value", number_index=0)],
            denominator=[],
            relative_tolerance=1e-9,
        ),
    )
    result = DeterministicSafetyGate(FakeOpenFOAMTools(), ws).validate_plan(_plan(intake, [binding]), intake)
    assert result.valid, result.failures
    ws.write_text(
        "0/U",
        foam_header("0/U", "volVectorField")
        + "dimensions [0 1 -1 0 0 0 0];\ninternalField uniform (1 0 0);\nboundaryField { inlet { type fixedValue; value uniform (1 0 0); } }\n",
    )
    bad = DeterministicSafetyGate(FakeOpenFOAMTools(), ws).validate_plan(_plan(intake, [binding]), intake)
    assert not bad.valid
    assert any("numeric semantic assertion" in item.lower() for item in bad.failures)


def test_staged_design_requirement_binding_survives_controller_seal():
    state = make_state()
    # Existing fixture facts are untyped, so this verifies transport/sealing rather
    # than mandatory typed-quantity policy.
    binding = ConfirmedFactBinding(
        fact_id="objective.primary",
        case_files=["0/U"],
        case_assertions=[CaseContentAssertion(path="0/U", anchor="internalField")],
    )
    design = EngineeringDesign(
        case_name="squareWake",
        execution=OpenFOAMExecutionSpec(
            driver="foamRun",
            driver_provider_id="driver.foamRun",
            solver_module="incompressibleFluid",
            solver_provider_id="solver.incompressibleFluid",
        ),
        problem_interpretation="Transient wake.",
        temporal_behavior="transient",
        motion_kind="static",
        mesh_motion_requirement="static",
        mesh_strategy="test",
        completion=CompletionContract(mode="transient", end_time=10.0, required_result_fields=["U"]),
        requirement_bindings=[binding],
        required_case_files=["0/U", "system/controlDict"],
    )
    plan = materialize_engineering_plan(design, state, _Catalog())
    assert plan.confirmed_fact_bindings == [binding]


def test_reynolds_requirement_false_acceptance_is_rejected(tmp_path):
    intake = CFDIntakeSpec(
        semantic_contract_version="2",
        title="Reynolds requirement",
        facts=[
            IntakeFact(
                id="request.summary", category="context", label="Request", value="Re=1000",
                source="derived", reason="Normalized request."
            ),
            IntakeFact(
                id="classification.problem_type", category="classification", label="Problem",
                value="external_flow", source="derived", reason="Routing interpretation.",
                depends_on=["request.summary"]
            ),
            IntakeFact(
                id="operating.reynolds_number", category="scale", label="Reynolds number", value="1000",
                unit="1",
                quantity=PhysicalQuantity(quantity="Reynolds number", value=1000.0, unit="1", dimensions=(0,0,0,0,0,0,0)),
                source="user", evidence="Re=1000"
            ),
        ],
        status="ready_for_review",
    )
    ws = CaseWorkspace(tmp_path)
    ws.write_text("system/controlDict", foam_header("system/controlDict") + "solver incompressibleFluid;\n")
    ws.write_text("0/U", foam_header("0/U", "volVectorField") + "dimensions [0 1 -1 0 0 0 0];\ninternalField uniform (1 0 0);\n")
    ws.write_text("constant/geometryProperties", foam_header("constant/geometryProperties") + "characteristicLength [0 1 0 0 0 0 0] 1;\n")
    ws.write_text("constant/physicalProperties", foam_header("constant/physicalProperties") + "nu [0 2 -1 0 0 0 0] 0.001;\n")
    binding = ConfirmedFactBinding(
        fact_id="operating.reynolds_number",
        case_files=["0/U", "constant/geometryProperties", "constant/physicalProperties"],
        numeric_relation=NumericRelationAssertion(
            numerator=[
                NumericEvidenceTerm(path="0/U", entry_path="internalField"),
                NumericEvidenceTerm(path="constant/geometryProperties", entry_path="characteristicLength"),
            ],
            denominator=[NumericEvidenceTerm(path="constant/physicalProperties", entry_path="nu")],
            relative_tolerance=1e-6,
        ),
    )
    plan = EngineeringPlan(
        case_name="reCase", solver="incompressibleFluid", solver_provider_id="solver.incompressibleFluid",
        openfoam_version="14", problem_interpretation="Re-controlled flow", temporal_behavior="steady",
        motion_kind="static", mesh_motion_requirement="static", mesh_strategy="test", decisions=[], assumptions=[],
        confirmed_fact_ids=["classification.problem_type", "operating.reynolds_number"],
        confirmed_fact_bindings=[binding],
        required_case_files=["0/U", "constant/geometryProperties", "constant/physicalProperties", "system/controlDict"],
        confirmed_intake_sha256=intake.digest(),
    )
    gate = DeterministicSafetyGate(FakeOpenFOAMTools(), ws)
    assert gate.validate_plan(plan, intake).valid
    ws.write_text("constant/physicalProperties", foam_header("constant/physicalProperties") + "nu [0 2 -1 0 0 0 0] 0.002;\n")
    failed = gate.validate_plan(plan, intake)
    assert not failed.valid
    assert any("recomputes to 500.0" in item and "target 1000.0" in item for item in failed.failures)


def test_untyped_user_reynolds_scale_is_still_mandatory():
    intake = CFDIntakeSpec(
        semantic_contract_version="2",
        title="Untyped Reynolds",
        facts=[
            IntakeFact(id="request.summary", category="context", label="Request", value="Re=1000", source="derived", reason="Normalized"),
            IntakeFact(id="classification.problem_type", category="classification", label="Problem", value="external_flow", source="derived", reason="Routing", depends_on=["request.summary"]),
            IntakeFact(id="operating.reynolds_number", category="scale", label="Reynolds", value="1000", source="user", evidence="Re=1000"),
        ],
        status="ready_for_review",
    )
    from openfoam_agent.verification.semantic_assurance import expectation_for_fact
    expectation = expectation_for_fact(intake.fact("operating.reynolds_number"))
    assert expectation.mode == "machine_assertion_required"
