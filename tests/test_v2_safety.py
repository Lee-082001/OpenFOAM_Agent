from __future__ import annotations

from openfoam_agent.schemas.common import ToolResult
from openfoam_agent.schemas.engineering import ConfirmedFactBinding
from openfoam_agent.tools.workspace import CaseWorkspace
from openfoam_agent.verification.safety import DeterministicSafetyGate, parse_check_mesh_evidence

from conftest import FakeOpenFOAMTools, make_intake, make_plan, mesh_ok_log


def test_safety_gate_does_not_require_template_files(tmp_path):
    intake = make_intake()
    plan = make_plan(intake)
    ws = CaseWorkspace(tmp_path)
    ws.write_text("system/controlDict", "solver incompressibleFluid;\n")
    gate = DeterministicSafetyGate(FakeOpenFOAMTools(), ws)
    result = gate.validate_plan(plan, intake)
    assert result.valid, result.failures
    assert ws.list_authored() == ["system/controlDict"]


def test_safety_gate_rejects_solver_case_mismatch(tmp_path):
    intake = make_intake()
    plan = make_plan(intake, solver="incompressibleFluid")
    ws = CaseWorkspace(tmp_path)
    ws.write_text("system/controlDict", "solver fluid;\n")
    result = DeterministicSafetyGate(FakeOpenFOAMTools(), ws).validate_plan(plan, intake)
    assert not result.valid
    assert any("solver disagrees" in item for item in result.failures)


def test_safety_gate_requires_exact_confirmed_fact_provenance(tmp_path):
    intake = make_intake()
    plan = make_plan(intake).model_copy(update={"confirmed_fact_ids": ["objective.primary"]})
    ws = CaseWorkspace(tmp_path)
    ws.write_text("system/controlDict", "solver incompressibleFluid;\n")
    result = DeterministicSafetyGate(FakeOpenFOAMTools(), ws).validate_plan(plan, intake)
    assert not result.valid
    assert any("fact provenance mismatch" in item for item in result.failures)


def test_checkmesh_parser_records_evidence_without_engineering_thresholds():
    result = ToolResult(
        success=True,
        command=["checkMesh"],
        return_code=0,
        stdout=mesh_ok_log(cells=4321),
    )
    evidence = parse_check_mesh_evidence(result)
    assert evidence.passed
    assert evidence.cell_count == 4321
    assert evidence.max_non_orthogonality == 21.5
    assert evidence.max_skewness == 1.2
    assert evidence.negative_volume_cells == 0


def test_safety_gate_binds_plan_to_exact_confirmed_intake_digest(tmp_path):
    intake = make_intake()
    plan = make_plan(intake).model_copy(update={"confirmed_intake_sha256": "0" * 64})
    ws = CaseWorkspace(tmp_path)
    ws.write_text("system/controlDict", "solver incompressibleFluid;\n")
    result = DeterministicSafetyGate(FakeOpenFOAMTools(), ws).validate_plan(plan, intake)
    assert not result.valid
    assert any("exact confirmed intake digest" in item for item in result.failures)


def test_safety_gate_requires_confirmed_fact_binding_coverage(tmp_path):
    intake = make_intake()
    plan = make_plan(intake).model_copy(update={"confirmed_fact_bindings": []})
    ws = CaseWorkspace(tmp_path)
    ws.write_text("system/controlDict", "solver incompressibleFluid;\n")
    result = DeterministicSafetyGate(FakeOpenFOAMTools(), ws).validate_plan(plan, intake)
    assert not result.valid
    assert any("implementation binding mismatch" in item for item in result.failures)


def test_safety_gate_rejects_binding_to_missing_case_file(tmp_path):
    intake = make_intake()
    plan = make_plan(intake)
    bindings = list(plan.confirmed_fact_bindings)
    bindings[0] = ConfirmedFactBinding(
        fact_id=bindings[0].fact_id,
        implementation_refs=["case:constant/physicalProperties"],
        explanation="This confirmed fact is implemented by physicalProperties.",
    )
    plan = plan.model_copy(update={"confirmed_fact_bindings": bindings})
    ws = CaseWorkspace(tmp_path)
    ws.write_text("system/controlDict", "solver incompressibleFluid;\n")
    result = DeterministicSafetyGate(FakeOpenFOAMTools(), ws).validate_plan(plan, intake)
    assert not result.valid
    assert any("references missing case file constant/physicalProperties" in item for item in result.failures)
