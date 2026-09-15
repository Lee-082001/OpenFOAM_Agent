from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import openfoam_agent
from conftest import control_dict, make_plan, make_state
from openfoam_agent.contracts.models import CompletionContract
from openfoam_agent.conversation import ConversationSession
from openfoam_agent.engineering.design_seal import DesignSealError, materialize_engineering_plan
from openfoam_agent.runtime.completion import compile_runtime_contract, verify_result_outputs
from openfoam_agent.schemas.engineering import EngineeringDesign, EngineeringPlanPatch, OpenFOAMExecutionSpec
from openfoam_agent.tools.parsers import parse_runtime_log
from openfoam_agent.tools.workspace import CaseWorkspace
from openfoam_agent.workflow.states import State


class _Catalog:
    def provider(self, provider_id):
        items = {
            "driver.foamRun": SimpleNamespace(
                id="driver.foamRun", name="foamRun", provider_type="execution_driver", openfoam_version="14"
            ),
            "solver.incompressibleFluid": SimpleNamespace(
                id="solver.incompressibleFluid", name="incompressibleFluid", provider_type="solver_module", openfoam_version="14"
            ),
        }
        return items.get(provider_id)


def _design(*, completion):
    return EngineeringDesign(
        case_name="resultContract",
        execution=OpenFOAMExecutionSpec(
            driver="foamRun",
            driver_provider_id="driver.foamRun",
            solver_module="incompressibleFluid",
            solver_provider_id="solver.incompressibleFluid",
        ),
        problem_interpretation="Transient synthetic result-contract regression.",
        temporal_behavior="transient",
        motion_kind="static",
        mesh_motion_requirement="static",
        mesh_strategy="Synthetic static mesh.",
        completion=completion,
        required_case_files=["system/controlDict", "0/U", "0/p"],
    )


def _field(value: str) -> str:
    return (
        "FoamFile\n{\n    version 2.0;\n    format ascii;\n    class volScalarField;\n}\n"
        f"internalField uniform {value};\n"
    )


def test_v522_staged_design_requires_explicit_result_fields():
    state = make_state()
    missing = _design(completion=None)
    try:
        materialize_engineering_plan(missing, state, _Catalog())
    except DesignSealError as exc:
        assert "completion.required_result_fields" in str(exc)
    else:
        raise AssertionError("staged design without a result-output contract was sealed")


def test_v522_initial_fields_are_not_inferred_as_required_final_outputs(tmp_path):
    state = make_state()
    plan = make_plan(state.intake)
    plan.required_case_files = ["system/controlDict", "0/U", "0/p"]
    plan.completion = None
    ws = CaseWorkspace(tmp_path)
    ws.write_text("system/controlDict", control_dict())
    contract = compile_runtime_contract(plan, ws, wall_seconds=60)
    assert contract.execution_bound.required_result_fields == []
    assert any("No explicit required result fields" in item for item in contract.execution_bound.warnings)


def test_v522_declared_temperature_outputs_do_not_require_initial_pressure_at_final_time(tmp_path):
    state = make_state()
    plan = make_plan(state.intake)
    plan.required_case_files = [
        "system/controlDict",
        "0/battery/T", "0/heater/T",
        "0/battery/p", "0/heater/p",
    ]
    plan.completion = CompletionContract(
        mode="transient", end_time=300.0,
        required_result_fields=["battery/T", "heater/T"],
    )
    ws = CaseWorkspace(tmp_path)
    ws.write_text("system/controlDict", control_dict().replace("endTime 10;", "endTime 300;"))
    contract = compile_runtime_contract(plan, ws, wall_seconds=3600)
    assert contract.execution_bound.required_result_fields == ["battery/T", "heater/T"]

    battery = ws.case_dir / "300" / "battery" / "T"
    heater = ws.case_dir / "300" / "heater" / "T"
    battery.parent.mkdir(parents=True, exist_ok=True)
    heater.parent.mkdir(parents=True, exist_ok=True)
    battery.write_text(_field("320"), encoding="utf-8")
    heater.write_text(_field("340"), encoding="utf-8")
    ok, failures, evidence = verify_result_outputs(ws, plan, contract, 300.0, before={})
    assert ok, failures
    assert {item["path"] for item in evidence} == {"300/battery/T", "300/heater/T"}
    assert not (ws.case_dir / "300/battery/p").exists()
    assert not (ws.case_dir / "300/heater/p").exists()


def test_v522_time_300_clean_run_with_declared_outputs_is_accept_eligible(tmp_path):
    state = make_state()
    state.current_state = State.RESULT_REVIEW_REQUIRED
    contract = CompletionContract(
        mode="transient", start_time=0.0, end_time=300.0,
        required_result_fields=["battery/T", "heater/T"],
    )
    result = parse_runtime_log("Time = 0.5\nTime = 300\nEnd\n", return_code=0, contract=contract)
    assert result.success and result.process_success and result.termination_verified
    state.simulation = result.model_copy(update={"outputs_verified": True})
    assert state.result_acceptance_blockers() == []
    state.accept_result()
    assert state.current_state == State.COMPLETE


def test_v522_accept_blocker_is_reported_without_cli_traceback(capsys):
    import openfoam_agent.cli as cli

    state = make_state()
    state.current_state = State.RESULT_REVIEW_REQUIRED
    contract = CompletionContract(mode="transient", end_time=300.0, required_result_fields=["battery/T"])
    result = parse_runtime_log("Time = 300\nEnd\n", return_code=0, contract=contract)
    state.simulation = result.model_copy(update={"success": False, "outputs_verified": False})

    session = ConversationSession()
    session.pending_workflow_state = state
    cli._accept_session(session, None, None, "codex", None)
    text = capsys.readouterr().out
    assert "[RESULT-ACCEPT] BLOCKED" in text
    assert "Required final result artifacts" in text
    assert state.current_state == State.RESULT_REVIEW_REQUIRED



def test_v522_revision_cannot_clear_result_artifact_contract():
    state = make_state()
    plan = make_plan(state.intake)
    plan.required_case_files = ["system/controlDict", "0/U"]
    plan.completion = CompletionContract(mode="transient", end_time=10.0, required_result_fields=["U"])
    patch = EngineeringPlanPatch(completion=None)
    try:
        patch.apply(plan)
    except ValueError as exc:
        assert "result-artifact contract" in str(exc)
    else:
        raise AssertionError("revision cleared the result-artifact contract")


def test_v522_version_metadata():
    assert openfoam_agent.__version__ == "5.2.2"
