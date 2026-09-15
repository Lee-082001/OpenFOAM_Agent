from __future__ import annotations

import pytest

from conftest import control_dict, make_plan, make_state
from openfoam_agent.contracts.execution import ExecutionApproval
from openfoam_agent.tools.workspace import CaseWorkspace


def _approved(tmp_path):
    state = make_state()
    plan = make_plan(state.intake)
    ws = CaseWorkspace(tmp_path)
    ws.write_text("system/controlDict", control_dict())
    ws.write_text("system/fvSolution", "FoamFile { version 2.0; format ascii; class dictionary; object fvSolution; }\nsolvers {}\n")
    ws.write_text("system/fvSchemes", "FoamFile { version 2.0; format ascii; class dictionary; object fvSchemes; }\nddtSchemes {}\n")
    seal = ws.seal(plan)
    return ws, plan, ExecutionApproval.issue(plan, seal)


def test_runtime_repair_allows_only_numerical_control_entries_in_control_dict(tmp_path):
    ws, _plan, approval = _approved(tmp_path)
    original = ws.read_text("system/controlDict")
    proposed = original.replace("deltaT 0.01;", "deltaT 0.005;")
    approval.check_repair_delta(ws, {"system/controlDict": proposed})


def test_runtime_repair_rejects_control_dict_execution_or_output_contract_changes(tmp_path):
    ws, _plan, approval = _approved(tmp_path)
    original = ws.read_text("system/controlDict")
    with pytest.raises(ValueError, match="unauthorized controlDict entries"):
        approval.check_repair_delta(
            ws,
            {"system/controlDict": original.replace("endTime 10;", "endTime 100;")},
        )
    with pytest.raises(ValueError, match="unauthorized controlDict entries"):
        approval.check_repair_delta(
            ws,
            {"system/controlDict": original.replace("solver incompressibleFluid;", "solver otherSolver;")},
        )


def test_runtime_repair_keeps_fvsolution_and_fvschemes_as_numerical_surfaces(tmp_path):
    ws, _plan, approval = _approved(tmp_path)
    approval.check_repair_delta(ws, {"system/fvSolution": ws.read_text("system/fvSolution") + "relaxationFactors {}\n"})
    approval.check_repair_delta(ws, {"system/fvSchemes": ws.read_text("system/fvSchemes") + "gradSchemes {}\n"})


def test_runtime_repair_rejects_physics_file_even_before_mutation(tmp_path):
    ws, _plan, approval = _approved(tmp_path)
    with pytest.raises(ValueError, match="outside numerical-control authorization"):
        approval.check_repair_delta(ws, {"constant/physicalProperties": "FoamFile {}\n"})


def test_user_bound_delta_t_cannot_be_changed_by_runtime_repair(tmp_path):
    from openfoam_agent.schemas.engineering import CaseContentAssertion, ConfirmedFactBinding

    state = make_state()
    base = make_plan(state.intake)
    binding = ConfirmedFactBinding(
        fact_id="objective.primary",
        case_files=["system/controlDict"],
        case_assertions=[CaseContentAssertion(path="system/controlDict", entry_path="deltaT", expected_value="0.01")],
    )
    plan = base.model_copy(update={
        "confirmed_fact_bindings": [*base.confirmed_fact_bindings, binding],
        "required_case_files": [*base.required_case_files, "system/controlDict"],
    })
    ws = CaseWorkspace(tmp_path)
    ws.write_text("system/controlDict", control_dict())
    approval = ExecutionApproval.issue(plan, ws.seal(plan))
    proposed = ws.read_text("system/controlDict").replace("deltaT 0.01;", "deltaT 0.005;")
    with pytest.raises(ValueError, match="frozen user requirement"):
        approval.check_repair_delta(ws, {"system/controlDict": proposed})


def test_authorized_control_dict_hash_passes_post_mutation_seal_check(tmp_path):
    ws, plan, approval = _approved(tmp_path)
    proposed = ws.read_text("system/controlDict").replace("deltaT 0.01;", "deltaT 0.005;")
    approval.check_repair_delta(ws, {"system/controlDict": proposed})
    ws.write_text("system/controlDict", proposed)
    approval.check_repair_files(ws.seal(plan))
