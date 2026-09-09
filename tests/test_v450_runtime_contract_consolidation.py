from __future__ import annotations

from pathlib import Path

from conftest import FakeOpenFOAMTools, ScriptedLLM, foam_header, make_plan, make_state
from openfoam_agent.cli import _current_state_message
from openfoam_agent.contracts.models import CompletionContract, ResidualThreshold
from openfoam_agent.engineering import CFDEngineeringAgent
from openfoam_agent.runtime.completion import compile_runtime_contract, completion_contract
from openfoam_agent.runtime.orchestrator import RuntimeOrchestrator
from openfoam_agent.schemas.engineering import (
    CaseAuthoringAction,
    CaseBundleFile,
    EngineeringDefaultAssumption,
    NativeOpenFOAMCommand,
    RepairCasePlanAction,
)
from openfoam_agent.schemas.simulation import RuntimePolicy
from openfoam_agent.tools.parsers import parse_runtime_log
from openfoam_agent.tools.workspace import CaseWorkspace
from openfoam_agent.workflow.states import State
from openfoam_agent.schemas.common import ToolResult


def _steady_control(end_time=2000):
    return foam_header("system/controlDict") + (
        "application foamRun;\n"
        "solver incompressibleFluid;\n"
        "startFrom startTime;\n"
        "startTime 0;\n"
        "stopAt endTime;\n"
        f"endTime {end_time};\n"
        "writeControl timeStep;\n"
        "writeInterval 100;\n"
    )


def _steady_default():
    return EngineeringDefaultAssumption(
        parameter="steady_iteration_controls",
        value=(
            "Maximum 2000 iterations; pressure relaxation 0.3, velocity relaxation 0.7; "
            "target residuals U 1e-7 and p 1e-6; outlet split stable within 0.1% over "
            "100 iterations and net volume imbalance below 0.1% of inlet flow."
        ),
        unit="dimensionless",
        basis="common_practice",
        rationale="Agent-selected bounded steady solve and review criteria.",
    )


def _steady_plan(tmp_path):
    state = make_state()
    plan = make_plan(state.intake)
    plan.temporal_behavior = "steady"
    plan.engineering_defaults = [_steady_default()]
    plan.required_case_files = ["system/controlDict", "0/U", "0/p"]
    ws = CaseWorkspace(tmp_path)
    ws.write_text("system/controlDict", _steady_control())
    ws.write_text("0/U", foam_header("0/U", "volVectorField") + "internalField uniform (0 0 0);\n")
    ws.write_text("0/p", foam_header("0/p", "volScalarField") + "internalField uniform 0;\n")
    return state, plan, ws


def test_v450_authoring_drops_redundant_foam_dictionary_native_probes():
    action = CaseAuthoringAction.model_validate({
        "type": "author_case",
        "goal": "author then validate",
        "files": [{"path": "system/controlDict", "content": "x"}],
        "native_pipeline": [
            {"command": "foamDictionary", "arguments": ["system/controlDict"], "role": "validation"},
            {"command": "foamDictionary", "arguments": ["system/fvSchemes"], "role": "validation"},
            {"command": "blockMesh", "role": "mesh"},
            {"command": "checkMesh", "role": "mesh_validation"},
        ],
    })
    assert [item.command for item in action.native_pipeline] == ["blockMesh", "checkMesh"]


def test_v450_repair_drops_redundant_foam_dictionary_probe():
    action = RepairCasePlanAction.model_validate({
        "type": "repair_case_plan",
        "diagnosis": "repair case",
        "replacement_files": [{"path": "system/controlDict", "content": "x"}],
        "native_pipeline": [
            {"command": "foamDictionary", "arguments": ["system/controlDict"], "role": "validation"},
            {"command": "checkMesh", "role": "mesh_validation"},
        ],
    })
    assert [item.command for item in action.native_pipeline] == ["checkMesh"]


def test_v450_compile_steady_runtime_contract_from_existing_agent_choices(tmp_path):
    _, plan, ws = _steady_plan(tmp_path)
    contract = compile_runtime_contract(plan, ws, wall_seconds=3600)
    assert contract.execution_bound.mode == "steady"
    assert contract.execution_bound.max_iterations == 2000
    assert contract.execution_bound.wall_seconds == 3600
    thresholds = {item.field: item.value for item in contract.result_acceptance.residual_thresholds}
    assert thresholds == {"U": 1e-7, "p": 1e-6}
    # The residual subset is compiled, but outlet-split/conservation prose remains review-only.
    assert not contract.result_acceptance.criteria_complete
    assert contract.result_acceptance.source == "engineering_defaults"


def test_v450_legacy_completion_adapter_no_longer_blocks_steady_without_explicit_contract(tmp_path):
    _, plan, ws = _steady_plan(tmp_path)
    contract = completion_contract(plan, ws)
    assert contract.mode == "steady"
    assert {item.field for item in contract.residual_thresholds} == {"U", "p"}


def test_v450_bounded_steady_execution_succeeds_even_when_result_acceptance_is_incomplete(tmp_path):
    _, plan, ws = _steady_plan(tmp_path)
    contract = compile_runtime_contract(plan, ws, wall_seconds=3600)
    result = parse_runtime_log(
        "Time = 1\n"
        "Solving for U, Initial residual = 1e-2, Final residual = 1e-3\n"
        "Solving for p, Initial residual = 1e-2, Final residual = 1e-3\n"
        "End\n",
        return_code=0,
        contract=contract,
    )
    assert result.success and result.completed and result.termination_verified
    assert not result.numerical_quality_verified
    assert result.acceptance_warnings
    assert not result.evidence_failures


def test_v450_explicit_complete_steady_acceptance_can_be_machine_verified(tmp_path):
    state = make_state()
    plan = make_plan(state.intake)
    plan.temporal_behavior = "steady"
    plan.required_case_files = ["system/controlDict"]
    plan.completion = CompletionContract(
        mode="steady",
        resolution_state="resolved",
        residual_thresholds=[ResidualThreshold(field="U", value=1e-7)],
        consecutive_samples=3,
    )
    ws = CaseWorkspace(tmp_path)
    ws.write_text("system/controlDict", _steady_control())
    contract = compile_runtime_contract(plan, ws, wall_seconds=3600)
    log = "".join(
        f"Time = {i}\nSolving for U, Initial residual = 1e-8, Final residual = 1e-9\n"
        for i in range(1, 4)
    ) + "End\n"
    result = parse_runtime_log(log, return_code=0, contract=contract)
    assert result.success
    assert result.numerical_quality_verified


def test_v450_current_state_message_ignores_stale_superseded_failure():
    state = make_state()
    state.current_state = State.INTAKE_REVIEW_REQUIRED
    state.history = [
        {"from": "INTAKE_ANALYSIS", "to": "INTAKE_REVIEW_REQUIRED", "note": "intake ready"},
        {"from": "ENGINEERING", "to": "FAILED", "note": "stale timeout from superseded attempt"},
    ]
    assert _current_state_message(state) == "intake ready"


def test_v450_interactive_solve_path_does_not_require_duplicate_steady_completion_contract(tmp_path, graph_path):
    state, plan, ws = _steady_plan(tmp_path)
    tools = FakeOpenFOAMTools(
        foam_runs=[ToolResult(
            success=True,
            command=["foamRun"],
            return_code=0,
            stdout="Time = 1\nEnd\n",
        )]
    )
    tools.synthetic_output_fields = ["U", "p"]
    agent = CFDEngineeringAgent(
        ScriptedLLM([]), workspace=tmp_path, capability_db=graph_path, tools=tools
    )
    # Agent owns a separate workspace object at the same path; write/adopt the case there.
    agent.workspace.write_text("system/controlDict", _steady_control())
    agent.workspace.write_text("0/U", foam_header("0/U", "volVectorField") + "internalField uniform (0 0 0);\n")
    agent.workspace.write_text("0/p", foam_header("0/p", "volScalarField") + "internalField uniform 0;\n")
    state.engineering_plan = plan
    state.case_dir = str(agent.workspace.case_dir)
    state.case_seal = agent.workspace.seal(plan)
    state.current_state = State.SOLVE_READY
    state.approve_solve()
    runtime = RuntimeOrchestrator(tools, agent, RuntimePolicy(max_attempts=1, solver_timeout_seconds=60))
    final = runtime.run(state)
    assert final.current_state == State.EXECUTION_DONE
    assert final.runtime_contract is not None
    assert final.runtime_contract.execution_bound.max_iterations == 2000
    assert final.simulation is not None and final.simulation.success
    assert not final.simulation.numerical_quality_verified
