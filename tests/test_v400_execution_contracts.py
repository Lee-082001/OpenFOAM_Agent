"""v4 audit 1-7,21-23: real subprocess fixtures, never real OpenFOAM."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import pytest
from openfoam_agent.contracts.models import ResourceLimits, CompletionContract
from openfoam_agent.tools.safe_runner import SafeRunner, UnsafeCommandError
from openfoam_agent.tools.execution_policy import ExecutionContext, ProcessBudgetExceeded, ProcessBudget
from openfoam_agent.tools.openfoam import OpenFOAMTools
from openfoam_agent.tools.workspace import CaseWorkspace, WorkspaceSafetyError
from openfoam_agent.tools.parsers import parse_runtime_log, parse_runtime_stream
from openfoam_agent.engineering import CFDEngineeringAgent, EngineeringPolicy
from openfoam_agent.schemas.engineering import RunNativeOpenFOAMAction, NativeOpenFOAMCommand
from openfoam_agent.workflow.states import State
from conftest import make_state, make_plan, control_dict, FakeOpenFOAMTools, ScriptedLLM


def synthetic_runner(tmp_path, commands, *, limits=None):
    bin_dir = tmp_path / "trusted-bin"; bin_dir.mkdir(parents=True)
    for name, body in commands.items():
        path = bin_dir / name
        path.write_text("#!/bin/sh\n" + body)
        path.chmod(0o700)
    workspace = CaseWorkspace(tmp_path / "run")
    runner = SafeRunner(allowed_commands=set(commands), workspace_root=workspace.root,
        trusted_executable_roots=[bin_dir], base_env={"PATH": f"{bin_dir}:/usr/bin:/bin", "LANG": "C"},
        resource_limits=limits or ResourceLimits())
    return runner, workspace


@pytest.mark.parametrize("phase", ["prepare", "runtime_repair", "human_revision"])
def test_a01_unapproved_generic_native_cannot_reach_even_fake_runner(tmp_path, graph_path, phase):
    state = make_state()
    tools = FakeOpenFOAMTools()
    tools.run_native_command = lambda *a, **k: pytest.fail("Unapproved command reached transport")
    tools.foam_dictionary_validate = lambda *a, **k: pytest.fail("Preflight spawned before approval denial")
    agent = CFDEngineeringAgent(ScriptedLLM([]), workspace=tmp_path, capability_db=graph_path, tools=tools)
    action = RunNativeOpenFOAMAction(type="run_openfoam_command", invocation=NativeOpenFOAMCommand(
        command="foamRun", role="utility", arguments=["-solver", "incompressibleFluid"]))
    event = agent._dispatch_tool_action(action, step=1, native_execution=True, phase=phase, state=state)
    assert not event.success
    assert "approved runtime" in event.summary


def test_a01_real_runner_requires_contract_before_spawn(tmp_path):
    runner, ws = synthetic_runner(tmp_path, {"foamRun": "echo should-not-run\n"})
    with pytest.raises(UnsafeCommandError, match="approval"):
        runner.run(["foamRun", "-solver", "incompressibleFluid"], cwd=ws.case_dir)
    assert runner.budget.used == 0
    assert runner.budget.attempts == 0


def test_a02_actual_argv_and_repair_scope_are_bound(tmp_path):
    runner, ws = synthetic_runner(tmp_path, {"foamRun": "echo 'Time = 1'; echo End\n"})
    state = make_state(); plan = make_plan(state.intake)
    ws.write_text("system/controlDict", control_dict())
    state.engineering_plan=plan; state.case_seal=ws.seal(plan); state.current_state=State.SOLVE_READY
    state.approve_solve()
    context = ExecutionContext(state.execution_approval, plan, state.case_seal, ws)
    with runner.approved_execution(context):
        with pytest.raises(UnsafeCommandError.__bases__[0], match="argv"):
            runner.run(["foamRun", "-solver", "fluid"], cwd=ws.case_dir)
        result = runner.run(["foamRun", "-solver", "incompressibleFluid"], cwd=ws.case_dir)
    assert result.success and runner.budget.used == 1
    ws.write_text("system/controlDict", control_dict().replace("endTime 10", "endTime 100"))
    with pytest.raises(ValueError, match="outside numerical"):
        state.execution_approval.check_repair_files(ws.seal(plan))


@pytest.mark.parametrize("layout", [
    'libs ("libUnknownAudit.so");',
    'functions { audit { libs (libUnknownAudit.so); } }',
    'functions{audit{libs/*comment*/(\n"libUnknownAudit.so"\n);}}',
    'functions { audit { "libs" ("libUnknownAudit.so"); } }',
])
def test_a03_libs_inline_nested_comments_cannot_bypass(tmp_path, layout):
    ws = CaseWorkspace(tmp_path)
    with pytest.raises((WorkspaceSafetyError, ValueError)):
        ws.write_text("system/controlDict", layout)


def test_a03_comment_only_unknown_library_is_not_loaded(tmp_path):
    ws = CaseWorkspace(tmp_path)
    ws.write_text("system/controlDict", '// libs ("libUnknownAudit.so");\nlibs (libforces.so);\n')


def test_a05_budget_is_actual_spawns_and_cache_hits_do_not_spend(tmp_path):
    runner, ws = synthetic_runner(tmp_path, {"foamDictionary": "echo keywords\n"}, limits=ResourceLimits(max_native_processes=1))
    tools = OpenFOAMTools(runner)
    ws.write_text("system/a", "a 1;")
    ws.write_text("system/b", "b 2;")
    first = tools.foam_dictionary_validate(ws.case_dir / "system/a", cwd=ws.case_dir)
    cached = tools.foam_dictionary_validate(ws.case_dir / "system/a", cwd=ws.case_dir)
    assert first.success and cached.cache_hit and runner.budget.used == 1
    with pytest.raises(ProcessBudgetExceeded):
        tools.foam_dictionary_validate(ws.case_dir / "system/b", cwd=ws.case_dir)
    assert runner.budget.used == 1
    reloaded = ProcessBudget(limit=1, ledger_path=runner.budget.ledger_path)
    assert reloaded.used == 1
    with pytest.raises(ProcessBudgetExceeded):
        reloaded.reserve(["foamDictionary"], "validation")


def test_a05_file_hash_invalidates_dictionary_cache(tmp_path):
    runner, ws = synthetic_runner(tmp_path, {"foamDictionary": "echo keywords\n"})
    tools = OpenFOAMTools(runner)
    ws.write_text("system/a", "a 1;")
    tools.foam_dictionary_validate(ws.case_dir/"system/a", cwd=ws.case_dir)
    ws.write_text("system/a", "a 2;")
    tools.foam_dictionary_validate(ws.case_dir/"system/a", cwd=ws.case_dir)
    assert runner.budget.used == 2


def test_a22_timeout_returns_partial_disk_log_and_counts_one_spawn(tmp_path):
    runner, ws = synthetic_runner(tmp_path, {"checkMesh": "echo 'Time = 0.1'; sleep 30\n"})
    result = runner.run(["checkMesh"], cwd=ws.case_dir, timeout=1)
    assert not result.success and result.return_code == 124
    assert result.termination_reason == "timeout"
    assert "Time = 0.1" in Path(result.log_path).read_text()
    assert result.process_group_terminated and runner.budget.used == 1


def test_a23_large_log_is_bounded_in_memory_and_complete_on_disk(tmp_path):
    runner, ws = synthetic_runner(tmp_path, {"checkMesh": "head -c 1000000 /dev/zero | tr '\\0' x; printf '\\nMesh OK.\\n'\n"})
    result = runner.run(["checkMesh"], cwd=ws.case_dir, timeout=10)
    assert result.success and result.output_truncated
    assert len(result.stdout.encode()) <= 65536
    assert result.output_bytes > 1_000_000
    assert hashlib.sha256(Path(result.log_path).read_bytes()).hexdigest() == result.log_sha256


def test_a23_output_budget_stops_process(tmp_path):
    runner, ws = synthetic_runner(tmp_path, {"checkMesh": "yes excessive-output\n"}, limits=ResourceLimits(max_output_bytes=1024))
    result = runner.run(["checkMesh"], cwd=ws.case_dir, timeout=5)
    assert result.termination_reason == "output_limit" and result.return_code == 125
    assert Path(result.log_path).stat().st_size <= 1024


@pytest.mark.parametrize("text", ["Time = 0\nEnd\n", "Time = 0.5\nEnd\n", "End\n"])
def test_a21_exit_is_not_requested_completion(text):
    result = parse_runtime_log(text, return_code=0, contract=CompletionContract(mode="transient", end_time=1))
    assert result.process_success
    assert not result.success and not result.completed


def test_a21_contract_progress_and_bounded_parser():
    contract = CompletionContract(mode="transient", end_time=1000)
    import io
    raw = "".join(f"Time = {i}\nSolving for p, Initial residual = 0.1, Final residual = 0.01\n" for i in range(1,1001)) + "End\n"
    result = parse_runtime_stream(io.BytesIO(raw.encode()), return_code=0, contract=contract)
    assert result.completed and result.success
    assert result.residual_sample_count == 1000 and len(result.residuals) == 256
    assert result.residuals_truncated
    assert not result.physical_goal_verified and not result.numerical_quality_verified
    assert result.outputs_verified is None


def test_a21_steady_requires_outer_not_inner_residual_convergence():
    contract = CompletionContract(mode="steady", residual_thresholds=[{"field":"p","value":1e-5}], consecutive_samples=3)
    log = "".join(f"Time = {i}\nSolving for p, Initial residual = 0.1, Final residual = 1e-10\n" for i in range(1,4))+"End\n"
    assert not parse_runtime_log(log, return_code=0, contract=contract).completed
    good=log.replace("Initial residual = 0.1", "Initial residual = 1e-6")
    assert parse_runtime_log(good, return_code=0, contract=contract).completed
