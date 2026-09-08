"""Final integration edges. CLI, outputs and resources use synthetic fixtures."""
from __future__ import annotations
import hashlib
from pathlib import Path
from types import SimpleNamespace
import pytest
from conftest import make_state, make_plan, control_dict, ScriptedLLM
from openfoam_agent.contracts.models import CompletionContract, ResourceLimits, QuantityOfInterest
from openfoam_agent.tools.workspace import CaseWorkspace
from openfoam_agent.tools.execution_policy import ExecutionContext
from openfoam_agent.runtime.completion import completion_contract, verify_result_outputs, snapshot_result_outputs
from openfoam_agent.tools.dictionary_policy import materialize_local_includes
from openfoam_agent.workflow.states import State
from openfoam_agent.tools.parsers import parse_runtime_log
from test_v400_execution_contracts import synthetic_runner


def sealed_state(tmp_path):
    state=make_state();plan=make_plan(state.intake)
    ws=CaseWorkspace(tmp_path);ws.write_text("system/controlDict",control_dict())
    state.engineering_plan=plan;state.case_seal=ws.seal(plan);state.case_dir=str(ws.case_dir)
    state.current_state=State.SOLVE_READY
    return state,plan,ws


def test_a02_approval_binds_postprocessing_goals_and_resource_limits(tmp_path):
    state,plan,ws=sealed_state(tmp_path)
    state.approve_solve()
    plan.quantities_of_interest=[QuantityOfInterest(id="q",quantity="pressure",unit="Pa",selection="inlet",source_path="postProcessing/p.dat",operation="time_mean")]
    with pytest.raises(ValueError,match="physical implementation"):state.execution_approval.check_plan(plan)


def test_a04_local_include_hash_cycle_and_escape(tmp_path):
    p=tmp_path/"input";p.write_text("value 1;\n")
    expanded,records=materialize_local_includes('#include "input"\n',parent=tmp_path,root=tmp_path)
    assert expanded=="value 1;\n" and records[0]["sha256"]==hashlib.sha256(p.read_bytes()).hexdigest()
    p.write_text('#include "input"\n')
    with pytest.raises(ValueError,match="cycle"):materialize_local_includes('#include "input"\n',parent=tmp_path,root=tmp_path)
    with pytest.raises(ValueError,match="Unsafe"):materialize_local_includes('#include "../outside"\n',parent=tmp_path,root=tmp_path)


def test_a04_actual_cpu_limit_is_enforced_on_synthetic_process(tmp_path):
    runner,ws=synthetic_runner(tmp_path,{"checkMesh":"while :; do :; done\n"},limits=ResourceLimits(cpu_seconds=1,wall_seconds=5))
    result=runner.run(["checkMesh"],cwd=ws.case_dir,timeout=5)
    assert not result.success and result.return_code != 0
    assert result.termination_reason != "timeout"  # RLIMIT_CPU, not the wall timer
    assert runner.budget.used==1


def test_a05_approval_budget_is_stricter_than_runner_budget(tmp_path):
    runner,ws=synthetic_runner(tmp_path,{"checkMesh":"echo OK\n"})
    state=make_state();plan=make_plan(state.intake);ws.write_text("system/controlDict",control_dict())
    state.engineering_plan=plan;state.case_seal=ws.seal(plan);state.current_state=State.SOLVE_READY
    state.approve_solve(ResourceLimits(max_native_processes=1))
    context=ExecutionContext(state.execution_approval,plan,state.case_seal,ws)
    from openfoam_agent.tools.execution_policy import ProcessBudgetExceeded
    with runner.approved_execution(context):
        runner.run(["checkMesh"],cwd=ws.case_dir)
        with pytest.raises(ProcessBudgetExceeded):runner.run(["checkMesh"],cwd=ws.case_dir)
    assert runner.budget.used==1


def test_a21_completion_cannot_shorten_literal_control_target(tmp_path):
    state,plan,ws=sealed_state(tmp_path)
    plan.completion=CompletionContract(mode="transient",end_time=1,required_result_fields=["U"])
    with pytest.raises(ValueError,match="disagrees"):completion_contract(plan,ws)
    plan.completion.end_time=10
    assert completion_contract(plan,ws).end_time==10


def test_a21_outputs_must_be_fresh_present_and_finite(tmp_path):
    state,plan,ws=sealed_state(tmp_path)
    contract=CompletionContract(mode="transient",end_time=10,required_result_fields=["U"])
    field=ws.case_dir/"10/U";field.parent.mkdir();field.write_text("FoamFile { format ascii; }\ninternalField uniform (1 0 0);\n")
    before=snapshot_result_outputs(ws,contract)
    ok,failures,_=verify_result_outputs(ws,plan,contract,10,before=before)
    assert not ok and "stale" in failures[0]
    field.write_text("FoamFile { format ascii; }\ninternalField uniform (2 0 0);\n")
    assert verify_result_outputs(ws,plan,contract,10,before=before)[0]
    field.write_text("internalField uniform (nan 0 0);\n")
    assert not verify_result_outputs(ws,plan,contract,10)[0]
    field.unlink()
    assert not verify_result_outputs(ws,plan,contract,10)[0]


def test_a21_accept_cannot_turn_incomplete_calculation_into_complete(tmp_path):
    state,_,_=sealed_state(tmp_path);state.current_state=State.RESULT_REVIEW_REQUIRED
    state.simulation=parse_runtime_log("Time = 0\nEnd\n",return_code=0,contract=CompletionContract(mode="transient",end_time=10))
    with pytest.raises(ValueError,match="incomplete"):state.accept_result()
    assert state.current_state==State.RESULT_REVIEW_REQUIRED


def test_a27_one_shot_solve_ready_reenters_workflow_after_common_approval(tmp_path,graph_path,monkeypatch):
    import openfoam_agent.cli as cli
    state,_,_=sealed_state(tmp_path/"fixture");calls=[]
    class Workflow:
        def __init__(self,**kwargs):pass
        def run(self,incoming):
            calls.append(incoming.current_state)
            if len(calls)==1:return state
            assert incoming is state and state.execution_approval is not None and state.solve_approved
            assert state.current_state==State.SIMULATION
            state.current_state=State.RESULT_REVIEW_REQUIRED
            return state
    monkeypatch.setattr(cli,"CFDWorkflow",Workflow)
    monkeypatch.setattr(cli,"build_report",lambda *a,**k:{"fixture":True})
    final,_=cli.run_prompt(state.user_request,llm=ScriptedLLM([]),backend="openai",model=None,capability_db=graph_path,workspace_root=tmp_path/"workflow",execute_solver=True)
    assert len(calls)==2 and final is state


def test_a27_interactive_solve_uses_same_approval_contract(tmp_path,graph_path,monkeypatch):
    import openfoam_agent.cli as cli
    from openfoam_agent.conversation import ConversationSession
    state,_,_=sealed_state(tmp_path)
    session=ConversationSession();session.add_turn(state.user_request.prompt);session.pending_workflow_state=state
    calls=[]
    class Workflow:
        def __init__(self,**kwargs):pass
        def run(self,incoming):
            assert incoming.execution_approval is not None and incoming.current_state==State.SIMULATION
            calls.append(incoming);return incoming
    monkeypatch.setattr(cli,"CFDWorkflow",Workflow);monkeypatch.setattr(cli,"build_report",lambda *a,**k:{})
    monkeypatch.setattr(cli,"_emit_report",lambda *a,**k:None)
    args=cli.build_parser().parse_args(["--capability-db",str(graph_path),"--json"])
    cli._solve_session(session,args,ScriptedLLM([]),"openai",None)
    assert calls==[state]


def test_a28_solve_ready_feedback_reaches_actual_review_agent(tmp_path):
    from openfoam_agent.review import CFDFeedbackReviewAgent
    state,_,_=sealed_state(tmp_path);calls=[]
    class Probe:
        def generate(self,*args,**kwargs):calls.append(True);raise RuntimeError("synthetic review outage")
    reviewer=CFDFeedbackReviewAgent(Probe())
    with pytest.raises(RuntimeError,match="synthetic"):
        reviewer.review(state,"Check the domain size before solve.")
    assert calls and state.current_state==State.SOLVE_READY
    assert not state.solve_approved


def test_a32_packaged_profiles_match_source_bytes():
    import openfoam_agent
    root=Path(__file__).resolve().parents[1]
    for version in ("13","14"):
        name=f"openfoam{version}_capability_graph.json"
        assert (Path(openfoam_agent.__file__).parent/"data"/name).read_bytes()==(root/"config"/name).read_bytes()


def test_a23_checkmesh_reads_full_bounded_disk_log_not_tail(tmp_path):
    from openfoam_agent.verification.safety import parse_check_mesh_evidence
    runner,ws=synthetic_runner(tmp_path,{"checkMesh":"echo 'cells: 42'; yes 'log detail' | head -n 20000; echo 'Mesh OK.'\n"})
    result=runner.run(["checkMesh"],cwd=ws.case_dir,timeout=10)
    assert "cells:" not in result.stdout and result.output_truncated
    evidence=parse_check_mesh_evidence(result)
    assert evidence.passed and evidence.cell_count==42 and evidence.raw_log_sha256==result.log_sha256
    with Path(result.log_path).open("ab") as f:f.write(b"changed\n")
    assert not parse_check_mesh_evidence(result).passed
