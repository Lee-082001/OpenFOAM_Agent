"""Real synthetic local subprocess tests plus mocked strict Linux/MPI contracts.

These tests are not executions of OpenFOAM, an MPI runtime or a live model.
"""
from copy import deepcopy
import json
import os
from pathlib import Path
from types import SimpleNamespace
import pytest
from conftest import make_state,make_plan,FakeOpenFOAMTools,ScriptedLLM,foam_header
from rc2_helpers import restart_case,field,cube,put
from test_v400_execution_contracts import synthetic_runner
from openfoam_agent.contracts.models import ResourceLimits,ParallelExecution
from openfoam_agent.contracts.execution import ExecutionApproval
from openfoam_agent.tools.safe_runner import SafeRunner,UnsafeCommandError
from openfoam_agent.tools.linux_isolation import LinuxIsolation,LinuxIsolationPolicy,IsolationUnavailable,workspace_execution_lock
from openfoam_agent.tools.workspace import CaseWorkspace
from openfoam_agent.runtime.restart import inspect_parallel_restart,prepare_parallel_restart,verify_restart,prepare_restart_state,rewrite_top_level
from openfoam_agent.runtime.parallel import prepare_parallel,verify_parallel_inputs,reconstruct_parallel
from openfoam_agent.runtime.completion import completion_contract
from openfoam_agent.engineering import CFDEngineeringAgent
from openfoam_agent.workflow.states import State
from openfoam_agent.workflow.checkpoint import CheckpointError


def test_rc2_actual_subprocess_total_storage_quota_not_only_file_size(tmp_path):
    runner,ws=synthetic_runner(tmp_path,{'foamDictionary':'mkdir burst; head -c 80000 /dev/zero > burst/a; head -c 80000 /dev/zero > burst/b\n'},limits=ResourceLimits(max_case_bytes=130000))
    result=runner.run(['foamDictionary'],cwd=ws.case_dir)
    assert not result.success and result.termination_reason=='workspace_quota'
    assert runner.budget.used==1
    with pytest.raises(UnsafeCommandError,match='quota'):runner.run(['foamDictionary'],cwd=ws.case_dir)
    assert runner.budget.used==1


def test_rc2_actual_cumulative_output_budget_across_subprocesses(tmp_path):
    runner,ws=synthetic_runner(tmp_path,{'foamDictionary':'head -c 800 /dev/zero\n'},limits=ResourceLimits(max_total_output_bytes=1200))
    first=runner.run(['foamDictionary'],cwd=ws.case_dir);second=runner.run(['foamDictionary'],cwd=ws.case_dir)
    assert first.success and not second.success and second.termination_reason=='output_limit'
    assert sum(row['output_bytes'] for row in runner.budget.records)==1200
    with pytest.raises(UnsafeCommandError,match='output budget'):runner.run(['foamDictionary'],cwd=ws.case_dir)


def test_rc2_actual_cumulative_wall_budget_timeout(tmp_path):
    runner,ws=synthetic_runner(tmp_path,{'foamDictionary':'sleep 1; echo done\n'},limits=ResourceLimits(total_wall_seconds=.15))
    result=runner.run(['foamDictionary'],cwd=ws.case_dir)
    assert result.termination_reason=='timeout' and not result.success
    with pytest.raises(UnsafeCommandError,match='wall budget'):runner.run(['foamDictionary'],cwd=ws.case_dir)


def test_rc2_cumulative_cpu_unknown_is_not_zero(tmp_path):
    runner,ws=synthetic_runner(tmp_path,{'foamDictionary':'echo measured-no-cgroup\n'})
    assert runner.run(['foamDictionary'],cwd=ws.case_dir).success
    runner.resource_limits=ResourceLimits(total_cpu_seconds=10)
    with pytest.raises(UnsafeCommandError,match='unknown'):runner.run(['foamDictionary'],cwd=ws.case_dir)
    assert runner.budget.used==1


def test_rc2_aggregate_cpu_contract_refuses_local_fallback(tmp_path):
    runner,ws=synthetic_runner(tmp_path,{'foamDictionary':'echo must-not-run\n'},limits=ResourceLimits(total_cpu_seconds=10))
    with pytest.raises(UnsafeCommandError,match='cgroup'):runner.run(['foamDictionary'],cwd=ws.case_dir)
    assert runner.budget.used==0


def test_rc2_workspace_lock_excludes_second_native_runner(tmp_path):
    runner,ws=synthetic_runner(tmp_path,{'foamDictionary':'echo locked\n'})
    with workspace_execution_lock(ws.root):
        with pytest.raises(IsolationUnavailable,match='Another'):runner.run(['foamDictionary'],cwd=ws.case_dir)
    assert runner.budget.used==0


def strict_policy(tmp_path):
    return LinuxIsolationPolicy(cgroup_parent=str(tmp_path/'cg'),writable_volume=str(tmp_path/'volume'),
        readonly_runtime_roots=['/opt/OpenFOAM-13'],limits=ResourceLimits(memory_bytes=128*1024*1024,total_cpu_seconds=5))


def test_rc2_strict_isolation_missing_prerequisites_refuses(tmp_path,monkeypatch):
    monkeypatch.setattr(os,'geteuid',lambda:1000)
    policy=strict_policy(tmp_path);policy.bwrap_path=str(tmp_path/'does-not-exist')
    with pytest.raises(IsolationUnavailable,match='bubblewrap'):LinuxIsolation(policy,tmp_path/'volume/run')


def test_rc2_strict_isolation_refuses_privileged_agent(tmp_path,monkeypatch):
    monkeypatch.setattr(os,'geteuid',lambda:0)
    with pytest.raises(IsolationUnavailable,match='unprivileged'):LinuxIsolation(strict_policy(tmp_path),tmp_path/'volume/run')


def test_rc2_strict_namespace_command_has_no_host_root_or_network(tmp_path):
    policy=strict_policy(tmp_path);policy.readonly_runtime_roots=[]
    backend=LinuxIsolation.__new__(LinuxIsolation);backend.policy=policy;backend.workspace=tmp_path/'run';backend.workspace.mkdir();(backend.workspace/'case').mkdir();backend.bwrap=Path('/usr/bin/bwrap')
    args=backend.command(['/usr/bin/true'],backend.workspace/'case',{'PATH':'/usr/bin','HOME':'/home/private'})
    for flag in ['--unshare-user','--unshare-net','--unshare-pid','--unshare-ipc','--unshare-cgroup','--disable-userns','--clearenv']:
        assert flag in args
    assert args[args.index('--cap-drop')+1]=='ALL'
    assert '/home/private' not in args
    assert not any(args[i:i+3]==['--ro-bind','/','/'] for i in range(len(args)))
    assert ['--setenv','HOME','/nonexistent'] in [args[i:i+3] for i in range(len(args))]


def test_rc2_cgroup_configuration_uses_aggregate_memory_cpu_pids(tmp_path,monkeypatch):
    policy=strict_policy(tmp_path);parent=tmp_path/'cg';parent.mkdir()
    backend=LinuxIsolation.__new__(LinuxIsolation);backend.policy=policy;backend.parent=parent;backend.active=None
    monkeypatch.setattr(backend,'preflight',lambda:None)
    real_mkdir=Path.mkdir
    def kernel_control_fixture(path,*a,**k):
        real_mkdir(path,*a,**k)
        if path.parent==parent:
            for name in ['memory.max','memory.swap.max','memory.oom.group','pids.max','cpu.max','cgroup.kill']:(path/name).write_text('')
    monkeypatch.setattr(Path,'mkdir',kernel_control_fixture)
    group=Path(backend.start(policy.limits))
    assert (group/'memory.max').read_text()==str(policy.limits.memory_bytes)
    assert (group/'memory.swap.max').read_text()=='0'
    assert (group/'pids.max').read_text()==str(policy.limits.max_processes)
    assert (group/'cpu.max').read_text()=='800000 100000'


def test_rc2_parallel_restart_latest_complete_preserves_future_outputs(tmp_path):
    ws=CaseWorkspace(tmp_path);state,plan=restart_case(ws)
    (ws.case_dir/'processor1/4/p').unlink()
    partial=(ws.case_dir/'processor0/4/p').read_bytes()
    selected=inspect_parallel_restart(ws,plan)
    assert selected['time_name']=='2' and selected['rejected_newer_snapshots']
    new,receipt=prepare_parallel_restart(ws,plan)
    assert plan.execution.parallel.restart is None and new.execution.parallel.restart.time_name=='2'
    assert new.completion==plan.completion
    assert (ws.root/receipt['receipt']['archive']/'outputs/processor0/4/p').read_bytes()==partial
    assert not (ws.case_dir/'processor0/4').exists()
    assert 'startTime 2;' in ws.read_text('system/controlDict')
    assert verify_restart(ws,new)['time_name']=='2'
    contract=completion_contract(new,ws)
    assert contract.start_time==2 and contract.end_time==10


@pytest.mark.parametrize('invalid',['rank_count','binary','field_count','nonfinite','dimensions','patch_missing','incomplete_mesh','duplicate_time','dynamic_mesh'])
def test_rc2_parallel_restart_rejects_invalid_rank_snapshot(tmp_path,invalid):
    ws=CaseWorkspace(tmp_path);_,plan=restart_case(ws);p=ws.case_dir/'processor1/4/p'
    if invalid=='rank_count':(ws.case_dir/'processor2').mkdir()
    elif invalid=='binary':p.write_text(p.read_text().replace('format ascii','format binary'))
    elif invalid=='field_count':p.write_text(p.read_text().replace('internalField uniform 5.0','internalField nonuniform List<scalar> 2 (1 2)'))
    elif invalid=='nonfinite':p.write_text(p.read_text().replace('uniform 5.0','uniform nan'))
    elif invalid=='dimensions':p.write_text(p.read_text().replace('[1 -1 -2 0 0 0 0]','[0 0 0 1 0 0 0]'))
    elif invalid=='patch_missing':p.write_text(p.read_text().replace('top {','missing {'))
    elif invalid=='incomplete_mesh':(ws.case_dir/'processor1/constant/polyMesh/owner').unlink()
    elif invalid=='duplicate_time':(ws.case_dir/'processor1/4.0').mkdir()
    elif invalid=='dynamic_mesh':plan.mesh_motion_requirement='moving'
    before=ws.read_text('system/controlDict')
    with pytest.raises((ValueError,OSError)):prepare_parallel_restart(ws,plan,time_name='4')
    assert ws.read_text('system/controlDict')==before and not (ws.root/'parallel-restart-intent.json').exists()


def test_rc2_parallel_restart_input_tampering_refuses_before_mpi(tmp_path):
    ws=CaseWorkspace(tmp_path);_,plan=restart_case(ws);new,_=prepare_parallel_restart(ws,plan,time_name='2')
    p=ws.case_dir/'processor0/2/p';p.write_text(p.read_text().replace('uniform 3.0','uniform 9.0'))
    with pytest.raises(ValueError,match='changed'):verify_restart(ws,new)


def test_rc2_parallel_restart_skips_decompose_and_requires_reapproval(tmp_path,graph_path):
    agent=CFDEngineeringAgent(ScriptedLLM([]),workspace=tmp_path,capability_db=graph_path,tools=FakeOpenFOAMTools())
    state,plan=restart_case(agent.workspace);state.engineering_plan=plan;state.case_seal=agent.workspace.seal(plan);state.current_state=State.SOLVE_READY;state.approve_solve()
    prepare_restart_state(agent,state,time_name='2')
    assert state.current_state==State.SOLVE_READY and not state.solve_approved and state.execution_approval is None
    tools=FakeOpenFOAMTools();tools.run_native_command=lambda *a,**k:pytest.fail('Restart must never rerun decomposePar')
    manifest=prepare_parallel(tools,agent.workspace,state.engineering_plan)
    assert manifest['restart']
    verify_parallel_inputs(agent.workspace,state.engineering_plan,manifest)
    state.approve_solve();assert state.execution_approval.execution['parallel']['restart']['time_name']=='2'


def test_rc2_restart_control_rewrite_preserves_nested_and_comments():
    text='// startTime 999;\nFoamFile { format ascii; } startFrom latestTime; startTime 0; functions { f { startTime 9; } } endTime 10;'
    changed=rewrite_top_level(text,{'startFrom':'startTime','startTime':'2'})
    assert '// startTime 999;' in changed and 'f { startTime 9; }' in changed and 'endTime 10;' in changed
    assert 'startFrom startTime; startTime 2;' in changed
    with pytest.raises(ValueError,match='Duplicate'):rewrite_top_level(text+' startTime 5;',{'startTime':'2'})


def test_rc2_restart_inflight_pid_is_not_killed_or_replayed(tmp_path,graph_path):
    agent=CFDEngineeringAgent(ScriptedLLM([]),workspace=tmp_path,capability_db=graph_path,tools=FakeOpenFOAMTools())
    state,plan=restart_case(agent.workspace);state.engineering_plan=plan;state.case_seal=agent.workspace.seal(plan)
    state.native_process_records=[{'id':1,'pid':os.getpid(),'status':'running'}];state.pending_action={'status':'intent','kind':'runtime'}
    agent.checkpoint(state,'inflight-test')
    with pytest.raises(CheckpointError,match='PID still exists'):agent.restore_checkpoint(reconcile_parallel_restart=True)
    assert os.kill(os.getpid(),0) is None


def test_rc2_restart_preparation_failure_journal_blocks_replay(tmp_path,graph_path,monkeypatch):
    agent=CFDEngineeringAgent(ScriptedLLM([]),workspace=tmp_path,capability_db=graph_path,tools=FakeOpenFOAMTools())
    state,plan=restart_case(agent.workspace);state.engineering_plan=plan;state.case_seal=agent.workspace.seal(plan);agent.checkpoint(state,'before')
    original=agent.workspace.write_text
    def fail_write(*a,**k):raise OSError('injected storage failure')
    monkeypatch.setattr(agent.workspace,'write_text',fail_write)
    with pytest.raises(OSError):prepare_parallel_restart(agent.workspace,plan,time_name='2')
    assert (agent.workspace.root/'parallel-restart-intent.json').exists()
    with pytest.raises(CheckpointError):agent.restore_checkpoint()


def test_rc2_real_cli_restart_preparation_does_not_configure_model(tmp_path,graph_path,monkeypatch,capsys):
    from openfoam_agent import cli
    from openfoam_agent.workflow.engine import CFDWorkflow
    workflow=CFDWorkflow(llm=ScriptedLLM([]),workspace=tmp_path,capability_db=graph_path,native_execution=False)
    state,plan=restart_case(workflow.engineering.workspace);state.engineering_plan=plan
    state.case_seal=workflow.engineering.workspace.seal(plan);state.current_state=State.SOLVE_READY
    workflow.engineering.checkpoint(state,'cli-restart-fixture')
    monkeypatch.setattr(cli,'_build_llm',lambda *_:pytest.fail('Restart preparation must not authenticate/call a model.'))
    assert cli.main(['--resume',str(tmp_path),'--prepare-parallel-restart','latest','--capability-db',str(graph_path),'--json','--dry-run'])==0
    report=json.loads(capsys.readouterr().out)
    assert report['backend']=='controller-only' and report['final_state']=='SOLVE_READY'
    assert report['execution_isolation']=={'requested_mode':'local_no_os_isolation','observed_modes':[],'native_results_observed':0,'requested_is_not_observed':True}
    assert report['conservation_analyses']==[]
    receipt=json.loads((tmp_path/'parallel-restart.json').read_text())
    assert receipt


def test_rc2_reconstruction_covers_physical_observation_times_not_only_last(tmp_path):
    from openfoam_agent.runtime.parallel import reconstruct_parallel
    from openfoam_agent.contracts.models import QuantityOfInterest,NativeFieldReduction
    from conftest import tool_result
    ws=CaseWorkspace(tmp_path);_,plan=restart_case(ws)
    plan.quantities_of_interest=[QuantityOfInterest(id='pmean',quantity='pressure',unit='Pa',selection='volume:all',operation='time_mean',
        native_field=NativeFieldReduction(quantity_kind='pressure',field='p',time_names=['0','2','4'],reduction='volume_mean',dimensions=(1,-1,-2,0,0,0,0)))]
    calls=[]
    class Tools:
        def run_native_command(self,*args,**kwargs):
            calls.append((args,kwargs));return tool_result('reconstructPar',success=True,stdout='End')
    before=(ws.case_dir/'0/p').read_bytes()
    reconstruct_parallel(Tools(),ws,plan,4)
    assert calls[0][1]['arguments']==['-time','2,4'] and (ws.case_dir/'0/p').read_bytes()==before
    import shutil
    shutil.rmtree(ws.case_dir/'processor1/2')
    with pytest.raises(ValueError,match='Requested-time'):reconstruct_parallel(Tools(),ws,plan,4)
    assert len(calls)==1
