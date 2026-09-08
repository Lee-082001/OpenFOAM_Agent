"""RC2 file/task/evidence/dependency regressions; no model/native CFD calls."""
from copy import deepcopy
import json
from types import SimpleNamespace
import pytest
from conftest import make_state,make_plan,FakeOpenFOAMTools,ScriptedLLM,foam_header,register_authoring_syntax
from openfoam_agent.engineering import CFDEngineeringAgent,EngineeringPolicy
from openfoam_agent.engineering.authoring_tasks import compile_tasks,accept_task,project,sha
from openfoam_agent.llm.context import ContextBudgetError,build_bounded_json_prompt
from openfoam_agent.contracts.evidence import implementation_evidence_pack,require_authoring_evidence,authoring_prompt_evidence
from openfoam_agent.contracts.mesh_dependencies import MeshDependencyGraph
from openfoam_agent.contracts.regions import region_mesh_digest
from openfoam_agent.schemas.engineering import (CaseAuthoringAction,CaseBundleFile,WriteCaseFileAction,PatchCaseFileAction,
    CaseFilePatch,ExecuteCasePlanAction,NativeOpenFOAMCommand,ReadReferenceAction,ImplementationEvidenceBinding,
    EngineeringEvidenceRecord,canonical_engineering_evidence_id)
from openfoam_agent.tools.workspace import CaseWorkspace
from openfoam_agent.workflow.states import State
from openfoam_agent.workflow.checkpoint import CheckpointError
from test_v290_execution_plan import _full_execution_plan


def partition_payload(count=8):
    state=make_state(syntax_evidence=False);plan=make_plan(state.intake)
    paths=[f'system/file{i}' for i in range(count)];plan.required_case_files=paths
    records=[];coverage=[]
    for i,p in enumerate(paths):
        eid=canonical_engineering_evidence_id('openfoam_reference',f'test:syntax{i}')
        plan.implementation_evidence_bindings.append(ImplementationEvidenceBinding(path=p,evidence_ids=[eid]))
        content=f'// syntax {i}\n'+'// observed detailed grammar\n'*150
        records.append({'evidence_id':eid,'content':content,'source':f'test:syntax{i}','excerpt_sha256':sha(content)})
        coverage.append({'path':p,'evidence_ids':[eid],'status':'explicit'})
    payload={'frozen_engineering_plan':plan.model_dump(mode='json'),'confirmed_intake':state.intake.model_dump(mode='json'),
        'implementation_evidence_pack':{'records':records,'file_coverage':coverage,'complete':True},'assets':[]}
    return state,plan,payload


def task_action(task):
    meta=task['authoring_task']
    return CaseAuthoringAction(type='author_case',goal='Complete exactly one assigned file group',
        task_id=meta['id'],defer_native=not meta['is_final'],required_case_files=meta['paths'],
        files=[CaseBundleFile(path=p,content=foam_header(p)+'value 1;\n') for p in meta['paths']],
        mesh_commands=['checkMesh'] if meta['is_final'] else [])


def test_rc2_partition_exact_file_fact_and_syntax_coverage():
    state,plan,payload=partition_payload();queue=compile_tasks('',payload,14000)
    assert 1<len(queue['tasks'])<=8
    assert [p for t in queue['tasks'] for p in t['authoring_task']['paths']]==plan.required_case_files
    ids=set()
    for task in queue['tasks']:
        assert len(build_bounded_json_prompt('',task,max_chars=14000).prompt)<=14000
        ids|={f['id'] for f in task['confirmed_intake']['facts']}
        for record in task['implementation_evidence_pack']['records']:assert record['content'].endswith('grammar\n')
        assert task['frozen_engineering_plan']['interfaces']==payload['frozen_engineering_plan']['interfaces']
    assert ids=={f.id for f in state.intake.facts}
    for task in queue['tasks']:
        result=accept_task(queue,task_action(task),plan)
        if not task['authoring_task']['is_final']:assert result is None
    assert result.plan==plan and [f.path for f in result.files]==plan.required_case_files


def test_rc2_partition_transitive_fact_dependencies_not_silently_dropped():
    _,_,payload=partition_payload(2)
    payload['confirmed_intake']['facts']=[{'id':'a','value':'A','depends_on':['b']},{'id':'b','value':'B','depends_on':['c']},{'id':'c','value':'C','depends_on':[]},{'id':'d','value':'D','depends_on':[]}]
    payload['frozen_engineering_plan']['confirmed_fact_bindings']=[{'fact_id':'a','case_files':['system/file0']},{'fact_id':'b','case_files':['system/file1']},{'fact_id':'c','case_files':['system/file1']},{'fact_id':'d','case_files':['system/file1']}]
    result=project(payload,['system/file0'])
    assert {f['id'] for f in result['confirmed_intake']['facts']}=={'a','b','c'}


@pytest.mark.parametrize('mutation',['task_id','missing_file','duplicate_file','changed_plan','early_native'])
def test_rc2_partition_rejects_bad_response_before_commit(mutation):
    _,plan,payload=partition_payload();queue=compile_tasks('',payload,14000);action=task_action(queue['tasks'][0]);before=deepcopy(queue)
    if mutation=='task_id':action.task_id='invented'
    if mutation=='missing_file':action.files=[]
    if mutation=='duplicate_file':action.files.append(action.files[0])
    if mutation=='changed_plan':plan.assumptions.append('changed')
    if mutation=='early_native':action.defer_native=False
    with pytest.raises(ValueError):accept_task(queue,action,plan)
    assert queue==before


def test_rc2_indivisible_global_contract_fails_without_truncation():
    _,_,payload=partition_payload();payload['frozen_engineering_plan']['assumptions']=['mandatory'*8000]
    with pytest.raises(ContextBudgetError,match='Indivisible'):compile_tasks('',payload,16000)


def test_rc2_partition_checkpoint_restores_pending_file_tasks(tmp_path,graph_path):
    state,plan,payload=partition_payload();agent=CFDEngineeringAgent(ScriptedLLM([]),workspace=tmp_path,capability_db=graph_path,tools=FakeOpenFOAMTools())
    agent._draft_design_plan=plan;agent._authoring_task_queue=compile_tasks('',payload,14000)
    accept_task(agent._authoring_task_queue,task_action(agent._authoring_task_queue['tasks'][0]),plan)
    agent.checkpoint(state,'partition-test')
    other=CFDEngineeringAgent(ScriptedLLM([]),workspace=tmp_path,capability_db=graph_path,tools=FakeOpenFOAMTools())
    other.restore_checkpoint()
    assert other._authoring_task_queue==agent._authoring_task_queue and other._draft_design_plan==plan
    assert not list(other.workspace.case_dir.rglob('file*'))


@pytest.mark.parametrize('phase',['prepare','human_revision','runtime_repair'])
@pytest.mark.parametrize('kind',['write','patch'])
def test_v421_primitive_mutations_do_not_need_documentary_syntax_evidence(tmp_path,graph_path,phase,kind):
    state=make_state(syntax_evidence=False);plan=make_plan(state.intake)
    agent=CFDEngineeringAgent(ScriptedLLM([]),workspace=tmp_path,capability_db=graph_path,tools=FakeOpenFOAMTools())
    path='system/fvSolution';original=foam_header(path)+'value 1;\n';agent.workspace.write_text(path,original)
    state.engineering_plan=plan;state.case_seal=agent.workspace.seal(plan);state.current_state=State.SOLVE_READY;state.approve_solve()
    action=WriteCaseFileAction(type='write_case_file',path=path,content=original.replace('value 1','value 2')) if kind=='write' else PatchCaseFileAction(type='patch_case_file',patch=CaseFilePatch(path=path,old='value 1',new='value 2'))
    event=agent._dispatch_tool_action(action,step=1,native_execution=False,phase=phase,state=state)
    assert event.success
    assert 'value 2' in agent.workspace.read_text(path)


def test_v421_progress_first_authoring_keeps_executable_content_hard_blocked(tmp_path,graph_path):
    state=make_state(syntax_evidence=False);agent=CFDEngineeringAgent(ScriptedLLM([]),workspace=tmp_path,capability_db=graph_path,tools=FakeOpenFOAMTools())
    action=WriteCaseFileAction(type='write_case_file',path='system/controlDict',content=foam_header('system/controlDict')+'#codeStream { code #{ system(\"touch /tmp/nope\"); #}; }\n')
    event=agent._dispatch_tool_action(action,step=1,native_execution=False,phase='prepare',state=state)
    assert not event.success
    assert 'unsafe' in event.summary.lower() or 'executable' in event.summary.lower()


def test_v421_legacy_raw_bundle_no_longer_needs_documentary_syntax_evidence(tmp_path,graph_path):
    state=make_state(syntax_evidence=False);agent=CFDEngineeringAgent(ScriptedLLM([]),workspace=tmp_path,capability_db=graph_path,tools=FakeOpenFOAMTools())
    action=_full_execution_plan(state)
    agent._execute_case_plan(state,action,llm_step=1,progress_phase='engineering',progress_step=1,progress_limit=20,native_execution=False)
    assert agent.workspace.file_seals()
    assert not any('syntax evidence required' in e.output_excerpt for e in state.engineering_events)


def test_rc2_reference_read_explicit_targets_unlock_legacy_mutation(tmp_path,graph_path):
    state=make_state(syntax_evidence=False);agent=CFDEngineeringAgent(ScriptedLLM([]),workspace=tmp_path,capability_db=graph_path,tools=FakeOpenFOAMTools())
    source=tmp_path/'syntax-source.foam';source.write_text('value 1;\n')
    agent.references.read=lambda *a,**k:source.read_text()
    read=ReadReferenceAction(type='read_reference',reference='test:syntax',target_case_files=['system/custom'])
    event=agent._dispatch_tool_action(read,step=1,native_execution=False,phase='prepare',state=state);assert event.success
    write=WriteCaseFileAction(type='write_case_file',path='system/custom',content=foam_header('system/custom')+'value 2;')
    assert agent._dispatch_tool_action(write,step=2,native_execution=False,phase='prepare',state=state).success
    pack=authoring_prompt_evidence(state)
    assert pack['records'][0]['content']==source.read_text() and pack['complete']


def test_rc2_search_summary_alone_cannot_unlock_authoring():
    state=make_state(syntax_evidence=False)
    state.engineering_evidence_records=[EngineeringEvidenceRecord(record_id='evrec_'+'1'*20,phase='prepare',step=1,action_type='search_references',payload={'reference':'test:syntax','snippet':'value 1;','target_case_files':['system/custom']})]
    with pytest.raises(ValueError,match='syntax evidence'):require_authoring_evidence(state,None,['system/custom'])


def test_rc2_multi_region_authoring_pipeline_accepts_distinct_checks():
    action=CaseAuthoringAction(type='author_case',goal='Validate each region',files=[CaseBundleFile(path='system/a',content='a 1;')],required_case_files=['system/a'],
        native_pipeline=[NativeOpenFOAMCommand(command='checkMesh',arguments=['-region',r]) for r in ['fluid','solid']])
    assert len(action.native_pipeline)==2
    with pytest.raises(ValueError,match='duplicate checkMesh'):
        CaseAuthoringAction.model_validate({**action.model_dump(),'native_pipeline':[x.model_dump() for x in [action.native_pipeline[0]]*2]})


def test_rc2_mesh_transitive_asset_conversion_invalidation(tmp_path):
    ws=CaseWorkspace(tmp_path)
    for p in ['constant/source.dat','constant/intermediate.dat','constant/final.dat']:ws.write_text(p,'original')
    graph=MeshDependencyGraph(ws)
    graph.register(inputs=['constant/source.dat'],outputs=['constant/intermediate.dat'],regions=[],command={'name':'convert'})
    graph.register(inputs=['constant/intermediate.dat'],outputs=['constant/final.dat'],regions=[],command={'name':'convert2'})
    graph.register(inputs=['constant/final.dat'],outputs=['constant/fluid/polyMesh'],regions=['fluid'],command={'name':'mesh'})
    before=graph.digest('fluid');paths,nodes=graph.dependencies('fluid')
    assert 'constant/source.dat' in paths and [n['id'] for n in nodes]==[0,1,2]
    ws.write_text('constant/source.dat','changed input')
    assert MeshDependencyGraph(ws).digest('fluid')!=before


@pytest.mark.parametrize('change',['add','delete','modify'])
def test_rc2_unknown_native_reads_conservative_roots(change,tmp_path):
    ws=CaseWorkspace(tmp_path);ws.write_text('system/arbitrary','x 1;')
    graph=MeshDependencyGraph(ws);graph.record_native('customMeshUtility',[])
    before=graph.digest('')
    if change=='add':ws.write_text('0/custom','new')
    elif change=='delete':(ws.case_dir/'system/arbitrary').unlink()
    else:ws.write_text('system/arbitrary','x 2;')
    assert MeshDependencyGraph(ws).digest('')!=before


def test_rc2_mesh_include_dependency_and_local_region_separation(tmp_path):
    ws=CaseWorkspace(tmp_path)
    # Shared literal include outside a recognized mesh basename must still matter.
    from rc2_helpers import put
    put(ws,'system/fluid/blockMeshDict','x 1; #include "geometry.inc"\n')  # operator-existing input, not LLM include authorization
    ws.write_text('system/fluid/geometry.inc','points 1;')
    ws.write_text('system/solid/blockMeshDict','points 2;')
    for region in ['fluid','solid']:
        path=ws.case_dir/f'constant/{region}/polyMesh/boundary';path.parent.mkdir(parents=True);path.write_text('boundary')
    graph=MeshDependencyGraph(ws)
    plan=SimpleNamespace(required_case_files=[],region_layouts=[SimpleNamespace(region=r) for r in ['fluid','solid']],execution=None)
    graph.record_native('blockMesh',['-region','fluid'],plan)
    graph.record_native('blockMesh',['-region','solid'],plan)
    before={r:graph.digest(r) for r in ['fluid','solid']}
    ws.write_text('system/fluid/geometry.inc','points 3;')
    assert graph.digest('fluid')!=before['fluid'] and graph.digest('solid')==before['solid']


def test_rc2_mesh_graph_tampering_blocks_checkpoint_restore(tmp_path,graph_path):
    state=make_state();agent=CFDEngineeringAgent(ScriptedLLM([]),workspace=tmp_path,capability_db=graph_path,tools=FakeOpenFOAMTools())
    graph=MeshDependencyGraph(agent.workspace);graph.record_native('blockMesh',[]);agent.checkpoint(state,'meshgraph')
    graph.register(inputs=['constant/a'],outputs=['constant/polyMesh'],regions=[''],command={'name':'changed'})
    with pytest.raises(CheckpointError,match='graph changed'):agent.restore_checkpoint()


def test_rc2_unbound_read_windows_all_reach_stateless_authoring_prompt():
    state=make_state(syntax_evidence=False)
    bodies=['// long observed prefix\n'*1200+'first_window_tail', 'second_distinct_window_tail']
    for index,body in enumerate(bodies):
        state.engineering_evidence_records.append(EngineeringEvidenceRecord(record_id='evrec_'+str(index)*20,
            phase='prepare',step=index+1,action_type='read_reference',payload={'reference':'test:unbound','content':body}))
    pack=authoring_prompt_evidence(state)
    assert len(pack['records'])==1 and len(pack['records'][0]['observed_windows'])==2
    for body in bodies: assert body in pack['records'][0]['content']
    eid=canonical_engineering_evidence_id('openfoam_reference','test:unbound')
    assert require_authoring_evidence(state,None,['system/newFile'],evidence_ids=[eid])['complete']
    assert build_bounded_json_prompt('',{'implementation_evidence_pack':pack},max_chars=40000).prompt.endswith('}')
    compacted=build_bounded_json_prompt('',{'implementation_evidence_pack':pack},max_chars=10000)
    assert compacted.compacted and len(compacted.prompt)<=10000


def test_rc2_controller_automatically_splits_routes_and_defers_bundle_commit(tmp_path,graph_path,monkeypatch):
    from openfoam_agent.schemas.engineering import CaseAuthoringTurn
    state,plan,payload=partition_payload()
    for index,record in enumerate(payload['implementation_evidence_pack']['records']):
        state.engineering_evidence_records.append(EngineeringEvidenceRecord(record_id='evrec_'+f'{index:020d}',phase='prepare',step=1,
            action_type='read_reference',payload={'reference':record['source'],'content':record['content']}))
    class Author:
        def __init__(self):self.prompts=[]
        def generate(self,schema,prompt,**kwargs):
            self.prompts.append(prompt);task=json.loads(prompt[prompt.index('{'):])
            return CaseAuthoringTurn(action=task_action(task))
    llm=Author();agent=CFDEngineeringAgent(llm,workspace=tmp_path,capability_db=graph_path,tools=FakeOpenFOAMTools(),
        policy=EngineeringPolicy(max_model_prompt_chars=14000,preload_capabilities=False,compact_phase_schemas=True,staged_case_authoring=True))
    agent._draft_design_plan=plan;calls=[]
    # Observe the existing transactional executor boundary, not a native CFD run.
    monkeypatch.setattr(agent,'_execute_case_plan',lambda state,action,**kw:calls.append(action))
    for step in range(1,10):
        turn=agent._generate_turn(state,step=step,phase='prepare',native_execution=False)
        assert len(llm.prompts[-1])<=23000
        agent._execute_prepare_decision(state,turn.action,llm_step=step,progress_phase='engineering',progress_step=step,progress_limit=10,native_execution=False)
        if calls:break
        assert not agent.workspace.file_seals()
        assert (tmp_path/'checkpoint.json').is_file()
    assert len(llm.prompts)>1 and len(calls)==1 and agent._authoring_task_queue is None
    assert [f.path for f in calls[0].files]==plan.required_case_files
