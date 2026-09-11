"""v4.0.2 prepare-design context partition regressions; no native CFD/model calls."""
import json

from conftest import make_state, FakeOpenFOAMTools
from openfoam_agent.agents.intake import confirmed_intake_definition
from openfoam_agent.engineering import CFDEngineeringAgent, EngineeringPolicy
from openfoam_agent.engineering.design_context import build_partitioned_design_prompt, project_design_capsule
from openfoam_agent.llm.context import build_bounded_json_prompt, ContextBudgetError
from openfoam_agent.llm.context_capsules import project_confirmed_intake
from openfoam_agent.schemas.engineering import BlockAction, EngineeringEvidenceRecord, ObservedEngineeringEvidence, canonical_engineering_evidence_id


def _oversized_payload(state):
    evidence=[]
    for i in range(10):
        evidence.append({
            'evidence_id':f'ev:{i}', 'kind':'openfoam_reference', 'reference':f'ref:{i}',
            'summary':('phase-change syntax summary '+str(i)+' ')*25,
            'detail':json.dumps({'body':('dictionary grammar '+str(i)+' ')*120}),
        })
    return {
        'state_mode':'bounded_engineering_design','phase':'prepare','step':3,
        'confirmed_intake':state.intake.model_dump(mode='json'),'intake_sha256':state.intake_digest,
        'engineering_assumption_policy':{'authorized':True,'allowed_when_authorized':['x']*20},
        'evidence_retrieval_policy':{'available':True},
        'environment_hint':{f'env_{i}':'installed capability description '*80 for i in range(40)},
        'capability_graph_hint':{f'provider_{i}':'provider detail '*120 for i in range(50)},
        'available_evidence':evidence,
        'evidence_context':{'shown':10,'total_observed':25,'truncated':True},
        'evidence_gap_status':[{'gap_id':f'G{1300+i}','status':'evidence_available','missing_evidence':'x'*250} for i in range(8)],
        'recent_observations':[{'summary':'old observation '*250} for _ in range(4)],
        'bindings':{'intake_sha256':state.intake_digest,'plan_sha256':None,'manifest_sha256':'0'*64},
        'budget':{'llm_limit':12,'llm_remaining':9,'native_limit':40,'native_used':0},
    }


def test_v402_design_context_partition_fits_18k_without_truncating_confirmed_intake():
    state=make_state();payload=_oversized_payload(state)
    try:
        build_bounded_json_prompt('design:\n',payload,max_chars=18000)
    except ContextBudgetError:
        pass
    result,capsule,metrics=build_partitioned_design_prompt('design:\n',payload,max_chars=18000,initial_evidence_limit=10)
    assert len(result.prompt)<=18000 and metrics['partitioned']==1
    assert capsule['confirmed_intake']==payload['confirmed_intake']
    assert capsule['context_partition']['active'] is True
    assert 'capability_graph_hint' not in capsule
    assert len(capsule['available_evidence'])<=10
    # Older evidence is digest-only; newest evidence retains bounded detail.
    detailed=[e for e in capsule['available_evidence'] if 'detail' in e]
    assert len(detailed)<=3


def test_v402_partition_projection_is_deterministic_and_does_not_claim_absent_evidence_unsupported():
    state=make_state();payload=_oversized_payload(state)
    a=project_design_capsule(payload,evidence_limit=4);b=project_design_capsule(payload,evidence_limit=4)
    assert a==b
    assert [e['evidence_id'] for e in a['available_evidence']]==[f'ev:{i}' for i in range(6,10)]
    assert 'absence from this capsule does not mean unsupported' in a['context_partition']['semantics']


def test_v402_agent_prepare_design_auto_partitions_instead_of_raising(tmp_path,graph_path):
    state=make_state()
    for i in range(12):
        eid=canonical_engineering_evidence_id('openfoam_reference',f'test:{i}')
        obs=ObservedEngineeringEvidence(evidence_id=eid,kind='openfoam_reference',reference=f'test:{i}',summary=('summary '*100))
        state.engineering_evidence_records.append(EngineeringEvidenceRecord(
            record_id='evrec_'+f'{i:020d}',phase='prepare',step=i+1,action_type='gather_evidence',
            observed_evidence=[obs],payload={'observed_details':{obs.evidence_id:{'body':'syntax '*400}}}))
    class LLM:
        model='codex-default';max_output_tokens=1000;store=False
        def __init__(self): self.prompts=[]
        def generate(self,schema,prompt,**kwargs):
            self.prompts.append(prompt)
            return schema(action=BlockAction(type='block',reason='fixture stop',needs_user_input=False))
    llm=LLM();tools=FakeOpenFOAMTools()
    tools.environment_snapshot=lambda:{f'env_{i}':'environment detail '*300 for i in range(50)}
    agent=CFDEngineeringAgent(llm,workspace=tmp_path,capability_db=graph_path,tools=tools,
        policy=EngineeringPolicy(max_model_prompt_chars=18000,max_prepare_model_evidence_items=10,
            max_decide_model_evidence_items=12,bounded_evidence_context=True,compact_phase_schemas=True,
            staged_case_authoring=True,preload_capabilities=False))
    agent.catalog.summary=lambda:{f'provider_{i}':'provider detail '*300 for i in range(60)}
    turn=agent._generate_turn(state,step=3,local_step=3,current_step_limit=12,phase='prepare',native_execution=False)
    assert turn.action.type=='block'
    assert len(llm.prompts)==1 and len(llm.prompts[0])<=18000
    sent=json.loads(llm.prompts[0][llm.prompts[0].index('{'):])
    assert sent['state_mode']=='partitioned_engineering_design'
    assert sent['confirmed_intake']==project_confirmed_intake(state.intake)
    assert sent['intake_sha256']==state.intake_digest
    assert sent['context_partition']['active'] is True


def test_v402_mandatory_design_contract_still_fails_closed_when_it_alone_exceeds_budget():
    state=make_state();payload=_oversized_payload(state)
    payload['confirmed_intake']['facts'][0]['value']='mandatory-user-condition '*2000
    try:
        build_partitioned_design_prompt('design:\n',payload,max_chars=18000,initial_evidence_limit=10)
    except ContextBudgetError as exc:
        assert 'no confirmed requirement was truncated' in str(exc)
    else:
        raise AssertionError('mandatory oversized intake must fail closed')
