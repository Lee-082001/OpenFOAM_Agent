"""Independent arithmetic/physics-contract tests on analytic synthetic cubes."""
import gzip
from types import SimpleNamespace
import pytest
from conftest import make_state,make_plan,FakeOpenFOAMTools,ScriptedLLM
from rc2_helpers import cube,field,put,PATCHES
from openfoam_agent.contracts.models import QuantityOfInterest,NativeFieldReduction,ConservationCheck,RegionBalance,RegionCaseLayout,RegionInterface
from openfoam_agent.postprocessing.quantities import analyze_quantity
from openfoam_agent.postprocessing.native_fields import read_mesh,read_field,check_sources_current
from openfoam_agent.postprocessing.conservation import analyze_conservation
from openfoam_agent.postprocessing.agent import CFDPostProcessingAgent
from openfoam_agent.schemas.postprocessing import AnalyzeQuantityAction,AnalyzeConservationAction,PostProcessingExecutionPlanAction
from openfoam_agent.tools.workspace import CaseWorkspace
from openfoam_agent.workflow.states import State


def temperature_spec(**update):
    data=dict(id='mean_temperature',quantity='temperature',selection='volume:all',unit='K',operation='time_mean',
        native_field=NativeFieldReduction(quantity_kind='temperature',field='T',time_names=['0','1'],reduction='volume_mean',dimensions=(0,0,0,1,0,0,0)))
    data.update(update);return QuantityOfInterest(**data)


def thermal_case(ws,region=''):
    cube(ws,region)
    for time,value in [('0',300),('1',320)]:field(ws,time,'T',region=region,value=value,dimensions=(0,0,0,1,0,0,0))


def mass_case(ws,*,steady=False,region='',bad=False):
    cube(ws,region)
    for t in ['0','1','2']:
        field(ws,t,'phi',region=region,kind='surfaceScalarField',value=0,dimensions=(1,0,-1,0,0,0,0),patch_values={'inlet':-1,'outlet':1 if steady else 0})
        field(ws,t,'rho',region=region,value=1+(0 if steady else float(t))+(1 if bad and t=='1' else 0),dimensions=(1,-3,0,0,0,0,0))


def mass_check(*,steady=False,regions=None):
    return ConservationCheck(id='mass_closure',kind='mass',time_names=['0','1','2'],absolute_tolerance=1e-10,relative_tolerance=1e-8,
        regions=regions or [RegionBalance(flux_field='phi',storage_mode='steady' if steady else 'density_field',storage_density_field=None if steady else 'rho',source_mode='zero')])


def test_rc2_native_mesh_volume_and_area_independent_geometry(tmp_path):
    ws=CaseWorkspace(tmp_path);cube(ws);sources=[];mesh=read_mesh(ws,'',sources)
    assert mesh.volumes==pytest.approx([1.0]) and mesh.areas==pytest.approx([1.0]*6)
    assert {p['path'].rsplit('/',1)[-1] for p in sources}=={'points','faces','owner','neighbour','boundary'}


def test_rc2_native_temperature_time_and_volume_average(tmp_path):
    ws=CaseWorkspace(tmp_path);thermal_case(ws)
    result=analyze_quantity(ws,temperature_spec())
    assert result['value']==pytest.approx(310) and result['unit']=='K'
    assert result['physical_semantics_verified'] and result['arithmetic_verified']
    assert not result['semantics']['solver_equation_correctness_verified']
    assert any(s['path']=='1/T' for s in result['sources'])


def test_rc2_native_outward_mass_flux_sign_is_preserved(tmp_path):
    ws=CaseWorkspace(tmp_path);mass_case(ws,steady=True)
    spec=QuantityOfInterest(id='inflow',quantity='mass_flow',selection='patches:inlet',unit='kg/s',operation='time_mean',
        native_field=NativeFieldReduction(quantity_kind='mass_flow',field='phi',time_names=['0','1'],reduction='patch_sum',patches=['inlet'],dimensions=(1,0,-1,0,0,0,0)))
    result=analyze_quantity(ws,spec)
    assert result['value']==pytest.approx(-1) and result['physical_semantics_verified']


def test_rc2_native_area_integral_checks_output_dimensions(tmp_path):
    ws=CaseWorkspace(tmp_path);cube(ws)
    for time in ['0','1']:field(ws,time,'q',value=10,dimensions=(1,0,-3,0,0,0,0))
    spec=QuantityOfInterest(id='heat',quantity='heat_rate',selection='patches:top,bottom',unit='W',operation='last',
        native_field=NativeFieldReduction(quantity_kind='heat_rate',field='q',time_names=['0','1'],reduction='area_integral',patches=['top','bottom'],dimensions=(1,0,-3,0,0,0,0)))
    assert analyze_quantity(ws,spec)['value']==pytest.approx(20)


@pytest.mark.parametrize('invalid',['unit','quantity_name','dimension','region','selection','binary','nan','field_count','patch_mismatch','trailing_tokens','dynamic_mesh','warped_face','open_cell','duplicate_patch'])
def test_rc2_native_semantics_rejects_false_or_unsupported_evidence(tmp_path,invalid):
    ws=CaseWorkspace(tmp_path);thermal_case(ws);spec=temperature_spec();p=ws.case_dir/'1/T'
    if invalid=='unit':spec.unit='Pa'
    elif invalid=='quantity_name':spec.quantity='pressure'
    elif invalid=='dimension':p.write_text(p.read_text().replace('[0 0 0 1 0 0 0]','[0 2 -2 0 0 0 0]'))
    elif invalid=='region':spec.region='not_this_region'
    elif invalid=='selection':spec.selection='patches:inlet'
    elif invalid=='binary':p.write_text(p.read_text().replace('format ascii','format binary'))
    elif invalid=='nan':p.write_text(p.read_text().replace('uniform 320','uniform nan'))
    elif invalid=='field_count':p.write_text(p.read_text().replace('internalField uniform 320','internalField nonuniform List<scalar> 2 (320 320)'))
    elif invalid=='patch_mismatch':p.write_text(p.read_text().replace('top {','wrong {'))
    elif invalid=='trailing_tokens':p.write_text(p.read_text().replace('internalField uniform 320','internalField uniform 320 junk'))
    elif invalid=='dynamic_mesh':(ws.case_dir/'1/polyMesh').mkdir()
    elif invalid=='warped_face':
        points=ws.case_dir/'constant/polyMesh/points';points.write_text(points.read_text().replace('(1 1 1)','(1 1 1.2)'))
    elif invalid=='open_cell':
        faces=ws.case_dir/'constant/polyMesh/faces';faces.write_text(faces.read_text().replace('4(0 4 7 3)','4(0 3 7 4)'))
    elif invalid=='duplicate_patch':
        boundary=ws.case_dir/'constant/polyMesh/boundary';boundary.write_text(boundary.read_text().replace('top {','bottom {'))
    with pytest.raises((ValueError,OSError)):analyze_quantity(ws,spec)


def test_rc2_gzip_native_field_has_raw_file_hash_and_finite_analysis(tmp_path):
    ws=CaseWorkspace(tmp_path);thermal_case(ws);p=ws.case_dir/'1/T';raw=p.read_bytes()
    p.with_name('T.gz').write_bytes(gzip.compress(raw));p.unlink()
    result=analyze_quantity(ws,temperature_spec())
    assert result['value']==pytest.approx(310)
    assert any(s['path']=='1/T.gz' for s in result['sources'])
    check_sources_current(ws,result['sources'])


def test_rc2_source_hash_tamper_invalidates_physical_analysis(tmp_path):
    ws=CaseWorkspace(tmp_path);thermal_case(ws);result=analyze_quantity(ws,temperature_spec())
    field(ws,'1','T',value=999,dimensions=(0,0,0,1,0,0,0))
    with pytest.raises(ValueError,match='changed'):check_sources_current(ws,result['sources'])


def test_rc2_table_label_alone_is_still_not_physical_verification(tmp_path):
    ws=CaseWorkspace(tmp_path);put(ws,'postProcessing/temperature.csv','0,300\n1,320\n')
    result=analyze_quantity(ws,QuantityOfInterest(id='untyped',quantity='temperature',selection='volume:all',unit='K',source_path='postProcessing/temperature.csv',operation='time_mean'))
    assert result['value']==310 and not result['physical_semantics_verified']


@pytest.mark.parametrize('steady',[True,False])
def test_rc2_closed_mass_balance_storage_and_boundary_flux_pass(tmp_path,steady):
    ws=CaseWorkspace(tmp_path);mass_case(ws,steady=steady);state=make_state();plan=make_plan(state.intake)
    result=analyze_conservation(ws,mass_check(steady=steady),plan)
    assert result['conservation_verified'] and result['physical_semantics_verified']
    assert all(abs(i['residual'])<1e-10 for i in result['intervals'])
    assert not result['governing_equation_completeness_verified']


def test_rc2_conservation_interval_failures_cannot_cancel_in_average(tmp_path):
    ws=CaseWorkspace(tmp_path);mass_case(ws,bad=True);plan=make_plan(make_state().intake)
    result=analyze_conservation(ws,mass_check(),plan)
    assert not result['conservation_verified']
    assert [i['residual'] for i in result['intervals']]==pytest.approx([1,-1])
    assert not any(i['passed'] for i in result['intervals'])


def test_rc2_mass_source_density_is_dimensionally_integrated(tmp_path):
    ws=CaseWorkspace(tmp_path);mass_case(ws,steady=True)
    for t in ['0','1','2']:
        field(ws,t,'rho',value=1+float(t),dimensions=(1,-3,0,0,0,0,0))
        field(ws,t,'source',value=1,dimensions=(1,-3,-1,0,0,0,0))
    check=mass_check();check.regions[0].source_mode='density_field';check.regions[0].source_density_field='source'
    result=analyze_conservation(ws,check,make_plan(make_state().intake))
    assert result['conservation_verified']
    for row in result['intervals']:assert row['storage_rate']==pytest.approx(row['source_rate'])


def test_rc2_incompressible_volume_flux_is_not_mass_flux_by_label(tmp_path):
    ws=CaseWorkspace(tmp_path);mass_case(ws)
    p=ws.case_dir/'1/phi';p.write_text(p.read_text().replace('[1 0 -1 0 0 0 0]','[0 3 -1 0 0 0 0]'))
    with pytest.raises(ValueError,match='dimensions'):analyze_conservation(ws,mass_check(),make_plan(make_state().intake))


def interface_case(ws,*,wrong=False):
    state=make_state();plan=make_plan(state.intake)
    plan.region_layouts=[RegionCaseLayout(region=r,required_fields=['T']) for r in ['fluid','solid']]
    plan.interfaces=[RegionInterface(region='fluid',patch='outlet',neighbour_region='solid',neighbour_patch='inlet')]
    for r in ['fluid','solid']:
        cube(ws,r)
        values={'inlet':-1,'outlet':1} if r=='fluid' else {'inlet':1 if wrong else -1,'outlet':-1 if wrong else 1}
        for t in ['0','1','2']:field(ws,t,'heatFlux',region=r,value=0,kind='surfaceScalarField',dimensions=(1,2,-3,0,0,0,0),patch_values=values)
    check=ConservationCheck(id='heat_balance',kind='energy',regions=[RegionBalance(region=r,flux_field='heatFlux',storage_mode='steady',source_mode='zero') for r in ['fluid','solid']],time_names=['0','1','2'],absolute_tolerance=1e-12,relative_tolerance=1e-8)
    return plan,check


@pytest.mark.parametrize('wrong',[False,True])
def test_rc2_interface_flux_sign_and_pairing_are_checked(tmp_path,wrong):
    ws=CaseWorkspace(tmp_path);plan,check=interface_case(ws,wrong=wrong)
    result=analyze_conservation(ws,check,plan)
    assert all(i['passed'] for i in result['intervals'])  # region closure alone is insufficient
    assert result['conservation_verified'] is (not wrong)
    assert len(result['interfaces'])==3


def test_rc2_interface_one_sided_evidence_is_rejected(tmp_path):
    ws=CaseWorkspace(tmp_path);plan,check=interface_case(ws);check.regions=check.regions[:1]
    with pytest.raises(ValueError,match='both adjacent'):analyze_conservation(ws,check,plan)


def test_rc2_postprocess_actions_connect_native_quantity_and_balance_report(tmp_path):
    agent=CFDPostProcessingAgent(ScriptedLLM([]),workspace=tmp_path,tools=FakeOpenFOAMTools());ws=agent.workspace
    mass_case(ws);state=make_state();plan=make_plan(state.intake);plan.conservation_checks=[mass_check()];state.engineering_plan=plan
    event,terminal=agent._dispatch(state,AnalyzeConservationAction(type='analyze_conservation',check_id='mass_closure'),step=1)
    assert event.success and state.conservation_analyses[0]['conservation_verified']
    report=agent._build_report(state,limitations=[])
    assert report.success and report.conservation_analyses
    field(ws,'1','rho',value=100,dimensions=(1,-3,0,0,0,0,0))
    report=agent._build_report(state,limitations=[])
    assert not report.success and not report.conservation_analyses
    assert any('stale' in text or 'changed' in text for text in report.limitations)


def test_rc2_requested_failed_balance_cannot_be_marked_successful_report(tmp_path):
    agent=CFDPostProcessingAgent(ScriptedLLM([]),workspace=tmp_path,tools=FakeOpenFOAMTools());mass_case(agent.workspace,bad=True)
    state=make_state();plan=make_plan(state.intake);plan.conservation_checks=[mass_check()];state.engineering_plan=plan
    event,_=agent._dispatch(state,AnalyzeConservationAction(type='analyze_conservation',check_id='mass_closure'),step=1)
    assert not event.success
    report=agent._build_report(state,limitations=[],scientific_confidence='high')
    assert not report.success and report.scientific_confidence=='low' and report.review_reasons


def test_rc2_quantity_definition_change_invalidates_report(tmp_path):
    agent=CFDPostProcessingAgent(ScriptedLLM([]),workspace=tmp_path,tools=FakeOpenFOAMTools());thermal_case(agent.workspace)
    state=make_state();state.engineering_plan=make_plan(state.intake);state.engineering_plan.quantities_of_interest=[temperature_spec()]
    event,_=agent._dispatch(state,AnalyzeQuantityAction(type='analyze_quantity',quantity_id='mean_temperature'),step=1)
    assert event.success and agent._build_report(state,limitations=[]).success
    state.engineering_plan.quantities_of_interest[0].selection='patches:inlet'
    report=agent._build_report(state,limitations=[])
    assert not report.success and not report.quantity_analyses
    assert any('plan changed' in item for item in report.limitations)


def test_rc2_energy_storage_and_heat_rate_closure(tmp_path):
    ws=CaseWorkspace(tmp_path);cube(ws)
    for t in ['0','1','2']:
        field(ws,t,'heatFlux',kind='surfaceScalarField',value=0,dimensions=(1,2,-3,0,0,0,0),patch_values={'inlet':-2})
        field(ws,t,'energyDensity',value=10+2*float(t),dimensions=(1,-1,-2,0,0,0,0))
    check=ConservationCheck(id='energy_closure',kind='energy',time_names=['0','1','2'],absolute_tolerance=1e-10,relative_tolerance=1e-8,
        regions=[RegionBalance(flux_field='heatFlux',storage_mode='density_field',storage_density_field='energyDensity',source_mode='zero')])
    result=analyze_conservation(ws,check,make_plan(make_state().intake))
    assert result['conservation_verified'] and all(abs(i['residual'])<1e-10 for i in result['intervals'])


def test_rc2_empty_mesh_points_rejected_without_unhandled_division(tmp_path):
    ws=CaseWorkspace(tmp_path);cube(ws)
    put(ws,'constant/polyMesh/points','FoamFile { format ascii; class vectorField; object points; }\n0\n(\n)\n')
    with pytest.raises(ValueError):read_mesh(ws,'',[])
