"""Analytical, synthetic OpenFOAM-format fixtures. No native CFD was run."""
from pathlib import Path
from types import SimpleNamespace
from conftest import foam_header,make_state,make_plan,control_dict
from openfoam_agent.contracts.models import ParallelExecution,CompletionContract,RegionCaseLayout
from openfoam_agent.schemas.engineering import OpenFOAMExecutionSpec

PATCHES=['inlet','outlet','front','back','bottom','top']
POINTS=[(0,0,0),(1,0,0),(1,1,0),(0,1,0),(0,0,1),(1,0,1),(1,1,1),(0,1,1)]
FACES=[(0,4,7,3),(1,2,6,5),(0,1,5,4),(3,7,6,2),(0,3,2,1),(4,5,6,7)]


def put(ws,relative,text):
    p=ws.case_dir/relative;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(text);return p


def cube(ws,region='',processor_patch=False):
    base='constant/'+(region+'/' if region else '')+'polyMesh/'
    def listing(name,rows,cls):
        put(ws,base+name,foam_header(base+name,cls)+str(len(rows))+'\n(\n'+'\n'.join(rows)+'\n)\n')
    listing('points',['('+' '.join(map(str,p))+')' for p in POINTS],'vectorField')
    listing('faces',['4('+' '.join(map(str,f))+')' for f in FACES],'faceList')
    listing('owner',['0']*6,'labelList');listing('neighbour',[],'labelList')
    listing('boundary',[f'{p} {{ type {"processor" if processor_patch and i==1 else "patch"}; nFaces 1; startFace {i}; }}' for i,p in enumerate(PATCHES)],'polyBoundaryMesh')


def field(ws,time,name,*,region='',value=1,dimensions=(0,0,0,0,0,0,0),kind='volScalarField',patch_values=None,internal=None):
    path=time+'/'+(region+'/' if region else '')+name
    patch_values=patch_values or {}
    text=foam_header(path,kind)+'dimensions ['+' '.join(map(str,dimensions))+'];\n'
    text+='internalField '+(internal or f'uniform {value}')+';\nboundaryField {\n'
    for patch in PATCHES:text+=f'{patch} {{ type calculated; value uniform {patch_values.get(patch,value)}; }}\n'
    return put(ws,path,text+'}\n')


def restart_case(ws):
    state=make_state();plan=make_plan(state.intake)
    plan.execution=OpenFOAMExecutionSpec(driver='foamRun',driver_provider_id='installed.application.foamRun',
        solver_module=plan.solver,solver_provider_id=plan.solver_provider_id,
        parallel=ParallelExecution(mode='local_mpi',ranks=2,decomposition_method='scotch'))
    plan.required_case_files=['0/p'];plan.region_layouts=[RegionCaseLayout(required_fields=['p'])]
    plan.completion=CompletionContract(mode='transient',start_time=0,end_time=10,required_result_fields=['p'])
    ws.write_text('system/controlDict',control_dict())
    ws.write_text('system/decomposeParDict',foam_header('system/decomposeParDict')+'numberOfSubdomains 2; method scotch;\n')
    cube(ws);field(ws,'0','p',dimensions=(1,-1,-2,0,0,0,0))
    for rank in range(2):
        sub=SimpleNamespace(case_dir=ws.case_dir/f'processor{rank}');cube(sub,processor_patch=True)
        for time in ['0','2','4']:
            field(sub,time,'p',value=float(time)+1,dimensions=(1,-1,-2,0,0,0,0))
    return state,plan
