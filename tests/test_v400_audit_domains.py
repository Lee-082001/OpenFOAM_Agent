"""Audit 6-20,24-30: deterministic contracts and synthetic data, no live CFD/model."""
from __future__ import annotations
import hashlib
import json
import math
import struct
from pathlib import Path
import pytest
from pydantic import ValidationError
from conftest import make_state, make_plan, make_intake, control_dict, foam_header, FakeOpenFOAMTools, ScriptedLLM, tool_result
from openfoam_agent.contracts.models import (RegionCaseLayout, RegionInterface, PhysicalQuantity, QuantityOfInterest,
    ImplementationEvidenceBinding, ParallelExecution, CompletionContract)
from openfoam_agent.contracts.regions import region_layouts, region_mesh_digest, validate_design
from openfoam_agent.contracts.quantities import si_value, assert_equivalent
from openfoam_agent.contracts.requirements import active_requirement_texts
from openfoam_agent.contracts.evidence import implementation_evidence_pack, evidence_coverage_failures
from openfoam_agent.agents.intake import validate_intake_provenance
from openfoam_agent.schemas.request import UserRequest
from openfoam_agent.schemas.intake import IntakeFact, CFDIntakeSpec, SemanticAmbiguity
from openfoam_agent.schemas.engineering import OpenFOAMExecutionSpec, RegionSolverAssignment, ReadReferenceAction, canonical_engineering_evidence_id
from openfoam_agent.schemas.installation import InstalledOpenFOAMIR, InstalledComponent, InstalledExecutable
from openfoam_agent.tools.workspace import CaseWorkspace, WorkspaceSafetyError
from openfoam_agent.tools.assets import AssetRegistry
from openfoam_agent.tools.references import OpenFOAMReferenceIndex
from openfoam_agent.tools.capability_catalog import CapabilityCatalog
from openfoam_agent.verification.presolve import PreSolveCompletenessGate
from openfoam_agent.engineering import CFDEngineeringAgent
from openfoam_agent.workflow.checkpoint import CheckpointStore, CheckpointError
from openfoam_agent.workflow.states import State
from openfoam_agent.runtime.parallel import prepare_parallel, verify_parallel_inputs, reconstruct_parallel
from openfoam_agent.postprocessing.context import resolve_postprocess_context
from openfoam_agent.postprocessing.quantities import analyze_quantity, clean_series, time_statistics
from openfoam_agent.postprocessing.analysis import analyze_force_coefficients
from openfoam_agent.llm.context import build_bounded_json_prompt, ContextBudgetError
from openfoam_agent.llm.codex_transport import authentication_kind, inspect_events
from openfoam_agent.llm.codex_client import _codex_environment


def multi_plan():
    state = make_state()
    data = make_plan(state.intake).model_dump()
    data.update(solver="foamMultiRun", solver_provider_id="installed.application.foamMultiRun",
        execution=OpenFOAMExecutionSpec(driver="foamMultiRun",driver_provider_id="installed.application.foamMultiRun",
            regions=[RegionSolverAssignment(region=r,solver_module=m,provider_id="installed.solver_module."+m)
                     for r,m in [("fluid","fluid"),("solid","solid")]]),
        required_case_files=["system/controlDict", *[p for r in ["fluid","solid"] for p in
            [f"0/{r}/T",f"system/{r}/fvSchemes",f"system/{r}/fvSolution"]]],
        region_layouts=[RegionCaseLayout(region=r,solver_module=r,required_fields=["T"]) for r in ["fluid","solid"]])
    return type(make_plan(state.intake)).model_validate(data)


def seed_regions(ws, plan):
    ws.write_text("system/controlDict", control_dict())
    for r in ["fluid","solid"]:
        for name in ["fvSchemes","fvSolution"]:
            p=f"system/{r}/{name}";ws.write_text(p,foam_header(p)+"solvers {}\n")
        p=f"0/{r}/T"
        ws.write_text(p,foam_header(p,"volScalarField")+"dimensions [0 0 0 1 0 0 0];\ninternalField uniform 300;\nboundaryField { interface { type zeroGradient; } }\n")
        path=ws.case_dir/f"constant/{r}/polyMesh/boundary";path.parent.mkdir(parents=True)
        path.write_text("FoamFile {}\n1\n(\ninterface { type wall; nFaces 1; startFace 0; }\n)\n")


def test_a06_two_regions_need_no_dummy_root_fv_files(tmp_path):
    ws=CaseWorkspace(tmp_path);plan=multi_plan();seed_regions(ws,plan)
    result=PreSolveCompletenessGate(FakeOpenFOAMTools(),ws).validate(plan)
    assert result.valid, result.failures
    assert set(result.regions)=={"fluid","solid"}
    assert "system/fvSchemes" not in result.checked_files


def test_a06_missing_solid_field_is_not_satisfied_by_fluid_field(tmp_path):
    ws=CaseWorkspace(tmp_path);plan=multi_plan();seed_regions(ws,plan)
    (ws.case_dir/"0/solid/T").unlink()
    result=PreSolveCompletenessGate(FakeOpenFOAMTools(),ws).validate(plan)
    assert not result.valid and any("0/solid/T" in f for f in result.failures)


def test_a07_mesh_digest_separates_regions_and_tracks_shared_geometry(tmp_path):
    ws=CaseWorkspace(tmp_path);plan=multi_plan();seed_regions(ws,plan)
    fluid,solid=region_mesh_digest(ws,"fluid"),region_mesh_digest(ws,"solid")
    (ws.case_dir/"constant/solid/polyMesh/boundary").write_text("changed mesh")
    assert region_mesh_digest(ws,"fluid")==fluid
    assert region_mesh_digest(ws,"solid")!=solid
    ws.write_text("constant/triSurface/shared.stl","solid a\nendsolid a\n")
    assert region_mesh_digest(ws,"fluid")!=fluid


def test_a06_interface_checks_membership_not_flux_truth(tmp_path):
    ws=CaseWorkspace(tmp_path);plan=multi_plan();seed_regions(ws,plan)
    plan.interfaces=[RegionInterface(region="fluid",patch="interface",neighbour_region="solid",neighbour_patch="missing",fields=["T"])]
    result=PreSolveCompletenessGate(FakeOpenFOAMTools(),ws).validate(plan)
    assert not result.valid and any("neighbour patch missing" in f for f in result.failures)
    assert any("flux conservation" in w for w in result.warnings)


def test_a08_binary_asset_hash_approval_and_immutability(tmp_path):
    source=tmp_path/"part.stl"
    source.write_bytes(b"binary".ljust(80,b"\0")+struct.pack("<I",1)+struct.pack("<12fH",0,0,1,0,0,0,1,0,0,0,1,0,0))
    ws=CaseWorkspace(tmp_path/"work");registry=AssetRegistry(ws)
    with pytest.raises(WorkspaceSafetyError,match="authorized"):
        registry.import_file(source,"constant/triSurface/part.stl",approved_sources=[])
    record=registry.import_file(source,"constant/triSurface/part.stl",approved_sources=[source],units="m")
    assert record["copy_sha256"]==hashlib.sha256(source.read_bytes()).hexdigest()
    assert record["quality"]["format"]=="binary_stl"
    assert not record["quality"]["watertightness_verified"]
    with pytest.raises(WorkspaceSafetyError):
        ws.write_text("constant/triSurface/part.stl","replacement")
    assert "source_path" not in registry.public_summary()[0]


def test_a08_asset_symlink_and_outside_case_rejected(tmp_path):
    source=tmp_path/"table.csv";source.write_text("0,1\n1,2\n")
    alias=tmp_path/"alias.csv";alias.symlink_to(source)
    registry=AssetRegistry(CaseWorkspace(tmp_path/"work"))
    with pytest.raises(WorkspaceSafetyError,match="symlink"):
        registry.import_file(alias,"constant/inputData/a.csv",approved_sources=[alias])
    with pytest.raises(WorkspaceSafetyError):
        registry.import_file(source,"../escaped.csv",approved_sources=[source])


def checkpoint_agent(tmp_path,graph_path):
    return CFDEngineeringAgent(ScriptedLLM([]),workspace=tmp_path,capability_db=graph_path,tools=FakeOpenFOAMTools())


def test_a09_draft_and_budget_checkpoint_restore(tmp_path,graph_path):
    agent=checkpoint_agent(tmp_path,graph_path);state=make_state()
    agent.workspace.write_text("system/controlDict",control_dict())
    agent._draft_design_plan=make_plan(state.intake);agent._draft_authoring_brief="preserve exact draft"
    agent._retrieval_cycles["prepare"]=2;state.engineering_next_step=17
    CheckpointStore(agent).save(state,reason="safe-boundary")
    restored_agent=checkpoint_agent(tmp_path,graph_path)
    restored=CheckpointStore(restored_agent).load()
    assert restored.engineering_next_step==17
    assert restored_agent._draft_design_plan.digest()==agent._draft_design_plan.digest()
    assert restored_agent._draft_authoring_brief=="preserve exact draft"
    assert restored_agent._retrieval_cycles["prepare"]==2
    assert not restored.solve_approved and restored.execution_approval is None


@pytest.mark.parametrize("mutation",["bytes","environment","pending","transaction"])
def test_a09_uncertain_or_changed_state_is_preserved_not_replayed(tmp_path,graph_path,mutation):
    agent=checkpoint_agent(tmp_path,graph_path);state=make_state()
    agent.workspace.write_text("system/controlDict",control_dict())
    if mutation=="pending": state.pending_action={"status":"intent","action":"native"}
    if mutation=="transaction": agent._active_transaction={"unknown":True}
    store=CheckpointStore(agent);store.save(state,reason="boundary")
    before=store.path.read_bytes()
    if mutation=="bytes": (agent.workspace.case_dir/"system/controlDict").write_text("changed")
    if mutation=="environment": agent.tools.version="13"
    with pytest.raises(CheckpointError): store.load()
    assert store.path.read_bytes()==before


def test_a10_mpi_structure_and_existing_outputs_preserved(tmp_path):
    plan=multi_plan();plan.execution.parallel=ParallelExecution(mode="local_mpi",ranks=2,decomposition_method="scotch")
    ws=CaseWorkspace(tmp_path);ws.write_text("system/decomposeParDict","numberOfSubdomains 2; method scotch;")
    calls=[]
    class Transport:
        def run_native_command(self,name,case,**kwargs):
            calls.append((name,kwargs))
            for rank in range(2):
                for layout in region_layouts(plan):
                    for rel in [*[f"{layout.mesh_dir}/{x}" for x in ["points","faces","owner","neighbour","boundary"]], f"{layout.field_dir}/T"]:
                        p=Path(case)/f"processor{rank}"/rel;p.parent.mkdir(parents=True,exist_ok=True);p.write_text("fixture")
            return tool_result(name,success=True)
    transport=Transport();manifest=prepare_parallel(transport,ws,plan)
    assert calls[0][1]["arguments"]==["-allRegions"]
    verify_parallel_inputs(ws,plan,manifest)
    with pytest.raises(ValueError,match="preserved"): prepare_parallel(transport,ws,plan)
    assert len(calls)==1
    (ws.case_dir/"processor1/0/solid/T").write_text("tampered")
    with pytest.raises(ValueError,match="changed"): verify_parallel_inputs(ws,plan,manifest)
    with pytest.raises(ValueError,match="Final-time"): reconstruct_parallel(transport,ws,plan,1.0)


def test_a11_numeric_override_preserves_latest_not_historical():
    active,history,current=active_requirement_texts(["Re=1000","Re=2000"])
    assert "1000" not in active[0] and history[0]["status"]=="superseded"
    spec=make_intake(reynolds="2000")
    # Other user evidence in the shared fixture must remain present.
    req=UserRequest(prompt="length 2 m square obstacle vortex shedding Re=1000",conversation_turns=["Re=2000"])
    spec.facts=[f for f in spec.facts if f.id in {"operating.reynolds_number"}]
    spec.status="needs_clarification"
    with pytest.raises(ValueError,match="missing"):
        validate_intake_provenance(spec,req)  # unrepresented length remains protected
    req.prompt="Re=1000"
    validate_intake_provenance(spec,req)
    assert spec.requirement_history[0].target=="operating.reynolds_number"


def test_a11_different_targets_questions_and_cancellation_are_not_silent_overrides():
    active,history,current=active_requirement_texts(["inlet pressure=100", "outlet pressure=200"])
    assert "100" in active[0] and not history
    active,history,current=active_requirement_texts(["Re=1000","Why Re=2000?"])
    assert not history and current["operating.reynolds_number"]["value"]=="1000"
    active,history,current=active_requirement_texts(["Re=1000","cancel Re"])
    assert not current and history[0]["status"]=="cancelled"


def test_a12_units_and_targets_must_both_be_preserved():
    a=PhysicalQuantity(quantity="length",value=1000,unit="mm",dimensions=(0,1,0,0,0,0,0),region="fluid")
    b=a.model_copy(update={"value":1,"unit":"m"});assert_equivalent(a,b)
    with pytest.raises(ValueError,match="region"): assert_equivalent(a,b.model_copy(update={"region":"solid"}))
    with pytest.raises(ValueError,match="dimensions"): si_value(a.model_copy(update={"unit":"Pa"}))


def test_a13_unresolved_physics_ambiguity_cannot_be_review_ready():
    spec=make_intake().model_dump()
    spec["ambiguities"]=[{"id":"heat","impact":"physics","alternatives":["heat source","fixed temperature"],"question":"Which physical input?"}]
    with pytest.raises(ValidationError): CFDIntakeSpec.model_validate(spec)
    with pytest.raises(ValidationError): SemanticAmbiguity(id="a",impact="geometry",alternatives=["solid","hole"],question="Which?",selected="solid")


def test_a14_read_syntax_is_projected_with_hash_not_just_search_summary(tmp_path,graph_path):
    refs=tmp_path/"refs";refs.mkdir();(refs/"controlDict").write_text("FoamFile { class dictionary; }\nendTime 1;\n")
    agent=checkpoint_agent(tmp_path/"work",graph_path);agent.references=OpenFOAMReferenceIndex({"tutorials":refs})
    state=make_state(syntax_evidence=False);plan=make_plan(state.intake);plan.required_case_files=["system/controlDict"]
    eid=canonical_engineering_evidence_id("openfoam_reference","tutorials:controlDict")
    plan.implementation_evidence_bindings=[ImplementationEvidenceBinding(path="system/controlDict",evidence_ids=[eid])]
    with pytest.raises(ValueError): implementation_evidence_pack(state,plan)
    event=agent._dispatch_tool_action(ReadReferenceAction(type="read_reference",reference="tutorials:controlDict",rationale="actual read"),step=1,native_execution=False,phase="prepare",state=state)
    assert event.success
    pack=implementation_evidence_pack(state,plan)
    assert pack["complete"] and "endTime 1" in pack["records"][0]["content"]
    assert len(pack["records"][0]["excerpt_sha256"])==64
    assert not evidence_coverage_failures(plan,state)
    plan.implementation_evidence_bindings=[]
    assert evidence_coverage_failures(plan,state)


def test_a15_mandatory_context_is_not_truncated():
    facts=[{"id":f"fact.{i}","value":"x"*30} for i in range(50)]
    result=build_bounded_json_prompt("",{"confirmed_intake":{"facts":facts},"observations":["z"*1000]*100},max_chars=14000)
    assert json.loads(result.prompt)["confirmed_intake"]["facts"]==facts
    with pytest.raises(ContextBudgetError): build_bounded_json_prompt("",{"confirmed_intake":{"facts":facts*10}},max_chars=4000)


def test_a16_early_design_rejects_undeclared_binding_and_duplicate_regions():
    state=make_state();plan=make_plan(state.intake)
    plan.confirmed_fact_bindings[0].case_files=["system/undeclared"]
    assert any("undeclared" in x for x in validate_design(plan,state.intake))
    plan=multi_plan();plan.region_layouts.append(plan.region_layouts[0])
    with pytest.raises(ValueError,match="Duplicate"): region_layouts(plan)


def test_a18_19_source_discovery_is_not_runtime_registration(graph_path):
    installation=InstalledOpenFOAMIR(version="14",installation_configured=True,
        executables=[InstalledExecutable(name="foamRun",category="execution_driver")],
        components=[InstalledComponent(name="surfaceFieldValue",category="function_object",source="installed_source")])
    catalog=CapabilityCatalog(graph_path,installation=installation)
    provider=catalog.provider("installed.function_object.surfaceFieldValue")
    assert provider.verification_level=="source_discovered" and not provider.metadata["runtime_load_verified"]
    assert catalog.provider("installed.application.foamRun").verification_level=="binary_present"


def test_a20_unicode_search_fair_scopes_and_actual_read_hash(tmp_path):
    roots={}
    for scope in ["source","tutorials","modules","etc"]:
        root=tmp_path/scope;root.mkdir();roots[scope]=root
        for i in range(5): (root/f"{i}.C").write_text("heat transfer temperature heatSource\n")
    refs=OpenFOAMReferenceIndex(roots)
    result=refs.search("\uc5f4\uc804\ub2ec",max_files=4)
    assert len(result)==4 and set(x["scope"] for x in result)==set(roots)
    assert all(n==1 for n in refs.last_search_metadata["inspected_by_scope"].values())
    text=refs.read("modules:0.C");assert text.startswith("1:")
    assert refs.last_read_metadata["source_sha256"]==hashlib.sha256((roots["modules"]/"0.C").read_bytes()).hexdigest()
    (roots["modules"]/"0.C").write_text("cold flow")
    assert not refs.search("heatSource",scope="modules",max_files=1)


def test_a24_postprocessing_uses_region_module_not_driver():
    plan=multi_plan()
    assert resolve_postprocess_context(plan,"solid")=="solid"
    assert resolve_postprocess_context(plan,"fluid")=="fluid"
    with pytest.raises(ValueError): resolve_postprocess_context(plan)
    with pytest.raises(ValueError): resolve_postprocess_context(plan,"unknown")


def test_a25_generic_scalar_quantities_provenance_and_declared_semantics(tmp_path):
    ws=CaseWorkspace(tmp_path);p=ws.case_dir/"postProcessing/inlet/0/p.dat";p.parent.mkdir(parents=True);p.write_text("# Time p\n0 0\n1 1\n3 3\n")
    spec=QuantityOfInterest(id="pressure.mean",quantity="pressure",region="fluid",selection="inlet",unit="Pa",source_path=p.relative_to(ws.case_dir).as_posix(),operation="time_mean")
    result=analyze_quantity(ws,spec)
    assert result["value"]==pytest.approx(1.5) and result["arithmetic_verified"]
    assert not result["physical_semantics_verified"]
    assert result["sources"][0]["sha256"]==hashlib.sha256(p.read_bytes()).hexdigest()
    with pytest.raises(ValueError): analyze_quantity(ws,spec.model_copy(update={"end_time":4}))


def test_a26_nonuniform_time_weighting_and_restart_reconciliation():
    series,cleanup=clean_series([[0,0],[1,1],[4,4],[1,10],[2,20],[2,21]])
    assert series==[(0,0),(1,10),(2,21)]
    assert cleanup=={"duplicate_times":1,"restart_segments":1}
    stats=time_statistics([(0,0),(1,1),(3,3)])
    assert stats["time_mean"]==pytest.approx(1.5)
    assert stats["rms"]==pytest.approx(math.sqrt(3))
    assert not stats["uniform_time_spacing"]
    with pytest.raises(ValueError): clean_series([[0,1],[1,float("nan")]])


@pytest.mark.parametrize("text,expected",[("Logged in using ChatGPT","chatgpt"),("Logged in using an API key","api_key"),("Logged in","unknown"),("Not logged in using ChatGPT","unknown")])
def test_a29_auth_kind_is_not_any_logged_in_message(text,expected):
    assert authentication_kind(text)==expected


def test_a30_environment_allowlist_and_event_usage(monkeypatch):
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY","private");monkeypatch.setenv("OPENAI_API_KEY","private")
    monkeypatch.setenv("BASH_ENV","/tmp/untrusted");monkeypatch.setenv("HOME","/home/user")
    env=_codex_environment();assert env["HOME"]=="/home/user"
    assert not {"AWS_SECRET_ACCESS_KEY","OPENAI_API_KEY","BASH_ENV"}&env.keys()
    contract,usage=inspect_events('{"type":"turn.completed","usage":{"input_tokens":10,"output_tokens":2}}\n')
    assert usage["totalTokens"]==12 and not contract["pre_execution_tool_prevention_verified"]
    _,unknown=inspect_events('{"type":"turn.completed"}\n');assert unknown is None
    for kind in ["command_execution","mcp_tool_call","file_change","web_search"]:
        with pytest.raises(ValueError,match="tool/action"):
            inspect_events(json.dumps({"type":"item.started","item":{"type":kind}}))
