"""Explicit, preserved-output local MPI restart preparation and verification.

Supported layout: uncollated processorN directories, ASCII/gzip fields, identical
rank topology and solver modules. Binary/collated/distributed layouts are refused,
not guessed. The chosen snapshot and decomposition are hash-bound to approval.
"""
from __future__ import annotations
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import uuid
from openfoam_agent.contracts.execution import digest, execution_document
from openfoam_agent.contracts.models import ParallelRestart
from openfoam_agent.contracts.regions import region_layouts
from openfoam_agent.verification.foam_semantics import parse_top_level_assignments


MESH_FILES=("points","faces","owner","neighbour","boundary")


def _safe(case,path):
    if case not in path.resolve().parents or any(p.is_symlink() for p in [path,*path.parents] if p!=case.parent):
        raise ValueError("Unsafe/symlink parallel restart path.")
    if not path.is_file() or path.stat().st_size<=0 or path.stat().st_size>256_000_000:
        raise ValueError(f"Missing, empty or oversized restart input: {path.name}")
    with path.open("rb") as stream: sha=hashlib.file_digest(stream,"sha256").hexdigest()
    return {"path":path.relative_to(case).as_posix(),"sha256":sha,"size_bytes":path.stat().st_size}


def _field(case,path,*,mesh=None,baseline=None):
    from openfoam_agent.postprocessing.native_fields import Tokens,header
    candidates=[p for p in (path,path.with_name(path.name+".gz")) if p.exists()]
    if len(candidates)!=1: raise ValueError("Missing or ambiguous compressed restart field.")
    target=candidates[0];record=_safe(case,target)
    reader=gzip.open if target.suffix==".gz" else open
    with reader(target,"rb") as stream: raw=stream.read(32_000_001)
    if len(raw)>32_000_000: raise ValueError("Restart field exceeds bounded ASCII verification limit.")
    from openfoam_agent.verification.foam_semantics.parser import strip_comments
    text=strip_comments(raw.decode("ascii"))
    if "$" in text or "#" in text:raise ValueError("Unexpanded restart fields need native qualification.")
    stream=Tokens(text);h=header(stream,object_name=path.name);data=stream.dictionary()
    kind=h.get("class",[""])[0]
    widths={"volScalarField":1,"volVectorField":3,"volSymmTensorField":6,"volTensorField":9,
        "surfaceScalarField":1,"surfaceVectorField":3,"pointVectorField":3}
    if kind not in widths:raise ValueError("Unsupported restart field class.")
    dimensions=data.get("dimensions",[])
    if len(dimensions)!=9 or dimensions[0]!="[" or dimensions[-1]!="]" or any(not re.fullmatch(r"[+-]?[0-9]+",x) for x in dimensions[1:-1]):
        raise ValueError("Restart field lacks explicit indexed dimensions.")
    if baseline is not None and (kind,dimensions)!=baseline:
        raise ValueError("Restart field class/dimensions changed from the sealed initial field.")
    width=widths[kind]
    def values(tokens,count):
        s=Tokens(" ".join(tokens));mode=s.pop()
        def value():
            if width>1:s.pop("(")
            for _ in range(width):s.number()
            if width>1:s.pop(")")
        if mode=="uniform":value()
        elif mode=="nonuniform":
            s.pop({1:"List<scalar>",3:"List<vector>",6:"List<symmTensor>",9:"List<tensor>"}[width])
            n=s.integer()
            if count is not None and n!=count:raise ValueError("Restart field count does not match rank mesh.")
            s.pop("(")
            for _ in range(n):value()
            s.pop(")")
        else:raise ValueError("Unsupported restart field encoding.")
        if s.peek() is not None:raise ValueError("Trailing restart field value tokens.")
    count=(len(mesh.volumes) if kind.startswith("vol") else len(mesh.points) if kind.startswith("point") else len(mesh.neighbour)) if mesh is not None else None
    values(data.get("internalField",[]),count)
    boundary=data.get("boundaryField")
    if not isinstance(boundary,dict):raise ValueError("Restart field lacks boundary dictionary.")
    if mesh is not None:
        if set(boundary)!=set(mesh.patches):raise ValueError("Restart patch fields do not match rank mesh.")
        for name,entry in boundary.items():
            if not isinstance(entry,dict) or not entry.get("type"):raise ValueError("Invalid restart boundary field.")
            if "value" in entry:
                if kind.startswith("point"):
                    patch=mesh.patches[name]
                    n=len({v for face in mesh.faces[patch["start"]:patch["start"]+patch["count"]] for v in face})
                else:n=mesh.patches[name]["count"]
                values(entry["value"],n)
    return record,(kind,dimensions)


def _times(root):
    times={}
    for path in root.iterdir():
        try: value=float(path.name)
        except ValueError: continue
        if not math.isfinite(value) or value<0 or not path.is_dir() or path.is_symlink():
            raise ValueError("Invalid numeric restart directory.")
        if value in times: raise ValueError("Ambiguous numeric time aliases on a rank.")
        times[value]=path.name
    return times


def _ranks(case,count):
    expected={f"processor{i}" for i in range(count)}
    observed={p.name for p in case.iterdir() if p.name.startswith("processor")}
    if observed!=expected or any((case/name).is_symlink() or not (case/name).is_dir() for name in expected):
        raise ValueError("Existing processor topology differs from the approved uncollated rank count.")
    return [case/f"processor{i}" for i in range(count)]


def _snapshot(workspace,plan,value):
    if plan.mesh_motion_requirement != "static":
        raise ValueError("Parallel restart currently requires a static mesh; dynamic topology needs native qualification.")
    case=workspace.case_dir;records=[];names=[]
    parallel=plan.execution.parallel
    for root in _ranks(case,parallel.ranks):
        times=_times(root)
        if value not in times: raise ValueError("Restart time is not common to every rank.")
        name=times[value];names.append(name)
        for layout in region_layouts(plan):
            from types import SimpleNamespace
            from openfoam_agent.postprocessing.native_fields import read_mesh
            mesh_records=[]
            mesh=read_mesh(SimpleNamespace(case_dir=root),layout.region,mesh_records,allow_processor=True)
            for record in mesh_records:
                records.append({**record,"path":root.name+"/"+record["path"]})
            fields=set(layout.required_fields)
            fields.update(Path(p).name for p in plan.required_case_files if str(Path(p).parent)==layout.field_dir)
            if not fields: raise ValueError("Restart fields are undeclared for a region.")
            for field in sorted(fields):
                rel=Path(name)/(layout.region or "")/field
                baseline_path=root/layout.field_dir/field
                if not baseline_path.is_file() and not baseline_path.with_name(field+".gz").is_file():
                    baseline_path=case/layout.field_dir/field
                baseline_record,baseline=_field(case,baseline_path)
                record,_=_field(case,root/rel,mesh=mesh,baseline=baseline)
                records.extend([baseline_record,record])
    if len(set(names))!=1:
        raise ValueError("All ranks must use the same literal restart directory name.")
    decompose=workspace.resolve_case_path("system/decomposeParDict",must_exist=True)
    entries,complete=parse_top_level_assignments(decompose.read_text())
    if not complete or str(entries.get("numberOfSubdomains","")).strip()!=str(parallel.ranks) or str(entries.get("method","")).strip()!=parallel.decomposition_method:
        raise ValueError("Decomposition settings do not match the restart contract.")
    records.append(_safe(case,decompose))
    execution=execution_document(plan)
    execution["parallel"]["restart"]=None
    return {"version":1,"time_name":names[0],"ranks":parallel.ranks,"files":sorted({r["path"]:r for r in records}.values(),key=lambda x:x["path"]),
        "execution_without_restart_sha256":digest(execution),"intake_sha256":plan.confirmed_intake_sha256,
        "scope":"ASCII/gzip field checks and mesh hashes; not a numerical correctness certificate"}


def inspect_parallel_restart(workspace,plan,time_name="latest"):
    if not plan.execution or plan.execution.parallel.mode!="local_mpi":
        raise ValueError("Parallel restart requires a local_mpi execution contract.")
    ranks=_ranks(workspace.case_dir,plan.execution.parallel.ranks)
    common=set.intersection(*(set(_times(root)) for root in ranks))
    candidates=sorted(common,reverse=True) if time_name=="latest" else [float(time_name)]
    rejected=[]
    for value in candidates:
        if not math.isfinite(value): raise ValueError("Non-finite restart time.")
        try:
            snapshot=_snapshot(workspace,plan,value)
            snapshot["rejected_newer_snapshots"]=rejected
            return snapshot
        except (ValueError,OSError,UnicodeError) as exc:
            rejected.append({"time":value,"reason":str(exc)})
            if time_name!="latest": raise
    raise ValueError("No complete common restart snapshot: "+str(rejected[:5]))


def verify_restart(workspace,plan):
    restart=plan.execution.parallel.restart
    if restart is None: raise ValueError("An explicit restart receipt is required.")
    path=workspace.root/"parallel-restart.json"
    if path.is_symlink() or not path.is_file() or path.stat().st_size>16_000_000:
        raise ValueError("Restart receipt is missing or unsafe.")
    envelope=json.loads(path.read_text());receipt=envelope["receipt"]
    if digest(receipt)!=envelope.get("sha256") or envelope["sha256"]!=restart.manifest_sha256:
        raise ValueError("Restart receipt differs from the approved hash.")
    current=_snapshot(workspace,plan,float(restart.time_name))
    for key in ("files","execution_without_restart_sha256","intake_sha256","ranks","time_name"):
        if current[key]!=receipt[key]: raise ValueError("Parallel restart input/contract changed: "+key)
    return receipt


def rewrite_top_level(text,replacements):
    """Replace only literal top-level entries; comments/inline/nested text survive."""
    token=re.compile(r'/\*.*?\*/|//[^\n]*|"(?:\\.|[^"\\])*"|[{}()\[\];]|[^\s{}()\[\];]+',re.S)
    matches=[m for m in token.finditer(text) if not m.group().startswith(("//","/*"))]
    stack=[];spans={};i=0
    while i<len(matches):
        m=matches[i];value=m.group()
        if not stack and value in replacements:
            if value in spans:raise ValueError("Duplicate literal restart control key.")
            j=i+1
            while j<len(matches) and matches[j].group()!=";":
                if matches[j].group() in {"{","}","(",")","[","]"}:raise ValueError("Restart control must be a literal scalar.")
                j+=1
            if j!=i+2:raise ValueError("Ambiguous restart control value.")
            spans[value]=(m.start(),matches[j].end());i=j+1;continue
        if value in {"{","(","["}:stack.append(value)
        if value in {"}",")","]"}:
            if not stack or stack.pop()!={"}":"{",")":"(","]":"["}[value]:raise ValueError("Unbalanced restart control dictionary.")
        i+=1
    if stack or set(spans)!=set(replacements):raise ValueError("Restart needs explicit unique startFrom/startTime entries.")
    for key,(start,end) in sorted(spans.items(),key=lambda x:x[1][0],reverse=True):
        text=text[:start]+key+" "+replacements[key]+";"+text[end:]
    return text


def prepare_parallel_restart(workspace,plan,*,time_name="latest"):
    """Operator-invoked transaction. Preserve future/partial outputs, do not run.

    An intent journal is written BEFORE mutations; an interrupted transaction is
    blocked on resume instead of replayed. The caller must re-seal/reapprove.
    """
    snapshot=inspect_parallel_restart(workspace,plan,time_name)
    selected=float(snapshot["time_name"])
    from .completion import compile_runtime_contract
    runtime_contract = compile_runtime_contract(plan, workspace)
    bound = runtime_contract.execution_bound
    if bound.mode != "transient" or bound.end_time is None:
        raise ValueError("Parallel restart requires a controller-compiled transient execution interval.")
    if selected < bound.start_time or selected >= bound.end_time:
        raise ValueError("Restart time must lie inside the controller-compiled execution interval.")
    journal=workspace.root/"parallel-restart-intent.json"
    if journal.exists(): raise ValueError("An interrupted restart transaction needs manual reconciliation.")
    control=workspace.resolve_case_path("system/controlDict",must_exist=True)
    original=control.read_text();entries,complete=parse_top_level_assignments(original)
    if not complete: raise ValueError("Dynamic controlDict cannot be changed mechanically for restart.")
    modified=rewrite_top_level(original,{"startFrom":"startTime","startTime":snapshot["time_name"]})
    archive=workspace.root/"restart-history"/uuid.uuid4().hex
    archive.mkdir(parents=True)
    moves=[]
    roots=_ranks(workspace.case_dir,plan.execution.parallel.ranks)+[workspace.case_dir]
    for root in roots:
        for value,name in _times(root).items():
            if value>selected:
                source=root/name;dest=archive/"outputs"/source.relative_to(workspace.case_dir)
                moves.append((source,dest))
    intent={"archive":str(archive),"selected_time":snapshot["time_name"],"moves":[[str(a),str(b)] for a,b in moves],"original_control_sha256":hashlib.sha256(original.encode()).hexdigest()}
    with journal.open("x") as stream:
        json.dump(intent,stream,sort_keys=True);stream.flush();os.fsync(stream.fileno())
    (archive/"controlDict.before").write_text(original)
    for source,dest in moves:
        dest.parent.mkdir(parents=True,exist_ok=True);os.replace(source,dest)
    workspace.write_text("system/controlDict",modified)
    receipt={**snapshot,"archive":str(archive.relative_to(workspace.root)),"original_control_sha256":intent["original_control_sha256"]}
    envelope={"receipt":receipt,"sha256":digest(receipt)}
    receipt_path=workspace.root/"parallel-restart.json"
    if receipt_path.is_symlink():raise ValueError("Unsafe restart receipt path.")
    temporary=receipt_path.with_suffix(".tmp")
    if temporary.is_symlink():raise ValueError("Unsafe restart receipt temporary path.")
    with temporary.open("w") as stream:
        json.dump(envelope,stream,sort_keys=True);stream.flush();os.fsync(stream.fileno())
    os.replace(temporary,receipt_path)
    updated=plan.model_copy(deep=True)
    updated.execution.parallel.restart=ParallelRestart(time_name=snapshot["time_name"],manifest_sha256=envelope["sha256"])
    # The requested start/end interval is preserved. Runtime uses an effective
    # progress start only after verifying the explicit restart receipt.
    verify_restart(workspace,updated)
    (archive/"transaction.json").write_text(json.dumps(intent,sort_keys=True))
    journal.unlink()
    return updated,envelope


def prepare_restart_state(agent,state,*,time_name="latest"):
    from openfoam_agent.workflow.states import State
    from openfoam_agent.tools.linux_isolation import workspace_execution_lock
    if state.engineering_plan is None or state.case_seal is None:
        raise ValueError("A previously sealed engineering plan is required for restart.")
    with workspace_execution_lock(agent.workspace.root):
        agent.safety.verify_seal(state.engineering_plan,state.case_seal)
        state.pending_action={"status":"intent","kind":"parallel_restart_prepare"}
        agent.checkpoint(state,"parallel-restart-preparation-intent")
        plan,receipt=prepare_parallel_restart(agent.workspace,state.engineering_plan,time_name=time_name)
        state.engineering_plan=plan
        state.case_seal=agent.workspace.seal(plan)
        state.parallel_evidence=receipt
        from .completion import compile_runtime_contract
        state.runtime_contract=compile_runtime_contract(plan,agent.workspace)
        state.solve_approved=False;state.execution_approval=None
        state.pending_action=None
        state.transition(State.SOLVE_READY,"Parallel restart inputs selected and preserved. Fresh /solve approval is required; no solver has run.")
        agent.checkpoint(state,"parallel-restart-prepared")
    return state
