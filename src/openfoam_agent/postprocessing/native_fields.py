"""Auditable reductions from actual saved ASCII/gzip fields, not table labels.

Scope: reconstructed static polyhedral mesh with planar convex faces, scalar
fields, explicit indexed dimensions, literal patch values. Unsupported binary,
regex, dynamic mesh, unit-annotated/macro fields are refused. This implementation
is intentionally independent of an LLM-declared column interpretation.
"""
from __future__ import annotations
from dataclasses import dataclass
import gzip
import hashlib
import math
from pathlib import Path
import re
from openfoam_agent.verification.foam_semantics.parser import strip_comments

MAX_BYTES=32_000_000
MAX_ITEMS=250_000
TOKEN=re.compile(r'"[^"\n]*"|[{}()\[\];]|[^\s{}()\[\];]+')


def source_path(workspace, relative):
    p=Path(relative)
    if p.is_absolute() or not p.parts or any(x in {"..", "."} for x in p.parts):
        raise ValueError("Unsafe physical evidence path.")
    path=workspace.case_dir/p
    if any(parent.is_symlink() for parent in [path,*path.parents] if parent!=workspace.case_dir.parent):
        raise ValueError("Symlink physical evidence is forbidden.")
    if workspace.case_dir.resolve() not in path.resolve().parents or not path.is_file():
        raise ValueError("Physical evidence is not a regular case file.")
    if path.stat().st_size>MAX_BYTES: raise ValueError("Physical evidence file exceeds the analysis limit.")
    return path


def read_source(workspace, relative, evidence):
    candidates=[p for p in (relative,relative+".gz") if (workspace.case_dir/p).exists()]
    if len(candidates)!=1: raise ValueError("Missing/ambiguous physical evidence: "+relative)
    path=source_path(workspace,candidates[0]);raw=path.read_bytes()
    record={"path":candidates[0],"sha256":hashlib.sha256(raw).hexdigest(),"size_bytes":len(raw)}
    previous=next((r for r in evidence if r["path"]==record["path"]),None)
    if previous is not None and previous!=record:raise ValueError("Source changed while analysis was running.")
    if previous is None:evidence.append(record)
    if path.suffix==".gz":
        with gzip.open(path,"rb") as stream:raw=stream.read(MAX_BYTES+1)
    if len(raw)>MAX_BYTES:raise ValueError("Decompressed physical evidence is too large.")
    text=strip_comments(raw.decode("ascii"))
    if "$" in text or "#" in text:raise ValueError("Physical evidence contains unresolved directives/macros.")
    return text


class Tokens:
    def __init__(self,text):
        self.values=TOKEN.findall(text);self.index=0
        if len(self.values)>3_000_000:raise ValueError("Physical evidence token limit exceeded.")
    def peek(self):return self.values[self.index] if self.index<len(self.values) else None
    def pop(self,expected=None):
        if self.index>=len(self.values):raise ValueError("Truncated physical evidence.")
        token=self.values[self.index];self.index+=1
        if expected is not None and token!=expected:raise ValueError("Unexpected physical evidence token: "+token)
        return token
    def integer(self):
        token=self.pop()
        if not re.fullmatch(r"[0-9]+",token):raise ValueError("Expected a nonnegative integer.")
        n=int(token)
        if n>MAX_ITEMS:raise ValueError("Physical mesh/list size exceeds bounded analysis.")
        return n
    def number(self):
        try:value=float(self.pop())
        except (ValueError,OverflowError):raise ValueError("Expected a literal scalar.") from None
        if not math.isfinite(value):raise ValueError("Non-finite physical evidence.")
        return value
    def dictionary(self,closing=None,depth=0):
        if depth>12:raise ValueError("Dictionary nesting exceeds bounded analysis.")
        result={}
        while self.peek() is not None and self.peek()!=closing:
            key=self.pop().strip('"')
            if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.-]*",key) or key in result:
                raise ValueError("Duplicate/nonliteral physical evidence dictionary key.")
            if self.peek()=="{":
                self.pop();result[key]=self.dictionary("}",depth+1);self.pop("}")
                if self.peek()==";":self.pop()
            else:
                value=[];stack=[]
                while self.peek() is not None:
                    token=self.pop()
                    if token==";" and not stack:break
                    if token in {"(","["}:stack.append(token)
                    if token in {")","]"}:
                        expected="(" if token==")" else "["
                        if not stack or stack.pop()!=expected:raise ValueError("Unbalanced physical entry.")
                    if token in {"{","}"}:raise ValueError("Malformed physical dictionary entry.")
                    value.append(token)
                else:raise ValueError("Missing physical entry semicolon.")
                if stack or not value:raise ValueError("Incomplete physical entry.")
                result[key]=value
        return result


def header(stream, expected_class=None, object_name=None):
    stream.pop("FoamFile");stream.pop("{");h=stream.dictionary("}");stream.pop("}")
    if h.get("format")!=["ascii"]:raise ValueError("Only explicitly ASCII field/mesh files can be physically checked.")
    if expected_class is not None and h.get("class")!=[expected_class]:raise ValueError("Native field/mesh class mismatch.")
    if object_name is not None and h.get("object")!=[object_name]:raise ValueError("Native field object name mismatch.")
    return h


def scalar_values(tokens,count):
    stream=Tokens(" ".join(tokens));kind=stream.pop()
    if kind=="uniform":values=[stream.number()]*count
    elif kind=="nonuniform":
        stream.pop("List<scalar>")
        if stream.integer()!=count:raise ValueError("Native field element count does not match its mesh.")
        stream.pop("(");values=[stream.number() for _ in range(count)];stream.pop(")")
    else:raise ValueError("Unsupported native field value encoding.")
    if stream.peek() is not None:raise ValueError("Trailing native field data/units are not supported.")
    return values


def sub(a,b):return tuple(x-y for x,y in zip(a,b))
def cross(a,b):return (a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0])
def dot(a,b):return sum(x*y for x,y in zip(a,b))
def norm(a):return math.sqrt(dot(a,a))
def add(a,b):return tuple(x+y for x,y in zip(a,b))


@dataclass
class Mesh:
    points:list
    faces:list
    owner:list
    neighbour:list
    patches:dict
    areas:list
    volumes:list


def read_mesh(workspace,region,evidence,*,allow_processor=False):
    if region and not re.fullmatch(r"[A-Za-z][A-Za-z0-9_.-]*",region):raise ValueError("Invalid region.")
    # Refuse dynamic topology/points instead of applying constant geometry to it.
    for path in workspace.case_dir.iterdir():
        try:float(path.name)
        except ValueError:continue
        if (path/region/"polyMesh").exists():raise ValueError("Moving/time-local meshes need native-qualified physical reduction.")
    base="constant/"+(region+"/" if region else "")+"polyMesh/"
    data={}
    for name in ("points","faces","owner","neighbour"):
        stream=Tokens(read_source(workspace,base+name,evidence));header(stream,object_name=name)
        n=stream.integer();stream.pop("(");values=[]
        for i in range(n):
            if name=="points":
                stream.pop("(");values.append(tuple(stream.number() for _ in range(3)));stream.pop(")")
            elif name=="faces":
                count=stream.integer();stream.pop("(");values.append([stream.integer() for _ in range(count)]);stream.pop(")")
            else:values.append(stream.integer())
        stream.pop(")")
        if stream.peek() is not None:raise ValueError("Trailing mesh list data.")
        data[name]=values
    points,faces,owner,neighbour=[data[n] for n in ("points","faces","owner","neighbour")]
    if not points or not faces or len(owner)!=len(faces) or len(neighbour)>len(faces):raise ValueError("Invalid mesh addressing lengths.")
    cells=max(owner+neighbour)+1
    if cells>MAX_ITEMS or set(owner+neighbour)!=set(range(cells)):raise ValueError("Noncontiguous/oversized mesh cells.")
    stream=Tokens(read_source(workspace,base+"boundary",evidence));header(stream,object_name="boundary")
    count=stream.integer();stream.pop("(");patches={};covered=[]
    for i in range(count):
        name=stream.pop().strip('"')
        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_.-]*",name) or name in patches:raise ValueError("Invalid/duplicate patch name.")
        stream.pop("{");attrs=stream.dictionary("}");stream.pop("}")
        start=Tokens(" ".join(attrs.get("startFace",[]))).integer()
        n=Tokens(" ".join(attrs.get("nFaces",[]))).integer()
        kind=attrs.get("type",[""])[0]
        if kind.startswith("processor") and not allow_processor:raise ValueError("Physical analysis requires reconstructed fields/mesh.")
        if start<len(neighbour) or start+n>len(faces):raise ValueError("Patch addressing exceeds boundary faces.")
        patches[name]={"type":kind,"start":start,"count":n};covered+=list(range(start,start+n))
    stream.pop(")")
    if stream.peek() is not None or sorted(covered)!=list(range(len(neighbour),len(faces))):raise ValueError("Mesh boundary does not cover each external face exactly once.")
    origin=tuple(sum(p[j] for p in points)/len(points) for j in range(3))
    volumes=[0.0]*cells;closure=[(0.,0.,0.)]*cells;areas=[];cell_area=[0.]*cells;edges=[{} for _ in range(cells)]
    for index,face in enumerate(faces):
        if len(face)<3 or len(face)!=len(set(face)) or any(i>=len(points) for i in face):raise ValueError("Degenerate mesh face.")
        p=[points[i] for i in face];area=(0.,0.,0.);volume=0.
        for i in range(1,len(p)-1):
            tri=cross(sub(p[i],p[0]),sub(p[i+1],p[0]))
            area=add(area,tuple(x/2 for x in tri));volume+=dot(sub(p[0],origin),tri)/6
        mag=norm(area)
        if mag<=0 or not math.isfinite(mag):raise ValueError("Zero/nonfinite face area.")
        scale=max(norm(sub(point,p[0])) for point in p)
        if any(abs(dot(sub(point,p[0]),area))>1e-9*mag*scale for point in p):raise ValueError("Warped faces require native-qualified geometry integration.")
        for i in range(len(p)):
            turn=cross(sub(p[(i+1)%len(p)],p[i]),sub(p[(i+2)%len(p)],p[(i+1)%len(p)]))
            if dot(turn,area)<-1e-12*mag*scale*scale:raise ValueError("Concave faces are outside this verified geometry subset.")
        areas.append(mag)
        targets=[(owner[index],1)]
        if index<len(neighbour):
            if owner[index]>=neighbour[index]:raise ValueError("Internal face owner/neighbour orientation is invalid.")
            targets.append((neighbour[index],-1))
        for cell,sign in targets:
            volumes[cell]+=sign*volume;closure[cell]=add(closure[cell],tuple(sign*x for x in area));cell_area[cell]+=mag
            for a,b in zip(face,face[1:]+face[:1]):
                key=tuple(sorted((a,b)));count,total=edges[cell].get(key,(0,0));edges[cell][key]=(count+1,total+(sign if a<b else -sign))
    if any(not math.isfinite(v) or v<=0 or norm(closure[c])>1e-9*cell_area[c] or any(count!=2 or total!=0 for count,total in edges[c].values()) for c,v in enumerate(volumes)):
        raise ValueError("Mesh cells are not positively oriented and geometrically/topologically closed.")
    return Mesh(points,faces,owner,neighbour,patches,areas,volumes)


def read_field(workspace,time,region,name,mesh,evidence,*,expected_dimensions=None,expected_class=None):
    rel=time+"/"+(region+"/" if region else "")+name
    stream=Tokens(read_source(workspace,rel,evidence));h=header(stream,object_name=name);data=stream.dictionary()
    kind=h.get("class",[""])[0]
    if kind not in {"volScalarField","surfaceScalarField"} or (expected_class is not None and kind!=expected_class):raise ValueError("Only the declared scalar field class is supported.")
    dims=data.get("dimensions",[])
    if len(dims)!=9 or dims[0]!="[" or dims[-1]!="]" or any(not re.fullmatch(r"[+-]?[0-9]+",v) for v in dims[1:-1]):raise ValueError("Indexed seven-base dimensions are required.")
    dimensions=tuple(map(int,dims[1:-1]))
    if expected_dimensions is not None and dimensions!=tuple(expected_dimensions):raise ValueError("Native field dimensions disagree with the physical contract.")
    internal=scalar_values(data.get("internalField",[]),len(mesh.volumes) if kind=="volScalarField" else len(mesh.neighbour))
    boundary=data.get("boundaryField")
    if not isinstance(boundary,dict) or set(boundary)!=set(mesh.patches):raise ValueError("Literal field patches do not match this region mesh.")
    return {"class":kind,"dimensions":dimensions,"internal":internal,"boundary":boundary}


def patch_values(field,mesh,name):
    if name not in mesh.patches:raise ValueError("Unknown patch selection: "+name)
    patch=mesh.patches[name];entry=field["boundary"][name]
    if not isinstance(entry,dict):raise ValueError("Invalid patch field entry.")
    if patch["type"]=="empty":
        if entry.get("type")!=["empty"]:raise ValueError("Empty mesh/field patch types disagree.")
        return [0.0]*patch["count"]
    if "value" not in entry:
        if field["class"]=="volScalarField" and entry.get("type")==["zeroGradient"]:
            return [field["internal"][mesh.owner[i]] for i in range(patch["start"],patch["start"]+patch["count"])]
        raise ValueError("No explicit saved patch value for physical reduction.")
    return scalar_values(entry["value"],patch["count"])


def check_sources_current(workspace,sources):
    for source in sources:
        path=source_path(workspace,source["path"])
        with path.open("rb") as stream:digest=hashlib.file_digest(stream,"sha256").hexdigest()
        if digest!=source["sha256"]:raise ValueError("Physical analysis source changed: "+source["path"])


def reduce_field(workspace,spec):
    from openfoam_agent.contracts.quantities import UNITS
    native=spec.native_field;evidence=[];mesh=read_mesh(workspace,spec.region,evidence);series=[]
    dimensions=native.dimensions
    outdims=list(dimensions)
    if native.reduction=="area_integral":outdims[1]+=2
    if native.reduction=="volume_integral":outdims[1]+=3
    quantity_dimensions={"temperature":(0,0,0,1,0,0,0),"pressure":(1,-1,-2,0,0,0,0),"density":(1,-3,0,0,0,0,0),
        "mass_flow":(1,0,-1,0,0,0,0),"volume_flow":(0,3,-1,0,0,0,0),"heat_rate":(1,2,-3,0,0,0,0),
        "mass":(1,0,0,0,0,0,0),"energy":(1,2,-2,0,0,0,0)}
    if tuple(outdims)!=quantity_dimensions[native.quantity_kind] or spec.quantity!=native.quantity_kind:
        raise ValueError("Physical quantity kind/name/dimensions disagree; use the canonical approved quantity kind.")
    unit=UNITS.get(spec.unit)
    if unit is None or tuple(outdims)!=unit[2] or unit[0]!=1.0 or unit[1]!=0.0:raise ValueError("Physical reduction requires a matching SI output unit, not a declared label alone.")
    expected_selection="patches:"+",".join(native.patches) if native.patches else "volume:all"
    if spec.selection!=expected_selection:raise ValueError("QoI selection does not match the native field selection.")
    for time in native.time_names:
        field=read_field(workspace,time,spec.region,native.field,mesh,evidence,expected_dimensions=dimensions)
        op=native.reduction
        if native.patches:
            if any(mesh.patches.get(p,{}).get("type")=="empty" for p in native.patches):raise ValueError("Empty patches cannot be measured surfaces.")
            values=[];weights=[]
            for name in native.patches:
                values+=patch_values(field,mesh,name);p=mesh.patches[name]
                weights+=mesh.areas[p["start"]:p["start"]+p["count"]]
            if not values:raise ValueError("Empty physical surface selection.")
            if op=="patch_sum":
                if field["class"]!="surfaceScalarField":raise ValueError("Oriented patch sum requires a native face-integrated surface scalar field.")
                value=sum(values)
            else:
                value=sum(v*w for v,w in zip(values,weights))
                if op=="area_mean":value/=sum(weights)
        else:
            if field["class"]!="volScalarField":raise ValueError("Volume reduction requires a volume scalar field.")
            values=field["internal"]
            if op=="cell_minimum":value=min(values)
            elif op=="cell_maximum":value=max(values)
            else:
                value=sum(v*w for v,w in zip(values,mesh.volumes))
                if op=="volume_mean":value/=sum(mesh.volumes)
        if not math.isfinite(value):raise ValueError("Physical reduction overflow.")
        series.append((float(time),value))
    check_sources_current(workspace,evidence)
    return series,evidence,{"field":native.field,"dimensions":list(dimensions),"output_dimensions":outdims,
        "region":spec.region,"selection":expected_selection,"spatial_operation":native.reduction,
        "semantic_scope":"Saved scalar field class, dimensions, literal region/patch, geometry and reduction verified; field name/quantity interpretation remains the approved model's declaration.",
        "solver_equation_correctness_verified":False}
