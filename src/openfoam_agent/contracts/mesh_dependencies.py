"""Persistent, versioned dependency DAG for mesh evidence.

Each native operation is a node; parents are earlier producers. This handles
in-place transforms without creating graph cycles. Uninstrumented native mesh
writers conservatively depend on ALL case inputs, not on a model's claimed read
set. Declared source/derived-asset edges can refine bootstrap dependencies; there
is no claim that arbitrary binary file accesses are inferred from tool names.
"""
from __future__ import annotations
import hashlib
import json
import os
import re
from pathlib import PurePosixPath
from openfoam_agent.contracts.execution_scopes import (
    execution_scopes,
    render_scope_template,
    scope_for_native_arguments,
    scope_paths,
)
from openfoam_agent.tools.native_contracts import native_command_region, native_tool_contract


def _sha(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",", ":")).encode()).hexdigest()


def _overlap(a,b):
    return a==b or a.startswith(b.rstrip("/")+"/") or b.startswith(a.rstrip("/")+"/")


class MeshDependencyGraph:
    def __init__(self,workspace):
        self.workspace=workspace
        self.path=workspace.root/"mesh-dependencies.json"
        self.operations=[]
        if self.path.exists():
            if self.path.is_symlink() or self.path.stat().st_size>16_000_000:
                raise ValueError("Unsafe mesh dependency registry.")
            envelope=json.loads(self.path.read_text())
            if envelope.get("version")!=1 or envelope.get("sha256")!=_sha(envelope.get("operations")):
                raise ValueError("Mesh dependency registry checksum/version mismatch.")
            self.operations=envelope["operations"]
            for index,node in enumerate(self.operations):
                if node["id"]!=index or any(parent>=index or parent<0 for parent in node["parents"]):
                    raise ValueError("Mesh dependency graph is not a chronological DAG.")
                for path in node["inputs"]+node["outputs"]: self._path(path)

    def _path(self,path):
        parts=PurePosixPath(path).parts
        if not parts or path.startswith("/") or ".." in parts or parts[0] not in {"0","constant","system"}:
            raise ValueError("Mesh dependency paths must be case-relative inputs.")
        return path.rstrip("/")

    def register(self,*,inputs,outputs,regions,command,conservative=False):
        inputs=sorted({self._path(p) for p in inputs})
        outputs=sorted({self._path(p) for p in outputs})
        if not outputs: raise ValueError("A mesh dependency operation must have output paths.")
        if len(self.operations)>=10000: raise ValueError("Mesh dependency history limit reached; explicit compaction required.")
        parents=[]
        for node in self.operations:
            if any(_overlap(p,q) for p in inputs for q in node["outputs"]): parents.append(node["id"])
        self.operations.append({"id":len(self.operations),"inputs":inputs,"outputs":outputs,
            "parents":parents,"regions":sorted(set(regions)),"command":command,"conservative":bool(conservative)})
        data={"version":1,"operations":self.operations,"sha256":_sha(self.operations)}
        temporary=self.path.with_suffix(".tmp")
        if temporary.is_symlink(): raise ValueError("Unsafe mesh graph temporary path.")
        with temporary.open("w") as stream:
            json.dump(data,stream,sort_keys=True);stream.flush();os.fsync(stream.fileno())
        os.replace(temporary,self.path)

    def record_native(self,command,arguments,plan=None):
        scopes = execution_scopes(plan) if plan is not None else []
        region = native_command_region(arguments)
        if region is not None:
            targets = [scope for scope in scopes if scope.name == region]
            if len(targets) != 1:
                raise ValueError("Native mesh operation targets an undeclared execution scope.")
        elif len(scopes) == 1:
            targets = scopes
        elif scopes:
            # An unscoped mesh writer may affect every declared scope. This is an
            # execution-side conservative effect, not topology inference.
            targets = scopes
        else:
            from openfoam_agent.schemas.engineering import ExecutionScope
            targets = [ExecutionScope(name=None)]

        contract = native_tool_contract(command)
        inputs: list[str] = []
        outputs: list[str] = []
        for scope in targets:
            for template in contract.mesh_dependency_inputs:
                rendered = render_scope_template(template, scope)
                if rendered not in inputs:
                    inputs.append(rendered)
            for template in contract.mesh_dependency_outputs:
                rendered = render_scope_template(template, scope)
                if rendered not in outputs:
                    outputs.append(rendered)
        conservative = not bool(contract.mesh_dependency_inputs)
        if conservative:
            inputs = ["0", "system", "constant"]
        if not outputs:
            outputs = [scope_paths(scope).constant_dir + "/polyMesh" for scope in targets]
        self.register(
            inputs=inputs,
            outputs=outputs,
            regions=[scope.name or "" for scope in targets],
            command={"name":command,"arguments":list(arguments)},
            conservative=conservative,
        )

    def _bootstrap(self,region):
        # The native operation DAG owns mesh dependencies. For imported/pre-existing
        # meshes, the sink polyMesh itself is sufficient freshness evidence. Do not
        # make material/thermo/numerical files accidental mesh dependencies.
        return []

    def dependencies(self,region):
        sink=f"constant/{region}/polyMesh" if region else "constant/polyMesh"
        selected={node["id"] for node in self.operations if region in node["regions"] or any(_overlap(sink,p) for p in node["outputs"])}
        todo=list(selected)
        while todo:
            for parent in self.operations[todo.pop()]["parents"]:
                if parent not in selected: selected.add(parent);todo.append(parent)
        paths=set(self._bootstrap(region))|{sink}
        for index in selected:
            paths.update(self.operations[index]["inputs"])
            paths.update(self.operations[index]["outputs"])
        # Literal include dependencies are traversed as files, not executed.
        # Missing files are retained as sentinels in the digest.
        sealed_paths=[item.path for item in self.workspace.execution_file_seals()]
        todo=list(paths)+[p for p in sealed_paths if any(p.startswith(root+"/") for root in paths)];seen=set()
        while todo:
            relative=todo.pop()
            if relative in seen: continue
            seen.add(relative)
            p=self.workspace.resolve_case_path(relative)
            if not p.is_file() or p.stat().st_size>2_000_000: continue
            try: text=p.read_text(encoding="utf-8")
            except UnicodeDecodeError: continue
            from openfoam_agent.verification.foam_semantics.parser import strip_comments
            text=strip_comments(text)
            for match in re.finditer(r'#include(?:IfPresent)?\s+"([^"\n]+)"',text):
                raw=match.group(1)
                target=(p.parent/raw).resolve()
                if self.workspace.case_dir not in target.parents:
                    raise ValueError("Mesh include dependency escapes the case.")
                child=target.relative_to(self.workspace.case_dir).as_posix()
                self._path(child);paths.add(child);todo.append(child)
        return paths,[self.operations[index] for index in sorted(selected)]

    def digest(self,region):
        paths,nodes=self.dependencies(region)
        seals={item.path:item.sha256 for item in self.workspace.execution_file_seals()}
        records=[]
        for path in sorted(paths):
            matches=sorted((name,value) for name,value in seals.items() if name==path or name.startswith(path+"/"))
            records.append((path,matches if matches else "MISSING"))
        return _sha({"region":region,"nodes":nodes,"artifacts":records})
