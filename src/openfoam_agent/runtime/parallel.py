"""Structured local MPI preparation/reconstruction; no arbitrary launch strings."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
from openfoam_agent.contracts.regions import region_layouts
from openfoam_agent.verification.foam_semantics import parse_top_level_assignments


def _input_manifest(case, execution, layouts):
    records = []
    for rank in range(execution.parallel.ranks):
        root = case / f"processor{rank}"
        if root.is_symlink() or not root.is_dir():
            raise ValueError(f"Missing uncollated processor directory: {root.name}.")
        for layout in layouts:
            paths = [f"{layout.mesh_dir}/{name}" for name in ("points", "faces", "owner", "neighbour", "boundary")]
            paths += [f"{layout.field_dir}/{name}" for name in layout.required_fields]
            for rel in paths:
                p = root / rel
                if p.is_symlink() or not p.is_file() or root.resolve() not in p.resolve().parents:
                    raise ValueError(f"Missing/unsafe decomposed input: {root.name}/{rel}")
                with p.open("rb") as handle:
                    sha = hashlib.file_digest(handle, "sha256").hexdigest()
                records.append({"path": p.relative_to(case).as_posix(), "sha256": sha})
    digest = hashlib.sha256(json.dumps(records, sort_keys=True).encode()).hexdigest()
    return {"files": records, "sha256": digest, "ranks": execution.parallel.ranks,
            "validation_scope": "input file presence and hashes; not partition numerical correctness"}


def prepare_parallel(tools, workspace, plan):
    execution = plan.execution
    if execution is None or execution.parallel.mode == "serial":
        return None
    parallel = execution.parallel
    path = workspace.resolve_case_path("system/decomposeParDict", must_exist=True)
    entries, complete = parse_top_level_assignments(path.read_text(encoding="utf-8"))
    if not complete:
        raise ValueError("Dynamic decomposition settings cannot be matched to MPI approval.")
    if str(entries.get("numberOfSubdomains", "")).strip() != str(parallel.ranks):
        raise ValueError("Decomposition ranks differ from the approved MPI contract.")
    if str(entries.get("method", "")).strip() != parallel.decomposition_method:
        raise ValueError("Decomposition method differs from the approved MPI contract.")
    if parallel.restart is not None:
        from .restart import verify_restart
        receipt = verify_restart(workspace, plan)
        return {"sha256": parallel.restart.manifest_sha256, "restart": True, "receipt": receipt}
    if any(p.name.startswith("processor") for p in workspace.case_dir.iterdir()):
        raise ValueError("Existing processor outputs are preserved; parallel restart/re-decomposition needs explicit reconciliation.")
    layouts = region_layouts(plan)
    arguments = ["-allRegions"] if execution.regions else []
    result = tools.run_native_command("decomposePar", workspace.case_dir, arguments=arguments)
    if not result.success:
        raise ValueError(f"decomposePar failed: {result.termination_reason}; {result.stderr or result.stdout[-1500:]}")
    return _input_manifest(workspace.case_dir, execution, layouts)


def verify_parallel_inputs(workspace, plan, manifest):
    if manifest.get("restart"):
        from .restart import verify_restart
        verify_restart(workspace, plan)
        if manifest["sha256"] != plan.execution.parallel.restart.manifest_sha256:
            raise ValueError("Restart selection changed before launch.")
        return
    current = _input_manifest(workspace.case_dir, plan.execution, region_layouts(plan))
    if current["sha256"] != manifest["sha256"]:
        raise ValueError("Decomposed inputs changed before MPI launch.")


def reconstruct_parallel(tools, workspace, plan, last_time):
    execution = plan.execution
    if execution is None or execution.parallel.mode == "serial":
        return None
    if not execution.parallel.reconstruct:
        raise ValueError("Distributed result validation without reconstruction is not implemented; outputs remain unverified.")
    if last_time is None:
        raise ValueError("Cannot reconstruct an unobserved final time.")
    import math
    if not math.isfinite(last_time) or last_time <= 0:
        raise ValueError("Reconstruction requires a positive finite observed final time.")
    requested = {float(last_time)}
    for quantity in plan.quantities_of_interest:
        if quantity.native_field:
            requested.update(float(t) for t in quantity.native_field.time_names if float(t)>0)
    for check in plan.conservation_checks:
        requested.update(float(t) for t in check.time_names if float(t)>0)
    if any(t>last_time and not math.isclose(t,last_time,rel_tol=1e-8,abs_tol=1e-10) for t in requested):
        raise ValueError("Requested observation exceeds the observed final solver time.")
    # 0/ is sealed input, not a reconstruction target. Analyses requiring time 0
    # use the original sealed fields and refuse missing ones. Never overwrite it.
    for rank in range(execution.parallel.ranks):
        root = workspace.case_dir / f"processor{rank}"
        if root.is_symlink() or not root.is_dir():
            raise ValueError("Missing/unsafe rank directory during reconstruction.")
        available=[]
        for p in root.iterdir():
            try:t=float(p.name)
            except ValueError:continue
            if not math.isfinite(t) or p.is_symlink():
                raise ValueError("Unsafe numeric rank result during reconstruction.")
            if p.is_dir():available.append(t)
        for target in requested:
            if sum(math.isclose(t,target,rel_tol=1e-8,abs_tol=1e-10) for t in available)!=1:
                label = "Final-time" if target == float(last_time) else "Requested-time"
                raise ValueError(f"{label} output missing or ambiguous on MPI rank {rank}: {target}.")
    selection = ",".join(format(t,".15g") for t in sorted(requested))
    arguments = (["-allRegions"] if execution.regions else []) + ["-time", selection]
    result = tools.run_native_command("reconstructPar", workspace.case_dir, arguments=arguments)
    if not result.success:
        raise ValueError("reconstructPar failed; distributed results were preserved for review.")
    return result
