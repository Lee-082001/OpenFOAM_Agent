"""Deterministic authoring partitioner. Full contracts remain controller-owned.

Every required file is assigned exactly once. All facts affecting a task and their
transitive fact dependencies travel with it. A fact with no file mapping is global
and travels with every task. Oversized indivisible units fail before a model call.
No file is committed and no native tool is run until all task responses assemble.
"""
from __future__ import annotations
from copy import deepcopy
import hashlib
import json
from openfoam_agent.llm.context import ContextBudgetError, build_bounded_json_prompt


def sha(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False,separators=(",", ":")).encode()).hexdigest()


def _fact_closure(facts, ids):
    by_id = {item["id"]: item for item in facts}
    selected = set(ids)
    todo = list(ids)
    while todo:
        item = by_id.get(todo.pop())
        if item is None:
            raise ValueError("Authoring binding references a missing confirmed fact.")
        for dependency in item.get("depends_on", []):
            if dependency not in selected:
                selected.add(dependency); todo.append(dependency)
    return [item for item in facts if item["id"] in selected]


def _binding_paths(binding):
    paths = set(binding.get("case_files", []))
    paths.update(item["path"] for item in binding.get("case_assertions", []))
    relation = binding.get("numeric_relation") or {}
    for item in relation.get("numerator", []) + relation.get("denominator", []):
        paths.add(item["path"])
    return paths


def project(payload, paths):
    full = payload["frozen_engineering_plan"]
    intake = deepcopy(payload["confirmed_intake"])
    if not isinstance(intake, dict) or not isinstance(intake.get("facts"), list):
        raise ValueError("Partitioning requires a structured frozen intake with facts.")
    all_facts = intake["facts"]
    bindings = full.get("confirmed_fact_bindings", [])
    mapped = {item["fact_id"] for item in bindings if _binding_paths(item)}
    relevant = [item for item in bindings if not _binding_paths(item) or _binding_paths(item).intersection(paths)]
    ids = {item["fact_id"] for item in relevant} | {item["id"] for item in all_facts if item["id"] not in mapped}
    closure = _fact_closure(all_facts + intake.get("provenance_dependencies", []), ids)
    primary_ids = {item["id"] for item in all_facts}
    intake["facts"] = [item for item in closure if item["id"] in primary_ids]
    if "provenance_dependencies" in intake:
        intake["provenance_dependencies"] = [item for item in closure if item["id"] not in primary_ids]
    selected_ids = {item["id"] for item in intake["facts"]}
    plan = deepcopy(full)
    plan["required_case_files"] = list(paths)
    plan["confirmed_fact_ids"] = [item for item in full.get("confirmed_fact_ids", []) if item in selected_ids]
    plan["confirmed_fact_bindings"] = [item for item in bindings if item["fact_id"] in selected_ids]
    plan["implementation_evidence_bindings"] = [item for item in full.get("implementation_evidence_bindings", []) if item["path"] in paths]
    evidence = payload["implementation_evidence_pack"]
    coverage = [item for item in evidence["file_coverage"] if item["path"] in paths]
    evidence_ids = {eid for item in coverage for eid in item["evidence_ids"]}
    evidence = {**evidence, "file_coverage":coverage, "records":[item for item in evidence["records"] if item["evidence_id"] in evidence_ids]}
    result = {key:deepcopy(value) for key,value in payload.items() if key not in {"frozen_engineering_plan", "confirmed_intake", "implementation_evidence_pack", "current_case_files"}}
    # Cross-region interfaces, execution topology and other global plan constraints
    # are NOT discarded, even if retaining them makes a single task too large.
    result.update(frozen_engineering_plan=plan,confirmed_intake=intake,implementation_evidence_pack=evidence,
        current_case_files=[item for item in payload.get("current_case_files",[]) if isinstance(item,dict) and item.get("path") in paths],
        authoring_contract={"full_plan_sha256":sha(full), "full_intake_sha256":sha(payload["confirmed_intake"]),
            "required_files_sha256":sha(full["required_case_files"]), "total_required_files":len(full["required_case_files"]),
            "projection_only":True, "instruction":"This is a file-scoped projection, not a replacement plan. Do not modify global contracts."})
    return result


def compile_tasks(instruction, payload, max_chars):
    if max_chars < 5200:
        raise ContextBudgetError("Authoring budget is too small for an indivisible contract.")
    paths = payload["frozen_engineering_plan"]["required_case_files"]
    if len(paths) != len(set(paths)) or not paths:
        raise ValueError("Authoring file manifest is empty or duplicated.")
    batches=[]; current=[]
    # Reserve protocol metadata bytes before choosing the greedy file grouping.
    def fits(items):
        try:
            build_bounded_json_prompt(instruction,project(payload,items),max_chars=max_chars-1200)
            return True
        except ContextBudgetError:
            return False
    for path in paths:
        if fits(current+[path]):
            current.append(path)
        else:
            if current: batches.append(current)
            current=[path]
            if not fits(current):
                raise ContextBudgetError(f"Indivisible file task {path} exceeds context budget; required facts/syntax were not truncated.")
    if current: batches.append(current)
    tasks=[]
    for index,items in enumerate(batches):
        task=project(payload,items)
        meta={"id":f"{sha(payload['frozen_engineering_plan'])[:16]}:{index+1}","index":index+1,"count":len(batches),
            "paths":items,"is_final":index==len(batches)-1,
            "response_contract":"Echo task_id. Return only these files. Intermediate tasks set defer_native=true with no commands. Final task supplies the complete native pipeline."}
        task["authoring_task"]=meta
        build_bounded_json_prompt(instruction,task,max_chars=max_chars)
        tasks.append(task)
    assigned=[p for task in tasks for p in task["authoring_task"]["paths"]]
    projected={f["id"] for task in tasks for f in task["confirmed_intake"]["facts"]}
    if assigned != paths or projected != {f["id"] for f in payload["confirmed_intake"]["facts"]}:
        raise ValueError("Authoring partition lost a required file or confirmed fact.")
    return {"version":1,"plan_sha256":sha(payload["frozen_engineering_plan"]),"tasks":tasks,"cursor":0,"responses":[],
        "assigned_files":assigned,"coverage_verified":True}


def accept_task(queue, action, plan):
    if queue["plan_sha256"] != sha(plan.model_dump(mode="json")):
        raise ValueError("Frozen plan changed while authoring tasks were pending.")
    task=queue["tasks"][queue["cursor"]]["authoring_task"]
    paths=[item.path for item in action.files]+[item.path for item in action.typed_dictionaries]
    if action.block_mesh is not None: paths.append(action.block_mesh.path)
    if action.task_id != task["id"] or set(paths)!=set(task["paths"]) or len(paths)!=len(set(paths)):
        raise ValueError("Authoring task ID/file coverage mismatch; no files were committed.")
    if set(action.required_case_files)!=set(task["paths"]):
        raise ValueError("Task required files differ from the controller assignment.")
    if action.defer_native == task["is_final"]:
        raise ValueError("Only the final task may supply a native pipeline.")
    queue["responses"].append(action.model_dump(mode="python")); queue["cursor"]+=1
    if queue["cursor"] < len(queue["tasks"]): return None
    final=deepcopy(queue["responses"][-1])
    for key in ("files","typed_dictionaries","validate_dictionaries","surface_checks"):
        final[key]=[item for response in queue["responses"] for item in response[key]]
    for key in ("validate_dictionaries","surface_checks"): final[key]=list(dict.fromkeys(final[key]))
    meshes=[response["block_mesh"] for response in queue["responses"] if response.get("block_mesh") is not None]
    if len(meshes)>1: raise ValueError("Duplicate structured mesh in assembled authoring tasks.")
    final.update(type="execute_case_plan",task_id=None,defer_native=False,block_mesh=meshes[0] if meshes else None,
        required_case_files=list(plan.required_case_files),plan=plan.model_dump(mode="python"))
    from openfoam_agent.schemas.engineering import ExecuteCasePlanAction
    return ExecuteCasePlanAction.model_validate(final)
