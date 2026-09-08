"""Deterministic authoring partitioner. Full contracts remain controller-owned.

Every required file is assigned exactly once. All facts affecting a task and their
transitive fact dependencies travel with it. A fact with no file mapping is global
and travels with every task. Oversized bundles are projected into compact file-scoped capsules before a model call.
Only a genuinely oversized mandatory file capsule fails. No file is committed and no native tool is run until all task responses assemble.
"""
from __future__ import annotations
from copy import deepcopy
import hashlib
import json
from openfoam_agent.llm.context import ContextBudgetError, build_bounded_json_prompt, compact_text


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




def _compact_binding(binding, paths):
    item = deepcopy(binding)
    item["case_files"] = [p for p in item.get("case_files", []) if p in paths]
    item["case_assertions"] = [a for a in item.get("case_assertions", []) if a.get("path") in paths]
    relation = item.get("numeric_relation")
    if isinstance(relation, dict):
        relation = deepcopy(relation)
        relation["numerator"] = [x for x in relation.get("numerator", []) if x.get("path") in paths]
        relation["denominator"] = [x for x in relation.get("denominator", []) if x.get("path") in paths]
        if relation["numerator"] or relation["denominator"]:
            item["numeric_relation"] = relation
        else:
            item["numeric_relation"] = None
    item["explanation"] = compact_text(str(item.get("explanation", "")), 160) if item.get("explanation") else ""
    return item


def _compact_plan_projection(full, paths, selected_ids, relevant_bindings):
    """Return an authoring-only plan capsule; the full plan stays controller-owned.

    The model does not return an EngineeringPlan during ``author_case``. Sending the
    entire frozen plan to every file task therefore wastes context and can make one
    ordinary file (for example ``system/controlDict``) indivisible under a small prompt
    budget. This capsule preserves execution identity, region/interface topology, all
    selected confirmed-fact bindings and concrete engineering defaults while summarizing
    advisory narrative/audit fields. ``authoring_contract.full_plan_sha256`` binds every
    task back to the immutable full plan used by Python when the responses are assembled.
    """
    essential = {
        "schema_version", "case_name", "solver", "solver_provider_id", "execution",
        "region_layouts", "interfaces", "completion", "openfoam_distribution",
        "openfoam_version", "temporal_behavior", "motion_kind",
        "mesh_motion_requirement", "confirmed_intake_sha256",
    }
    bound_plan_fields = {
        field
        for binding in relevant_bindings
        if binding.get("fact_id") in selected_ids
        for field in binding.get("plan_fields", [])
    }
    # These fields are useful for almost every case file but are allowed to be
    # compact projections because the full controller-held plan remains authoritative.
    useful = {"problem_interpretation", "mesh_strategy", "engineering_defaults"}
    keep = essential | bound_plan_fields | useful
    plan = {key: deepcopy(value) for key, value in full.items() if key in keep}
    if "problem_interpretation" in plan:
        plan["problem_interpretation"] = compact_text(str(plan["problem_interpretation"]), 1200)
    if "mesh_strategy" in plan:
        plan["mesh_strategy"] = compact_text(str(plan["mesh_strategy"]), 700)

    defaults = []
    for item in full.get("engineering_defaults", []):
        if not isinstance(item, dict):
            continue
        defaults.append({
            "parameter": item.get("parameter"),
            "value": item.get("value"),
            "unit": item.get("unit", ""),
            "basis": item.get("basis", "representative"),
        })
    plan["engineering_defaults"] = defaults

    # Decisions/assumptions are advisory authoring context. Keep the engineering
    # choices, not repeated long rationales/evidence notes.
    plan["decisions"] = [
        {
            "area": item.get("area"),
            "choice": compact_text(str(item.get("choice", "")), 320),
            "risk_level": item.get("risk_level", "medium"),
            "verification_stage": item.get("verification_stage", "pre_validation"),
        }
        for item in full.get("decisions", [])
        if isinstance(item, dict)
    ]
    plan["assumptions"] = [compact_text(str(item), 240) for item in full.get("assumptions", [])]
    plan["postprocess_strategy"] = [compact_text(str(item), 200) for item in full.get("postprocess_strategy", [])]

    # Runtime-result intent is relevant mainly to controlDict/function-object authoring;
    # otherwise it is deferred to post-processing and need not consume every file task.
    if "system/controlDict" in paths:
        plan["quantities_of_interest"] = deepcopy(full.get("quantities_of_interest", []))
        plan["conservation_checks"] = deepcopy(full.get("conservation_checks", []))
    else:
        plan["quantities_of_interest"] = []
        plan["conservation_checks"] = []

    plan["required_case_files"] = list(paths)
    plan["confirmed_fact_ids"] = [x for x in full.get("confirmed_fact_ids", []) if x in selected_ids]
    plan["confirmed_fact_bindings"] = [
        _compact_binding(item, set(paths))
        for item in relevant_bindings
        if item.get("fact_id") in selected_ids
    ]
    plan["implementation_evidence_bindings"] = [
        deepcopy(item) for item in full.get("implementation_evidence_bindings", [])
        if item.get("path") in paths
    ]
    # Evidence pointers are audit metadata; documentary bodies are advisory in v4.2+.
    plan["evidence"] = [
        {"evidence_id": item.get("evidence_id")}
        for item in full.get("evidence", [])[:16]
        if isinstance(item, dict) and item.get("evidence_id")
    ]
    plan["projection_notice"] = (
        "Authoring-only projection. Omitted narrative/result-analysis fields remain immutable "
        "in the controller-held full plan identified by authoring_contract.full_plan_sha256."
    )
    return plan


def _compact_evidence_pack(evidence, paths):
    coverage = [deepcopy(item) for item in evidence.get("file_coverage", []) if item.get("path") in paths]
    evidence_ids = {eid for item in coverage for eid in item.get("evidence_ids", [])}
    records = []
    for item in evidence.get("records", []):
        if item.get("evidence_id") not in evidence_ids:
            continue
        records.append({
            "evidence_id": item.get("evidence_id"),
            "source": item.get("source"),
            "record_id": item.get("record_id"),
            "openfoam_version": item.get("openfoam_version"),
            "excerpt_sha256": item.get("excerpt_sha256"),
            "content_excerpt": compact_text(str(item.get("content", "")), 900) if item.get("content") else "",
        })
    return {
        "records": records,
        "file_coverage": coverage,
        "complete": bool(evidence.get("complete")),
        "scope": "Advisory file-scoped projection; native/deterministic validation authorizes progress.",
    }

def project(payload, paths, *, compact=False):
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
    if compact:
        plan = _compact_plan_projection(full, set(paths), selected_ids, relevant)
        evidence = _compact_evidence_pack(payload["implementation_evidence_pack"], set(paths))
    else:
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
    # The full controller-held plan remains immutable; compact tasks preserve the
    # authoring-relevant execution/region/interface projection plus its full-plan hash.
    result.update(frozen_engineering_plan=plan,confirmed_intake=intake,implementation_evidence_pack=evidence,
        current_case_files=[item for item in payload.get("current_case_files",[]) if isinstance(item,dict) and item.get("path") in paths],
        authoring_contract={"full_plan_sha256":sha(full), "full_intake_sha256":sha(payload["confirmed_intake"]),
            "required_files_sha256":sha(full["required_case_files"]), "total_required_files":len(full["required_case_files"]),
            "projection_only":True, "compact_projection":bool(compact), "instruction":"This is a file-scoped projection, not a replacement plan. Do not modify global contracts."})
    return result


def compile_tasks(instruction, payload, max_chars):
    if max_chars < 5200:
        raise ContextBudgetError("Authoring budget is too small for an indivisible contract.")
    paths = payload["frozen_engineering_plan"]["required_case_files"]
    if len(paths) != len(set(paths)) or not paths:
        raise ValueError("Authoring file manifest is empty or duplicated.")
    batches=[]; current=[]
    # Protocol metadata is small; reserve less than the legacy 1200-char margin and
    # let the final exact build below enforce the real cap.
    grouping_limit=max_chars-600
    def projection_that_fits(items):
        for compact in (False, True):
            candidate=project(payload,items,compact=compact)
            try:
                build_bounded_json_prompt(instruction,candidate,max_chars=grouping_limit)
                return candidate,compact
            except ContextBudgetError:
                pass
        return None,None
    projection_modes=[]
    for path in paths:
        candidate,compact=projection_that_fits(current+[path])
        if candidate is not None:
            current.append(path)
            continue
        if current:
            final_candidate,final_compact=projection_that_fits(current)
            if final_candidate is None:
                raise ContextBudgetError("Authoring task projection unexpectedly exceeded its previously verified budget.")
            batches.append(current);projection_modes.append(final_compact)
        current=[path]
        single,single_compact=projection_that_fits(current)
        if single is None:
            raise ContextBudgetError(
                f"Minimal file-scoped authoring capsule for {path} exceeds context budget. "
                "Increase the authoring context cap; no confirmed requirement was truncated."
            )
    if current:
        final_candidate,final_compact=projection_that_fits(current)
        if final_candidate is None:
            raise ContextBudgetError("Final authoring task projection exceeded budget.")
        batches.append(current);projection_modes.append(final_compact)
    tasks=[]
    for index,(items,compact_mode) in enumerate(zip(batches,projection_modes)):
        task=project(payload,items,compact=bool(compact_mode))
        meta={"id":f"{sha(payload['frozen_engineering_plan'])[:16]}:{index+1}","index":index+1,"count":len(batches),
            "paths":items,"is_final":index==len(batches)-1,"compact_plan_projection":bool(compact_mode),
            "response_contract":"Echo task_id. Return only these files. Intermediate tasks set defer_native=true with no commands. Final task supplies the complete native pipeline."}
        task["authoring_task"]=meta
        # If protocol metadata tips the task over the grouping reserve, the exact cap is
        # still allowed. Rebuild with compact projection before giving up.
        try:
            build_bounded_json_prompt(instruction,task,max_chars=max_chars)
        except ContextBudgetError:
            if not compact_mode:
                task=project(payload,items,compact=True)
                task["authoring_task"]={**meta,"compact_plan_projection":True}
                build_bounded_json_prompt(instruction,task,max_chars=max_chars)
            else:
                raise
        tasks.append(task)
    assigned=[p for task in tasks for p in task["authoring_task"]["paths"]]
    projected={f["id"] for task in tasks for f in task["confirmed_intake"]["facts"]}
    if assigned != paths or projected != {f["id"] for f in payload["confirmed_intake"]["facts"]}:
        raise ValueError("Authoring partition lost a required file or confirmed fact.")
    return {"version":2,"plan_sha256":sha(payload["frozen_engineering_plan"]),"tasks":tasks,"cursor":0,"responses":[],
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
