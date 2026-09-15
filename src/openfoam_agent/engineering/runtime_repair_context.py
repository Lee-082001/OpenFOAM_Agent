from __future__ import annotations

from copy import deepcopy
import re

from openfoam_agent.llm.context import ContextBudgetError, PromptBuildResult, build_bounded_json_prompt, compact_text


def _paths_from_files(items: list[dict[str, object]]) -> list[str]:
    paths: list[str] = []
    for item in items:
        path = str(item.get("path") or "").strip()
        if path and path not in paths:
            paths.append(path)
    return paths


def _scope_names(execution: object, diagnostic: str, paths: list[str]) -> list[str]:
    if not isinstance(execution, dict):
        return []
    scopes = [item for item in list(execution.get("scopes") or []) if isinstance(item, dict)]
    named = [str(item.get("name")) for item in scopes if item.get("name") not in (None, "")]
    selected: list[str] = []
    text = diagnostic or ""
    for name in named:
        if re.search(rf"(?<![A-Za-z0-9_.-]){re.escape(name)}(?![A-Za-z0-9_.-])", text):
            selected.append(name)
            continue
        if any(f"/{name}/" in f"/{path}/" for path in paths):
            selected.append(name)
    # Path-poor diagnostics (for example <LOCAL_PATH:solvers>) still need the
    # execution topology. Keep all scopes, but only the compact scope records.
    if not selected:
        selected = named[:12]
    return selected


def _record_mentions_scope(record: object, selected: list[str]) -> bool:
    if not selected:
        return True
    if not isinstance(record, dict):
        return False
    values = {str(record.get(key) or "") for key in ("region", "neighbour_region", "name")}
    return bool(values.intersection(selected))


def _project_execution(execution: object, selected: list[str]) -> object:
    if not isinstance(execution, dict):
        return execution
    projected = {
        "driver": execution.get("driver"),
        "driver_provider_id": execution.get("driver_provider_id"),
        "arguments": deepcopy(execution.get("arguments") or []),
        "parallel": deepcopy(execution.get("parallel")),
    }
    scopes = [item for item in list(execution.get("scopes") or []) if isinstance(item, dict)]
    if selected:
        filtered = [item for item in scopes if str(item.get("name") or "") in selected]
        projected["scopes"] = deepcopy(filtered or scopes[:12])
    else:
        projected["scopes"] = deepcopy(scopes[:12])
    return projected


def _project_required_files(required: object, implicated: list[str]) -> list[str]:
    values = [str(path) for path in list(required or []) if str(path).strip()]
    keep: list[str] = []
    core = {"system/controlDict", "system/fvSchemes", "system/fvSolution"}
    for path in values:
        if path in implicated or path in core:
            keep.append(path)
    if not keep:
        keep = values[:24]
    return list(dict.fromkeys(keep))[:40]


def _project_bindings(bindings: object, implicated: list[str], required: list[str]) -> list[dict[str, object]]:
    allowed_paths = set(implicated) | set(required)
    out: list[dict[str, object]] = []
    for item in list(bindings or []):
        if not isinstance(item, dict):
            continue
        case_files = [str(path) for path in list(item.get("case_files") or []) if str(path) in allowed_paths]
        plan_fields = [str(value) for value in list(item.get("plan_fields") or [])[:8]]
        # Runtime repair is file-local unless the native failure explicitly points
        # at a plan change. Solver/execution identity is already present in the plan
        # projection, so unrelated provenance bindings need not be repeated.
        if not case_files:
            continue
        out.append({
            "fact_id": item.get("fact_id"),
            "case_files": case_files[:8],
            "plan_fields": plan_fields,
        })
        if len(out) >= 24:
            break
    return out


def project_runtime_repair_plan(approved_plan: object, *, diagnostic: str, implicated_files: list[str]) -> tuple[object, list[str]]:
    """Project the Python-held approved plan to the failure-local model view.

    The controller keeps the complete EngineeringPlan as authority.  This projection
    carries only execution identity, implicated region topology and files needed to
    reason about the observed runtime failure.
    """
    if not isinstance(approved_plan, dict):
        return approved_plan, []
    execution = approved_plan.get("execution")
    selected = _scope_names(execution, diagnostic, implicated_files)
    required = _project_required_files(approved_plan.get("required_case_files"), implicated_files)
    layouts = [item for item in list(approved_plan.get("region_layouts") or []) if _record_mentions_scope(item, selected)]
    interfaces = [item for item in list(approved_plan.get("interfaces") or []) if _record_mentions_scope(item, selected)]
    projected = {
        "solver": approved_plan.get("solver"),
        "solver_provider_id": approved_plan.get("solver_provider_id"),
        "execution": _project_execution(execution, selected),
        "region_layouts": deepcopy(layouts[:16]),
        "interfaces": deepcopy(interfaces[:16]),
        "temporal_behavior": approved_plan.get("temporal_behavior"),
        "motion_kind": approved_plan.get("motion_kind"),
        "mesh_motion_requirement": approved_plan.get("mesh_motion_requirement"),
        "required_case_files": required,
        "confirmed_fact_bindings": _project_bindings(
            approved_plan.get("confirmed_fact_bindings"), implicated_files, required
        ),
    }
    return projected, selected


def _bounded_files(items: list[dict[str, object]], *, max_files: int, total_content_chars: int) -> list[dict[str, object]]:
    if max_files <= 0 or total_content_chars <= 0:
        return []
    selected = items[:max_files]
    remaining = total_content_chars
    out: list[dict[str, object]] = []
    for item in selected:
        content = str(item.get("content") or "")
        # Reserve useful content for every remaining file rather than allowing the
        # first dictionary to consume the entire partition budget.
        slots = max(1, len(selected) - len(out))
        allowance = max(256, remaining // slots)
        bounded = compact_text(content, max(256, min(allowance, 5000))) if content else ""
        remaining = max(0, remaining - len(bounded))
        out.append({
            "path": item.get("path"),
            "sha256": item.get("sha256"),
            "content": bounded,
            "truncated": bool(item.get("truncated")) or len(bounded) < len(content),
        })
    return out


def _compact_contract_scan(scan: object) -> object:
    if not isinstance(scan, dict):
        return scan
    return {
        "checked_count": scan.get("checked_count"),
        "invalid_count": scan.get("invalid_count"),
        "invalid": deepcopy(list(scan.get("invalid") or [])[:12]),
    }


def _compact_mesh_bindings(bindings: object, selected: list[str]) -> object:
    if not isinstance(bindings, dict):
        return bindings
    projected = {
        "intake_sha256": bindings.get("intake_sha256"),
        "plan_sha256": bindings.get("plan_sha256"),
        "manifest_sha256": bindings.get("manifest_sha256"),
        "check_mesh_passed": bindings.get("check_mesh_passed"),
    }
    evidence = bindings.get("mesh_scope_evidence")
    if isinstance(evidence, dict):
        if selected:
            projected["mesh_scope_evidence"] = {
                key: deepcopy(value)
                for key, value in evidence.items()
                if key in selected or key.removeprefix("region:") in selected
            }
        else:
            projected["mesh_scope_evidence"] = dict(list(evidence.items())[:12])
    return projected


def project_runtime_repair_capsule(
    payload: dict[str, object],
    *,
    file_limit: int,
    total_file_content_chars: int,
    evidence_limit: int,
    failure_chars: int,
) -> dict[str, object]:
    files = list(payload.get("relevant_case_files") or [])
    implicated = _paths_from_files(files)
    diagnostic = str(payload.get("native_failure") or "")
    plan, selected = project_runtime_repair_plan(
        payload.get("approved_plan"), diagnostic=diagnostic, implicated_files=implicated
    )
    evidence = list(payload.get("available_evidence") or [])
    confirmed_facts: list[dict[str, object]] = []
    for item in list(payload.get("confirmed_facts") or [])[:32]:
        if not isinstance(item, dict):
            continue
        confirmed_facts.append({
            "id": item.get("id"),
            "value": compact_text(str(item.get("value") or ""), 360),
            "source": item.get("source"),
        })
    capsule = {
        "state_mode": "runtime_failure_partition_v1",
        "phase": payload.get("phase"),
        "step": payload.get("step"),
        "confirmed_facts": confirmed_facts,
        "approved_plan_sha256": payload.get("approved_plan_sha256"),
        "approved_plan": plan,
        "native_failure": compact_text(diagnostic, max(512, failure_chars)) if diagnostic else None,
        "case_file_contract_scan": _compact_contract_scan(payload.get("case_file_contract_scan")),
        "relevant_case_files": _bounded_files(
            files, max_files=file_limit, total_content_chars=total_file_content_chars
        ),
        "mesh_evidence": deepcopy(payload.get("mesh_evidence")),
        "available_evidence": deepcopy(evidence[-evidence_limit:] if evidence_limit else []),
        "evidence_gap_status": deepcopy(list(payload.get("evidence_gap_status") or [])[-3:]),
        "bindings": _compact_mesh_bindings(payload.get("bindings"), selected),
        "engineering_assumption_policy": deepcopy(payload.get("engineering_assumption_policy")),
        "evidence_retrieval_policy": deepcopy(payload.get("evidence_retrieval_policy")),
        "budget": deepcopy(payload.get("budget")),
        "context_partition": {
            "active": True,
            "authority": "full approved EngineeringPlan remains Python-held; this is a failure-local projection only",
            "selected_regions": selected,
            "implicated_files": implicated,
        },
    }
    return capsule


def build_partitioned_runtime_repair_prompt(
    instruction: str,
    payload: dict[str, object],
    *,
    max_chars: int,
) -> tuple[PromptBuildResult, dict[str, object], dict[str, int]]:
    """Fit runtime repair under a hard budget by region/file projection.

    No plan authority is transferred to the projection.  If even the smallest
    failure-local capsule cannot fit, the caller keeps the runtime failure intact and
    raises ContextBudgetError rather than silently truncating required diagnostic data.
    """
    attempts = [
        (6, 9000, 8, 6000),
        (4, 6500, 4, 4500),
        (2, 4500, 2, 3200),
        (1, 2600, 0, 2200),
    ]
    last_error: ContextBudgetError | None = None
    for index, (file_limit, content_chars, evidence_limit, failure_chars) in enumerate(attempts, start=1):
        capsule = project_runtime_repair_capsule(
            payload,
            file_limit=file_limit,
            total_file_content_chars=content_chars,
            evidence_limit=evidence_limit,
            failure_chars=failure_chars,
        )
        try:
            built = build_bounded_json_prompt(instruction, capsule, max_chars=max_chars)
            files = list(capsule.get("relevant_case_files") or [])
            partition = capsule.get("context_partition") or {}
            return built, capsule, {
                "runtimeRepairPartitioned": 1,
                "runtimeRepairPartitionAttempts": index,
                "runtimeRepairFocusedFiles": len(files),
                "runtimeRepairFileContentChars": sum(
                    len(str(item.get("content") or "")) for item in files if isinstance(item, dict)
                ),
                "runtimeRepairRegions": len(list(partition.get("selected_regions") or [])) if isinstance(partition, dict) else 0,
                "runtimeRepairEvidenceVisible": len(list(capsule.get("available_evidence") or [])),
            }
        except ContextBudgetError as exc:
            last_error = exc
    raise ContextBudgetError(
        "Runtime-repair failure-local projection still exceeds the deterministic model-context budget. "
        "The full approved plan remains controller-held and the runtime failure/case are unchanged; no repair mutation was authorized."
    ) from last_error
