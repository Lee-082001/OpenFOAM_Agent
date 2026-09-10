from __future__ import annotations

from copy import deepcopy

from openfoam_agent.llm.context import ContextBudgetError, PromptBuildResult, compact_text, build_bounded_json_prompt


def project_validation_repair_plan(plan: object) -> dict[str, object] | None:
    """Project only the plan fields needed to repair an already-authored case.

    The complete EngineeringPlan remains controller state.  Validation repair must not
    re-send QoI/result-analysis/audit bodies merely to fix one dictionary or field.
    """
    if plan is None:
        return None
    raw = plan.model_dump(mode="json") if hasattr(plan, "model_dump") else deepcopy(plan)
    if not isinstance(raw, dict):
        return None

    defaults: list[dict[str, object]] = []
    for item in list(raw.get("engineering_defaults") or [])[:20]:
        if not isinstance(item, dict):
            continue
        defaults.append({
            "parameter": compact_text(str(item.get("parameter") or ""), 120),
            "value": compact_text(str(item.get("value") or ""), 220),
            "unit": compact_text(str(item.get("unit") or ""), 80),
            "basis": item.get("basis"),
        })

    bindings: list[dict[str, object]] = []
    for item in list(raw.get("confirmed_fact_bindings") or []):
        if not isinstance(item, dict):
            continue
        bindings.append({
            "fact_id": item.get("fact_id"),
            "plan_fields": list(item.get("plan_fields") or []),
            "case_files": list(item.get("case_files") or []),
        })

    return {
        "case_name": raw.get("case_name"),
        "solver": raw.get("solver"),
        "solver_provider_id": raw.get("solver_provider_id"),
        "execution": deepcopy(raw.get("execution")),
        "openfoam_distribution": raw.get("openfoam_distribution"),
        "openfoam_version": raw.get("openfoam_version"),
        "temporal_behavior": raw.get("temporal_behavior"),
        "motion_kind": raw.get("motion_kind"),
        "mesh_motion_requirement": raw.get("mesh_motion_requirement"),
        "problem_interpretation": compact_text(str(raw.get("problem_interpretation") or ""), 700),
        "mesh_strategy": compact_text(str(raw.get("mesh_strategy") or ""), 700),
        "region_layouts": deepcopy(raw.get("region_layouts") or []),
        "interfaces": deepcopy(raw.get("interfaces") or []),
        "engineering_defaults": defaults,
        "required_case_files": list(raw.get("required_case_files") or []),
        "confirmed_intake_sha256": raw.get("confirmed_intake_sha256"),
        "confirmed_fact_ids": list(raw.get("confirmed_fact_ids") or []),
        "confirmed_fact_bindings": bindings,
        "projection_note": (
            "Failure-local repair projection. Python retains the complete EngineeringPlan; "
            "unchanged plan/audit/result fields are intentionally omitted."
        ),
    }


def _bounded_files(
    files: list[dict[str, object]], *, max_files: int, total_content_chars: int
) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    remaining = max(0, total_content_chars)
    for item in files[:max_files]:
        if not isinstance(item, dict):
            continue
        content = str(item.get("content") or "")
        if remaining <= 0:
            projected_content = ""
        else:
            cap = min(6000, remaining)
            projected_content = compact_text(content, max(64, cap)) if len(content) > cap else content
        remaining -= len(projected_content)
        result.append({
            "path": item.get("path"),
            "sha256": item.get("sha256"),
            "content": projected_content,
            "truncated": bool(item.get("truncated")) or len(projected_content) < len(content),
        })
    return result


def build_partitioned_validation_repair_prompt(
    instruction: str,
    payload: dict[str, object],
    *,
    max_chars: int,
) -> tuple[PromptBuildResult, dict[str, object], dict[str, int]]:
    """Fit a committed-case validation repair without dropping the failure contract.

    The failure diagnostic and compact baseline plan remain present in every attempt.
    Only supporting evidence and file bodies are reduced.  The exact full case remains
    in the controller workspace and every returned delta is revalidated there.
    """
    attempts = [
        (4, 9000, 4, 4),
        (3, 7000, 3, 3),
        (2, 5000, 2, 2),
        (1, 3500, 1, 1),
    ]
    last_error: ContextBudgetError | None = None
    source_files = list(payload.get("relevant_case_files") or [])
    source_evidence = list(payload.get("supporting_evidence") or [])
    source_observations = list(payload.get("supporting_observations") or [])
    for index, (file_limit, content_chars, evidence_limit, observation_limit) in enumerate(attempts, start=1):
        capsule = dict(payload)
        capsule["relevant_case_files"] = _bounded_files(
            source_files,
            max_files=file_limit,
            total_content_chars=content_chars,
        )
        capsule["supporting_evidence"] = source_evidence[-evidence_limit:]
        capsule["supporting_observations"] = source_observations[-observation_limit:]
        try:
            built = build_bounded_json_prompt(instruction, capsule, max_chars=max_chars)
            return built, capsule, {
                "repairPartitioned": 1,
                "repairPartitionAttempts": index,
                "repairFocusedFiles": len(capsule["relevant_case_files"]),
                "repairFileContentChars": sum(
                    len(str(item.get("content") or ""))
                    for item in capsule["relevant_case_files"]
                    if isinstance(item, dict)
                ),
            }
        except ContextBudgetError as exc:
            last_error = exc

    raise ContextBudgetError(
        "Failure-local case repair could not fit the deterministic model-context budget even after "
        "reducing supporting files/evidence. The authored case and primary validation diagnostic "
        "remain unchanged; no repair mutation was authorized."
    ) from last_error
