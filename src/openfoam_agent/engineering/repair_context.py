from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import re

from openfoam_agent.llm.context import ContextBudgetError, PromptBuildResult, compact_text, build_bounded_json_prompt


# v4.8.2 model-visible repair budgets. These are intentionally independent from
# the controller-side evidence/archive history: evidence may remain durable in
# Python state without becoming prompt memory on every repair turn.
REPAIR_CURRENT_FAILURE_CHARS = 3500
REPAIR_ROOT_FAILURE_SUMMARY_CHARS = 1200
REPAIR_FILE_CONTENT_CHARS = 7000
REPAIR_SUPPORT_CHARS = 1800
REPAIR_HISTORY_CHARS = 900
REPAIR_SUPPORT_ITEMS = 4
REPAIR_HISTORY_ITEMS = 4
REPAIR_IMPLICATED_FILE_LIMIT = 8


_EXPLICIT_ALTERNATIVES_RE = re.compile(
    r"(?is)\b(?:unknown|unsupported|invalid)\b.{0,500}?"
    r"\b(?:supported|valid|available|allowed)\b[^\n:]{0,120}[:\n]"
)


def _json_chars(value: object) -> int:
    return len(json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str))


def _compact_record(value: object, max_chars: int) -> object:
    """Bound one model-visible record without mutating the controller archive."""
    if value is None:
        return None
    if isinstance(value, str):
        return compact_text(value, max_chars)
    if isinstance(value, dict):
        # Preserve the most useful compact-event fields first. Unknown fields are
        # permitted, but only while the section budget remains available.
        preferred = (
            "action_type", "success", "summary", "output_excerpt", "failure_signature",
            "failure_scope", "failure_category", "validation_status", "generation",
            "diagnosis", "changed_files",
        )
        out: dict[str, object] = {}
        for key in preferred:
            if key not in value:
                continue
            child = value[key]
            if isinstance(child, str):
                child = compact_text(child, min(max_chars, 1200))
            elif isinstance(child, list):
                child = child[:12]
            out[key] = child
            if _json_chars(out) >= max_chars:
                break
        if not out:
            text = compact_text(json.dumps(value, ensure_ascii=False, default=str), max_chars)
            return {"digest": hashlib.sha256(text.encode("utf-8")).hexdigest(), "summary": text}
        while out and _json_chars(out) > max_chars:
            key = next(reversed(out))
            child = out[key]
            if isinstance(child, str) and len(child) > 96:
                out[key] = compact_text(child, max(64, len(child) // 2))
            else:
                out.pop(key)
        return out
    text = compact_text(str(value), max_chars)
    return text


def _bounded_records(
    records: list[object], *, max_items: int, total_chars: int, per_item_chars: int = 600
) -> list[object]:
    """Newest-first relevance window with a hard aggregate JSON-character budget."""
    selected: list[object] = []
    seen: set[str] = set()
    remaining = max(0, total_chars)
    for raw in reversed(records):
        if len(selected) >= max_items or remaining < 64:
            break
        token = json.dumps(raw, ensure_ascii=True, sort_keys=True, default=str)
        if token in seen:
            continue
        seen.add(token)
        item = _compact_record(raw, min(per_item_chars, remaining))
        cost = _json_chars(item)
        if cost > remaining:
            item = _compact_record(raw, remaining)
            cost = _json_chars(item)
        if cost <= remaining:
            selected.append(item)
            remaining -= cost
    selected.reverse()
    return selected


def project_repair_episode(episode: object | None) -> dict[str, object] | None:
    """Return a bounded model projection of RepairEpisode.

    The authoritative RepairEpisode remains Python state. In particular, raw/older
    supporting observations are an archive, not prompt memory. The projection keeps
    a digest + compact root, one bounded current failure, a compact repair history,
    and a small support window.
    """
    if episode is None:
        return None
    raw = episode.model_dump(mode="json") if hasattr(episode, "model_dump") else deepcopy(episode)
    if not isinstance(raw, dict):
        return None

    root = raw.get("root_failure")
    current = raw.get("current_failure")
    previous = list(raw.get("previous_repairs") or [])
    support = list(raw.get("supporting_observations") or [])
    implicated = list(dict.fromkeys(str(x) for x in (raw.get("current_implicated_files") or []) if str(x)))
    root_serialized = json.dumps(root, ensure_ascii=True, sort_keys=True, default=str)

    return {
        "episode_id": raw.get("episode_id"),
        "validation_generation": raw.get("validation_generation", 0),
        "root_failure_digest": hashlib.sha256(root_serialized.encode("utf-8")).hexdigest(),
        "root_failure_summary": _compact_record(root, REPAIR_ROOT_FAILURE_SUMMARY_CHARS),
        "current_failure": _compact_record(current, REPAIR_CURRENT_FAILURE_CHARS),
        "current_implicated_files": implicated[:REPAIR_IMPLICATED_FILE_LIMIT],
        "previous_repairs_summary": _bounded_records(
            previous,
            max_items=REPAIR_HISTORY_ITEMS,
            total_chars=REPAIR_HISTORY_CHARS,
            per_item_chars=360,
        ),
        "support_evidence_window": _bounded_records(
            support,
            max_items=REPAIR_SUPPORT_ITEMS,
            total_chars=REPAIR_SUPPORT_CHARS,
            per_item_chars=500,
        ),
        "archive_counts": {
            "previous_repairs": len(previous),
            "supporting_observations": len(support),
        },
        "projection_note": "Controller archive retained; only bounded repair memory is model-visible.",
    }


def diagnostic_has_explicit_alternatives(diagnostic: object) -> bool:
    """Detect native diagnostics that already enumerate acceptable alternatives.

    This is a sufficiency signal only. It never chooses one of the alternatives.
    """
    if isinstance(diagnostic, dict):
        text = "\n".join(
            str(diagnostic.get(key) or "") for key in ("summary", "output_excerpt")
        )
    else:
        text = str(diagnostic or "")
    if not _EXPLICIT_ALTERNATIVES_RE.search(text):
        return False
    # Require at least one non-empty line after the supported/valid heading so a
    # bare heading cannot accidentally suppress useful retrieval.
    match = re.search(r"(?is)\b(?:supported|valid|available|allowed)\b[^\n:]{0,120}[:\n](.+)", text)
    if match is None:
        return False
    candidates = [line.strip(" \\t-*,:;") for line in match.group(1).splitlines()]
    return any(candidate and len(candidate) <= 160 for candidate in candidates[:12])


def direct_repair_evidence_is_sufficient(diagnostic: object, relevant_files: list[dict[str, object]]) -> bool:
    """True when native evidence + implicated files justify a direct repair attempt."""
    has_file = any(str(item.get("path") or "").strip() for item in relevant_files if isinstance(item, dict))
    return has_file and diagnostic_has_explicit_alternatives(diagnostic)


def repair_episode_requires_direct_attempt(episode: object | None) -> bool:
    """Gate retrieval until one direct repair has been attempted for this failure generation."""
    if episode is None:
        return False
    raw = episode.model_dump(mode="json") if hasattr(episode, "model_dump") else deepcopy(episode)
    if not isinstance(raw, dict):
        return False
    implicated = [{"path": path} for path in list(raw.get("current_implicated_files") or [])]
    if not direct_repair_evidence_is_sufficient(raw.get("current_failure"), implicated):
        return False
    generation = int(raw.get("validation_generation") or 0)
    for item in list(raw.get("previous_repairs") or []):
        if not isinstance(item, dict) or item.get("generation") is None:
            continue
        if int(item["generation"]) == generation:
            return False
    return True


def project_validation_repair_plan(plan: object) -> dict[str, object] | None:
    """Project only the plan fields needed to repair an already-authored case.

    The complete EngineeringPlan remains controller state. Validation repair must not
    re-send QoI/result-analysis/audit bodies merely to fix one dictionary or field.
    """
    if plan is None:
        return None
    raw = plan.model_dump(mode="json") if hasattr(plan, "model_dump") else deepcopy(plan)
    if not isinstance(raw, dict):
        return None

    defaults: list[dict[str, object]] = []
    for item in list(raw.get("engineering_defaults") or [])[:12]:
        if not isinstance(item, dict):
            continue
        defaults.append({
            "parameter": compact_text(str(item.get("parameter") or ""), 120),
            "value": compact_text(str(item.get("value") or ""), 220),
            "unit": compact_text(str(item.get("unit") or ""), 80),
            "basis": item.get("basis"),
        })

    bindings: list[dict[str, object]] = []
    for item in list(raw.get("confirmed_fact_bindings") or [])[:32]:
        if not isinstance(item, dict):
            continue
        bindings.append({
            "fact_id": item.get("fact_id"),
            "plan_fields": list(item.get("plan_fields") or [])[:12],
            "case_files": list(item.get("case_files") or [])[:16],
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
        "problem_interpretation": compact_text(str(raw.get("problem_interpretation") or ""), 450),
        "mesh_strategy": compact_text(str(raw.get("mesh_strategy") or ""), 450),
        "region_layouts": deepcopy(list(raw.get("region_layouts") or [])[:8]),
        "interfaces": deepcopy(list(raw.get("interfaces") or [])[:8]),
        "engineering_defaults": defaults,
        "required_case_files": list(raw.get("required_case_files") or [])[:48],
        "confirmed_intake_sha256": raw.get("confirmed_intake_sha256"),
        "confirmed_fact_ids": list(raw.get("confirmed_fact_ids") or [])[:64],
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
            cap = min(5000, remaining)
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
    """Build a budget-first committed-case repair capsule.

    v4.8.2 bounds support/history before whole-prompt fitting. Reducing focused files
    is now a last-mile partition, not the only mechanism preventing context growth.
    """
    attempts = [
        (4, REPAIR_FILE_CONTENT_CHARS, 4),
        (3, 5500, 3),
        (2, 4500, 2),
        (1, 3500, 1),
    ]
    last_error: ContextBudgetError | None = None
    source_files = list(payload.get("relevant_case_files") or [])
    source_evidence = list(payload.get("supporting_evidence") or [])
    source_observations = list(payload.get("supporting_observations") or [])

    # Backward compatibility for callers that still supply an authoritative
    # RepairEpisode: project it before any prompt build. Never model_dump the
    # archive directly into a repair prompt.
    projected_episode = project_repair_episode(payload.get("repair_episode"))
    if projected_episode is None and isinstance(payload.get("repair_episode_projection"), dict):
        projected_episode = deepcopy(payload["repair_episode_projection"])
    if isinstance(projected_episode, dict) and source_observations:
        existing_support = list(projected_episode.get("support_evidence_window") or [])
        projected_episode["support_evidence_window"] = _bounded_records(
            existing_support + source_observations,
            max_items=REPAIR_SUPPORT_ITEMS,
            total_chars=REPAIR_SUPPORT_CHARS,
            per_item_chars=500,
        )

    bounded_evidence = _bounded_records(
        source_evidence,
        max_items=REPAIR_SUPPORT_ITEMS,
        total_chars=REPAIR_SUPPORT_CHARS,
        per_item_chars=500,
    )

    for index, (file_limit, content_chars, evidence_limit) in enumerate(attempts, start=1):
        capsule = dict(payload)
        capsule.pop("supporting_observations", None)
        capsule.pop("repair_episode_projection", None)
        capsule["repair_episode"] = deepcopy(projected_episode)
        capsule["relevant_case_files"] = _bounded_files(
            source_files,
            max_files=file_limit,
            total_content_chars=content_chars,
        )
        capsule["supporting_evidence"] = bounded_evidence[-evidence_limit:]
        try:
            built = build_bounded_json_prompt(instruction, capsule, max_chars=max_chars)
            support_window = (
                list(projected_episode.get("support_evidence_window") or [])
                if isinstance(projected_episode, dict) else []
            )
            history_window = (
                list(projected_episode.get("previous_repairs_summary") or [])
                if isinstance(projected_episode, dict) else []
            )
            return built, capsule, {
                "repairPartitioned": 1,
                "repairPartitionAttempts": index,
                "repairFocusedFiles": len(capsule["relevant_case_files"]),
                "repairFileContentChars": sum(
                    len(str(item.get("content") or ""))
                    for item in capsule["relevant_case_files"]
                    if isinstance(item, dict)
                ),
                "repairSupportVisible": len(support_window) + len(capsule["supporting_evidence"]),
                "repairHistoryVisible": len(history_window),
                "repairSupportChars": _json_chars(support_window) + _json_chars(capsule["supporting_evidence"]),
            }
        except ContextBudgetError as exc:
            last_error = exc

    raise ContextBudgetError(
        "Failure-local case repair could not fit the deterministic model-context budget after "
        "bounded episode/support projection and focused-file partitioning. The authored case and "
        "primary native diagnostic remain unchanged; no repair mutation was authorized."
    ) from last_error
