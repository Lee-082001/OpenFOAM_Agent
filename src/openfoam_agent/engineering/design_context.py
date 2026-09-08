from __future__ import annotations

from copy import deepcopy
from typing import Any, Callable

from openfoam_agent.llm.context import ContextBudgetError, PromptBuildResult, build_bounded_json_prompt, compact_text


def _summary_only_evidence(records: list[dict[str, object]], *, keep_detail: int = 3) -> list[dict[str, object]]:
    """Keep full syntax/detail only for the newest evidence slice.

    Durable evidence remains in CFDState. Older records are represented by identity,
    reference and summary so evidence retrieval cannot make every later design prompt
    monotonically larger.
    """
    projected: list[dict[str, object]] = []
    detail_start = max(0, len(records) - max(0, keep_detail))
    for index, record in enumerate(records):
        item = {
            "evidence_id": record.get("evidence_id"),
            "kind": record.get("kind"),
            "reference": compact_text(str(record.get("reference", "")), 240),
            "summary": compact_text(str(record.get("summary", "")), 360),
        }
        if index >= detail_start and record.get("detail") not in (None, ""):
            item["detail"] = compact_text(str(record.get("detail")), 520)
        projected.append(item)
    return projected


def _minimal_environment_hint(value: object) -> object:
    """Project environment identity rather than an unbounded installation inventory."""
    if not isinstance(value, dict):
        return value
    allowed = {
        "openfoam_version", "version", "foam_version", "WM_PROJECT_VERSION",
        "foam_api", "installation", "platform", "checkMesh", "native_available",
    }
    result = {key: deepcopy(val) for key, val in value.items() if key in allowed}
    if result:
        return result
    # Unknown tool adapters may expose different names. Keep a bounded identity slice.
    for key, val in list(value.items())[:8]:
        if isinstance(val, (str, int, float, bool)) or val is None:
            result[str(key)] = val
    return result


def project_design_capsule(payload: dict[str, object], *, evidence_limit: int) -> dict[str, object]:
    """Create a bounded design/evidence capsule without truncating authoritative inputs."""
    evidence = list(payload.get("available_evidence", []) or [])
    if evidence_limit >= 0:
        evidence = evidence[-evidence_limit:] if evidence_limit else []
    capsule: dict[str, object] = {
        # Authoritative requirements/contracts are copied exactly.
        "state_mode": "partitioned_engineering_design",
        "phase": payload.get("phase"),
        "step": payload.get("step"),
        "confirmed_intake": deepcopy(payload.get("confirmed_intake")),
        "intake_sha256": payload.get("intake_sha256"),
        "engineering_assumption_policy": deepcopy(payload.get("engineering_assumption_policy")),
        "evidence_policy": deepcopy(payload.get("evidence_policy")),
        "verified_execution_candidates": deepcopy(payload.get("verified_execution_candidates", [])),
        "evidence_retrieval_policy": deepcopy(payload.get("evidence_retrieval_policy")),
        "bindings": deepcopy(payload.get("bindings")),
        "budget": deepcopy(payload.get("budget")),
        # Optional observations are explicitly projected, not silently truncated.
        "available_evidence": _summary_only_evidence(evidence, keep_detail=min(3, len(evidence))),
        "evidence_context": deepcopy(payload.get("evidence_context", {})),
        "evidence_gap_status": list(payload.get("evidence_gap_status", []) or [])[-3:],
        "recent_observations": list(payload.get("recent_observations", []) or [])[-2:],
        "environment_hint": _minimal_environment_hint(payload.get("environment_hint")),
        "context_partition": {
            "active": True,
            "evidence_shown": len(evidence),
            "evidence_total": (payload.get("evidence_context") or {}).get("total_observed")
                if isinstance(payload.get("evidence_context"), dict) else None,
            "semantics": (
                "Confirmed intake/contracts are complete. Evidence is a relevance/recency projection of a durable registry; "
                "absence from this capsule does not mean unsupported. Request only a genuinely missing tool/version fact."
            ),
        },
    }
    return capsule


def build_partitioned_design_prompt(
    instruction: str,
    payload: dict[str, object],
    *,
    max_chars: int,
    initial_evidence_limit: int,
) -> tuple[PromptBuildResult, dict[str, object], dict[str, int]]:
    """Fit prepare-design context by deterministic evidence projection before failing.

    This is intentionally different from generic truncation: authoritative requirements
    remain byte-for-byte represented in the JSON object while optional evidence/detail is
    paged down. If those mandatory inputs alone cannot fit, ContextBudgetError is retained.
    """
    attempts: list[int] = []
    limits: list[int] = []
    for candidate in (initial_evidence_limit, min(initial_evidence_limit, 6), 4, 2, 0):
        candidate = max(0, int(candidate))
        if candidate not in limits:
            limits.append(candidate)
    last_error: ContextBudgetError | None = None
    for evidence_limit in limits:
        attempts.append(evidence_limit)
        capsule = project_design_capsule(payload, evidence_limit=evidence_limit)
        try:
            result = build_bounded_json_prompt(instruction, capsule, max_chars=max_chars)
            return result, capsule, {
                "partitioned": 1,
                "partitionEvidenceLimit": evidence_limit,
                "partitionAttempts": len(attempts),
            }
        except ContextBudgetError as exc:
            last_error = exc
    raise ContextBudgetError(
        "Mandatory confirmed design requirements/contracts exceed the Engineering prompt budget even after "
        "evidence/environment partitioning. Increase --engineering-context-chars or reduce authoritative input; "
        "no confirmed requirement was truncated."
    ) from last_error
