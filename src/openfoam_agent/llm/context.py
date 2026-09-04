from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel


class ContextBudgetError(RuntimeError):
    """Raised before an API call when a bounded model context still cannot fit."""


@dataclass(frozen=True)
class PromptBuildResult:
    prompt: str
    prompt_chars: int
    prompt_bytes: int
    approximate_tokens: int
    compacted: bool
    compaction_level: int


def compact_text(value: str, max_chars: int) -> str:
    """Keep bounded head/tail evidence while making omission explicit."""

    if max_chars < 64:
        raise ValueError("max_chars must be >= 64")
    if len(value) <= max_chars:
        return value
    marker = "\n... [model-context compacted] ...\n"
    remaining = max_chars - len(marker)
    if remaining <= 0:
        return value[:max_chars]
    head = max(1, int(remaining * 0.6))
    tail = max(1, remaining - head)
    return value[:head] + marker + value[-tail:]


def compact_event_for_model(
    event: BaseModel,
    *,
    excerpt_chars: int,
    summary_chars: int = 900,
) -> dict[str, object]:
    """Project a durable event into a bounded LLM observation.

    The original event remains untouched in CFDState and is still used by deterministic
    provenance validation.  Only the remote-model projection is compacted.
    """

    raw = event.model_dump(mode="json")
    projected: dict[str, object] = {
        "step": raw.get("step"),
        "action_type": raw.get("action_type"),
        "success": raw.get("success"),
        "summary": compact_text(str(raw.get("summary", "")), summary_chars),
        "output_excerpt": compact_text(
            str(raw.get("output_excerpt", "")), excerpt_chars
        ),
    }
    for key in (
        "payload_ref",
        "artifact_sha256",
        "artifact_path",
        "native_command_executed",
        "mesh_command_executed",
    ):
        if key in raw and raw.get(key) not in (None, False, ""):
            projected[key] = raw.get(key)
    return projected


def compact_runtime_result(result: BaseModel, *, max_fields: int = 16, recent_samples: int = 8) -> dict[str, object]:
    """Summarize runtime evidence without transmitting every residual sample."""

    raw = result.model_dump(mode="json")
    residuals = list(raw.pop("residuals", []) or [])

    # Keep the latest residual sample per recently observed field. This preserves the
    # information post-processing/review needs without O(time-steps x fields) growth.
    latest_by_field: dict[str, dict[str, object]] = {}
    field_order: list[str] = []
    for sample in residuals:
        if not isinstance(sample, dict):
            continue
        field = str(sample.get("field", "")).strip()
        if not field:
            continue
        if field in latest_by_field:
            try:
                field_order.remove(field)
            except ValueError:
                pass
        field_order.append(field)
        latest_by_field[field] = sample
    selected_fields = field_order[-max_fields:]

    raw["residual_summary"] = {
        "total_samples": len(residuals),
        "field_count": len(latest_by_field),
        "latest_by_field": [latest_by_field[field] for field in selected_fields],
        "recent_samples": residuals[-recent_samples:] if recent_samples else [],
        "truncated": len(residuals) > max(len(selected_fields), recent_samples),
    }
    return raw


def compact_runtime_report(report: BaseModel, *, max_attempts: int = 4) -> dict[str, object]:
    """Compact a RuntimeReport while retaining final and recent-attempt evidence."""

    raw = report.model_dump(mode="json")
    attempts = list(raw.get("attempts", []) or [])
    compact_attempts: list[dict[str, object]] = []
    for attempt in attempts[-max_attempts:]:
        if not isinstance(attempt, dict):
            continue
        result = attempt.get("result")
        compact_attempt = {
            "attempt": attempt.get("attempt"),
            "repair_requested": attempt.get("repair_requested", False),
        }
        if isinstance(result, dict):
            compact_attempt["result"] = _compact_runtime_result_dict(result)
        compact_attempts.append(compact_attempt)
    final = raw.get("final_result")
    return {
        "success": raw.get("success"),
        "attempt_count": len(attempts),
        "recent_attempts": compact_attempts,
        "final_result": _compact_runtime_result_dict(final) if isinstance(final, dict) else None,
    }


def compact_inventory(
    items: list[dict[str, object]], *, max_items: int
) -> dict[str, object]:
    """Bound result/file inventories while keeping both early and latest entries."""

    total = len(items)
    if total <= max_items:
        selected = items
    else:
        first_count = max_items // 3
        last_count = max_items - first_count
        selected = items[:first_count] + items[-last_count:]
    return {
        "total_files": total,
        "shown_files": len(selected),
        "truncated": total > len(selected),
        "files": selected,
    }


def build_bounded_json_prompt(
    instruction: str,
    payload: dict[str, object],
    *,
    max_chars: int,
) -> PromptBuildResult:
    """Serialize JSON under a hard character budget before any remote API call.

    Domain-specific callers should already compact high-volume fields.  The generic
    fallback only activates if an unusual user/reference payload still exceeds the
    cap. It preserves dict keys and marks omitted list/string content explicitly.
    """

    if max_chars < 4_000:
        raise ValueError("max_chars must be >= 4000")

    levels: list[tuple[int | None, int | None]] = [
        (None, None),
        (4_000, 80),
        (2_000, 48),
        (1_000, 24),
        (500, 12),
    ]
    for level, (string_limit, list_limit) in enumerate(levels):
        candidate = payload if level == 0 else _compact_json_value(
            payload,
            string_limit=int(string_limit),
            list_limit=int(list_limit),
        )
        body = json.dumps(candidate, ensure_ascii=False, indent=2)
        prompt = instruction + body
        if len(prompt) <= max_chars:
            return _prompt_result(prompt, compacted=level > 0, level=level)

    raise ContextBudgetError(
        f"Model context remained above the deterministic {max_chars}-character budget "
        "after aggressive compaction; refusing the API call."
    )


def structured_request_metrics(
    schema: type[BaseModel],
    prompt: str,
    *,
    system_prompt: str = "",
) -> dict[str, int]:
    """Return conservative telemetry, not provider-billed token accounting."""

    schema_text = json.dumps(
        schema.model_json_schema(), ensure_ascii=False, separators=(",", ":")
    )
    combined = system_prompt + prompt + schema_text
    byte_count = len(combined.encode("utf-8"))
    # A deliberately conservative, tokenizer-free heuristic. English JSON usually
    # consumes fewer tokens than bytes/2; Korean/mixed text can be denser. This is
    # telemetry only and is never used as exact billing data.
    approximate_tokens = int(math.ceil(byte_count / 2.0))
    return {
        "promptChars": len(prompt),
        "schemaChars": len(schema_text),
        "approxTokens": approximate_tokens,
    }


def _compact_runtime_result_dict(raw: dict[str, Any]) -> dict[str, object]:
    residuals = list(raw.get("residuals", []) or [])
    latest_by_field: dict[str, dict[str, object]] = {}
    field_order: list[str] = []
    for sample in residuals:
        if not isinstance(sample, dict):
            continue
        field = str(sample.get("field", "")).strip()
        if not field:
            continue
        if field in latest_by_field:
            try:
                field_order.remove(field)
            except ValueError:
                pass
        field_order.append(field)
        latest_by_field[field] = sample
    fields = field_order[-16:]
    projected = {key: value for key, value in raw.items() if key != "residuals"}
    projected["residual_summary"] = {
        "total_samples": len(residuals),
        "field_count": len(latest_by_field),
        "latest_by_field": [latest_by_field[field] for field in fields],
        "recent_samples": residuals[-8:],
        "truncated": len(residuals) > max(len(fields), 8),
    }
    return projected


def _compact_json_value(value: object, *, string_limit: int, list_limit: int) -> object:
    if isinstance(value, str):
        return compact_text(value, max(64, string_limit))
    if isinstance(value, dict):
        return {
            str(key): _compact_json_value(
                child, string_limit=string_limit, list_limit=list_limit
            )
            for key, child in value.items()
        }
    if isinstance(value, list):
        items = value
        if len(items) > list_limit:
            first_count = max(1, list_limit // 2)
            last_count = max(1, list_limit - first_count)
            omitted = len(items) - first_count - last_count
            items = (
                items[:first_count]
                + [{"_model_context_omitted_items": omitted}]
                + items[-last_count:]
            )
        return [
            _compact_json_value(
                child, string_limit=string_limit, list_limit=list_limit
            )
            for child in items
        ]
    return value


def _prompt_result(prompt: str, *, compacted: bool, level: int) -> PromptBuildResult:
    encoded = prompt.encode("utf-8")
    return PromptBuildResult(
        prompt=prompt,
        prompt_chars=len(prompt),
        prompt_bytes=len(encoded),
        approximate_tokens=int(math.ceil(len(encoded) / 2.0)),
        compacted=compacted,
        compaction_level=level,
    )
