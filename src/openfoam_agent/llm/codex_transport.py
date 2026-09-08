"""Observable Codex transport contract for model-only structured calls.

The adapter requests a no-tool Codex exec profile *before* the model call.  The JSON
stream is still inspected afterwards because CLI/model regressions can re-expose a
capability despite the requested profile.  Pre-execution prevention is therefore a
requested configuration contract, while the event inspector remains the fail-closed
runtime backstop.
"""
from __future__ import annotations

import json
import re
from typing import Any


_MODEL_ONLY_ALLOWED_ITEMS = frozenset({"agent_message", "reasoning", "plan", "todo_list"})


class CodexTransportViolation(ValueError):
    """Raised when a Codex event violates the model-only transport contract."""

    def __init__(
        self,
        *,
        event_type: str,
        item_type: str | None = None,
        detail: str | None = None,
    ) -> None:
        self.event_type = event_type or "<missing>"
        self.item_type = item_type
        self.detail = _bounded_detail(detail)
        fields = [f"event={self.event_type}"]
        if item_type:
            fields.append(f"item.type={item_type}")
        if self.detail:
            fields.append(self.detail)
        super().__init__(
            "CodexTransportViolation: "
            + ", ".join(fields)
            + "; rejecting model-only output before OpenFOAM Agent acts on it."
        )


def _bounded_detail(value: str | None, *, limit: int = 600) -> str:
    text = (value or "").replace("\x00", "").strip().replace("\n", " ")
    if len(text) > limit:
        text = text[:limit] + "...<truncated>"
    return text


def _item_detail(item: dict[str, Any]) -> str:
    """Expose the useful tool diagnostic without dumping arbitrary event payloads."""

    item_type = str(item.get("type") or "")
    if item_type == "command_execution":
        command = _bounded_detail(str(item.get("command") or item.get("cmd") or ""), limit=500)
        return f'command={json.dumps(command, ensure_ascii=False)}' if command else ""
    if item_type in {"web_search", "web_search_call", "search"}:
        query = _bounded_detail(str(item.get("query") or item.get("q") or ""), limit=500)
        return f'query={json.dumps(query, ensure_ascii=False)}' if query else ""
    if item_type in {"mcp_tool_call", "tool_call"}:
        name = _bounded_detail(str(item.get("name") or item.get("tool_name") or ""), limit=300)
        return f'tool={json.dumps(name, ensure_ascii=False)}' if name else ""
    if item_type == "file_change":
        path = _bounded_detail(str(item.get("path") or item.get("file_path") or ""), limit=400)
        return f'path={json.dumps(path, ensure_ascii=False)}' if path else ""
    return ""


def authentication_kind(status: str) -> str:
    text = status.casefold()
    if re.search(r"api[ -_]?key|\bwith api\b|\busing api\b", text):
        return "api_key"
    if "logged in" in text and "chatgpt" in text and not any(x in text for x in ("not logged", "failed", "error")):
        return "chatgpt"
    return "unknown"


def inspect_events(output: str):
    if len(output) > 8_000_000:
        raise ValueError("Codex event output exceeds the transport inspection limit.")
    completed = False
    usage = None
    count = 0
    for line in output.splitlines():
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except (ValueError, TypeError):
            raise ValueError("Codex --json returned a non-JSON event; transport contract is unverified.") from None
        if not isinstance(event, dict):
            raise ValueError("Invalid Codex event envelope.")
        count += 1
        kind = str(event.get("type") or "")
        if kind in {"error", "turn.failed"}:
            raise ValueError("Codex reported a failed turn.")
        if kind in {"item.started", "item.updated", "item.completed"}:
            item = event.get("item", {})
            if not isinstance(item, dict):
                raise CodexTransportViolation(event_type=kind, detail="item=<invalid-envelope>")
            item_type = str(item.get("type") or "<missing>")
            if item_type not in _MODEL_ONLY_ALLOWED_ITEMS:
                raise CodexTransportViolation(
                    event_type=kind,
                    item_type=item_type,
                    detail=_item_detail(item),
                )
        elif kind == "turn.completed":
            completed = True
            raw = event.get("usage")
            if raw is not None:
                if not isinstance(raw, dict):
                    raise ValueError("Invalid Codex usage object.")
                keys = {
                    "input_tokens": "inputTokens",
                    "output_tokens": "outputTokens",
                    "cached_input_tokens": "cachedInputTokens",
                    "reasoning_output_tokens": "reasoningOutputTokens",
                }
                if not {"input_tokens", "output_tokens"} <= raw.keys():
                    raise ValueError("Codex usage is incomplete; not substituting zero.")
                usage = {}
                for source, target in keys.items():
                    if source in raw:
                        value = raw[source]
                        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                            raise ValueError("Invalid Codex token count.")
                        usage[target] = value
                usage["totalTokens"] = usage["inputTokens"] + usage["outputTokens"]
        elif kind not in {"thread.started", "turn.started"}:
            raise CodexTransportViolation(event_type=kind or "<missing>", detail="unknown event envelope")
    if not completed:
        raise ValueError("Codex event stream has no completed turn.")
    return {
        "event_count": count,
        "turn_completed": True,
        "tool_events_observed": False,
        "model_only_profile_requested": True,
        "post_execution_event_guard": True,
        # We cannot introspect the exact server-side tool surface from the completed
        # JSON stream, so do not overclaim that prevention itself was independently verified.
        "pre_execution_tool_prevention_verified": False,
    }, usage
