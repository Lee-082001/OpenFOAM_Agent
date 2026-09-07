"""Observable Codex transport contract; read-only is not equivalent to no tools."""
from __future__ import annotations
import json
import re


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
        kind = event.get("type")
        if kind in {"error", "turn.failed"}:
            raise ValueError("Codex reported a failed turn.")
        if kind in {"item.started", "item.updated", "item.completed"}:
            item = event.get("item", {})
            if not isinstance(item, dict) or item.get("type") not in {"agent_message", "reasoning", "plan", "todo_list"}:
                raise ValueError("Codex emitted a tool/action or unknown item event; rejecting model-only output.")
        elif kind == "turn.completed":
            completed = True
            raw = event.get("usage")
            if raw is not None:
                if not isinstance(raw, dict):
                    raise ValueError("Invalid Codex usage object.")
                keys = {"input_tokens": "inputTokens", "output_tokens": "outputTokens", "cached_input_tokens": "cachedInputTokens", "reasoning_output_tokens": "reasoningOutputTokens"}
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
            raise ValueError("Unknown Codex event type; verify the installed CLI version before use.")
    if not completed:
        raise ValueError("Codex event stream has no completed turn.")
    return {"event_count": count, "turn_completed": True, "tool_events_observed": False,
            "pre_execution_tool_prevention_verified": False}, usage
