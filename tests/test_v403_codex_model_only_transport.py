from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from pydantic import BaseModel

from openfoam_agent.llm.codex_client import (
    _CODEX_MODEL_ONLY_DISABLED_FEATURES,
    CodexCLIStatus,
    CodexLLM,
)
from openfoam_agent.llm.codex_transport import CodexTransportViolation, inspect_events


class _Output(BaseModel):
    ok: bool


def _status() -> CodexCLIStatus:
    return CodexCLIStatus(
        binary="/usr/bin/codex",
        version="codex-cli 0.153.0",
        login_status="Logged in using ChatGPT",
        supports_ignore_user_config=True,
    )


def test_codex_model_only_exec_requests_tool_disable_profile(monkeypatch):
    import openfoam_agent.llm.codex_client as codex

    captured: dict[str, object] = {}

    def fake_run(command, **kwargs):
        captured["command"] = list(command)
        output_path = command[command.index("--output-last-message") + 1]
        with open(output_path, "w", encoding="utf-8") as handle:
            json.dump({"ok": True}, handle)
        return SimpleNamespace(
            returncode=0,
            stdout='{"type":"turn.completed","usage":{"input_tokens":1,"output_tokens":1}}\n',
            stderr="",
        )

    monkeypatch.setattr(codex.subprocess, "run", fake_run)
    assert CodexLLM(status=_status()).generate(_Output, "Return ok.").ok is True

    command = captured["command"]
    assert command[command.index("--sandbox") + 1] == "read-only"
    assert '--ignore-user-config' in command
    assert 'approval_policy="never"' in command
    disabled = {
        command[index + 1]
        for index, token in enumerate(command[:-1])
        if token == "--disable"
    }
    assert set(_CODEX_MODEL_ONLY_DISABLED_FEATURES) <= disabled
    assert {"shell_tool", "unified_exec", "code_mode", "web_search", "apps", "plugins"} <= disabled


def test_codex_transport_violation_exposes_exact_command_event():
    stream = json.dumps(
        {
            "type": "item.completed",
            "item": {
                "type": "command_execution",
                "command": "/bin/bash -lc 'pwd'",
                "status": "completed",
            },
        }
    )
    with pytest.raises(CodexTransportViolation) as exc_info:
        inspect_events(stream)
    text = str(exc_info.value)
    assert "event=item.completed" in text
    assert "item.type=command_execution" in text
    assert "pwd" in text


def test_codex_transport_violation_exposes_web_search_query():
    stream = json.dumps(
        {
            "type": "item.started",
            "item": {"type": "web_search", "query": "OpenFOAM pressure outlet"},
        }
    )
    with pytest.raises(CodexTransportViolation) as exc_info:
        inspect_events(stream)
    text = str(exc_info.value)
    assert "item.type=web_search" in text
    assert "OpenFOAM pressure outlet" in text


def test_codex_success_contract_records_requested_model_only_profile():
    contract, usage = inspect_events(
        '{"type":"turn.completed","usage":{"input_tokens":10,"output_tokens":2}}\n'
    )
    assert usage["totalTokens"] == 12
    assert contract["model_only_profile_requested"] is True
    assert contract["post_execution_event_guard"] is True
    # Do not overclaim independent verification of the server-side tool surface.
    assert contract["pre_execution_tool_prevention_verified"] is False


def test_codex_nonterminal_error_item_is_recorded_not_rejected():
    stream = "\n".join(
        [
            json.dumps({"type": "thread.started", "thread_id": "t"}),
            json.dumps({"type": "item.completed", "item": {"type": "error", "message": "non-fatal diagnostic"}}),
            json.dumps({"type": "turn.started"}),
            json.dumps({"type": "item.completed", "item": {"type": "agent_message", "text": '{"ok":true}'}}),
            json.dumps({"type": "turn.completed", "usage": {"input_tokens": 3, "output_tokens": 2}}),
        ]
    )
    contract, usage = inspect_events(stream)
    assert contract["turn_completed"] is True
    assert contract["diagnostic_items_observed"] == 1
    assert contract["diagnostics"] == ["non-fatal diagnostic"]
    assert usage["totalTokens"] == 5


def test_codex_terminal_error_exposes_message():
    stream = json.dumps({"type": "error", "message": "provider temporarily unavailable"})
    with pytest.raises(ValueError) as exc_info:
        inspect_events(stream)
    text = str(exc_info.value)
    assert "CodexTurnError" in text
    assert "event=error" in text
    assert "provider temporarily unavailable" in text


def test_codex_turn_failed_exposes_nested_message():
    stream = json.dumps({"type": "turn.failed", "error": {"message": "shared rollout token budget exhausted"}})
    with pytest.raises(ValueError) as exc_info:
        inspect_events(stream)
    text = str(exc_info.value)
    assert "event=turn.failed" in text
    assert "shared rollout token budget exhausted" in text


def test_codex_client_accepts_nonterminal_error_item_when_turn_completes(monkeypatch):
    import openfoam_agent.llm.codex_client as codex

    def fake_run(command, **kwargs):
        output_path = command[command.index("--output-last-message") + 1]
        with open(output_path, "w", encoding="utf-8") as handle:
            json.dump({"ok": True}, handle)
        stdout = "\n".join(
            [
                json.dumps({"type": "item.completed", "item": {"type": "error", "message": "warning only"}}),
                json.dumps({"type": "turn.completed", "usage": {"input_tokens": 1, "output_tokens": 1}}),
            ]
        )
        return SimpleNamespace(returncode=0, stdout=stdout, stderr="")

    monkeypatch.setattr(codex.subprocess, "run", fake_run)
    llm = CodexLLM(status=_status())
    assert llm.generate(_Output, "Return ok.").ok is True
    assert llm.last_transport["diagnostic_items_observed"] == 1
    assert llm.last_transport["diagnostics"] == ["warning only"]
