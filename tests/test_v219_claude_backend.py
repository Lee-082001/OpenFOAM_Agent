from __future__ import annotations

import json
import os
from types import SimpleNamespace

import pytest
from pydantic import BaseModel

from openfoam_agent.cli import (
    _build_llm,
    _resolve_backend_model_names,
    _validate_args,
    build_parser,
)
from openfoam_agent.llm.claude_client import ClaudeCLIStatus, ClaudeLLM, check_claude_cli
from openfoam_agent.llm.openai_client import LLMConfigurationError, StructuredOutputError


class DemoOutput(BaseModel):
    action: str
    reason: str


def _status() -> ClaudeCLIStatus:
    return ClaudeCLIStatus(
        binary="/usr/bin/claude",
        version="2.1.221 (Claude Code)",
        auth_method="claude.ai",
        subscription_type="max",
        api_provider="firstParty",
        supports_safe_mode=True,
    )


def test_claude_cli_check_requires_subscription_auth_and_strips_api_routing(monkeypatch):
    import openfoam_agent.llm.claude_client as claude

    monkeypatch.setattr(claude.shutil, "which", lambda binary: "/usr/bin/claude")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "must-not-leak")
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "must-not-leak")
    monkeypatch.setenv("CLAUDE_CODE_USE_BEDROCK", "1")
    captured_envs = []

    def fake_run(command, **kwargs):
        captured_envs.append(dict(kwargs["env"]))
        if command[-1] == "--version" and "--safe-mode" not in command:
            return SimpleNamespace(returncode=0, stdout="2.1.221 (Claude Code)\n", stderr="")
        if command[-1] == "--version" and "--safe-mode" in command:
            return SimpleNamespace(returncode=0, stdout="2.1.221 (Claude Code)\n", stderr="")
        if command[-1] == "--help":
            return SimpleNamespace(
                returncode=0,
                stdout=(
                    "--json-schema --output-format --no-session-persistence --tools "
                    "--strict-mcp-config --system-prompt --safe-mode"
                ),
                stderr="",
            )
        if command[1:3] == ["auth", "status"]:
            return SimpleNamespace(
                returncode=0,
                stdout=json.dumps({
                    "loggedIn": True,
                    "authMethod": "claude.ai",
                    "apiProvider": "firstParty",
                    "subscriptionType": "max",
                }),
                stderr="",
            )
        raise AssertionError(command)

    monkeypatch.setattr(claude.subprocess, "run", fake_run)
    status = check_claude_cli()
    assert status.auth_method == "claude.ai"
    assert status.subscription_type == "max"
    assert status.supports_safe_mode is True
    assert captured_envs
    for env in captured_envs:
        assert "ANTHROPIC_API_KEY" not in env
        assert "ANTHROPIC_AUTH_TOKEN" not in env
        assert "CLAUDE_CODE_USE_BEDROCK" not in env


def test_claude_cli_rejects_api_key_auth(monkeypatch):
    import openfoam_agent.llm.claude_client as claude

    monkeypatch.setattr(claude.shutil, "which", lambda binary: "/usr/bin/claude")

    def fake_run(command, **kwargs):
        del kwargs
        if command[-1] == "--version" and "--safe-mode" not in command:
            return SimpleNamespace(returncode=0, stdout="2.1.221\n", stderr="")
        if command[-1] == "--version" and "--safe-mode" in command:
            return SimpleNamespace(returncode=0, stdout="2.1.221\n", stderr="")
        if command[-1] == "--help":
            return SimpleNamespace(returncode=0, stdout="--safe-mode", stderr="")
        if command[1:3] == ["auth", "status"]:
            return SimpleNamespace(
                returncode=0,
                stdout=json.dumps({"loggedIn": True, "authMethod": "api_key"}),
                stderr="",
            )
        raise AssertionError(command)

    monkeypatch.setattr(claude.subprocess, "run", fake_run)
    with pytest.raises(LLMConfigurationError, match="authMethod=claude.ai"):
        check_claude_cli()


def test_claude_llm_disables_tools_mcp_persistence_and_revalidates(monkeypatch):
    import openfoam_agent.llm.claude_client as claude

    captured = {}
    monkeypatch.setenv("ANTHROPIC_API_KEY", "must-not-leak")
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "must-not-leak")
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://proxy.invalid")

    def fake_run(command, **kwargs):
        captured["command"] = list(command)
        captured["cwd"] = kwargs["cwd"]
        captured["env"] = dict(kwargs["env"])
        envelope = {
            "structured_output": {"action": "ok", "reason": "validated"},
            "usage": {
                "input_tokens": 123,
                "output_tokens": 45,
                "cache_read_input_tokens": 10,
                "cache_creation_input_tokens": 7,
            },
        }
        return SimpleNamespace(returncode=0, stdout=json.dumps(envelope), stderr="")

    monkeypatch.setattr(claude.subprocess, "run", fake_run)
    llm = ClaudeLLM(status=_status(), model=None)
    result = llm.generate(DemoOutput, "Return a tiny result.", system_prompt="System")

    assert result.action == "ok"
    command = captured["command"]
    assert command[:2] == ["/usr/bin/claude", "-p"]
    assert command[command.index("--output-format") + 1] == "json"
    assert "--json-schema" in command
    assert "--no-session-persistence" in command
    assert command[command.index("--tools") + 1] == ""
    assert "--strict-mcp-config" in command
    assert "--safe-mode" in command
    assert command[command.index("--max-turns") + 1] == "1"
    assert "ANTHROPIC_API_KEY" not in captured["env"]
    assert "ANTHROPIC_AUTH_TOKEN" not in captured["env"]
    assert "ANTHROPIC_BASE_URL" not in captured["env"]
    assert os.path.isdir(captured["cwd"]) is False
    assert llm.last_usage == {
        "inputTokens": 123,
        "outputTokens": 45,
        "cachedInputTokens": 10,
        "cacheWriteTokens": 7,
        "totalTokens": 168,
    }


def test_claude_llm_rejects_missing_structured_output(monkeypatch):
    import openfoam_agent.llm.claude_client as claude

    monkeypatch.setattr(
        claude.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=0,
            stdout=json.dumps({"result": "not structured"}),
            stderr="",
        ),
    )
    llm = ClaudeLLM(status=_status())
    with pytest.raises(StructuredOutputError, match="structured_output"):
        llm.generate(DemoOutput, "Return a tiny result.")


def test_claude_model_routing_precedence_and_cli_construction(monkeypatch):
    import openfoam_agent.cli as cli

    parser = build_parser()
    args = parser.parse_args([
        "demo",
        "--backend",
        "claude",
        "--confirm-api-calls",
        "--model",
        "sonnet",
        "--engineering-model",
        "opus",
    ])
    assert _validate_args(args, parser) == "demo"
    default_model, names = _resolve_backend_model_names(
        args,
        backend="claude",
        environ={
            "CLAUDE_MODEL": "haiku",
            "CLAUDE_REVIEW_MODEL": "fable",
        },
    )
    assert default_model == "sonnet"
    assert names == {
        "intake": "sonnet",
        "engineering": "opus",
        "postprocessing": "sonnet",
        "review": "fable",
    }

    monkeypatch.setattr(cli, "check_claude_cli", _status)

    class DummyClaudeLLM:
        created = []

        def __init__(self, *, model, status):
            self.model = model or "claude-default"
            self.status = status
            self.last_usage = None
            self.__class__.created.append(self.model)

    class ForbiddenOpenAI:
        def __init__(self, **kwargs):
            raise AssertionError("--backend claude must not construct OpenAILLM")

    monkeypatch.setattr(cli, "ClaudeLLM", DummyClaudeLLM)
    monkeypatch.setattr(cli, "OpenAILLM", ForbiddenOpenAI)
    llms, backend, selected = _build_llm(args)
    assert backend == "claude"
    assert selected == "sonnet"
    assert llms.intake.model == "sonnet"
    assert llms.engineering.model == "opus"
    assert len(DummyClaudeLLM.created) == 2


def test_cli_claude_requires_explicit_cloud_authorization():
    parser = build_parser()
    args = parser.parse_args(["demo", "--backend", "claude"])
    with pytest.raises(SystemExit):
        _validate_args(args, parser)

    args = parser.parse_args(["demo", "--backend", "claude", "--confirm-api-calls"])
    assert _validate_args(args, parser) == "demo"
