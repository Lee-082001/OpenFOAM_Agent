from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel

from .openai_client import (
    DEFAULT_SYSTEM_PROMPT,
    LLMConfigurationError,
    StructuredOutputError,
    validate_structured_output_schema,
)
from .structured_schema import compile_transport_schema

T = TypeVar("T", bound=BaseModel)

DEFAULT_CLAUDE_MODEL = "claude-default"
DEFAULT_CLAUDE_TIMEOUT_SECONDS = 900
DEFAULT_CLAUDE_STRUCTURED_REPAIRS = 1

# --backend claude is intentionally the Claude Code subscription/OAuth path. Claude Code
# gives API/provider environment variables precedence over a logged-in subscription in -p
# mode, so remove those routing variables from both startup checks and model subprocesses.
# CLAUDE_CODE_OAUTH_TOKEN is deliberately preserved because Anthropic documents it as a
# subscription OAuth credential for headless/CI use.
_CLAUDE_API_ROUTING_ENV = {
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_AUTH_TOKEN",
    "ANTHROPIC_BASE_URL",
    "CLAUDE_CODE_USE_BEDROCK",
    "CLAUDE_CODE_USE_VERTEX",
    "CLAUDE_CODE_USE_FOUNDRY",
}


@dataclass(frozen=True)
class ClaudeCLIStatus:
    binary: str
    version: str
    auth_method: str
    subscription_type: str | None
    api_provider: str | None
    supports_safe_mode: bool


def _safe_process_text(value: str | None, *, limit: int = 4000) -> str:
    text = (value or "").strip()
    if len(text) > limit:
        return text[:limit] + "...<truncated>"
    return text


def _claude_environment() -> dict[str, str]:
    env = dict(os.environ)
    for key in _CLAUDE_API_ROUTING_ENV:
        env.pop(key, None)
    return env



def _parse_cli_version(text: str) -> tuple[int, int, int] | None:
    match = re.search(r"(?<!\d)(\d+)\.(\d+)\.(\d+)(?!\d)", text)
    if match is None:
        return None
    return tuple(int(part) for part in match.groups())


def _load_json_object(text: str, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise LLMConfigurationError(f"{label} did not return valid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise LLMConfigurationError(f"{label} must return a JSON object.")
    return value


def check_claude_cli(
    *,
    binary: str = "claude",
    timeout: float = 5.0,
) -> ClaudeCLIStatus:
    """Verify a Claude Code CLI authenticated through a Claude subscription/OAuth path."""

    resolved = shutil.which(binary)
    if resolved is None:
        raise LLMConfigurationError(
            "Claude Code CLI was not found on PATH. Install Claude Code, run `claude auth login`, "
            "then retry --backend claude."
        )
    env = _claude_environment()
    try:
        version_proc = subprocess.run(
            [resolved, "--version"],
            capture_output=True,
            text=True,
            stdin=subprocess.DEVNULL,
            timeout=timeout,
            check=False,
            env=env,
        )
        help_proc = subprocess.run(
            [resolved, "--help"],
            capture_output=True,
            text=True,
            stdin=subprocess.DEVNULL,
            timeout=timeout,
            check=False,
            env=env,
        )
        safe_mode_proc = subprocess.run(
            [resolved, "--safe-mode", "--version"],
            capture_output=True,
            text=True,
            stdin=subprocess.DEVNULL,
            timeout=timeout,
            check=False,
            env=env,
        )
        auth_proc = subprocess.run(
            [resolved, "auth", "status"],
            capture_output=True,
            text=True,
            stdin=subprocess.DEVNULL,
            timeout=timeout,
            check=False,
            env=env,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise LLMConfigurationError(f"Could not inspect Claude Code CLI: {exc}") from exc

    version = _safe_process_text(version_proc.stdout or version_proc.stderr, limit=300)
    help_text = "\n".join(part for part in (help_proc.stdout, help_proc.stderr) if part)
    if version_proc.returncode != 0 or help_proc.returncode != 0:
        raise LLMConfigurationError(
            "Could not inspect the installed Claude Code CLI. "
            f"Version output: {version or '<unknown>'}"
        )
    if safe_mode_proc.returncode != 0:
        diagnostic = _safe_process_text(safe_mode_proc.stderr or safe_mode_proc.stdout, limit=1000)
        raise LLMConfigurationError(
            "--backend claude requires a Claude Code CLI with --safe-mode so project/user "
            "customizations are not loaded into model-transport calls. Update Claude Code. "
            f"Diagnostic: {diagnostic or '<none>'}"
        )
    parsed_version = _parse_cli_version(version)
    if parsed_version is not None and parsed_version < (2, 1, 205):
        raise LLMConfigurationError(
            "--backend claude requires Claude Code >= 2.1.205 for reliable strict "
            f"--json-schema behavior; found {version or '<unknown>'}."
        )

    auth_text = _safe_process_text(auth_proc.stdout or auth_proc.stderr, limit=2000)
    if auth_proc.returncode != 0:
        raise LLMConfigurationError(
            "--backend claude requires an authenticated Claude Code CLI. Run `claude auth login` "
            "and verify with `claude auth status`. API/provider routing environment variables are "
            "intentionally ignored so this backend does not silently become pay-as-you-go API billing."
        )
    auth = _load_json_object(auth_text, label="claude auth status")
    logged_in = bool(auth.get("loggedIn"))
    auth_method = str(auth.get("authMethod") or "").strip()
    subscription_type = str(auth.get("subscriptionType") or "").strip() or None
    api_provider = str(auth.get("apiProvider") or "").strip() or None
    if not logged_in or auth_method.casefold() != "claude.ai":
        raise LLMConfigurationError(
            "--backend claude is the Claude subscription/OAuth transport and requires "
            "`claude auth status` to report loggedIn=true and authMethod=claude.ai. "
            f"Observed authMethod={auth_method or '<none>'!r}. Remove API/provider routing "
            "credentials and run `claude auth login` if you intend to use the subscription path."
        )

    return ClaudeCLIStatus(
        binary=resolved,
        version=version or "claude",
        auth_method=auth_method,
        subscription_type=subscription_type,
        api_provider=api_provider,
        supports_safe_mode=True,
    )


class ClaudeLLM:
    """Structured model adapter backed by subscription-authenticated Claude Code print mode.

    Claude Code is used only as a model transport. Built-in tools and MCP are disabled,
    customizations are suppressed when the installed CLI supports safe mode, sessions are not
    persisted, and each invocation runs from an empty temporary working directory. CFD case
    mutation and native OpenFOAM execution remain exclusively owned by deterministic Python.
    """

    store = False

    def __init__(
        self,
        *,
        model: str | None = None,
        binary: str = "claude",
        timeout_seconds: int = DEFAULT_CLAUDE_TIMEOUT_SECONDS,
        structured_repair_attempts: int = DEFAULT_CLAUDE_STRUCTURED_REPAIRS,
        status: ClaudeCLIStatus | None = None,
    ) -> None:
        normalized = (model or "").strip()
        if timeout_seconds < 1:
            raise LLMConfigurationError("Claude timeout_seconds must be positive.")
        if structured_repair_attempts < 0:
            raise LLMConfigurationError("Claude structured_repair_attempts must be non-negative.")
        self.cli_model = normalized or None
        self.model = normalized or DEFAULT_CLAUDE_MODEL
        # Claude Code print mode does not expose a stable per-response max-output-token flag.
        self.max_output_tokens = None
        self.timeout_seconds = timeout_seconds
        self.structured_repair_attempts = structured_repair_attempts
        self.status = status or check_claude_cli(binary=binary)
        self.binary = self.status.binary
        self.last_usage: dict[str, int] | None = None

    def generate(
        self,
        schema: type[T],
        prompt: str,
        *,
        system_prompt: str | None = None,
        conversation_key: str | None = None,
        use_previous_response: bool = False,
        prompt_cache_key: str | None = None,
    ) -> T:
        if not isinstance(schema, type) or not issubclass(schema, BaseModel):
            raise TypeError("schema must be a Pydantic BaseModel class.")
        validate_structured_output_schema(schema)
        normalized_prompt = prompt.strip()
        if not normalized_prompt:
            raise ValueError("prompt must not be empty.")
        # The calling agent already supplies bounded full/delta state capsules. Claude print-mode
        # sessions are intentionally stateless and non-persistent for this transport.
        del conversation_key, use_previous_response, prompt_cache_key

        effective_system = DEFAULT_SYSTEM_PROMPT if system_prompt is None else system_prompt.strip()
        attempts = 1 + self.structured_repair_attempts
        last_error = "Claude Code did not return valid structured JSON."
        previous = ""
        self.last_usage = None

        for attempt in range(attempts):
            current_prompt = normalized_prompt
            if attempt > 0:
                current_prompt += (
                    "\n\nPROTOCOL REPAIR:\nYour previous structured output failed deterministic "
                    "Python/Pydantic validation. Return the same intended engineering decision "
                    "with only the JSON/schema shape corrected. Do not use Markdown and do not "
                    "change confirmed user facts.\nVALIDATION ERROR:\n"
                    + last_error
                    + "\nPREVIOUS STRUCTURED OUTPUT:\n"
                    + previous[:12000]
                )
            try:
                structured, usage = self._run_once(schema, current_prompt, effective_system)
            except StructuredOutputError:
                raise
            previous = json.dumps(structured, ensure_ascii=False, separators=(",", ":"))
            if usage:
                self.last_usage = usage
            try:
                return schema.model_validate(structured)
            except Exception as exc:
                last_error = _validation_error(exc)

        raise StructuredOutputError(
            f"Claude Code model {self.model!r} failed Python/Pydantic structured-output "
            f"validation after {attempts} attempt(s): {last_error}"
        )

    def _run_once(
        self,
        schema: type[BaseModel],
        prompt: str,
        system_prompt: str,
    ) -> tuple[dict[str, Any], dict[str, int] | None]:
        env = _claude_environment()
        schema_json = json.dumps(
            compile_transport_schema(schema, backend="claude"),
            ensure_ascii=True,
            sort_keys=True,
        )
        with tempfile.TemporaryDirectory(prefix="openfoam-agent-claude-") as temp_name:
            temp = Path(temp_name)
            command = [
                self.binary,
                "-p",
                "--output-format",
                "json",
                "--json-schema",
                schema_json,
                "--no-session-persistence",
                "--tools",
                "",
                "--strict-mcp-config",
                "--system-prompt",
                system_prompt,
                "--max-turns",
                "1",
            ]
            if self.status.supports_safe_mode:
                command.append("--safe-mode")
            if self.cli_model is not None:
                command.extend(["--model", self.cli_model])
            command.append(prompt)
            try:
                proc = subprocess.run(
                    command,
                    capture_output=True,
                    text=True,
                    cwd=temp,
                    timeout=self.timeout_seconds,
                    check=False,
                    env=env,
                    stdin=subprocess.DEVNULL,
                )
            except subprocess.TimeoutExpired as exc:
                raise StructuredOutputError(
                    f"Claude Code CLI timed out after {self.timeout_seconds}s."
                ) from exc
            except OSError as exc:
                raise StructuredOutputError(f"Could not execute Claude Code CLI: {exc}") from exc

            if proc.returncode != 0:
                diagnostic = _safe_process_text(proc.stderr or proc.stdout, limit=5000)
                raise StructuredOutputError(
                    f"claude -p failed with exit code {proc.returncode}: {diagnostic or '<no diagnostic>'}"
                )
            outer_text = (proc.stdout or "").strip()
            if not outer_text:
                raise StructuredOutputError("claude -p returned an empty JSON envelope.")
            try:
                envelope = json.loads(outer_text)
            except json.JSONDecodeError as exc:
                raise StructuredOutputError(
                    "claude -p --output-format json returned invalid JSON: "
                    + _safe_process_text(outer_text, limit=3000)
                ) from exc
            if not isinstance(envelope, dict):
                raise StructuredOutputError("claude -p JSON envelope was not an object.")
            structured = envelope.get("structured_output")
            if not isinstance(structured, dict):
                diagnostic = _safe_process_text(
                    str(envelope.get("result") or envelope.get("error") or outer_text), limit=3000
                )
                raise StructuredOutputError(
                    "claude -p exited successfully but did not return a structured_output object. "
                    f"Diagnostic: {diagnostic or '<none>'}"
                )
            return structured, _claude_usage(envelope)


def _claude_usage(envelope: dict[str, Any]) -> dict[str, int] | None:
    usage = envelope.get("usage")
    if not isinstance(usage, dict):
        return None

    def integer(*names: str) -> int:
        for name in names:
            value = usage.get(name)
            if isinstance(value, bool):
                continue
            if isinstance(value, (int, float)):
                return int(value)
        return 0

    mapped = {
        "inputTokens": integer("input_tokens", "inputTokens"),
        "outputTokens": integer("output_tokens", "outputTokens"),
        "cachedInputTokens": integer("cache_read_input_tokens", "cached_input_tokens", "cachedInputTokens"),
        "cacheWriteTokens": integer("cache_creation_input_tokens", "cache_write_input_tokens", "cacheWriteTokens"),
    }
    if not any(mapped.values()):
        return None
    mapped["totalTokens"] = mapped["inputTokens"] + mapped["outputTokens"]
    return mapped


def _validation_error(exc: Exception, *, limit: int = 4000) -> str:
    text = str(exc).strip() or type(exc).__name__
    return text[:limit] + ("...<truncated>" if len(text) > limit else "")
