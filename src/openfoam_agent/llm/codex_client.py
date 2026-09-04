from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, TypeVar

from pydantic import BaseModel

from .openai_client import (
    DEFAULT_SYSTEM_PROMPT,
    LLMConfigurationError,
    StructuredOutputError,
    validate_structured_output_schema,
)
from .structured_schema import compile_transport_schema

T = TypeVar("T", bound=BaseModel)

DEFAULT_CODEX_MODEL = "codex-default"
DEFAULT_CODEX_TIMEOUT_SECONDS = 900
DEFAULT_CODEX_WAIT_HEARTBEAT_SECONDS = 15.0
DEFAULT_CODEX_STRUCTURED_REPAIRS = 1

# `--backend codex` is specifically the ChatGPT/Codex-login path. Strip API-key
# routing variables so an exported API credential cannot silently turn this backend
# into ordinary API billing. The Codex CLI's cached ChatGPT login remains available.
_CODEX_API_ROUTING_ENV = {
    "OPENAI_API_KEY",
    "CODEX_API_KEY",
    "OPENAI_BASE_URL",
    "OPENAI_ORG_ID",
    "OPENAI_ORGANIZATION",
    "OPENAI_PROJECT",
    "OPENAI_PROJECT_ID",
}


@dataclass(frozen=True)
class CodexCLIStatus:
    binary: str
    version: str
    login_status: str
    supports_ignore_user_config: bool


def _safe_process_text(value: str | None, *, limit: int = 4000) -> str:
    text = (value or "").strip()
    if len(text) > limit:
        return text[:limit] + "...<truncated>"
    return text


def _codex_environment() -> dict[str, str]:
    env = dict(os.environ)
    for key in _CODEX_API_ROUTING_ENV:
        env.pop(key, None)
    return env


def check_codex_cli(
    *,
    binary: str = "codex",
    timeout: float = 5.0,
) -> CodexCLIStatus:
    """Verify a usable, ChatGPT-authenticated Codex CLI without making a model call."""

    resolved = shutil.which(binary)
    if resolved is None:
        raise LLMConfigurationError(
            "Codex CLI was not found on PATH. Install @openai/codex, run `codex login`, "
            "then retry --backend codex."
        )
    env = _codex_environment()
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
            [resolved, "exec", "--help"],
            capture_output=True,
            text=True,
            stdin=subprocess.DEVNULL,
            timeout=timeout,
            check=False,
            env=env,
        )
        login_proc = subprocess.run(
            [resolved, "login", "status"],
            capture_output=True,
            text=True,
            stdin=subprocess.DEVNULL,
            timeout=timeout,
            check=False,
            env=env,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise LLMConfigurationError(f"Could not inspect Codex CLI: {exc}") from exc

    version = _safe_process_text(version_proc.stdout or version_proc.stderr, limit=300)
    help_text = "\n".join(part for part in (help_proc.stdout, help_proc.stderr) if part)
    required = ("--output-schema", "--output-last-message", "--ephemeral", "--sandbox")
    missing = [flag for flag in required if flag not in help_text]
    if version_proc.returncode != 0 or help_proc.returncode != 0 or missing:
        detail = f" missing flags={missing}" if missing else ""
        raise LLMConfigurationError(
            "Installed Codex CLI does not expose the required non-interactive structured-output "
            f"surface.{detail} Version output: {version or '<unknown>'}"
        )

    login_text = _safe_process_text(login_proc.stdout or login_proc.stderr, limit=500)
    if login_proc.returncode != 0 or "logged in" not in login_text.casefold():
        raise LLMConfigurationError(
            "--backend codex requires an authenticated Codex CLI. Run `codex login` and verify "
            "with `codex login status`. API-key environment variables are intentionally ignored "
            "by this backend so it uses the ChatGPT/Codex login path."
        )

    return CodexCLIStatus(
        binary=resolved,
        version=version or "codex",
        login_status=login_text,
        supports_ignore_user_config="--ignore-user-config" in help_text,
    )


class CodexLLM:
    """Structured LLM adapter backed by subscription-authenticated `codex exec`.

    Codex is used only as a model transport here. It runs in an empty temporary working
    directory with a read-only sandbox and ephemeral session, so CFD filesystem/tool
    execution remains owned by deterministic OpenFOAM Agent Python.
    """

    store = False

    def __init__(
        self,
        *,
        model: str | None = None,
        binary: str = "codex",
        timeout_seconds: int = DEFAULT_CODEX_TIMEOUT_SECONDS,
        structured_repair_attempts: int = DEFAULT_CODEX_STRUCTURED_REPAIRS,
        status: CodexCLIStatus | None = None,
        wait_callback: Callable[[float, float], None] | None = None,
        wait_heartbeat_seconds: float = DEFAULT_CODEX_WAIT_HEARTBEAT_SECONDS,
    ) -> None:
        normalized = (model or "").strip()
        if timeout_seconds < 1:
            raise LLMConfigurationError("Codex timeout_seconds must be positive.")
        if structured_repair_attempts < 0:
            raise LLMConfigurationError("Codex structured_repair_attempts must be non-negative.")
        if wait_heartbeat_seconds <= 0:
            raise LLMConfigurationError("Codex wait_heartbeat_seconds must be positive.")
        self.cli_model = normalized or None
        self.model = normalized or DEFAULT_CODEX_MODEL
        # Codex CLI currently has no stable per-response max-output-token exec flag.
        self.max_output_tokens = None
        self.timeout_seconds = timeout_seconds
        self.structured_repair_attempts = structured_repair_attempts
        self.status = status or check_codex_cli(binary=binary)
        self.binary = self.status.binary
        self.wait_callback = wait_callback
        self.wait_heartbeat_seconds = float(wait_heartbeat_seconds)
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
        # Codex exec sessions are deliberately ephemeral and stateless here. The calling
        # agent already supplies bounded full/delta state capsules in each prompt.
        del conversation_key, use_previous_response, prompt_cache_key

        effective_system = DEFAULT_SYSTEM_PROMPT if system_prompt is None else system_prompt.strip()
        base_prompt = self._combined_prompt(effective_system, normalized_prompt)
        attempts = 1 + self.structured_repair_attempts
        last_error = "Codex did not return valid structured JSON."
        previous = ""
        self.last_usage = None

        for attempt in range(attempts):
            current_prompt = base_prompt
            if attempt > 0:
                current_prompt += (
                    "\n\nPROTOCOL REPAIR:\nYour previous final JSON failed deterministic "
                    "Python/Pydantic validation. Return the same intended engineering decision "
                    "with only the JSON/schema shape corrected. Do not use Markdown and do not "
                    "change confirmed user facts.\nVALIDATION ERROR:\n"
                    + last_error
                    + "\nPREVIOUS FINAL OUTPUT:\n"
                    + previous[:12000]
                )
            try:
                text = self._run_once(schema, current_prompt)
            except StructuredOutputError:
                raise
            previous = text
            try:
                return schema.model_validate_json(text)
            except Exception as exc:
                last_error = _validation_error(exc)

        raise StructuredOutputError(
            f"Codex CLI model {self.model!r} failed Python/Pydantic structured-output "
            f"validation after {attempts} attempt(s): {last_error}"
        )

    @staticmethod
    def _combined_prompt(system_prompt: str, prompt: str) -> str:
        if not system_prompt:
            return prompt
        return (
            "SYSTEM INSTRUCTION (authoritative for this model-only call):\n"
            + system_prompt
            + "\n\nUSER/WORKFLOW INPUT:\n"
            + prompt
        )

    def _run_once(self, schema: type[BaseModel], prompt: str) -> str:
        env = _codex_environment()
        with tempfile.TemporaryDirectory(prefix="openfoam-agent-codex-") as temp_name:
            temp = Path(temp_name)
            schema_path = temp / "output.schema.json"
            output_path = temp / "final.json"
            schema_path.write_text(
                json.dumps(
                    compile_transport_schema(schema, backend="codex"),
                    ensure_ascii=True,
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
            command = [
                self.binary,
                "exec",
                "--ephemeral",
                "--skip-git-repo-check",
                "--sandbox",
                "read-only",
            ]
            if self.status.supports_ignore_user_config:
                # Avoid user MCP/tool configuration interfering with strict final JSON.
                command.append("--ignore-user-config")
            if self.cli_model is not None:
                command.extend(["--model", self.cli_model])
            command.extend(
                [
                    "--output-schema",
                    str(schema_path),
                    "--output-last-message",
                    str(output_path),
                    "-",
                ]
            )
            try:
                proc = _run_with_wait_heartbeat(
                    command,
                    input_text=prompt,
                    cwd=temp,
                    timeout_seconds=self.timeout_seconds,
                    env=env,
                    wait_callback=self.wait_callback,
                    heartbeat_seconds=self.wait_heartbeat_seconds,
                )
            except subprocess.TimeoutExpired as exc:
                raise StructuredOutputError(
                    f"Codex CLI timed out after {self.timeout_seconds}s."
                ) from exc
            except OSError as exc:
                raise StructuredOutputError(f"Could not execute Codex CLI: {exc}") from exc

            if proc.returncode != 0:
                stderr = _safe_process_text(proc.stderr or proc.stdout, limit=5000)
                raise StructuredOutputError(
                    f"codex exec failed with exit code {proc.returncode}: {stderr or '<no diagnostic>'}"
                )
            if not output_path.is_file():
                diagnostic = _safe_process_text(proc.stderr or proc.stdout, limit=3000)
                raise StructuredOutputError(
                    "codex exec exited successfully but did not write --output-last-message. "
                    f"Diagnostic: {diagnostic or '<none>'}"
                )
            text = output_path.read_text(encoding="utf-8", errors="replace").strip()
            if not text:
                raise StructuredOutputError("codex exec returned an empty final structured output.")
            return text


def _validation_error(exc: Exception, *, limit: int = 4000) -> str:
    text = str(exc).strip() or type(exc).__name__
    return text[:limit] + ("...<truncated>" if len(text) > limit else "")


def _run_with_wait_heartbeat(
    command: list[str],
    *,
    input_text: str,
    cwd: Path,
    timeout_seconds: float,
    env: dict[str, str],
    wait_callback: Callable[[float, float], None] | None,
    heartbeat_seconds: float,
):
    """Run one blocking CLI call while emitting bounded wait heartbeats.

    The subprocess contract stays identical to subprocess.run; the watchdog only reports
    wall-clock waiting and never reads model output or interferes with structured JSON.
    """
    stop = threading.Event()
    started = time.monotonic()
    watcher: threading.Thread | None = None

    if wait_callback is not None:
        def watch() -> None:
            while not stop.wait(heartbeat_seconds):
                elapsed = time.monotonic() - started
                try:
                    wait_callback(elapsed, timeout_seconds)
                except Exception:
                    # Progress reporting must never change model-call semantics.
                    continue

        watcher = threading.Thread(target=watch, name="codex-wait-heartbeat", daemon=True)
        watcher.start()

    try:
        return subprocess.run(
            command,
            input=input_text,
            capture_output=True,
            text=True,
            cwd=cwd,
            timeout=timeout_seconds,
            check=False,
            env=env,
        )
    finally:
        stop.set()
        if watcher is not None:
            watcher.join(timeout=0.2)
