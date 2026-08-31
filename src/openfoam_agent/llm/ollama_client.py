from __future__ import annotations

import json
import os
import socket
from typing import Any, TypeVar
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from pydantic import BaseModel

from .openai_client import (
    DEFAULT_SYSTEM_PROMPT,
    LLMConfigurationError,
    StructuredOutputError,
)

T = TypeVar("T", bound=BaseModel)

DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434/v1"
DEFAULT_OLLAMA_MODEL = "gemma4:31b"
DEFAULT_OLLAMA_API_KEY = "ollama"
DEFAULT_OLLAMA_HEALTH_TIMEOUT = 3.0
DEFAULT_OLLAMA_STRUCTURED_REPAIRS = 2
_ALLOWED_LOOPBACK_HOSTS = {"localhost", "127.0.0.1", "::1"}


class OllamaConnectionError(RuntimeError):
    """Raised when the local Ollama endpoint cannot be reached."""


class OllamaModelError(RuntimeError):
    """Raised when Ollama does not expose a requested local model."""


def normalize_ollama_base_url(base_url: str) -> str:
    value = base_url.strip().rstrip("/")
    if not value:
        raise LLMConfigurationError("An Ollama base URL is required.")
    parsed = urlparse(value)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise LLMConfigurationError(
            "Ollama base URL must be an http(s) URL such as http://localhost:11434/v1."
        )
    if parsed.hostname.lower() not in _ALLOWED_LOOPBACK_HOSTS:
        raise LLMConfigurationError(
            "Ollama backend base URL must use a loopback host (localhost/127.0.0.1/::1). "
            "Do not expose the remote Ollama port directly; connect through SSH local "
            "port forwarding instead."
        )
    if parsed.path not in {"", "/", "/v1"}:
        raise LLMConfigurationError(
            "Ollama OpenAI-compatible base URL must end at /v1 (for example "
            "http://localhost:11434/v1)."
        )
    if not parsed.path or parsed.path == "/":
        value = f"{value}/v1"
    return value.rstrip("/")


def ollama_models_url(base_url: str) -> str:
    return f"{normalize_ollama_base_url(base_url)}/models"


def _connection_message(base_url: str) -> str:
    endpoint = normalize_ollama_base_url(base_url)
    root = endpoint[:-3] if endpoint.endswith("/v1") else endpoint
    root = root.rstrip("/")
    return (
        f"Cannot connect to Ollama at {root}. Check that the SSH tunnel to "
        "mlfm4.knu.ac.kr and the Ollama service are running."
    )


def check_ollama_health(
    *,
    base_url: str,
    models: list[str] | tuple[str, ...] = (),
    api_key: str = DEFAULT_OLLAMA_API_KEY,
    timeout: float = DEFAULT_OLLAMA_HEALTH_TIMEOUT,
) -> list[str]:
    """Check the loopback OpenAI-compatible Ollama endpoint and requested models.

    This intentionally uses a tiny stdlib HTTP request instead of any Ollama-specific
    SDK. It verifies the same /v1 endpoint the OpenAI-compatible adapter will use and
    makes the SSH tunnel requirement fail fast before the workflow starts.
    """

    endpoint = normalize_ollama_base_url(base_url)
    if timeout <= 0:
        raise LLMConfigurationError("Ollama health-check timeout must be positive.")
    request = Request(
        f"{endpoint}/models",
        headers={"Authorization": f"Bearer {api_key}", "Accept": "application/json"},
        method="GET",
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (URLError, HTTPError, TimeoutError, socket.timeout, OSError, json.JSONDecodeError) as exc:
        raise LLMConfigurationError(_connection_message(endpoint)) from exc

    data = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(data, list):
        raise LLMConfigurationError(
            f"Ollama at {endpoint} did not return a valid OpenAI-compatible /v1/models response."
        )
    available = sorted(
        {
            item.get("id")
            for item in data
            if isinstance(item, dict) and isinstance(item.get("id"), str) and item.get("id")
        }
    )
    requested = sorted({model.strip() for model in models if model and model.strip()})
    missing = [model for model in requested if model not in available]
    if missing:
        names = ", ".join(missing)
        raise LLMConfigurationError(
            f"Ollama is reachable at {endpoint}, but model(s) {names} were not found. "
            "On mlfm4.knu.ac.kr, verify `ollama list` and pull/run the requested model first."
        )
    return available


class OllamaLLM:
    """Structured-output adapter for Ollama's OpenAI-compatible Chat Completions API.

    Ollama is only the LLM backend. Intake, engineering, evidence validation, native
    OpenFOAM tools, repair orchestration and review remain owned by OpenFOAM Agent.
    """

    def __init__(
        self,
        *,
        model: str = DEFAULT_OLLAMA_MODEL,
        base_url: str = DEFAULT_OLLAMA_BASE_URL,
        api_key: str = DEFAULT_OLLAMA_API_KEY,
        client: Any | None = None,
        default_system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        max_output_tokens: int | None = None,
        structured_repair_attempts: int = DEFAULT_OLLAMA_STRUCTURED_REPAIRS,
    ) -> None:
        normalized_model = model.strip()
        if not normalized_model:
            raise LLMConfigurationError("An Ollama model name is required.")
        if max_output_tokens is not None and max_output_tokens <= 0:
            raise LLMConfigurationError("max_output_tokens must be positive when set.")
        if structured_repair_attempts < 0:
            raise LLMConfigurationError("structured_repair_attempts must be non-negative.")

        self.model = normalized_model
        self.base_url = normalize_ollama_base_url(base_url)
        self.api_key = api_key or DEFAULT_OLLAMA_API_KEY
        self.default_system_prompt = default_system_prompt.strip()
        self.max_output_tokens = max_output_tokens
        self.structured_repair_attempts = structured_repair_attempts
        self.last_usage: dict[str, int] | None = None
        self._client = client if client is not None else self._build_client()

    @classmethod
    def from_env(cls, **kwargs: Any) -> "OllamaLLM":
        return cls(
            model=os.getenv("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL),
            base_url=os.getenv("OLLAMA_BASE_URL", DEFAULT_OLLAMA_BASE_URL),
            api_key=os.getenv("OLLAMA_API_KEY", DEFAULT_OLLAMA_API_KEY),
            **kwargs,
        )

    def _build_client(self) -> Any:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise LLMConfigurationError(
                "The OpenAI SDK is not installed. Run `pip install -e .` first."
            ) from exc
        return OpenAI(base_url=self.base_url, api_key=self.api_key)

    def generate(
        self,
        schema: type[T],
        prompt: str,
        *,
        system_prompt: str | None = None,
    ) -> T:
        """Generate JSON, validate it in Python, and repair invalid local-model output.

        The Ollama adapter intentionally does *not* send the full Pydantic model as a
        constrained-decoding grammar. Complex Agent schemas (notably EngineeringTurn)
        can exceed what some Ollama/llama.cpp grammar paths accept. Instead Ollama is
        asked for generic JSON mode, while Pydantic remains the authoritative schema
        validator before any Agent action can execute.
        """

        if not isinstance(schema, type) or not issubclass(schema, BaseModel):
            raise TypeError("schema must be a Pydantic BaseModel class.")

        normalized_prompt = prompt.strip()
        if not normalized_prompt:
            raise ValueError("prompt must not be empty.")

        effective_system_prompt = (
            self.default_system_prompt if system_prompt is None else system_prompt.strip()
        )
        schema_json = json.dumps(
            schema.model_json_schema(),
            ensure_ascii=False,
            separators=(",", ":"),
        )
        output_contract = (
            "\n\nOUTPUT CONTRACT FOR LOCAL JSON MODE:\n"
            "Return exactly one JSON object and no Markdown/code fences or commentary. "
            "The JSON must validate against the following schema. This schema is guidance "
            "for generation only; Python/Pydantic performs the authoritative validation "
            "after the response:\n"
            f"{schema_json}"
        )

        messages: list[dict[str, str]] = []
        if effective_system_prompt:
            messages.append({"role": "system", "content": effective_system_prompt})
        messages.append({"role": "user", "content": normalized_prompt + output_contract})

        self.last_usage = None
        cumulative_usage: dict[str, int] = {}
        last_error = "The model did not return valid structured JSON."
        last_content = ""
        max_attempts = 1 + self.structured_repair_attempts

        for attempt in range(max_attempts):
            request: dict[str, Any] = {
                "model": self.model,
                "messages": messages,
                "response_format": {"type": "json_object"},
                "temperature": 0,
            }
            if self.max_output_tokens is not None:
                # Ollama's OpenAI-compatible /v1/chat/completions endpoint documents max_tokens.
                request["max_tokens"] = self.max_output_tokens

            try:
                completion = self._client.chat.completions.create(**request)
            except Exception as exc:
                if _looks_like_connection_error(exc):
                    raise OllamaConnectionError(_connection_message(self.base_url)) from exc
                raise

            cumulative_usage = _add_usage(cumulative_usage, _chat_completion_usage(completion))
            self.last_usage = cumulative_usage or None

            choices = getattr(completion, "choices", None) or []
            message = getattr(choices[0], "message", None) if choices else None
            content = getattr(message, "content", None) if message is not None else None
            refusal = getattr(message, "refusal", None) if message is not None else None
            last_content = content if isinstance(content, str) else ""

            if not isinstance(content, str) or not content.strip():
                last_error = "Response content was empty or missing."
                if refusal:
                    last_error += f" refusal={refusal!r}"
            else:
                try:
                    return schema.model_validate_json(content)
                except Exception as exc:
                    last_error = _structured_validation_error(exc)

            if attempt >= max_attempts - 1:
                break

            messages.append(
                {
                    "role": "assistant",
                    "content": last_content or "{}",
                }
            )
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "Your previous JSON failed deterministic Python/Pydantic validation. "
                        "Correct the JSON only; do not explain the correction and do not use "
                        "Markdown. Preserve the intended engineering decision unless the "
                        "validation error requires changing it.\n\n"
                        f"VALIDATION ERROR:\n{last_error}"
                    ),
                }
            )

        raise StructuredOutputError(
            f"Ollama model {self.model!r} failed Python/Pydantic structured-output "
            f"validation after {max_attempts} attempt(s): {last_error}"
        )


def _structured_validation_error(exc: Exception, *, limit: int = 4000) -> str:
    text = str(exc).strip() or type(exc).__name__
    if len(text) > limit:
        return text[:limit] + "...<truncated>"
    return text


def _add_usage(
    total: dict[str, int],
    usage: dict[str, int] | None,
) -> dict[str, int]:
    if not usage:
        return dict(total)
    merged = dict(total)
    for key, value in usage.items():
        if isinstance(value, int) and value >= 0:
            merged[key] = merged.get(key, 0) + value
    return merged


def _looks_like_connection_error(exc: Exception) -> bool:
    names = {
        "APIConnectionError",
        "APITimeoutError",
        "ConnectError",
        "ConnectTimeout",
        "ReadTimeout",
        "TimeoutException",
    }
    return type(exc).__name__ in names or isinstance(exc, (ConnectionError, TimeoutError, OSError))


def _chat_completion_usage(completion: Any) -> dict[str, int] | None:
    usage = getattr(completion, "usage", None)
    if usage is None:
        return None

    def value(name: str) -> int | None:
        raw = usage.get(name) if isinstance(usage, dict) else getattr(usage, name, None)
        return raw if isinstance(raw, int) and raw >= 0 else None

    mapped = {
        "inputTokens": value("prompt_tokens"),
        "outputTokens": value("completion_tokens"),
        "totalTokens": value("total_tokens"),
    }
    return {key: item for key, item in mapped.items() if item is not None} or None
