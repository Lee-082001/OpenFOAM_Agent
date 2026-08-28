from __future__ import annotations

import os
from typing import Any, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


DEFAULT_SYSTEM_PROMPT = (
    "You are a CFD requirements engineer. Extract only information supported by "
    "the user's request. Represent unknown or omitted information explicitly using "
    "the supplied schema; do not invent physical values."
)


class LLMConfigurationError(RuntimeError):
    """Raised when the OpenAI LLM client cannot be configured."""


class StructuredOutputError(RuntimeError):
    """Raised when a response does not contain the requested parsed object."""


class StructuredOutputSchemaError(ValueError):
    """Raised when a Pydantic schema cannot be represented in strict JSON Schema."""


def _dynamic_object_paths(value: object, path: str = "$") -> list[str]:
    """Find arbitrary-key mappings unsupported by strict Structured Outputs."""

    if isinstance(value, dict):
        violations: list[str] = []
        additional = value.get("additionalProperties")
        if additional is True or isinstance(additional, dict):
            violations.append(path)
        for key, child in value.items():
            violations.extend(_dynamic_object_paths(child, f"{path}.{key}"))
        return violations
    if isinstance(value, list):
        violations = []
        for index, child in enumerate(value):
            violations.extend(_dynamic_object_paths(child, f"{path}[{index}]"))
        return violations
    return []


def _keyword_paths(value: object, keyword: str, path: str = "$") -> list[str]:
    """Return JSON-schema paths containing a specific keyword."""

    if isinstance(value, dict):
        violations: list[str] = []
        if keyword in value:
            violations.append(path)
        for key, child in value.items():
            violations.extend(_keyword_paths(child, keyword, f"{path}.{key}"))
        return violations
    if isinstance(value, list):
        violations = []
        for index, child in enumerate(value):
            violations.extend(_keyword_paths(child, keyword, f"{path}[{index}]"))
        return violations
    return []


def validate_structured_output_schema(schema: type[BaseModel]) -> None:
    """Fail locally for schema constructs rejected by strict Structured Outputs."""

    json_schema = schema.model_json_schema()
    dynamic_paths = _dynamic_object_paths(json_schema)
    if dynamic_paths:
        joined = ", ".join(dynamic_paths)
        raise StructuredOutputSchemaError(
            f"{schema.__name__} contains arbitrary-key object fields at {joined}. "
            "Strict Structured Outputs requires fixed object keys; model these "
            "fields as lists of key/value objects instead."
        )

    one_of_paths = _keyword_paths(json_schema, "oneOf")
    if one_of_paths:
        joined = ", ".join(one_of_paths)
        raise StructuredOutputSchemaError(
            f"{schema.__name__} contains unsupported `oneOf` at {joined}. "
            "Use a plain Union that emits nested `anyOf` instead of a "
            "discriminated union."
        )

    # Structured Outputs requires the root schema to be an object. Nested anyOf
    # is supported, but a root anyOf is not.
    if "anyOf" in json_schema or json_schema.get("type") != "object":
        raise StructuredOutputSchemaError(
            f"{schema.__name__} must have an object root without root-level `anyOf`."
        )


class OpenAILLM:
    """Structured-output adapter for the OpenAI Responses API.

    The adapter deliberately owns only model I/O. CFD engineering decisions belong to
    CFDEngineeringAgent; deterministic Python owns workflow permissions, sandboxing,
    bounded execution and evidence validation.
    """

    def __init__(
        self,
        *,
        model: str,
        client: Any | None = None,
        default_system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        store: bool = False,
        max_output_tokens: int | None = None,
    ) -> None:
        normalized_model = model.strip()
        if not normalized_model:
            raise LLMConfigurationError("An OpenAI model name is required.")
        if max_output_tokens is not None and max_output_tokens <= 0:
            raise LLMConfigurationError("max_output_tokens must be positive when set.")

        self.model = normalized_model
        self.default_system_prompt = default_system_prompt.strip()
        self.store = store
        self.max_output_tokens = max_output_tokens
        self._client = client if client is not None else self._build_client()

    @classmethod
    def from_env(cls, **kwargs: Any) -> "OpenAILLM":
        """Create a client using OPENAI_MODEL and the SDK's OPENAI_API_KEY support."""

        model = os.getenv("OPENAI_MODEL", "").strip()
        if not model:
            raise LLMConfigurationError(
                "OPENAI_MODEL is not set. Choose an available model explicitly."
            )
        return cls(model=model, **kwargs)

    @staticmethod
    def _build_client() -> Any:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise LLMConfigurationError(
                "The OpenAI SDK is not installed. Run `pip install -e .` first."
            ) from exc
        return OpenAI()

    def generate(
        self,
        schema: type[T],
        prompt: str,
        *,
        system_prompt: str | None = None,
    ) -> T:
        """Generate and validate one Pydantic object from a user prompt."""

        if not isinstance(schema, type) or not issubclass(schema, BaseModel):
            raise TypeError("schema must be a Pydantic BaseModel class.")
        validate_structured_output_schema(schema)

        normalized_prompt = prompt.strip()
        if not normalized_prompt:
            raise ValueError("prompt must not be empty.")

        effective_system_prompt = (
            self.default_system_prompt if system_prompt is None else system_prompt.strip()
        )
        input_messages: list[dict[str, str]] = []
        if effective_system_prompt:
            input_messages.append({"role": "system", "content": effective_system_prompt})
        input_messages.append({"role": "user", "content": normalized_prompt})

        request: dict[str, Any] = {
            "model": self.model,
            "input": input_messages,
            "text_format": schema,
            "store": self.store,
        }
        if self.max_output_tokens is not None:
            request["max_output_tokens"] = self.max_output_tokens

        response = self._client.responses.parse(**request)
        parsed = getattr(response, "output_parsed", None)
        if parsed is None:
            response_id = getattr(response, "id", None)
            suffix = f" (response_id={response_id})" if response_id else ""
            raise StructuredOutputError(
                "The model response had no parsed structured output; it may have "
                f"been refused or incomplete{suffix}."
            )

        if isinstance(parsed, schema):
            return parsed
        return schema.model_validate(parsed)
