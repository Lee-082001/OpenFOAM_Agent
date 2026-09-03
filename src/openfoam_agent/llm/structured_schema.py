from __future__ import annotations

import copy
from typing import Any, Literal

from pydantic import BaseModel


TransportBackend = Literal["canonical", "claude", "codex"]


class StructuredOutputSchemaError(ValueError):
    """Raised when a Pydantic schema cannot be represented safely for a backend."""


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
        violations: list[str] = []
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
        violations: list[str] = []
        for index, child in enumerate(value):
            violations.extend(_keyword_paths(child, keyword, f"{path}[{index}]"))
        return violations
    return []


def validate_structured_output_schema(schema: type[BaseModel]) -> None:
    """Fail locally for canonical constructs unsupported by strict transports.

    This validates the Pydantic/domain schema, not a specific CLI dialect. Backend
    compatibility rewrites are performed separately by ``compile_transport_schema``.
    """

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


def _dedupe_schemas(values: list[object]) -> list[object]:
    unique: list[object] = []
    for value in values:
        if value not in unique:
            unique.append(value)
    return unique


def _tuple_items_superset(prefix_items: list[object]) -> object:
    """Return a backend-portable item schema for a 2020-12 tuple.

    Homogeneous tuples preserve the transport constraint exactly. Heterogeneous
    tuples are widened to an ``anyOf`` item schema because the target CLI dialects
    do not reliably support positional tuple keywords. Final Pydantic validation
    remains authoritative and therefore prevents the widened transport schema from
    accepting an invalid domain object.
    """

    if not prefix_items:
        return {}
    unique = _dedupe_schemas(prefix_items)
    if len(unique) == 1:
        return copy.deepcopy(unique[0])
    return {"anyOf": copy.deepcopy(unique)}


def _normalize_prefix_items(value: object) -> None:
    if isinstance(value, list):
        for child in value:
            _normalize_prefix_items(child)
        return
    if not isinstance(value, dict):
        return

    prefix = value.pop("prefixItems", None)
    if isinstance(prefix, list):
        # Pydantic fixed tuples include minItems/maxItems. Preserve those exact
        # cardinality constraints and replace only the positional keyword.
        replacement = _tuple_items_superset(prefix)
        existing_items = value.get("items")
        if existing_items is None or existing_items is False:
            value["items"] = replacement
        elif existing_items is True:
            value["items"] = replacement
        elif isinstance(existing_items, dict):
            # A general 2020-12 schema may combine prefixItems with trailing items.
            # Use a safe transport superset and rely on final Pydantic validation.
            value["items"] = {
                "anyOf": _dedupe_schemas([replacement, copy.deepcopy(existing_items)])
            }
        # If no cardinality was supplied, do not invent one: prefixItems alone does
        # not require every positional schema to be present.

    for child in value.values():
        _normalize_prefix_items(child)


def _compile_openai_strict(value: object) -> None:
    """Normalize a schema for Codex/OpenAI strict response-format requirements."""

    if isinstance(value, list):
        for child in value:
            _compile_openai_strict(child)
        return
    if not isinstance(value, dict):
        return

    # ``default`` describes omission behavior in normal JSON Schema. OpenAI strict
    # response formats require every declared property to be present, so defaults are
    # transport-inapplicable and can trigger unsupported-keyword rejection.
    value.pop("default", None)

    properties = value.get("properties")
    if isinstance(properties, dict):
        value["required"] = list(properties.keys())
        value["additionalProperties"] = False
    elif value.get("type") == "object":
        value.setdefault("additionalProperties", False)

    for child in value.values():
        _compile_openai_strict(child)


def compile_transport_schema(
    schema: type[BaseModel],
    *,
    backend: TransportBackend,
) -> dict[str, Any]:
    """Compile canonical Pydantic JSON Schema for a model transport.

    The canonical Pydantic model is never modified. CLI-specific rewrites only
    constrain model decoding; every response is still validated against the original
    Pydantic model after transport.
    """

    if not isinstance(schema, type) or not issubclass(schema, BaseModel):
        raise TypeError("schema must be a Pydantic BaseModel class.")
    validate_structured_output_schema(schema)
    compiled = copy.deepcopy(schema.model_json_schema())

    if backend == "canonical":
        return compiled
    if backend == "claude":
        _normalize_prefix_items(compiled)
        return compiled
    if backend == "codex":
        # Codex CLI forwards --output-schema as a strict OpenAI response-format
        # schema. Normalize tuple syntax first, then enforce strict object shape.
        _normalize_prefix_items(compiled)
        _compile_openai_strict(compiled)
        return compiled
    raise ValueError(f"Unsupported structured-output backend: {backend!r}")
