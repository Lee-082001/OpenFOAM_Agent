from __future__ import annotations

import json
from types import SimpleNamespace

from pydantic import BaseModel

from openfoam_agent.llm.claude_client import ClaudeCLIStatus, ClaudeLLM
from openfoam_agent.llm.codex_client import CodexCLIStatus, CodexLLM
from openfoam_agent.llm.structured_schema import compile_transport_schema
from openfoam_agent.schemas.engineering import EngineeringTurn
from openfoam_agent.schemas.intake import CFDIntakeSpec


def _walk(value: object):
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from _walk(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk(child)


def test_codex_compiler_makes_every_object_property_required_and_removes_defaults():
    canonical = CFDIntakeSpec.model_json_schema()
    strict = compile_transport_schema(CFDIntakeSpec, backend="codex")

    # Canonical Pydantic omission/default semantics are not mutated.
    assert "semantic_contract_version" not in canonical["required"]
    assert "suggested_default" not in canonical["$defs"]["BlockingUnknown"]["required"]

    # Codex/OpenAI strict response formats require all properties to be present.
    for node in _walk(strict):
        properties = node.get("properties")
        if isinstance(properties, dict):
            assert node.get("required") == list(properties.keys())
            assert node.get("additionalProperties") is False
        assert "default" not in node

    blocking = strict["$defs"]["BlockingUnknown"]
    assert "suggested_default" in blocking["required"]
    assert {entry.get("type") for entry in blocking["properties"]["suggested_default"]["anyOf"]} == {
        "string",
        "null",
    }


def test_claude_compiler_rewrites_pydantic_prefix_items_without_changing_fixed_lengths():
    canonical = EngineeringTurn.model_json_schema()
    compiled = compile_transport_schema(EngineeringTurn, backend="claude")

    assert any("prefixItems" in node for node in _walk(canonical))
    assert not any("prefixItems" in node for node in _walk(compiled))

    vertex = compiled["$defs"]["BlockMeshVertex"]["properties"]["coordinates"]
    assert vertex["items"]["type"] == "number"
    assert vertex["minItems"] == 3
    assert vertex["maxItems"] == 3

    block_vertices = compiled["$defs"]["BlockMeshBlock"]["properties"]["vertices"]
    assert block_vertices["items"]["type"] == "integer"
    assert block_vertices["minItems"] == 8
    assert block_vertices["maxItems"] == 8


def test_codex_compiler_also_normalizes_engineering_tuples():
    compiled = compile_transport_schema(EngineeringTurn, backend="codex")
    assert not any("prefixItems" in node for node in _walk(compiled))
    for node in _walk(compiled):
        properties = node.get("properties")
        if isinstance(properties, dict):
            assert node.get("required") == list(properties.keys())
            assert node.get("additionalProperties") is False
        assert "default" not in node


class _TupleOutput(BaseModel):
    point: tuple[float, float, float]
    note: str | None = None


def test_claude_client_passes_compiled_schema_to_cli(monkeypatch):
    import openfoam_agent.llm.claude_client as claude

    captured: dict[str, object] = {}

    def fake_run(command, **kwargs):
        captured["schema"] = json.loads(command[command.index("--json-schema") + 1])
        return SimpleNamespace(
            returncode=0,
            stdout=json.dumps(
                {"structured_output": {"point": [1.0, 2.0, 3.0], "note": None}}
            ),
            stderr="",
        )

    monkeypatch.setattr(claude.subprocess, "run", fake_run)
    status = ClaudeCLIStatus(
        binary="/usr/bin/claude",
        version="2.1.221",
        auth_method="claude.ai",
        subscription_type="max",
        api_provider="firstParty",
        supports_safe_mode=True,
    )
    result = ClaudeLLM(status=status).generate(_TupleOutput, "Return one point.")
    assert result.point == (1.0, 2.0, 3.0)
    sent = captured["schema"]
    assert isinstance(sent, dict)
    assert not any("prefixItems" in node for node in _walk(sent))


def test_codex_client_writes_openai_strict_compiled_schema(monkeypatch):
    import openfoam_agent.llm.codex_client as codex

    captured: dict[str, object] = {}

    def fake_run(command, **kwargs):
        schema_path = command[command.index("--output-schema") + 1]
        with open(schema_path, encoding="utf-8") as handle:
            captured["schema"] = json.load(handle)
        output_path = command[command.index("--output-last-message") + 1]
        with open(output_path, "w", encoding="utf-8") as handle:
            json.dump({"point": [1.0, 2.0, 3.0], "note": None}, handle)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(codex.subprocess, "run", fake_run)
    status = CodexCLIStatus(
        binary="/usr/bin/codex",
        version="codex-cli 0.153.0",
        login_status="Logged in using ChatGPT",
        supports_ignore_user_config=True,
    )
    result = CodexLLM(status=status).generate(_TupleOutput, "Return one point.")
    assert result.point == (1.0, 2.0, 3.0)
    sent = captured["schema"]
    assert isinstance(sent, dict)
    assert sent["required"] == ["point", "note"]
    assert sent["additionalProperties"] is False
    assert "default" not in sent["properties"]["note"]
    assert not any("prefixItems" in node for node in _walk(sent))
