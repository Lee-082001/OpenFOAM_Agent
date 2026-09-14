from __future__ import annotations

from types import SimpleNamespace

import pytest
from pydantic import BaseModel, ConfigDict

from openfoam_agent.contracts.models import RegionCaseLayout
from openfoam_agent.contracts.regions import region_layouts
from openfoam_agent.llm.codex_client import CodexLLM
from openfoam_agent.llm.codex_transport import (
    CodexTransportParseError,
    CodexTransportViolation,
    inspect_events,
)
from openfoam_agent.llm.openai_client import StructuredOutputError


def _plan(layout_solver: str | None = "stale"):
    assignment = SimpleNamespace(region="battery", solver_module="solid")
    execution = SimpleNamespace(regions=[assignment])
    return SimpleNamespace(
        required_case_files=["0/battery/T", "system/battery/fvSchemes"],
        execution=execution,
        region_layouts=[
            RegionCaseLayout(region="battery", solver_module=layout_solver, required_fields=["T"])
        ],
    )


def test_region_layout_solver_mirror_is_not_llm_schema_authority():
    schema = RegionCaseLayout.model_json_schema()
    assert "solver_module" not in schema.get("properties", {})


def test_region_layout_solver_mirror_is_canonicalized_from_execution():
    [layout] = region_layouts(_plan())
    assert layout.region == "battery"
    assert layout.solver_module == "solid"


def test_region_name_disagreement_still_fails_closed():
    plan = _plan()
    plan.execution = SimpleNamespace(
        regions=[SimpleNamespace(region="heater", solver_module="solid")]
    )
    with pytest.raises(ValueError, match="do not exactly match"):
        region_layouts(plan)


def test_non_json_codex_event_is_retryable_parse_class():
    with pytest.raises(CodexTransportParseError):
        inspect_events("not-json")


def test_tool_event_stays_fail_closed_not_retryable_parse_error():
    line = '{"type":"item.completed","item":{"type":"command_execution","command":"pwd"}}'
    with pytest.raises(CodexTransportViolation):
        inspect_events(line)


class _Reply(BaseModel):
    model_config = ConfigDict(extra="forbid")
    value: int


def test_codex_generate_retries_one_transport_parse_corruption(monkeypatch):
    llm = object.__new__(CodexLLM)
    llm.structured_repair_attempts = 0
    llm.transport_retries = 1
    llm.model = "test-codex"
    llm.last_usage = None
    calls = {"count": 0}

    def fake_run_once(schema, prompt):
        calls["count"] += 1
        if calls["count"] == 1:
            try:
                raise CodexTransportParseError("corrupt jsonl")
            except CodexTransportParseError as cause:
                raise StructuredOutputError(str(cause)) from cause
        return '{"value":7}'

    monkeypatch.setattr(llm, "_run_once", fake_run_once)
    result = llm.generate(_Reply, "return a value", system_prompt="")
    assert result.value == 7
    assert calls["count"] == 2
    assert llm.last_transport_retry_count == 1
