from __future__ import annotations

import json

from openfoam_agent.engineering.agent import CFDEngineeringAgent, EngineeringPolicy
from openfoam_agent.llm.prompts.engineering import PREPARE_SYSTEM_PROMPT
from openfoam_agent.llm.structured_schema import compile_transport_schema
from openfoam_agent.schemas.engineering import (
    BlockAction,
    EngineeringDefaultAssumption,
    EvidenceGapRequest,
    GatherEvidenceAction,
    PrepareTurn,
)
from openfoam_agent.workflow.states import State

from conftest import FakeOpenFOAMTools, make_plan, make_state


class EmptyLLM:
    store = False

    def generate(self, *args, **kwargs):  # pragma: no cover
        raise AssertionError("LLM should not be called")


def _agent(tmp_path, graph_path) -> CFDEngineeringAgent:
    return CFDEngineeringAgent(
        EmptyLLM(),
        workspace=tmp_path / "case",
        capability_db=graph_path,
        tools=FakeOpenFOAMTools(),
        policy=EngineeringPolicy(
            compact_phase_schemas=True,
            max_prepare_retrieval_cycles=3,
            max_observation_chars=12_000,
        ),
    )


def test_large_evidence_payload_is_stored_outside_engineering_event(tmp_path, graph_path):
    agent = _agent(tmp_path, graph_path)
    state = make_state()

    def huge_search(query: str, limit: int = 8):
        del query
        return [
            {
                "provider_id": f"solver.synthetic{i}",
                "name": f"synthetic{i}",
                "provider_type": "solver",
                "openfoam_version": "14",
                "capabilities": ["x" * 5000, "y" * 5000],
            }
            for i in range(limit)
        ]

    agent.catalog.search = huge_search  # type: ignore[method-assign]
    action = GatherEvidenceAction(
        type="gather_evidence",
        gaps=[
            EvidenceGapRequest(
                gap_id="G01",
                missing_evidence="Synthetic capability evidence.",
                why_required="Regression test for large structured retrieval payloads.",
                capability_queries=["synthetic"],
                reference_queries=[],
                read_top_reference_matches=0,
            )
        ],
    )
    event = agent._dispatch_tool_action(
        action, step=1, native_execution=False, phase="prepare", state=state
    )

    assert event.success
    assert len(event.output_excerpt) < 12_000
    compact = json.loads(event.output_excerpt)
    assert compact["gaps"][0]["status"] == "new_evidence"
    assert "new_evidence" not in compact["gaps"][0]
    assert event.payload_ref is not None
    assert len(state.engineering_evidence_records) == 1
    record = state.engineering_evidence_records[0]
    assert record.record_id == event.payload_ref
    assert len(json.dumps(record.payload)) > 50_000


def test_event_truncation_never_exceeds_pydantic_limit(tmp_path, graph_path):
    agent = _agent(tmp_path, graph_path)
    event = agent._event(1, "test", True, "ok", "x" * 80_000)
    assert len(event.output_excerpt) <= 12_000
    assert "model-context compacted" in event.output_excerpt


def test_evidence_infrastructure_failure_disables_retrieval_and_escalates_phase(tmp_path, graph_path):
    agent = _agent(tmp_path, graph_path)
    state = make_state()
    action = GatherEvidenceAction(
        type="gather_evidence",
        gaps=[
            EvidenceGapRequest(
                gap_id="G01",
                missing_evidence="Need one reference.",
                why_required="Exercise deterministic recovery.",
                reference_queries=["fvSchemes"],
            )
        ],
    )

    def fail_store(*args, **kwargs):
        raise ValueError("synthetic evidence store failure")

    agent._store_evidence_payload = fail_store  # type: ignore[method-assign]
    first = agent._dispatch_tool_action(
        action, step=1, native_execution=False, phase="prepare", state=state
    )
    assert not first.success
    assert first.failure_signature == "evidence_retrieval:prepare:infrastructure"
    assert "synthetic evidence store failure" in first.summary
    assert "prepare" in agent._evidence_retrieval_disabled
    assert agent._retrieval_cycles["prepare"] == agent.policy.max_prepare_retrieval_cycles
    assert agent._phase_contract(state, "prepare")[2] == "prepare_decide"

    second = agent._dispatch_tool_action(
        action, step=2, native_execution=False, phase="prepare", state=state
    )
    assert not second.success
    assert second.failure_signature == "evidence_retrieval:prepare:disabled"


def test_authorized_engineering_defaults_have_explicit_provenance(tmp_path, graph_path):
    agent = _agent(tmp_path, graph_path)
    state = make_state()
    default = EngineeringDefaultAssumption(
        parameter="battery diameter",
        value="21",
        unit="mm",
        basis="common_practice",
        rationale="Representative cylindrical-cell geometry selected for an exploratory case.",
    )
    plan = make_plan(state.intake).model_copy(update={"engineering_defaults": [default]})

    assert agent._validate_engineering_defaults(plan, state) == []
    dumped = plan.model_dump(mode="json")["engineering_defaults"][0]
    assert dumped["source"] == "engineering_default"

    unauthorized = state.model_copy(
        update={
            "user_request": state.user_request.model_copy(
                update={"exploratory_completion_authorized": False}
            )
        }
    )
    failures = agent._validate_engineering_defaults(plan, unauthorized)
    assert any("not authorized" in item for item in failures)


def test_delegated_engineering_choice_block_is_rejected_when_assumptions_authorized(tmp_path, graph_path):
    agent = _agent(tmp_path, graph_path)
    state = make_state()
    state.transition(State.ENGINEERING, "test")
    event, terminal = agent._dispatch_prepare(
        state,
        BlockAction(
            type="block",
            reason="Battery dimensions and inlet temperature are missing.",
            block_kind="engineering_choice_missing",
            missing_items=["battery dimensions", "inlet temperature"],
            needs_user_input=True,
        ),
        step=1,
        native_execution=False,
    )
    assert terminal is False
    assert not event.success
    assert state.current_state == State.ENGINEERING
    assert "engineering_defaults" in event.summary


def test_prepare_contract_tells_model_to_use_defaults_and_compiles_for_cli_backends():
    assert "EngineeringPlan.engineering_defaults" in PREPARE_SYSTEM_PROMPT
    assert "source=engineering_default" in PREPARE_SYSTEM_PROMPT
    for backend in ("codex", "claude"):
        compiled = compile_transport_schema(PrepareTurn, backend=backend)
        encoded = json.dumps(compiled)
        assert "engineering_defaults" in encoded
        assert "block_kind" in encoded
