from __future__ import annotations

from types import SimpleNamespace

from openfoam_agent.engineering.agent import CFDEngineeringAgent
from openfoam_agent.schemas.engineering import (
    EngineeringEvent,
    EngineeringEvidenceRecord,
    ObservedEngineeringEvidence,
)


def _observed(n: int):
    return [
        ObservedEngineeringEvidence(
            evidence_id=f"ev_cap_{i:020x}"[-27:],
            kind="capability",
            reference=f"provider-{i:03d}",
            summary=f"provider {i}",
        )
        for i in range(n)
    ]


def test_durable_evidence_record_is_not_cardinality_bounded():
    observed = _observed(100)
    record = EngineeringEvidenceRecord(
        record_id="evrec_0123456789abcdefabcd",
        phase="prepare",
        step=1,
        action_type="gather_evidence",
        payload={"count": len(observed)},
        observed_evidence=observed,
    )
    assert len(record.observed_evidence) == 100


def test_engineering_event_is_bounded_projection_not_durable_store():
    agent = object.__new__(CFDEngineeringAgent)
    agent.policy = SimpleNamespace(max_observation_chars=12000)
    observed = _observed(100)

    event = agent._event(
        1,
        "gather_evidence",
        True,
        "retrieved evidence",
        "ok",
        payload_ref="evrec_0123456789abcdefabcd",
        observed_evidence=observed,
    )

    assert len(event.observed_evidence) == 24
    assert event.payload_ref == "evrec_0123456789abcdefabcd"
    assert [x.reference for x in event.observed_evidence] == [f"provider-{i:03d}" for i in range(24)]


def test_engineering_event_model_itself_is_fail_soft_for_projection_overflow():
    observed = _observed(100)
    event = EngineeringEvent(
        step=1,
        action_type="gather_evidence",
        success=True,
        summary="ok",
        observed_evidence=observed,
    )
    assert len(event.observed_evidence) == 24

from openfoam_agent.engineering.agent import EngineeringPolicy
from openfoam_agent.schemas.engineering import EvidenceGapRequest, GatherEvidenceAction
from conftest import FakeOpenFOAMTools, make_state


class _NoopLLM:
    store = False

    def generate(self, *args, **kwargs):  # pragma: no cover - not used by this test
        raise AssertionError("LLM should not be called")


def test_gather_evidence_large_batch_keeps_full_store_and_bounded_event(tmp_path, graph_path):
    agent = CFDEngineeringAgent(
        _NoopLLM(),
        workspace=tmp_path / "case",
        capability_db=graph_path,
        tools=FakeOpenFOAMTools(),
        policy=EngineeringPolicy(compact_phase_schemas=True),
    )
    providers = [
        {
            "provider_id": f"provider-{i:03d}",
            "name": f"Provider {i}",
            "provider_type": "utility",
            "openfoam_version": "14",
        }
        for i in range(100)
    ]
    agent.catalog.search = lambda query, limit=8: providers[: min(len(providers), max(limit, 100))]
    state = make_state()
    agent._evidence_gap_ledger["prepare"] = {}
    agent._retrieval_cycles["prepare"] = 0
    action = GatherEvidenceAction(
        type="gather_evidence",
        gaps=[
            EvidenceGapRequest(
                gap_id="G01",
                missing_evidence="Many valid installed providers.",
                why_required="Stress the evidence projection boundary.",
                capability_queries=["provider"],
                reference_queries=[],
                read_top_reference_matches=0,
            )
        ],
    )

    event = agent._dispatch_tool_action(
        action,
        step=1,
        native_execution=False,
        phase="prepare",
        state=state,
    )

    assert event.success
    assert len(event.observed_evidence) == 24
    assert len(state.engineering_evidence_records) == 1
    # GatherEvidence currently requests a bounded 8 providers per capability query;
    # the durable store still owns every item actually returned by the tool call.
    assert len(state.engineering_evidence_records[0].observed_evidence) == 100
