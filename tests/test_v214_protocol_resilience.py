from __future__ import annotations

import json

from pydantic import BaseModel, ConfigDict, Field

from openfoam_agent.engineering.agent import CFDEngineeringAgent, EngineeringPolicy
from openfoam_agent.llm.openai_client import OpenAILLM
from openfoam_agent.schemas.engineering import (
    EvidenceGapRequest,
    GatherEvidenceAction,
    RepairCasePlanAction,
    RuntimeCaseRepairAction,
    PrepareTurn,
)
from openfoam_agent.tools.references import OpenFOAMReferenceIndex

from conftest import FakeOpenFOAMTools, make_plan, make_state


class EmptyLLM:
    store = False

    def generate(self, *args, **kwargs):  # pragma: no cover - should not be called
        raise AssertionError("LLM should not be called in this test")


def test_evidence_gap_normalizes_soft_protocol_fields_instead_of_failing_validation():
    gap = EvidenceGapRequest.model_validate(
        {
            "gap_id": "gap-1812",
            "missing_evidence": " exact OpenFOAM syntax " + ("x" * 600),
            "why_required": " required " + ("y" * 600),
            "capability_queries": ["   "],
            "reference_queries": ["  "],
            "reference_scope": "unexpected",
            "read_top_reference_matches": 99,
        }
    )

    assert gap.gap_id == "G1812"
    assert len(gap.missing_evidence) <= 400
    assert len(gap.why_required) <= 400
    assert gap.capability_queries == []
    assert len(gap.reference_queries) == 1
    assert gap.reference_queries[0]
    assert len(gap.reference_queries[0]) <= 500
    assert gap.reference_scope == "all"
    assert gap.read_top_reference_matches == 2


def test_prepare_turn_accepts_overlong_or_blank_evidence_queries_after_normalization():
    turn = PrepareTurn.model_validate(
        {
            "action": {
                "type": "gather_evidence",
                "gaps": [
                    {
                        "gap_id": "G1812",
                        "missing_evidence": "Exact release syntax",
                        "why_required": "Needed before authoring",
                        "capability_queries": ["   "],
                        "reference_queries": ["q" * 1400],
                        "read_top_reference_matches": 1,
                    }
                ],
            }
        }
    )
    gap = turn.action.gaps[0]
    assert gap.reference_queries
    assert len(gap.reference_queries[0]) == 500


def test_gather_evidence_merges_duplicate_gap_ids_as_protocol_noise():
    action = GatherEvidenceAction(
        type="gather_evidence",
        gaps=[
            EvidenceGapRequest(
                gap_id="G01",
                missing_evidence="release syntax",
                why_required="needed",
                reference_queries=["query one"],
            ),
            EvidenceGapRequest(
                gap_id="G01",
                missing_evidence="release syntax",
                why_required="needed",
                reference_queries=["query two"],
            ),
        ],
    )
    assert len(action.gaps) == 1
    assert action.gaps[0].reference_queries == ["query one", "query two"]


def test_evidence_gap_lifecycle_closes_on_proceed_and_blocks_refining_satisfied_gap(tmp_path, graph_path):
    refs = tmp_path / "refs"
    refs.mkdir()
    (refs / "fvSchemes.example").write_text(
        "divSchemes { div(phi,U) Gauss linear; }\n", encoding="utf-8"
    )
    agent = CFDEngineeringAgent(
        EmptyLLM(),
        workspace=tmp_path / "case",
        capability_db=graph_path,
        tools=FakeOpenFOAMTools(),
        policy=EngineeringPolicy(max_prepare_retrieval_cycles=3, compact_phase_schemas=True),
    )
    agent.references = OpenFOAMReferenceIndex({"tutorials": refs})
    state = make_state()

    first = GatherEvidenceAction(
        type="gather_evidence",
        gaps=[
            EvidenceGapRequest(
                gap_id="G01",
                missing_evidence="Exact div scheme syntax.",
                why_required="Needed for release-specific dictionary syntax.",
                reference_queries=["div(phi,U)"],
                reference_scope="tutorials",
            )
        ],
    )
    event = agent._dispatch_tool_action(first, step=1, native_execution=False, phase="prepare", state=state)
    assert json.loads(event.output_excerpt)["gaps"][0]["status"] == "new_evidence"

    agent._mark_evidence_gaps_satisfied("prepare")
    status = agent._evidence_gap_status("prepare")[0]
    assert status["status"] == "satisfied"
    assert status["satisfied"] is True

    refined = GatherEvidenceAction(
        type="gather_evidence",
        gaps=[
            EvidenceGapRequest(
                gap_id="G02",
                refines_gap_id="G01",
                missing_evidence="More syntax.",
                why_required="Still needed.",
                reference_queries=["more syntax"],
            )
        ],
    )
    before = agent._retrieval_cycles["prepare"]
    blocked = agent._dispatch_tool_action(refined, step=2, native_execution=False, phase="prepare", state=state)
    payload = json.loads(blocked.output_excerpt)
    assert payload["gaps"][0]["status"] == "satisfied_parent_blocked"
    assert agent._retrieval_cycles["prepare"] == before


def test_runtime_noop_repair_is_controlled_outcome_not_pydantic_failure(tmp_path, graph_path):
    state = make_state()
    state.engineering_plan = make_plan(state.intake)
    agent = CFDEngineeringAgent(
        EmptyLLM(),
        workspace=tmp_path,
        capability_db=graph_path,
        tools=FakeOpenFOAMTools(),
        policy=EngineeringPolicy(compact_phase_schemas=True),
    )
    repair = RuntimeCaseRepairAction(
        type="repair_runtime_case",
        diagnosis="No concrete delta was supplied.",
        retry_solver=True,
    )
    outcome = agent._execute_runtime_repair_plan(
        state,
        repair,
        approved_solver=state.engineering_plan.solver,
        llm_step=1,
        native_execution=False,
        runtime_event_start=0,
    )
    assert outcome is not None
    assert outcome.retry is False
    assert "no case-file change" in outcome.reason


def test_prepare_noop_repair_is_controlled_event_not_schema_failure(tmp_path, graph_path):
    state = make_state()
    state.engineering_plan = make_plan(state.intake)
    agent = CFDEngineeringAgent(
        EmptyLLM(),
        workspace=tmp_path,
        capability_db=graph_path,
        tools=FakeOpenFOAMTools(),
        policy=EngineeringPolicy(compact_phase_schemas=True),
    )
    repair = RepairCasePlanAction(
        type="repair_case_plan",
        diagnosis="No concrete delta was supplied.",
        validate_pre_solve=False,
    )
    terminal = agent._execute_prepare_repair_plan(
        state,
        repair,
        llm_step=1,
        progress_phase="engineering",
        native_execution=False,
    )
    assert terminal is False
    assert state.engineering_events[-1].success is False
    assert "no artifact or EngineeringPlan change" in state.engineering_events[-1].summary


def test_openai_adapter_repairs_one_pydantic_protocol_failure_before_giving_up():
    class TinyResponse(BaseModel):
        model_config = ConfigDict(extra="forbid")
        value: str = Field(min_length=1, max_length=8)

    class FakeResponses:
        def __init__(self):
            self.requests = []

        def parse(self, **request):
            self.requests.append(request)
            if len(self.requests) == 1:
                # Simulate the same class of Responses.parse/Pydantic validation error
                # seen in production logs before the workflow got a chance to recover.
                TinyResponse.model_validate({"value": ""})
            return type(
                "Response",
                (),
                {"id": "resp_ok", "output_parsed": TinyResponse(value="fixed"), "usage": None},
            )()

    class FakeClient:
        def __init__(self):
            self.responses = FakeResponses()

    client = FakeClient()
    llm = OpenAILLM(model="gpt-5.6-sol", client=client, structured_repair_attempts=1)
    result = llm.generate(TinyResponse, "Return a value")

    assert result.value == "fixed"
    assert len(client.responses.requests) == 2
    correction = client.responses.requests[1]["input"][-1]["content"]
    assert "failed deterministic Pydantic validation" in correction
    assert "confirmed CFD facts" in correction
