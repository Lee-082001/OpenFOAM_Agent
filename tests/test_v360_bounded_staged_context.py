from __future__ import annotations

from pathlib import Path

from conftest import FakeOpenFOAMTools, make_state, mesh_ok_log, tool_result
from test_v210_token_optimization import FlexibleScriptedLLM, _boundary_file, _compact_plan

from openfoam_agent.engineering.agent import CFDEngineeringAgent, EngineeringPolicy
from openfoam_agent.llm.context import structured_request_metrics
from openfoam_agent.llm.prompts.engineering import (
    CASE_AUTHORING_SYSTEM_PROMPT,
    PREPARE_DECISION_DESIGN_SYSTEM_PROMPT,
    PREPARE_DECISION_ONLY_SYSTEM_PROMPT,
)
from openfoam_agent.schemas.engineering import (
    CaseAuthoringAction,
    CaseAuthoringTurn,
    DesignCaseAction,
    EngineeringEvidenceRecord,
    ObservedEngineeringEvidence,
    PrepareDecisionDesignTurn,
    PrepareDecisionOnlyTurn,
    PrepareDesignTurn,
    canonical_engineering_evidence_id,
)
from openfoam_agent.workflow.states import State


def _staged_actions(state):
    legacy = _compact_plan(state)
    design = DesignCaseAction(
        type="design_case",
        plan=legacy.plan,
        authoring_brief="Implement the accepted simple single-region case.",
    )
    data = legacy.model_dump(mode="python", exclude={"type", "plan"})
    data["type"] = "author_case"
    author = CaseAuthoringAction.model_validate(data)
    return design, author


def test_staged_schemas_split_decision_from_large_case_authoring_contract():
    legacy = structured_request_metrics(
        PrepareDecisionOnlyTurn, "{}", system_prompt=PREPARE_DECISION_ONLY_SYSTEM_PROMPT
    )["approxTokens"]
    design = structured_request_metrics(
        PrepareDecisionDesignTurn, "{}", system_prompt=PREPARE_DECISION_DESIGN_SYSTEM_PROMPT
    )["approxTokens"]
    author = structured_request_metrics(
        CaseAuthoringTurn, "{}", system_prompt=CASE_AUTHORING_SYSTEM_PROMPT
    )["approxTokens"]

    assert design < legacy
    assert author < legacy
    assert design <= int(legacy * 0.75)
    assert author <= int(legacy * 0.6)


def test_staged_cli_style_flow_reaches_solve_ready_with_two_small_contracts(tmp_path, graph_path):
    state = make_state()
    design, author = _staged_actions(state)
    llm = FlexibleScriptedLLM([design, author])
    tools = FakeOpenFOAMTools(
        mesh_results={
            "blockMesh": [tool_result("blockMesh", success=True, stdout="ok\n")],
            "checkMesh": [tool_result("checkMesh", success=True, stdout=mesh_ok_log())],
        }
    )
    agent = CFDEngineeringAgent(
        llm,
        workspace=tmp_path,
        capability_db=graph_path,
        tools=tools,
        policy=EngineeringPolicy(
            max_agent_steps=4,
            hard_max_agent_steps=4,
            require_solve_ready_gate=True,
            preload_capabilities=True,
            max_preloaded_capabilities=24,
            compact_phase_schemas=True,
            state_delta_context=True,
            bounded_evidence_context=True,
            staged_case_authoring=True,
            max_model_prompt_chars=18_000,
            max_prepare_model_evidence_items=10,
            max_decide_model_evidence_items=12,
            max_model_evidence_detail_chars=600,
        ),
    )
    agent.workspace.write_text("constant/polyMesh/boundary", _boundary_file())

    agent.prepare(state, native_execution=True)

    assert state.current_state == State.SOLVE_READY
    assert llm.schemas == [PrepareDesignTurn, CaseAuthoringTurn]
    assert all(len(prompt) <= 18_000 for prompt in llm.prompts)
    assert '"state_mode": "bounded_engineering_design"' in llm.prompts[0]
    assert '"state_mode": "staged_case_authoring"' in llm.prompts[1]
    assert '"available_evidence"' not in llm.prompts[1]
    assert all(item["use_previous_response"] is False for item in llm.kwargs)


def test_bounded_evidence_projection_does_not_grow_with_durable_store(tmp_path, graph_path):
    state = make_state()
    llm = FlexibleScriptedLLM([])
    agent = CFDEngineeringAgent(
        llm,
        workspace=tmp_path,
        capability_db=graph_path,
        tools=FakeOpenFOAMTools(),
        policy=EngineeringPolicy(
            preload_capabilities=False,
            bounded_evidence_context=True,
            max_prepare_model_evidence_items=10,
            max_model_evidence_detail_chars=600,
        ),
    )

    observed = []
    for index in range(80):
        reference = f"provider.synthetic.{index:03d}"
        observed.append(
            ObservedEngineeringEvidence(
                evidence_id=canonical_engineering_evidence_id("capability", reference),
                kind="capability",
                reference=reference,
                summary=f"Synthetic deterministic capability evidence {index}",
            )
        )
    state.engineering_evidence_records.append(
        EngineeringEvidenceRecord(
            record_id="evrec_0123456789abcdefabcd",
            phase="prepare",
            step=1,
            action_type="gather_evidence",
            payload={"gaps": []},
            observed_evidence=observed,
        )
    )

    assert len(agent._observed_evidence_registry(state)) == 80
    capsule = agent._bounded_evidence_for_model(state, phase="prepare", max_items=10)
    assert len(capsule) == 10
    assert {item["evidence_id"] for item in capsule}.issubset(
        {item.evidence_id for item in observed}
    )


def test_two_retrieval_cycles_stay_bounded_and_switch_to_small_decision_schema(tmp_path, graph_path):
    from openfoam_agent.schemas.engineering import EvidenceGapRequest, GatherEvidenceAction

    state = make_state()
    legacy = _compact_plan(state)
    gather_one = GatherEvidenceAction(
        type="gather_evidence",
        gaps=[
            EvidenceGapRequest(
                gap_id="G1000",
                missing_evidence="single-region incompressible solver support",
                why_required="select a solver module",
                capability_queries=["internal flow"],
            ),
            EvidenceGapRequest(
                gap_id="G1001",
                missing_evidence="execution driver support",
                why_required="select the execution driver",
                capability_queries=["execution driver"],
            ),
        ],
    )
    gather_two = GatherEvidenceAction(
        type="gather_evidence",
        gaps=[
            EvidenceGapRequest(
                gap_id="G1002",
                missing_evidence="momentum transport model support",
                why_required="select turbulence treatment",
                capability_queries=["momentum transport"],
            ),
            EvidenceGapRequest(
                gap_id="G1003",
                missing_evidence="mesh utility support",
                why_required="construct the mesh",
                capability_queries=["utility mesh"],
            ),
        ],
    )
    design = DesignCaseAction(type="design_case", plan=legacy.plan, authoring_brief="simple")
    author_data = legacy.model_dump(mode="python", exclude={"type", "plan"})
    author_data["type"] = "author_case"
    author = CaseAuthoringAction.model_validate(author_data)
    llm = FlexibleScriptedLLM([gather_one, gather_two, design, author])
    tools = FakeOpenFOAMTools(
        mesh_results={
            "blockMesh": [tool_result("blockMesh", success=True, stdout="ok\n")],
            "checkMesh": [tool_result("checkMesh", success=True, stdout=mesh_ok_log())],
        }
    )
    agent = CFDEngineeringAgent(
        llm,
        workspace=tmp_path,
        capability_db=graph_path,
        tools=tools,
        policy=EngineeringPolicy(
            max_agent_steps=6,
            hard_max_agent_steps=6,
            require_solve_ready_gate=True,
            preload_capabilities=True,
            max_preloaded_capabilities=24,
            compact_phase_schemas=True,
            state_delta_context=True,
            bounded_evidence_context=True,
            staged_case_authoring=True,
            max_prepare_retrieval_cycles=2,
            max_model_prompt_chars=18_000,
            max_prepare_model_evidence_items=10,
            max_decide_model_evidence_items=12,
            max_model_evidence_detail_chars=600,
        ),
    )
    agent.workspace.write_text("constant/polyMesh/boundary", _boundary_file())
    agent.prepare(state, native_execution=True)

    assert state.current_state == State.SOLVE_READY
    assert llm.schemas[:3] == [PrepareDesignTurn, PrepareDesignTurn, PrepareDecisionDesignTurn]
    assert len(agent._observed_evidence_registry(state)) > 20
    assert all(len(prompt) <= 18_000 for prompt in llm.prompts)
    decide_metrics = structured_request_metrics(
        llm.schemas[2], llm.prompts[2], system_prompt=llm.kwargs[2]["system_prompt"] or ""
    )
    assert decide_metrics["approxTokens"] < 15_000
    assert '"shown": 12' in llm.prompts[2]
