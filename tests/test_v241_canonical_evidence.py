from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from openfoam_agent.engineering.agent import CFDEngineeringAgent, EngineeringPolicy
from openfoam_agent.schemas.engineering import (
    BlockAction,
    EngineeringEvidence,
    EngineeringTurn,
    SearchCapabilitiesAction,
    canonical_engineering_evidence_id,
)

from conftest import FakeOpenFOAMTools, ScriptedLLM, make_plan, make_state


def test_engineering_evidence_accepts_only_canonical_ids() -> None:
    with pytest.raises(ValidationError):
        EngineeringEvidence.model_validate(
            {
                "kind": "tool_result",
                "reference": "checkMesh at preparation step 98",
                "note": "legacy free-form evidence claim",
            }
        )
    with pytest.raises(ValidationError):
        EngineeringEvidence(evidence_id="user_fact:confirmed_intake")


def test_capability_search_issues_canonical_evidence_records(tmp_path, graph_path) -> None:
    state = make_state()
    llm = ScriptedLLM([])
    agent = CFDEngineeringAgent(
        llm,
        workspace=tmp_path / "workspace",
        capability_db=graph_path,
        tools=FakeOpenFOAMTools(),
    )

    event = agent._dispatch_tool_action(
        SearchCapabilitiesAction(
            type="search_capabilities",
            query="incompressible transient",
            rationale="Observe solver capability evidence.",
        ),
        step=1,
        native_execution=True,
        phase="prepare",
        state=state,
    )
    assert event.success
    assert event.observed_evidence
    expected = canonical_engineering_evidence_id(
        "capability", "solver.incompressibleFluid"
    )
    issued = {item.evidence_id: item for item in event.observed_evidence}
    assert expected in issued
    assert issued[expected].reference == "solver.incompressibleFluid"
    assert issued[expected].kind == "capability"


def test_prompt_exposes_registry_but_keeps_python_bindings_separate(tmp_path, graph_path) -> None:
    state = make_state()
    llm = ScriptedLLM(
        [
            EngineeringTurn(
                action=BlockAction(
                    type="block",
                    reason="stop after prompt capture",
                    needs_user_input=False,
                    rationale="test",
                )
            )
        ]
    )
    agent = CFDEngineeringAgent(
        llm,
        workspace=tmp_path / "workspace",
        capability_db=graph_path,
        tools=FakeOpenFOAMTools(),
        policy=EngineeringPolicy(max_agent_steps=4),
    )
    event = agent._dispatch_tool_action(
        SearchCapabilitiesAction(
            type="search_capabilities",
            query="incompressible transient",
            rationale="Observe solver capability evidence.",
        ),
        step=1,
        native_execution=True,
        phase="prepare",
        state=state,
    )
    state.engineering_events.append(event)

    agent._generate_turn(
        state,
        step=2,
        local_step=2,
        current_step_limit=4,
        phase="prepare",
        native_execution=True,
    )
    prompt = llm.prompts[-1]
    payload = json.loads(prompt[prompt.index("{") :])
    evidence_ids = {item["evidence_id"] for item in payload["available_evidence"]}
    assert canonical_engineering_evidence_id(
        "capability", "solver.incompressibleFluid"
    ) in evidence_ids
    assert payload["deterministic_bindings"]["confirmed_intake"]["bound_by"] == "python"
    assert payload["deterministic_bindings"]["check_mesh"]["bound_by"] == "python"
    assert payload["deterministic_bindings"]["case_manifest"]["bound_by"] == "python"
    assert all(item["kind"] in {"capability", "openfoam_reference"} for item in payload["available_evidence"])


def test_unissued_canonical_evidence_id_is_rejected_without_free_form_matching(
    tmp_path, graph_path
) -> None:
    state = make_state()
    agent = CFDEngineeringAgent(
        ScriptedLLM([]),
        workspace=tmp_path / "workspace",
        capability_db=graph_path,
        tools=FakeOpenFOAMTools(),
    )
    event = agent._dispatch_tool_action(
        SearchCapabilitiesAction(
            type="search_capabilities",
            query="incompressible transient",
            rationale="Observe solver capability evidence.",
        ),
        step=1,
        native_execution=True,
        phase="prepare",
        state=state,
    )
    state.engineering_events.append(event)

    plan = make_plan(state.intake).model_copy(
        update={
            "evidence": [
                EngineeringEvidence(
                    evidence_id=canonical_engineering_evidence_id(
                        "openfoam_reference", "source:not-observed.C"
                    )
                )
            ]
        }
    )
    failures = agent._validate_observed_provenance(plan, state)
    assert failures == [
        "Engineering evidence ID was not issued by the deterministic evidence registry "
        f"in this run: {plan.evidence[0].evidence_id}"
    ]


def test_finish_preview_needs_no_llm_claim_for_checkmesh_or_confirmed_intake(
    tmp_path, graph_path
) -> None:
    from openfoam_agent.schemas.common import ToolResult
    from openfoam_agent.schemas.engineering import (
        FinishPreviewAction,
        RunMeshCommandAction,
        WriteCaseFileAction,
    )
    from openfoam_agent.workflow.states import State
    from conftest import mesh_ok_log

    state = make_state()
    plan = make_plan(state.intake).model_copy(update={"evidence": []})
    llm = ScriptedLLM(
        [
            SearchCapabilitiesAction(
                type="search_capabilities",
                query="incompressible transient",
                rationale="Observe solver provider.",
            ),
            WriteCaseFileAction(
                type="write_case_file",
                path="system/controlDict",
                content=(
                    "solver incompressibleFluid;\n"
                    "startFrom startTime;\nstartTime 0;\nendTime 1;\ndeltaT 0.01;\n"
                ),
                rationale="Create minimal bounded runtime control.",
            ),
            RunMeshCommandAction(
                type="run_mesh_command",
                command="checkMesh",
                rationale="Establish native mesh evidence.",
            ),
            FinishPreviewAction(
                type="finish_preview",
                plan=plan,
                rationale="Seal using Python-owned intake/checkMesh bindings.",
            ),
        ]
    )
    tools = FakeOpenFOAMTools(
        mesh_results={
            "checkMesh": [
                ToolResult(
                    success=True,
                    command=["checkMesh"],
                    return_code=0,
                    stdout=mesh_ok_log(cells=1200),
                )
            ]
        }
    )
    agent = CFDEngineeringAgent(
        llm,
        workspace=tmp_path / "workspace",
        capability_db=graph_path,
        tools=tools,
        policy=EngineeringPolicy(max_agent_steps=8),
    )
    agent.prepare(state, native_execution=True)
    assert state.current_state == State.MESH_READY
    assert state.case_seal is not None
    assert state.mesh_evidence is not None and state.mesh_evidence.passed
    assert not any(
        "user_fact:confirmed_intake" in event.output_excerpt
        or "tool_result:checkMesh" in event.output_excerpt
        for event in state.engineering_events
    )
