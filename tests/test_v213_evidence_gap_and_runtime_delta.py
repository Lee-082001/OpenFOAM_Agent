from __future__ import annotations

from collections import deque
import json

from openfoam_agent.engineering.agent import CFDEngineeringAgent, EngineeringPolicy
from openfoam_agent.schemas.engineering import (
    CaseFilePatch,
    CaseFilePatchGroup,
    EvidenceGapRequest,
    ExactCaseFileEdit,
    GatherEvidenceAction,
    PrepareDecisionOnlyTurn,
    RuntimeCaseRepairAction,
    RuntimeRepairTurn,
)
from openfoam_agent.tools.references import OpenFOAMReferenceIndex

from conftest import FakeOpenFOAMTools, make_plan, make_state


class AnySchemaLLM:
    def __init__(self, actions):
        self.actions = deque(actions)
        self.prompts: list[str] = []
        self.schemas: list[type] = []
        self.store = False

    def generate(
        self,
        schema,
        prompt: str,
        *,
        system_prompt: str | None = None,
        conversation_key: str | None = None,
        use_previous_response: bool = False,
        prompt_cache_key: str | None = None,
    ):
        del system_prompt, conversation_key, use_previous_response, prompt_cache_key
        self.prompts.append(prompt)
        self.schemas.append(schema)
        if not self.actions:
            raise AssertionError("AnySchemaLLM exhausted")
        action = self.actions.popleft()
        return schema(action=action)


def _gap(gap_id: str = "G01", query: str = "div(phi,U)") -> EvidenceGapRequest:
    return EvidenceGapRequest(
        gap_id=gap_id,
        missing_evidence="Exact OpenFOAM release syntax for the missing divergence scheme.",
        why_required="The native error names a release-sensitive dictionary entry.",
        reference_queries=[query],
        reference_scope="tutorials",
        read_top_reference_matches=1,
    )


def test_prepare_batch_retrieval_tracks_novelty_and_stagnation(tmp_path, graph_path):
    refs = tmp_path / "refs"
    refs.mkdir()
    (refs / "fvSchemes.example").write_text(
        "divSchemes\n{\n    div(phi,U) Gauss linearUpwind grad(U);\n}\n",
        encoding="utf-8",
    )
    agent = CFDEngineeringAgent(
        AnySchemaLLM([]),
        workspace=tmp_path / "case",
        capability_db=graph_path,
        tools=FakeOpenFOAMTools(),
        policy=EngineeringPolicy(
            max_prepare_retrieval_cycles=3,
            compact_phase_schemas=True,
            preload_capabilities=True,
        ),
    )
    agent.references = OpenFOAMReferenceIndex({"tutorials": refs})
    state = make_state()
    agent._evidence_gap_ledger["prepare"] = {}
    agent._retrieval_cycles["prepare"] = 0

    first = GatherEvidenceAction(type="gather_evidence", gaps=[_gap()])
    event1 = agent._dispatch_tool_action(first, step=1, native_execution=False, phase="prepare", state=state)
    assert event1.success
    payload1 = json.loads(event1.output_excerpt)
    assert payload1["gaps"][0]["status"] == "new_evidence"
    assert payload1["gaps"][0]["new_evidence_ids"]

    second = GatherEvidenceAction(type="gather_evidence", gaps=[_gap(query="fvSchemes div(phi,U)")])
    event2 = agent._dispatch_tool_action(second, step=2, native_execution=False, phase="prepare", state=state)
    assert event2.success
    payload2 = json.loads(event2.output_excerpt)
    assert payload2["gaps"][0]["status"] == "already_retrieved_blocked"
    # A blocked repeat does not consume the retrieval hard-fuse budget.
    assert agent._retrieval_cycles["prepare"] == 1

    refined = EvidenceGapRequest(
        gap_id="G02",
        refines_gap_id="G01",
        missing_evidence="A more specific release example for the same scheme.",
        why_required="The first evidence did not expose the exact desired variant.",
        reference_queries=["another query"],
        reference_scope="tutorials",
        read_top_reference_matches=1,
    )
    event3 = agent._dispatch_tool_action(
        GatherEvidenceAction(type="gather_evidence", gaps=[refined]),
        step=3,
        native_execution=False,
        phase="prepare",
        state=state,
    )
    payload3 = json.loads(event3.output_excerpt)
    assert payload3["gaps"][0]["status"] in {"new_evidence", "no_new_evidence"}
    status = {item["gap_id"]: item for item in agent._evidence_gap_status("prepare")}
    assert status["G01"]["status"] == "superseded"
    assert status["G01"]["superseded_by"] == "G02"


def test_prepare_gather_evidence_batches_multiple_independent_gaps(tmp_path, graph_path):
    refs = tmp_path / "refs"
    refs.mkdir()
    (refs / "momentumTransport").write_text("simulationType laminar;\n", encoding="utf-8")
    agent = CFDEngineeringAgent(
        AnySchemaLLM([]),
        workspace=tmp_path / "case",
        capability_db=graph_path,
        tools=FakeOpenFOAMTools(),
        policy=EngineeringPolicy(compact_phase_schemas=True),
    )
    agent.references = OpenFOAMReferenceIndex({"tutorials": refs})
    state = make_state()
    action = GatherEvidenceAction(
        type="gather_evidence",
        gaps=[
            EvidenceGapRequest(
                gap_id="G01",
                missing_evidence="Solver provider evidence.",
                why_required="Need an observed provider before selecting the solver.",
                capability_queries=["incompressibleFluid"],
                reference_queries=[],
                read_top_reference_matches=0,
            ),
            EvidenceGapRequest(
                gap_id="G02",
                missing_evidence="OpenFOAM laminar momentumTransport syntax.",
                why_required="Release-specific dictionary syntax is needed.",
                capability_queries=[],
                reference_queries=["momentumTransport laminar"],
                reference_scope="tutorials",
                read_top_reference_matches=1,
            ),
        ],
    )
    event = agent._dispatch_tool_action(action, step=1, native_execution=False, phase="prepare", state=state)
    payload = json.loads(event.output_excerpt)
    assert {item["gap_id"] for item in payload["gaps"]} == {"G01", "G02"}
    assert len(event.observed_evidence) >= 2
    assert agent._retrieval_cycles["prepare"] == 1


def test_runtime_grouped_patch_allows_multiple_edits_to_same_file_and_retries(tmp_path, graph_path):
    state = make_state()
    plan = make_plan(state.intake)
    repair = RuntimeCaseRepairAction(
        type="repair_runtime_case",
        diagnosis="fvSchemes is missing/incorrect runtime scheme entries.",
        file_patches=[
            CaseFilePatchGroup(
                path="system/fvSchemes",
                edits=[
                    ExactCaseFileEdit(old="default Euler;", new="default backward;"),
                    ExactCaseFileEdit(old="default none;", new="default Gauss linear;"),
                ],
            )
        ],
        validate_dictionaries=[],
        validate_pre_solve=False,
        retry_solver=True,
    )
    llm = AnySchemaLLM([repair])
    agent = CFDEngineeringAgent(
        llm,
        workspace=tmp_path,
        capability_db=graph_path,
        tools=FakeOpenFOAMTools(),
        policy=EngineeringPolicy(
            max_runtime_repair_steps=2,
            compact_phase_schemas=True,
            state_delta_context=True,
            preload_capabilities=True,
        ),
    )
    agent.workspace.write_text(
        "system/fvSchemes",
        "ddtSchemes { default Euler; }\ndivSchemes { default none; }\n",
    )
    agent.workspace.write_text(
        "system/controlDict",
        "solver incompressibleFluid;\nstartTime 0;\nendTime 1;\ndeltaT 0.01;\n",
    )
    state.engineering_plan = plan
    state.case_seal = agent.workspace.seal(plan)

    outcome = agent.repair_runtime(
        state,
        runtime_log=(
            "--> FOAM FATAL IO ERROR:\n"
            "Cannot find scheme for div(phi,U) in dictionary system/fvSchemes/divSchemes\n"
        ),
        attempt=1,
        native_execution=False,
    )

    assert outcome.retry is True
    text = agent.workspace.read_text("system/fvSchemes")
    assert "default backward;" in text
    assert "default Gauss linear;" in text
    assert llm.schemas == [RuntimeRepairTurn]
    assert '"state_mode": "runtime_failure_slice"' in llm.prompts[0]
    assert '"path": "system/fvSchemes"' in llm.prompts[0]
    assert "recent_observations" not in llm.prompts[0]


def test_legacy_delta_schema_allows_multiple_exact_patches_same_file():
    # Backward-compatible RepairCasePlanAction no longer rejects same-file patches;
    # exact-match application still validates each edit sequentially at execution.
    from openfoam_agent.schemas.engineering import RepairCasePlanAction

    action = RepairCasePlanAction(
        type="repair_case_plan",
        diagnosis="two independent edits in one dictionary",
        patches=[
            CaseFilePatch(path="system/fvSchemes", old="a", new="b"),
            CaseFilePatch(path="system/fvSchemes", old="c", new="d"),
        ],
        validate_pre_solve=False,
    )
    assert len(action.patches) == 2
