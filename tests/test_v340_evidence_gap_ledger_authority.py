from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from openfoam_agent.engineering.agent import CFDEngineeringAgent, EngineeringPolicy
from openfoam_agent.schemas.engineering import EvidenceGapRequest, GatherEvidenceAction, PrepareTurn
from openfoam_agent.tools.references import OpenFOAMReferenceIndex

from conftest import FakeOpenFOAMTools, make_state


class EmptyLLM:
    store = False

    def generate(self, *args, **kwargs):  # pragma: no cover
        raise AssertionError("LLM should not be called")


def _agent(tmp_path, graph_path):
    return CFDEngineeringAgent(
        EmptyLLM(),
        workspace=tmp_path / "case",
        capability_db=graph_path,
        tools=FakeOpenFOAMTools(),
        policy=EngineeringPolicy(max_prepare_retrieval_cycles=4, compact_phase_schemas=True),
    )


def test_self_refining_gap_is_protocol_valid_before_ledger_resolution():
    turn = PrepareTurn.model_validate(
        {
            "action": {
                "type": "gather_evidence",
                "gaps": [
                    {
                        "gap_id": "G1086",
                        "refines_gap_id": "G1086",
                        "missing_evidence": "Exact phase-change model syntax.",
                        "why_required": "Needed before authoring the model dictionary.",
                        "reference_queries": ["phase change model"],
                    }
                ],
            }
        }
    )
    gap = turn.action.gaps[0]
    assert gap.gap_id == "G1086"
    assert gap.refines_gap_id == "G1086"


def test_existing_self_refining_gap_gets_fresh_python_owned_child_id(tmp_path, graph_path):
    refs = tmp_path / "refs"
    refs.mkdir()
    (refs / "phaseChange.txt").write_text("phase change model syntax\n", encoding="utf-8")
    agent = _agent(tmp_path, graph_path)
    agent.references = OpenFOAMReferenceIndex({"source": refs})
    state = make_state()

    first = GatherEvidenceAction(
        type="gather_evidence",
        gaps=[
            EvidenceGapRequest(
                gap_id="G1086",
                missing_evidence="Initial phase-change evidence.",
                why_required="Need release-specific syntax.",
                reference_queries=["phase change"],
                reference_scope="source",
            )
        ],
    )
    event1 = agent._dispatch_tool_action(
        first, step=1, native_execution=False, phase="prepare", state=state
    )
    assert event1.success
    assert "G1086" in agent._evidence_gap_ledger["prepare"]

    follow_up = GatherEvidenceAction(
        type="gather_evidence",
        gaps=[
            EvidenceGapRequest(
                gap_id="G1086",
                refines_gap_id="G1086",
                missing_evidence="More specific phase-change dictionary syntax.",
                why_required="The first evidence did not establish the exact dictionary form.",
                reference_queries=["phase change model syntax"],
                reference_scope="source",
            )
        ],
    )
    event2 = agent._dispatch_tool_action(
        follow_up, step=2, native_execution=False, phase="prepare", state=state
    )
    assert event2.success
    display = json.loads(event2.output_excerpt)
    item = display["gaps"][0]
    assert item["requested_gap_id"] == "G1086"
    assert item["gap_id"] != "G1086"
    assert any("reissued" in note for note in item["protocol_notes"])
    child_id = item["gap_id"]
    child = agent._evidence_gap_ledger["prepare"][child_id]
    assert child["refines_gap_id"] == "G1086"
    assert agent._evidence_gap_ledger["prepare"]["G1086"]["superseded_by"] == child_id


def test_duplicate_in_batch_gap_ids_are_reissued_not_semantically_merged(tmp_path, graph_path):
    agent = _agent(tmp_path, graph_path)
    action = GatherEvidenceAction(
        type="gather_evidence",
        gaps=[
            EvidenceGapRequest(
                gap_id="G20",
                missing_evidence="solver syntax",
                why_required="solver evidence",
                reference_queries=["solver syntax"],
            ),
            EvidenceGapRequest(
                gap_id="G20",
                missing_evidence="fvModel syntax",
                why_required="model evidence",
                reference_queries=["fvModel syntax"],
            ),
        ],
    )
    normalized = agent._normalize_evidence_gap_batch(action, phase="prepare")
    assert len(normalized) == 2
    assert normalized[0][0].gap_id == "G20"
    assert normalized[1][0].gap_id != "G20"
    assert normalized[0][0].missing_evidence == "solver syntax"
    assert normalized[1][0].missing_evidence == "fvModel syntax"


def test_unknown_refinement_parent_is_cleared_as_protocol_noise(tmp_path, graph_path):
    agent = _agent(tmp_path, graph_path)
    action = GatherEvidenceAction(
        type="gather_evidence",
        gaps=[
            EvidenceGapRequest(
                gap_id="G31",
                refines_gap_id="G99",
                missing_evidence="tool syntax",
                why_required="tool evidence",
                reference_queries=["tool syntax"],
            )
        ],
    )
    normalized = agent._normalize_evidence_gap_batch(action, phase="prepare")
    gap, requested, notes = normalized[0]
    assert requested == "G31"
    assert gap.gap_id == "G31"
    assert gap.refines_gap_id is None
    assert any("unknown refinement parent" in note for note in notes)


def test_prepare_union_reports_only_declared_action_branch_errors():
    with pytest.raises(ValidationError) as exc_info:
        PrepareTurn.model_validate(
            {"action": {"type": "gather_evidence", "gaps": "not-a-list"}}
        )
    errors = exc_info.value.errors()
    assert len(errors) == 1
    assert errors[0]["type"] == "list_type"
    assert "ReadCaseFileAction" not in str(exc_info.value)
    assert "ExecuteCasePlanAction" not in str(exc_info.value)


def test_codex_wait_heartbeat_reports_long_blocking_cli_call(monkeypatch):
    import time
    from types import SimpleNamespace

    from pydantic import BaseModel
    import openfoam_agent.llm.codex_client as codex
    from openfoam_agent.llm.codex_client import CodexCLIStatus, CodexLLM

    class Output(BaseModel):
        ok: bool

    beats: list[tuple[float, float]] = []

    def fake_run(command, **kwargs):
        time.sleep(0.03)
        output_path = command[command.index("--output-last-message") + 1]
        with open(output_path, "w", encoding="utf-8") as handle:
            handle.write('{"ok":true}')
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(codex.subprocess, "run", fake_run)
    status = CodexCLIStatus(
        binary="/usr/bin/codex",
        version="codex-cli 0.153.0",
        login_status="Logged in using ChatGPT",
        supports_ignore_user_config=True,
    )
    llm = CodexLLM(
        status=status,
        wait_callback=lambda elapsed, timeout: beats.append((elapsed, timeout)),
        wait_heartbeat_seconds=0.005,
        timeout_seconds=2,
    )
    assert llm.generate(Output, "Return ok.").ok is True
    assert beats
    assert all(timeout == 2 for _, timeout in beats)


def test_claude_wait_heartbeat_reports_long_blocking_cli_call(monkeypatch):
    import json
    import time
    from types import SimpleNamespace

    from pydantic import BaseModel
    import openfoam_agent.llm.claude_client as claude
    from openfoam_agent.llm.claude_client import ClaudeCLIStatus, ClaudeLLM

    class Output(BaseModel):
        ok: bool

    beats: list[tuple[float, float]] = []

    def fake_run(command, **kwargs):
        time.sleep(0.03)
        return SimpleNamespace(
            returncode=0,
            stdout=json.dumps({"structured_output": {"ok": True}}),
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
    llm = ClaudeLLM(
        status=status,
        wait_callback=lambda elapsed, timeout: beats.append((elapsed, timeout)),
        wait_heartbeat_seconds=0.005,
        timeout_seconds=2,
    )
    assert llm.generate(Output, "Return ok.").ok is True
    assert beats
    assert all(timeout == 2 for _, timeout in beats)
