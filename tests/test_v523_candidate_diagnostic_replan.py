from __future__ import annotations

from pathlib import Path

import openfoam_agent
from conftest import FakeOpenFOAMTools, ScriptedLLM, foam_header, make_plan, make_state
from openfoam_agent.engineering import CFDEngineeringAgent
from openfoam_agent.engineering.candidate_replan_context import build_partitioned_candidate_replan_prompt
from openfoam_agent.schemas.engineering import (
    CandidateCasePlanRepairAction,
    CaseBundleFile,
    ExecuteCasePlanAction,
)


def _unsafe_u() -> str:
    return foam_header("0/U", "volVectorField") + (
        "dimensions [0 1 -1 0 0 0 0];\n"
        "internalField uniform (1 0 0);\n"
        "boundaryField\n"
        "{\n"
        "    inlet\n"
        "    {\n"
        "        type codedFixedValue;\n"
        "        value uniform (1 0 0);\n"
        "        #codeStream\n"
        "    }\n"
        "}\n"
    )


def _safe_u() -> str:
    return foam_header("0/U", "volVectorField") + (
        "dimensions [0 1 -1 0 0 0 0];\n"
        "internalField uniform (1 0 0);\n"
        "boundaryField\n"
        "{\n"
        "    inlet\n"
        "    {\n"
        "        type fixedValue;\n"
        "        value uniform (1 0 0);\n"
        "    }\n"
        "}\n"
    )


def _unsafe_candidate(state):
    plan = make_plan(state.intake)
    plan.required_case_files = ["0/U"]
    return ExecuteCasePlanAction(
        type="execute_case_plan",
        goal="author a vortex-shedding inlet field",
        files=[CaseBundleFile(path="0/U", content=_unsafe_u())],
        plan=plan,
    )


def test_v523_precommit_security_failure_is_retained_with_exact_diagnostic(tmp_path, graph_path):
    state = make_state(syntax_evidence=False)
    execution = _unsafe_candidate(state)
    agent = CFDEngineeringAgent(
        ScriptedLLM([]),
        workspace=tmp_path,
        capability_db=graph_path,
        tools=FakeOpenFOAMTools(),
    )

    terminal = agent._execute_case_plan(
        state,
        execution,
        llm_step=1,
        progress_phase="engineering",
        progress_step=1,
        progress_limit=12,
        native_execution=False,
    )

    assert terminal is False
    assert agent.workspace.list_authored() == []
    assert agent._pending_candidate_failed_paths == ("0/U",)
    diagnostic = agent._pending_candidate_failure_diagnostic
    assert diagnostic is not None
    assert diagnostic["action_type"] == "case_bundle_preflight"
    assert diagnostic["category"] == "security"
    assert diagnostic["failed_paths"] == ["0/U"]
    assert "0/U contains executable/unsafe directives" in diagnostic["diagnostic"]
    assert "#codestream" in diagnostic["diagnostic"].casefold()

    capsule = agent._candidate_repair_context()
    assert capsule is not None
    assert capsule["deterministic_failure"] == diagnostic
    assert capsule["failed_artifacts"][0]["path"] == "0/U"


def test_v523_partitioned_replan_keeps_bounded_diagnostic_and_failed_artifact():
    payload = {
        "phase": "prepare",
        "step": 5,
        "retained_candidate": {
            "goal": "repair unsafe inlet",
            "failed_paths": ["0/U"],
            "deterministic_failure": {
                "action_type": "case_bundle_preflight",
                "category": "security",
                "validation_status": "fail",
                "summary": "Case bundle rejected before commit",
                "diagnostic": "- 0/U contains executable/unsafe directives: #codestream" + " x" * 12000,
                "failed_paths": ["0/U"],
            },
            "missing_required_files": [],
            "deferred_native_required_files": [],
            "manifest": [{"path": "0/U", "kind": "raw", "chars": 50000}],
            "failed_artifacts": [{"path": "0/U", "kind": "raw", "content": _unsafe_u() + "x" * 50000}],
            "controller_build_policy": {"manifest_authority": "EngineeringPlan.required_case_files"},
            "plan_capsule": {
                "solver": "incompressibleFluid",
                "solver_provider_id": "solver.incompressibleFluid",
                "confirmed_intake_sha256": "a" * 64,
                "required_case_files": ["0/U"],
            },
        },
        "deterministic_bindings": {"confirmed_intake": {"sha256": "a" * 64}},
        "budget": {"steps_remaining_in_current_window": 4},
    }

    built, capsule, metrics = build_partitioned_candidate_replan_prompt(
        "Repair retained candidate:\n", payload, max_chars=18_000
    )

    assert len(built.prompt) <= 18_000
    failure = capsule["retained_candidate"]["deterministic_failure"]
    assert failure["category"] == "security"
    assert failure["failed_paths"] == ["0/U"]
    assert "executable/unsafe directives" in failure["diagnostic"]
    assert len(failure["diagnostic"]) <= 6000
    assert metrics["candidateReplanDiagnosticChars"] > 0
    assert capsule["retained_candidate"]["failed_artifacts"][0]["path"] == "0/U"


def test_v523_unsafe_u_can_be_repaired_only_at_implicated_path_without_weakening_policy(tmp_path, graph_path):
    state = make_state(syntax_evidence=False)
    execution = _unsafe_candidate(state)
    agent = CFDEngineeringAgent(
        ScriptedLLM([]),
        workspace=tmp_path,
        capability_db=graph_path,
        tools=FakeOpenFOAMTools(),
    )
    agent._execute_case_plan(
        state,
        execution,
        llm_step=1,
        progress_phase="engineering",
        progress_step=1,
        progress_limit=12,
        native_execution=False,
    )

    repaired = agent._apply_candidate_case_plan_repair(
        CandidateCasePlanRepairAction(
            type="repair_candidate_case_plan",
            diagnosis="Replace coded inlet implementation with ordinary declarative fixedValue syntax.",
            replacement_files=[CaseBundleFile(path="0/U", content=_safe_u())],
        )
    )

    assert [item.path for item in repaired.files] == ["0/U"]
    assert "codedfixedvalue" not in repaired.files[0].content.casefold()
    assert "#codestream" not in repaired.files[0].content.casefold()
    assert agent.workspace.validate_candidate_bundle({"0/U": repaired.files[0].content}) == []


def test_v523_release_metadata():
    import tomllib

    root = Path(__file__).resolve().parents[1]
    assert openfoam_agent.__version__ == "5.2.3"
    assert tomllib.loads((root / "pyproject.toml").read_text())["project"]["version"] == "5.2.3"
    assert (root / "V5_2_3_RELEASE.md").is_file()
