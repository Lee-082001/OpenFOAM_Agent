from __future__ import annotations

import json
import os
import subprocess
from types import SimpleNamespace

import pytest
from pydantic import BaseModel

from conftest import FakeOpenFOAMTools, ScriptedLLM, control_dict, make_plan, make_state, mesh_ok_log, tool_result
from openfoam_agent.cli import _build_llm, _validate_args, build_parser
from openfoam_agent.engineering import CFDEngineeringAgent, EngineeringPolicy
from openfoam_agent.llm.codex_client import CodexCLIStatus, CodexLLM, check_codex_cli
from openfoam_agent.runtime import RuntimeOrchestrator
from openfoam_agent.schemas.common import ToolResult
from openfoam_agent.schemas.engineering import (
    BlockAction,
    CaseSeal,
    FinishPreviewAction,
    MeshEvidence,
    RunMeshCommandAction,
    SearchCapabilitiesAction,
    WriteCaseFileAction,
)
from openfoam_agent.schemas.simulation import RuntimePolicy, RuntimeRepairDecision
from openfoam_agent.tools.workspace import CaseWorkspace
from openfoam_agent.verification.presolve import PreSolveCompletenessGate
from openfoam_agent.workflow.states import State


BOUNDARY = '''FoamFile {}\n1\n(\nfrontAndBack\n{\n    type empty;\n}\n)\n'''


def _field(name: str, patch_type: str) -> str:
    return f'''FoamFile {{ object {name}; }}\ndimensions [0 0 0 0 0 0 0];\ninternalField uniform 0;\nboundaryField\n{{\n    frontAndBack\n    {{\n        type {patch_type};\n    }}\n}}\n'''


def _prepared_runtime_agent(tmp_path, graph_path, tools, repair_actions):
    state = make_state()
    plan = make_plan(state.intake)
    llm = ScriptedLLM([
        SearchCapabilitiesAction(type="search_capabilities", query="incompressibleFluid", rationale="capability"),
        WriteCaseFileAction(type="write_case_file", path="system/controlDict", content=control_dict(), rationale="control"),
        RunMeshCommandAction(type="run_mesh_command", command="checkMesh", rationale="mesh"),
        FinishPreviewAction(type="finish_preview", plan=plan, rationale="seal"),
        *repair_actions,
    ])
    agent = CFDEngineeringAgent(
        llm,
        workspace=tmp_path,
        capability_db=graph_path,
        tools=tools,
        policy=EngineeringPolicy(max_agent_steps=12),
    )
    agent.prepare(state, native_execution=True)
    assert state.current_state == State.MESH_READY
    return state, agent


def test_runtime_repair_decision_closes_transient_state_in_orchestrator(tmp_path, graph_path):
    fail = ToolResult(
        success=False,
        command=["foamRun"],
        return_code=1,
        stderr="--> FOAM FATAL IO ERROR: patch mismatch\n",
    )
    tools = FakeOpenFOAMTools(
        mesh_results={"checkMesh": [tool_result("checkMesh", success=True, stdout=mesh_ok_log())]},
        foam_runs=[fail],
    )
    state, agent = _prepared_runtime_agent(
        tmp_path,
        graph_path,
        tools,
        [BlockAction(type="block", reason="Runtime change needs review.", needs_user_input=True, rationale="review")],
    )
    state.approve_solve()
    RuntimeOrchestrator(tools, agent, RuntimePolicy(max_attempts=2, solver_timeout_seconds=30)).run(state)

    assert state.current_state == State.ENGINEERING_REVIEW_REQUIRED
    assert state.current_state != State.RUNTIME_REPAIR
    assert state.runtime_report is not None and not state.runtime_report.success
    assert all("No v2 handler for RUNTIME_REPAIR" not in item["note"] for item in state.history)


def test_runtime_repair_outcome_uses_explicit_decision_enum(tmp_path, graph_path):
    tools = FakeOpenFOAMTools(
        mesh_results={"checkMesh": [tool_result("checkMesh", success=True, stdout=mesh_ok_log())]}
    )
    state, agent = _prepared_runtime_agent(tmp_path, graph_path, tools, [])
    agent.llm = ScriptedLLM([
        BlockAction(type="block", reason="Need human review.", needs_user_input=True, rationale="review")
    ])
    outcome = agent.repair_runtime(
        state,
        runtime_log="--> FOAM FATAL ERROR: model mismatch\n",
        attempt=1,
        native_execution=True,
    )
    assert outcome.decision == RuntimeRepairDecision.NEEDS_USER_REVIEW
    assert outcome.retry is False
    assert state.current_state == State.ENGINEERING_REVIEW_REQUIRED


def test_presolve_rejects_mesh_field_constraint_patch_mismatch(tmp_path):
    workspace = CaseWorkspace(tmp_path)
    boundary = workspace.case_dir / "constant/polyMesh/boundary"
    boundary.parent.mkdir(parents=True, exist_ok=True)
    boundary.write_text(BOUNDARY, encoding="utf-8")
    workspace.write_text("system/controlDict", control_dict())
    workspace.write_text("system/fvSchemes", "FoamFile {}\nddtSchemes {}\n")
    workspace.write_text("system/fvSolution", "FoamFile {}\nsolvers {}\n")
    workspace.write_text("0/U", _field("U", "empty"))
    workspace.write_text("0/p", _field("p", "patch"))
    state = make_state()
    plan = make_plan(state.intake).model_copy(update={"required_case_files": ["0/U", "0/p"]})

    result = PreSolveCompletenessGate(FakeOpenFOAMTools(), workspace).validate(plan)
    assert result.valid is False
    assert result.mesh_patch_types["frontAndBack"] == "empty"
    assert any("mesh=empty, field=patch" in failure for failure in result.failures)

    workspace.write_text("0/p", _field("p", "empty"))
    fixed = PreSolveCompletenessGate(FakeOpenFOAMTools(), workspace).validate(plan)
    assert fixed.valid, fixed.failures


def test_createpatch_invalidates_mesh_presolve_and_runtime_seal(tmp_path, graph_path):
    tools = FakeOpenFOAMTools(
        mesh_results={
            "checkMesh": [tool_result("checkMesh", success=True, stdout=mesh_ok_log())],
            "createPatch": [tool_result("createPatch", success=True, stdout="patches changed\n")],
        }
    )
    state, agent = _prepared_runtime_agent(tmp_path, graph_path, tools, [])
    assert state.case_seal is not None
    assert state.mesh_evidence is not None
    seal = state.case_seal
    agent._presolve_case_manifest = "old-case"
    agent._presolve_required_case_files = ("0/U", "0/p")
    agent._checkmesh_mesh_manifest = "old-mesh"

    event = agent._dispatch_tool_action(
        RunMeshCommandAction(type="run_mesh_command", command="createPatch", rationale="change topology"),
        step=99,
        native_execution=True,
        phase="runtime_repair",
        state=state,
    )
    assert event.success
    assert agent._presolve_case_manifest is None
    assert agent._presolve_required_case_files is None
    assert agent._checkmesh_mesh_manifest is None
    assert state.mesh_evidence is None
    assert state.case_seal is None
    assert seal is not None


class DemoOutput(BaseModel):
    action: str
    reason: str


def test_codex_cli_check_requires_structured_exec_and_chatgpt_login(monkeypatch):
    import openfoam_agent.llm.codex_client as codex

    monkeypatch.setattr(codex.shutil, "which", lambda binary: "/usr/bin/codex")

    def fake_run(command, **kwargs):
        del kwargs
        if command[-1] == "--version":
            return SimpleNamespace(returncode=0, stdout="codex-cli 1.2.3\n", stderr="")
        if command[1:3] == ["exec", "--help"]:
            return SimpleNamespace(
                returncode=0,
                stdout="--output-schema --output-last-message --ephemeral --sandbox --ignore-user-config",
                stderr="",
            )
        if command[1:3] == ["login", "status"]:
            return SimpleNamespace(returncode=0, stdout="Logged in using ChatGPT\n", stderr="")
        raise AssertionError(command)

    monkeypatch.setattr(codex.subprocess, "run", fake_run)
    status = check_codex_cli()
    assert status.binary == "/usr/bin/codex"
    assert status.supports_ignore_user_config is True


def test_codex_llm_isolated_readonly_exec_strips_api_env_and_revalidates(monkeypatch):
    import openfoam_agent.llm.codex_client as codex

    captured = {}
    monkeypatch.setenv("OPENAI_API_KEY", "must-not-leak")
    monkeypatch.setenv("CODEX_API_KEY", "must-not-leak")
    status = CodexCLIStatus(
        binary="/usr/bin/codex",
        version="codex 1.2.3",
        login_status="Logged in using ChatGPT",
        supports_ignore_user_config=True,
    )

    def fake_run(command, **kwargs):
        captured["command"] = list(command)
        captured["cwd"] = kwargs["cwd"]
        captured["env"] = dict(kwargs["env"])
        captured["input"] = kwargs["input"]
        output_index = command.index("--output-last-message") + 1
        with open(command[output_index], "w", encoding="utf-8") as handle:
            json.dump({"action": "ok", "reason": "validated"}, handle)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(codex.subprocess, "run", fake_run)
    llm = CodexLLM(status=status, model=None)
    result = llm.generate(DemoOutput, "Return a tiny result.", system_prompt="System")

    assert result.action == "ok"
    command = captured["command"]
    assert "--ephemeral" in command
    assert command[command.index("--sandbox") + 1] == "read-only"
    assert "--output-schema" in command and "--output-last-message" in command
    assert "--ignore-user-config" in command
    assert "OPENAI_API_KEY" not in captured["env"]
    assert "CODEX_API_KEY" not in captured["env"]
    assert os.path.isdir(captured["cwd"]) is False  # TemporaryDirectory is cleaned after return.


def test_cli_codex_requires_confirmation_and_builds_no_openai_adapter(monkeypatch):
    import openfoam_agent.cli as cli

    parser = build_parser()
    args = parser.parse_args(["demo", "--backend", "codex"])
    with pytest.raises(SystemExit):
        _validate_args(args, parser)

    args = parser.parse_args(["demo", "--backend", "codex", "--confirm-api-calls"])
    assert _validate_args(args, parser) == "demo"

    status = CodexCLIStatus(
        binary="/usr/bin/codex",
        version="codex 1.2.3",
        login_status="Logged in using ChatGPT",
        supports_ignore_user_config=False,
    )
    monkeypatch.setattr(cli, "check_codex_cli", lambda: status)

    class DummyCodexLLM:
        created = []
        def __init__(self, *, model, status):
            self.model = model or "codex-default"
            self.status = status
            self.last_usage = None
            self.__class__.created.append(self.model)

    class ForbiddenOpenAI:
        def __init__(self, **kwargs):
            raise AssertionError("--backend codex must not construct OpenAILLM")

    monkeypatch.setattr(cli, "CodexLLM", DummyCodexLLM)
    monkeypatch.setattr(cli, "OpenAILLM", ForbiddenOpenAI)
    llms, backend, default_model = _build_llm(args)
    assert backend == "codex"
    assert default_model == "codex-default"
    assert llms.intake.model == "codex-default"
    assert len(DummyCodexLLM.created) == 1
