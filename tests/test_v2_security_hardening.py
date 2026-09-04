from __future__ import annotations

import json
from pathlib import Path

from openfoam_agent.engineering import CFDEngineeringAgent
from openfoam_agent.schemas.engineering import (
    BlockAction,
    EngineeringEvent,
    FinishPreviewAction,
    WriteCaseFileAction,
)
from openfoam_agent.tools.openfoam import OpenFOAMTools
from openfoam_agent.tools.parsers import parse_runtime_log
from openfoam_agent.tools.references import OpenFOAMReferenceIndex
from openfoam_agent.tools.safe_runner import SafeRunner

from conftest import (
    FakeOpenFOAMTools,
    ScriptedLLM,
    control_dict,
    make_plan,
    make_state,
)


def _write_executable(path: Path, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("#!/bin/sh\nset -eu\n" + body, encoding="utf-8")
    path.chmod(0o755)


def test_safe_runner_does_not_inherit_api_key_or_user_library_paths(tmp_path):
    trusted = tmp_path / "openfoam"
    exe = trusted / "bin" / "checkMesh"
    _write_executable(
        exe,
        'printf "%s|%s|%s\\n" "${OPENAI_API_KEY-unset}" "${FOAM_TEST-unset}" "${FOAM_USER_LIBBIN-unset}"\n',
    )
    env = {
        "WM_PROJECT_DIR": str(trusted),
        "PATH": f"{exe.parent}:/usr/bin:/bin",
        "OPENAI_API_KEY": "sk-must-not-reach-openfoam",
        "FOAM_TEST": "kept-openfoam-runtime-value",
        "FOAM_USER_LIBBIN": str(tmp_path / "user-controlled-libs"),
        "HOME": str(tmp_path / "home"),
    }
    runner = SafeRunner(
        workspace_root=tmp_path,
        trusted_executable_roots=[trusted],
        base_env=env,
    )
    result = runner.run(["checkMesh"], cwd=tmp_path, timeout=5)
    assert result.success
    assert result.stdout.strip() == "unset|kept-openfoam-runtime-value|unset"
    assert "OPENAI_API_KEY" not in runner.sanitized_environment()
    assert "FOAM_USER_LIBBIN" not in runner.sanitized_environment()


def test_safe_runner_ignores_path_shadow_outside_trusted_installation(tmp_path):
    trusted = tmp_path / "openfoam"
    malicious = tmp_path / "shadow"
    _write_executable(malicious / "checkMesh", 'echo MALICIOUS\n')
    _write_executable(trusted / "bin" / "checkMesh", 'echo TRUSTED\n')
    env = {
        "WM_PROJECT_DIR": str(trusted),
        "PATH": f"{malicious}:{trusted / 'bin'}:/usr/bin:/bin",
    }
    runner = SafeRunner(
        workspace_root=tmp_path,
        trusted_executable_roots=[trusted],
        base_env=env,
    )
    result = runner.run(["checkMesh"], cwd=tmp_path, timeout=5)
    assert result.success
    assert result.stdout.strip() == "TRUSTED"


def test_environment_reference_roots_cannot_escape_wm_project_dir(tmp_path, monkeypatch):
    project = tmp_path / "openfoam14"
    source = project / "src"
    outside = tmp_path / "private"
    source.mkdir(parents=True)
    outside.mkdir()
    (source / "safe.C").write_text("displacementLaplacian\n", encoding="utf-8")
    (outside / "secret.txt").write_text("TOP SECRET displacementLaplacian\n", encoding="utf-8")

    monkeypatch.setenv("WM_PROJECT_DIR", str(project))
    monkeypatch.setenv("FOAM_SRC", str(outside))
    index = OpenFOAMReferenceIndex()
    assert index.summary() == {}
    assert index.search("displacementLaplacian", scope="source") == []

    monkeypatch.setenv("FOAM_SRC", str(source))
    index = OpenFOAMReferenceIndex()
    summary_text = json.dumps(index.summary())
    assert str(project) not in summary_text
    results = index.search("displacementLaplacian", scope="source")
    assert results and results[0]["reference"] == "source:safe.C"
    assert str(project) not in json.dumps(results)


def test_environment_snapshot_does_not_expose_absolute_openfoam_paths(tmp_path, monkeypatch):
    project = tmp_path / "openfoam14"
    appbin = project / "platforms" / "linux64" / "bin"
    _write_executable(appbin / "checkMesh", "echo ok\n")
    monkeypatch.setenv("WM_PROJECT_DIR", str(project))
    monkeypatch.setenv("WM_PROJECT", "OpenFOAM")
    monkeypatch.setenv("WM_PROJECT_VERSION", "14")
    monkeypatch.setenv("FOAM_SRC", str(project / "src"))
    monkeypatch.setenv("PATH", f"{appbin}:/usr/bin:/bin")
    runner = SafeRunner(workspace_root=tmp_path)
    snapshot = OpenFOAMTools(runner).environment_snapshot()
    encoded = json.dumps(snapshot)
    assert str(project) not in encoded
    assert "foam_src" not in snapshot
    assert snapshot["trusted_installation_configured"] is True
    assert "commands" not in snapshot
    assert snapshot["installed_executable_count"] == 1
    assert snapshot["capability_inventory_queryable"] is True


def test_model_prompt_redacts_known_local_paths(tmp_path, graph_path, monkeypatch):
    state = make_state()
    home = tmp_path / "sensitive-home"
    foam_root = tmp_path / "openfoam14"
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("WM_PROJECT_DIR", str(foam_root))
    llm = ScriptedLLM([
        BlockAction(type="block", reason="test", needs_user_input=False, rationale="test")
    ])
    agent = CFDEngineeringAgent(
        llm,
        workspace=tmp_path / "workspace",
        capability_db=graph_path,
        tools=FakeOpenFOAMTools(),
    )
    state.engineering_events.append(
        EngineeringEvent(
            step=1,
            action_type="checkMesh",
            success=False,
            summary=f"failure under {agent.workspace.case_dir}",
            output_excerpt=f"source {foam_root}/src/file.C home {home}/note.txt",
        )
    )
    agent._generate_turn(
        state,
        step=2,
        phase="runtime_repair",
        runtime_log=f"fatal at {agent.workspace.case_dir}/system/fvSolution {home}/secret",
    )
    prompt = llm.prompts[-1]
    assert str(agent.workspace.root) not in prompt
    assert str(home) not in prompt
    assert str(foam_root) not in prompt
    assert "<WORKSPACE>" in prompt or "<CASE_DIR>" in prompt
    assert "<HOME>" in prompt
    assert "<OPENFOAM_ROOT>" in prompt


def test_runtime_end_marker_without_time_progress_is_not_success():
    result = parse_runtime_log("End\n", return_code=0)
    assert not result.success
    assert any("no Time progress evidence" in item for item in result.evidence_failures)


def test_finish_preview_cannot_claim_unobserved_solver_capability(tmp_path, graph_path):
    state = make_state()
    plan = make_plan(state.intake)
    llm = ScriptedLLM([
        WriteCaseFileAction(
            type="write_case_file",
            path="system/controlDict",
            content=control_dict(),
            rationale="write case",
        ),
        FinishPreviewAction(
            type="finish_preview",
            plan=plan,
            rationale="claim completion without capability observation",
        ),
    ])
    agent = CFDEngineeringAgent(
        llm,
        workspace=tmp_path / "workspace",
        capability_db=graph_path,
        tools=FakeOpenFOAMTools(),
    )
    agent.policy.max_agent_steps = 2
    agent.prepare(state, native_execution=False)
    assert state.current_state.value == "ENGINEERING_BLOCKED"
    assert any(
        "no successful capability-graph observation" in event.output_excerpt
        for event in state.engineering_events
        if event.action_type == "finish_preview"
    )


def test_workspace_directories_are_private(tmp_path):
    from openfoam_agent.tools.workspace import CaseWorkspace

    ws = CaseWorkspace(tmp_path / "run")
    assert (ws.root.stat().st_mode & 0o777) == 0o700
    assert (ws.case_dir.stat().st_mode & 0o777) == 0o700
    assert (ws.log_dir.stat().st_mode & 0o777) == 0o700
    ws.write_text("system/controlDict", "solver incompressibleFluid;\n")
    assert (ws.resolve_case_path("system/controlDict").stat().st_mode & 0o077) == 0
    log = ws.write_log("solver.log", "private log\n")
    assert (log.stat().st_mode & 0o077) == 0
