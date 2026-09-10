from __future__ import annotations

import subprocess
from pathlib import Path

from conftest import FakeOpenFOAMTools, ScriptedLLM, foam_header, make_plan, make_state
from openfoam_agent.engineering import CFDEngineeringAgent
from openfoam_agent.schemas.common import ToolResult
from openfoam_agent.schemas.engineering import CaseBundleFile, ExecuteCasePlanAction
from openfoam_agent.tools.diagnostics import classify_native_validation
from openfoam_agent.tools.safe_runner import SafeRunner
from openfoam_agent.workflow.states import State


def test_loader_missing_shared_library_is_infra_not_case_failure():
    result = ToolResult(
        success=False,
        command=["snappyHexMesh", "-overwrite"],
        return_code=127,
        stderr=(
            "/opt/OpenFOAM-13/platforms/linux64GccDPInt32Opt/bin/snappyHexMesh: "
            "error while loading shared libraries: libscotch.so: cannot open shared object file: "
            "No such file or directory\n"
        ),
    )
    assessment = classify_native_validation(result, command_name="snappyHexMesh", probe=False)
    assert assessment.status == "inconclusive"
    assert assessment.category == "infra"
    assert assessment.diagnostic is not None
    assert assessment.diagnostic.kind == "shared_library_missing"
    assert assessment.workflow_success


def test_exit_127_without_foam_fatal_is_still_infra():
    result = ToolResult(
        success=False,
        command=["snappyHexMesh"],
        return_code=127,
        stderr="native loader failed before application startup\n",
    )
    assessment = classify_native_validation(result, command_name="snappyHexMesh", probe=False)
    assert assessment.status == "inconclusive"
    assert assessment.category == "infra"


def test_explicit_foam_fatal_stays_case_repair_eligible():
    result = ToolResult(
        success=False,
        command=["snappyHexMesh"],
        return_code=1,
        stderr="FOAM FATAL IO ERROR\nIllegal dictionary keyword\n",
    )
    assessment = classify_native_validation(result, command_name="snappyHexMesh", probe=False)
    assert assessment.status == "fail"
    assert assessment.category == "case"


def test_sanitized_loader_path_preserves_trusted_thirdparty_but_drops_untrusted(tmp_path):
    inst = tmp_path / "OpenFOAM"
    project = inst / "OpenFOAM-13"
    third = inst / "ThirdParty-13"
    appbin = project / "platforms/linux64/bin"
    extlib = third / "platforms/linux64/lib"
    bad = tmp_path / "untrusted-libs"
    for path in (appbin, extlib, bad):
        path.mkdir(parents=True)
    env = {
        "WM_PROJECT_DIR": str(project),
        "WM_PROJECT_INST_DIR": str(inst),
        "WM_THIRD_PARTY_DIR": str(third),
        "FOAM_APPBIN": str(appbin),
        "FOAM_EXT_LIBBIN": str(extlib),
        "PATH": f"{appbin}:/usr/bin:/bin",
        "LD_LIBRARY_PATH": f"{extlib}:{bad}:/usr/lib",
        "LANG": "C",
    }
    runner = SafeRunner(
        workspace_root=tmp_path / "workspace",
        trusted_executable_roots=[project],
        base_env=env,
    )
    sanitized = runner.sanitized_environment()
    loader_paths = sanitized.get("LD_LIBRARY_PATH", "").split(":")
    assert str(extlib.resolve()) in loader_paths
    assert str(bad.resolve()) not in loader_paths
    assert sanitized.get("FOAM_EXT_LIBBIN") == str(extlib.resolve())


def test_dependency_status_reports_missing_loader_library_without_running_cfd(tmp_path, monkeypatch):
    project = tmp_path / "OpenFOAM-13"
    project.mkdir()
    executable = project / "snappyHexMesh"
    executable.write_bytes(b"\x7fELF" + b"\x00" * 32)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    runner = SafeRunner(
        allowed_commands={"snappyHexMesh"},
        workspace_root=workspace,
        trusted_executable_roots=[project],
        base_env={"WM_PROJECT_DIR": str(project), "PATH": "/usr/bin:/bin", "LANG": "C"},
    )
    monkeypatch.setattr(runner, "resolve_trusted_executable", lambda exe, env=None: executable)
    monkeypatch.setattr(
        "openfoam_agent.tools.safe_runner.shutil.which",
        lambda name, path=None: "/usr/bin/ldd" if name == "ldd" else str(executable),
    )
    monkeypatch.setattr(
        "openfoam_agent.tools.safe_runner.subprocess.run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args=args[0], returncode=0, stdout="libscotch.so => not found\nlibc.so.6 => /lib/libc.so.6\n"
        ),
    )
    status = runner.executable_dependency_status("snappyHexMesh")
    assert status["dependency_checked"] is True
    assert status["runtime_ready"] is False
    assert status["missing_libraries"] == ["libscotch.so"]


class _BrokenSnappyTools(FakeOpenFOAMTools):
    def native_command_preflight(self, command: str):
        if command == "snappyHexMesh":
            return {
                "name": command,
                "available": True,
                "trusted": True,
                "runtime_ready": False,
                "dependency_checked": True,
                "missing_libraries": ["libscotch.so"],
                "reason": "Missing shared libraries under the sanitized OpenFOAM runtime environment: libscotch.so",
            }
        return {
            "name": command,
            "available": True,
            "trusted": True,
            "runtime_ready": True,
            "dependency_checked": True,
        }


def test_case_build_toolchain_preflight_blocks_before_first_candidate_write(tmp_path, graph_path):
    state = make_state(syntax_evidence=False)
    plan = make_plan(state.intake)
    plan.required_case_files = ["system/controlDict", "system/snappyHexMeshDict"]
    execution = ExecuteCasePlanAction(
        type="execute_case_plan",
        goal="author snappy case",
        files=[
            CaseBundleFile(path="system/controlDict", content=foam_header("system/controlDict") + "application foamRun;\n"),
            CaseBundleFile(path="system/snappyHexMeshDict", content=foam_header("system/snappyHexMeshDict") + "castellatedMesh true;\n"),
        ],
        plan=plan,
    )
    agent = CFDEngineeringAgent(
        ScriptedLLM([]), workspace=tmp_path, capability_db=graph_path, tools=_BrokenSnappyTools()
    )
    terminal = agent._execute_case_plan(
        state,
        execution,
        llm_step=2,
        progress_phase="engineering",
        progress_step=2,
        progress_limit=12,
        native_execution=True,
    )
    assert terminal
    assert state.current_state == State.ENGINEERING_BLOCKED
    assert agent.workspace.list_authored() == []
    event = state.engineering_events[-1]
    assert event.action_type == "native_toolchain_preflight"
    assert event.failure_category == "infra"
    assert event.validation_status == "inconclusive"
    assert "libscotch.so" in event.output_excerpt
    assert state.primary_failure is not None
    assert state.primary_failure["category"] == "infra"

from openfoam_agent.schemas.engineering import RepairCasePlanAction, RuntimeCaseRepairAction


def _seed_snappy_baseline(agent, state):
    plan = make_plan(state.intake)
    plan.required_case_files = ["system/controlDict", "system/snappyHexMeshDict"]
    state.engineering_plan = plan
    control = foam_header("system/controlDict") + "application foamRun;\n"
    snappy = foam_header("system/snappyHexMeshDict") + "castellatedMesh true;\n"
    agent.workspace.write_text("system/controlDict", control)
    agent.workspace.write_text("system/snappyHexMeshDict", snappy)
    return plan, snappy


def test_prepare_repair_toolchain_failure_does_not_commit_delta(tmp_path, graph_path):
    state = make_state(syntax_evidence=False)
    state.current_state = State.ENGINEERING
    agent = CFDEngineeringAgent(
        ScriptedLLM([]), workspace=tmp_path, capability_db=graph_path, tools=_BrokenSnappyTools()
    )
    _, original = _seed_snappy_baseline(agent, state)
    replacement = foam_header("system/snappyHexMeshDict") + "castellatedMesh false;\n"
    repair = RepairCasePlanAction(
        type="repair_case_plan",
        diagnosis="change snappy setup",
        replacement_files=[CaseBundleFile(path="system/snappyHexMeshDict", content=replacement)],
        validate_pre_solve=True,
    )
    terminal = agent._execute_prepare_repair_plan(
        state, repair, llm_step=3, progress_phase="engineering", native_execution=True
    )
    assert terminal
    assert agent.workspace.read_text("system/snappyHexMeshDict") == original
    assert state.engineering_events[-1].action_type == "native_toolchain_preflight"
    assert state.engineering_events[-1].failure_category == "infra"


def test_runtime_repair_toolchain_failure_does_not_commit_delta(tmp_path, graph_path):
    state = make_state(syntax_evidence=False)
    state.current_state = State.ENGINEERING
    agent = CFDEngineeringAgent(
        ScriptedLLM([]), workspace=tmp_path, capability_db=graph_path, tools=_BrokenSnappyTools()
    )
    plan, original = _seed_snappy_baseline(agent, state)
    replacement = foam_header("system/snappyHexMeshDict") + "castellatedMesh false;\n"
    repair = RuntimeCaseRepairAction(
        type="repair_runtime_case",
        diagnosis="runtime mesh repair",
        replacement_files=[CaseBundleFile(path="system/snappyHexMeshDict", content=replacement)],
        validate_pre_solve=True,
        retry_solver=True,
    )
    outcome = agent._execute_runtime_repair_plan(
        state,
        repair,
        approved_solver=plan.solver,
        llm_step=1,
        native_execution=True,
        runtime_event_start=0,
    )
    assert outcome is not None
    assert outcome.decision.value == "BLOCKED"
    assert agent.workspace.read_text("system/snappyHexMeshDict") == original
    assert state.engineering_events[-1].action_type == "native_toolchain_preflight"
    assert state.engineering_events[-1].failure_category == "infra"
