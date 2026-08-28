from __future__ import annotations

import pytest

from openfoam_agent.tools.safe_runner import SafeRunner, UnsafeCommandError
from openfoam_agent.tools.workspace import CaseWorkspace, WorkspaceSafetyError


def test_workspace_rejects_path_escape(tmp_path):
    ws = CaseWorkspace(tmp_path)
    with pytest.raises(WorkspaceSafetyError):
        ws.write_text("../system/controlDict", "solver incompressibleFluid;\n")
    with pytest.raises(WorkspaceSafetyError):
        ws.write_text("/etc/passwd", "x")


def test_workspace_rejects_runtime_code_directives(tmp_path):
    ws = CaseWorkspace(tmp_path)
    with pytest.raises(WorkspaceSafetyError, match="unsafe directives"):
        ws.write_text("system/controlDict", "#codeStream\nsolver incompressibleFluid;\n")


def test_workspace_rejects_untrusted_dynamic_library(tmp_path):
    ws = CaseWorkspace(tmp_path)
    with pytest.raises(WorkspaceSafetyError, match="non-allowlisted"):
        ws.write_text("system/controlDict", 'libs ("libEvil.so");\nsolver incompressibleFluid;\n')


def test_workspace_seal_detects_tampering(tmp_path, make_state=None):
    from conftest import make_intake, make_plan

    ws = CaseWorkspace(tmp_path)
    intake = make_intake()
    plan = make_plan(intake)
    ws.write_text("system/controlDict", "solver incompressibleFluid;\n")
    seal = ws.seal(plan)
    ws.resolve_case_path("system/controlDict").write_text("solver fluid;\n", encoding="utf-8")
    with pytest.raises(WorkspaceSafetyError, match="changed"):
        ws.verify_seal(seal, plan)


def test_safe_runner_rejects_non_allowlisted_executable(tmp_path):
    runner = SafeRunner(workspace_root=tmp_path)
    with pytest.raises(UnsafeCommandError, match="not allowlisted"):
        runner.run(["bash", "-c", "echo no"], cwd=tmp_path, timeout=1)


def test_safe_runner_rejects_cwd_escape(tmp_path):
    runner = SafeRunner(workspace_root=tmp_path)
    with pytest.raises(UnsafeCommandError, match="escapes workspace"):
        runner.run(["checkMesh"], cwd=tmp_path.parent, timeout=1)


def test_seal_includes_native_generated_mesh_inputs(tmp_path):
    from conftest import make_intake, make_plan

    ws = CaseWorkspace(tmp_path)
    plan = make_plan(make_intake())
    ws.write_text("system/controlDict", "solver incompressibleFluid;\n")
    native_mesh = ws.case_dir / "constant" / "polyMesh" / "points"
    native_mesh.parent.mkdir(parents=True, exist_ok=True)
    native_mesh.write_text("(0 0 0)\n", encoding="utf-8")
    seal = ws.seal(plan)
    entries = {item.path: item for item in seal.files}
    assert entries["system/controlDict"].origin == "agent"
    assert entries["constant/polyMesh/points"].origin == "native"

    native_mesh.write_text("(1 0 0)\n", encoding="utf-8")
    with pytest.raises(WorkspaceSafetyError, match="execution input changed"):
        ws.verify_seal(seal, plan)
