from __future__ import annotations

import os
from pathlib import Path

import pytest

from openfoam_agent.cli import _default_capability_db
from openfoam_agent.schemas.engineering import OpenFOAMExecutionSpec, RegionSolverAssignment
from openfoam_agent.tools.capability_catalog import CapabilityCatalog
from openfoam_agent.tools.openfoam import OpenFOAMTools
from openfoam_agent.tools.safe_runner import SafeRunner, UnsafeCommandError
from openfoam_agent.verification.foam_semantics.parser import parse_named_dictionary_assignments


def _make_executable(path: Path, body: str = '#!/bin/sh\nprintf "%s\\n" "$@"\n') -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
    path.chmod(0o755)


def _fake_install(tmp_path: Path, version: str, *executables: str):
    root = tmp_path / f"OpenFOAM-{version}"
    appbin = root / "platforms" / "linux64" / "bin"
    modules = root / "applications" / "modules"
    src = root / "src"
    for name in executables:
        _make_executable(appbin / name)
    # One arbitrary non-profile solver module proves source discovery is not a static enum.
    custom = modules / "customThermalModule" / "Make"
    custom.mkdir(parents=True, exist_ok=True)
    (custom / "files").write_text("customThermalModule.C\n", encoding="utf-8")

    # Runtime-selection registrations prove that installed fvModel/functionObject types
    # are discovered from OpenFOAM's own registration semantics rather than directory
    # names or a Python enum.
    fv_model = src / "fvModels" / "general" / "customLatentModel" / "customLatentModel.C"
    fv_model.parent.mkdir(parents=True, exist_ok=True)
    fv_model.write_text(
        "addToRunTimeSelectionTable(fvModel, customLatentModel, dictionary);\n",
        encoding="utf-8",
    )
    named_model = src / "fvModels" / "general" / "namedModel" / "namedModel.C"
    named_model.parent.mkdir(parents=True, exist_ok=True)
    named_model.write_text(
        "addNamedToRunTimeSelectionTable(fvModel, namedModel, dictionary, customLookupModel);\n",
        encoding="utf-8",
    )
    function_object = src / "functionObjects" / "field" / "customFunction" / "customFunction.C"
    function_object.parent.mkdir(parents=True, exist_ok=True)
    function_object.write_text(
        "addToRunTimeSelectionTable(functionObject, customFunction, dictionary);\n",
        encoding="utf-8",
    )
    env = {
        "WM_PROJECT": "OpenFOAM",
        "WM_PROJECT_VERSION": version,
        "WM_PROJECT_DIR": str(root),
        "FOAM_APPBIN": str(appbin),
        "FOAM_MODULES": str(modules),
        "FOAM_SRC": str(src),
        "PATH": os.pathsep.join((str(appbin), "/usr/bin", "/bin")),
    }
    return root, appbin, env


@pytest.mark.parametrize("version", ["13", "14"])
def test_sourced_installation_discovers_every_trusted_application_not_static_allowlist(tmp_path, version):
    root, _, env = _fake_install(
        tmp_path,
        version,
        "foamRun",
        "foamMultiRun",
        "checkMesh",
        "splitMeshRegions",
        "brandNewUtility",
    )
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    runner = SafeRunner(
        workspace_root=workspace,
        trusted_executable_roots=[root],
        base_env=env,
    )

    assert "brandNewUtility" in runner.allowed_commands
    assert "splitMeshRegions" in runner.allowed_commands
    assert "foamMultiRun" in runner.allowed_commands
    assert "ls" not in runner.allowed_commands
    assert runner.installation.version == version
    assert "customThermalModule" in runner.installation.solver_modules
    assert "solid" not in runner.installation.solver_modules  # docs are not installed evidence

    tools = OpenFOAMTools(runner)
    result = tools.run_native_command("brandNewUtility", workspace, arguments=["hello"])
    assert result.success
    assert result.stdout.strip() == "hello"


@pytest.mark.parametrize("version", ["13", "14"])
def test_discovered_catalog_merges_installed_tools_modules_and_documented_graph(tmp_path, version):
    root, _, env = _fake_install(tmp_path, version, "foamRun", "foamMultiRun", "newMeshThing")
    runner = SafeRunner(
        workspace_root=tmp_path / "workspace",
        trusted_executable_roots=[root],
        base_env=env,
    )
    catalog = CapabilityCatalog(
        Path(__file__).resolve().parents[1] / "config" / f"openfoam{version}_capability_graph.json",
        installation=runner.installation,
    )

    assert catalog.provider("execution.foamMultiRun").name == "foamMultiRun"
    assert catalog.provider("solver.solid").name == "solid"
    assert catalog.provider("model.heatSource").name == "heatSource"
    assert any(item["name"] == "newMeshThing" for item in catalog.search("newMeshThing"))
    assert any(item["name"] == "customThermalModule" for item in catalog.search("customThermalModule"))


@pytest.mark.parametrize("version", ["13", "14"])
def test_foundation_phase_change_models_are_available_as_documented_fallback(tmp_path, version):
    root, _, env = _fake_install(tmp_path, version, "foamRun")
    runner = SafeRunner(
        workspace_root=tmp_path / "workspace",
        trusted_executable_roots=[root],
        base_env=env,
    )
    assert "solidificationMelting" not in runner.installation.fv_models
    assert "VoFSolidificationMelting" not in runner.installation.fv_models

    catalog = CapabilityCatalog(
        Path(__file__).resolve().parents[1] / "config" / f"openfoam{version}_capability_graph.json",
        installation=runner.installation,
    )
    melting = {item["name"] for item in catalog.search("melting", limit=12)}
    assert "solidificationMelting" in melting
    assert "VoFSolidificationMelting" in melting
    assert catalog.provider("model.solidificationMelting").verification_level == "documented"
    assert catalog.provider("installed.fv_model.solidificationMelting") is None



@pytest.mark.parametrize("version", ["13", "14"])
def test_sourced_installation_discovers_runtime_selection_types_from_openfoam_source(tmp_path, version):
    root, _, env = _fake_install(tmp_path, version, "foamRun")
    runner = SafeRunner(
        workspace_root=tmp_path / "workspace",
        trusted_executable_roots=[root],
        base_env=env,
    )

    assert "customLatentModel" in runner.installation.fv_models
    assert "customLookupModel" in runner.installation.fv_models
    assert any(
        item.category == "function_object" and item.name == "customFunction"
        for item in runner.installation.components
    )

    catalog = CapabilityCatalog(
        Path(__file__).resolve().parents[1] / "config" / f"openfoam{version}_capability_graph.json",
        installation=runner.installation,
    )
    installed = catalog.provider("installed.fv_model.customLatentModel")
    assert installed is not None
    assert installed.verification_level == "installed"
    assert catalog.provider("installed.fv_model.customLookupModel") is not None

def test_native_generic_tool_cannot_escape_case_or_override_case_root(tmp_path):
    root, _, env = _fake_install(tmp_path, "14", "arbitraryFoamTool")
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    tools = OpenFOAMTools(
        SafeRunner(workspace_root=workspace, trusted_executable_roots=[root], base_env=env)
    )
    with pytest.raises((UnsafeCommandError, ValueError)):
        tools.run_native_command("arbitraryFoamTool", workspace, arguments=["../escape"])
    with pytest.raises(ValueError):
        tools.run_native_command("arbitraryFoamTool", workspace, arguments=["-case", "/tmp"])


def test_execution_ir_supports_single_and_multi_region_foundation_drivers(tmp_path):
    root, _, env = _fake_install(tmp_path, "13", "foamRun", "foamMultiRun")
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    tools = OpenFOAMTools(
        SafeRunner(workspace_root=workspace, trusted_executable_roots=[root], base_env=env)
    )

    single = OpenFOAMExecutionSpec(
        driver="foamRun",
        driver_provider_id="execution.foamRun",
        solver_module="incompressibleFluid",
        solver_provider_id="solver.incompressibleFluid",
    )
    result = tools.run_execution(workspace, single)
    assert result.success
    assert result.stdout.splitlines()[:2] == ["-solver", "incompressibleFluid"]

    multi = OpenFOAMExecutionSpec(
        driver="foamMultiRun",
        driver_provider_id="execution.foamMultiRun",
        regions=[
            RegionSolverAssignment(region="water", solver_module="fluid", provider_id="solver.fluid"),
            RegionSolverAssignment(region="battery", solver_module="solid", provider_id="solver.solid"),
        ],
    )
    result = tools.run_execution(workspace, multi)
    assert result.success
    assert result.stdout.strip() == ""  # regionSolvers comes from controlDict, no fake -solver argument


def test_region_solvers_is_interpreted_as_semantic_mapping_not_substring():
    text = """
    FoamFile { format ascii; class dictionary; object controlDict; }
    regionSolvers
    {
        water       fluid;
        battery     solid;
    }
    """
    mapping, complete = parse_named_dictionary_assignments(text, "regionSolvers")
    assert complete
    assert mapping == {"water": "fluid", "battery": "solid"}


def test_cli_default_capability_profile_tracks_sourced_foundation_version(monkeypatch):
    monkeypatch.setenv("WM_PROJECT_VERSION", "13")
    assert _default_capability_db().name == "openfoam13_capability_graph.json"
    monkeypatch.setenv("WM_PROJECT_VERSION", "14")
    assert _default_capability_db().name == "openfoam14_capability_graph.json"


def test_dynamic_command_authority_requires_sourced_foundation_13_or_14(tmp_path):
    root, appbin, env = _fake_install(tmp_path, "14", "brandNewUtility")
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    wrong_version = dict(env, WM_PROJECT_VERSION="12")
    runner = SafeRunner(
        workspace_root=workspace,
        trusted_executable_roots=[root],
        base_env=wrong_version,
    )
    assert runner.installation.version is None
    assert not runner.installation.installation_configured
    assert "brandNewUtility" not in runner.allowed_commands

    wrong_project = dict(env, WM_PROJECT="NotOpenFOAM")
    runner = SafeRunner(
        workspace_root=workspace,
        trusted_executable_roots=[root],
        base_env=wrong_project,
    )
    assert runner.installation.version == "14"
    assert not runner.installation.installation_configured
    assert "brandNewUtility" not in runner.allowed_commands


def test_safety_gate_validates_multiregion_region_solver_semantics(tmp_path):
    from conftest import FakeOpenFOAMTools, make_intake, make_plan
    from openfoam_agent.tools.workspace import CaseWorkspace
    from openfoam_agent.verification.safety import DeterministicSafetyGate

    intake = make_intake()
    base = make_plan(intake)
    execution = OpenFOAMExecutionSpec(
        driver="foamMultiRun",
        driver_provider_id="execution.foamMultiRun",
        regions=[
            RegionSolverAssignment(region="water", solver_module="fluid", provider_id="solver.fluid"),
            RegionSolverAssignment(region="battery", solver_module="solid", provider_id="solver.solid"),
        ],
    )
    plan = base.model_copy(
        update={
            "solver": "foamMultiRun",
            "solver_provider_id": "execution.foamMultiRun",
            "execution": execution,
        }
    )
    ws = CaseWorkspace(tmp_path)
    ws.write_text(
        "system/controlDict",
        "regionSolvers\n{\n    water fluid;\n    battery solid;\n}\n",
    )
    gate = DeterministicSafetyGate(FakeOpenFOAMTools(), ws)
    result = gate.validate_plan(plan, intake)
    assert result.valid, result.failures

    ws.write_text(
        "system/controlDict",
        "regionSolvers\n{\n    water incompressibleFluid;\n    battery solid;\n}\n",
    )
    result = gate.validate_plan(plan, intake)
    assert not result.valid
    assert any("regionSolvers disagrees" in item for item in result.failures)
