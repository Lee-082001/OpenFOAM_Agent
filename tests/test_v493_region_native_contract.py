from __future__ import annotations

from types import SimpleNamespace

import pytest

from openfoam_agent.engineering.case_build_graph import compile_case_build_graph
from openfoam_agent.schemas.engineering import NativeOpenFOAMCommand
from openfoam_agent.tools.native_contracts import native_command_region, required_dictionary


def _execution(paths: list[str], *, regions: list[str], pipeline=None):
    plan = SimpleNamespace(
        required_case_files=paths,
        region_layouts=[SimpleNamespace(region=name) for name in regions],
        execution=SimpleNamespace(regions=[]),
    )
    return SimpleNamespace(
        plan=plan,
        native_pipeline=list(pipeline or []),
        mesh_commands=[],
        validate_dictionaries=[],
        surface_checks=[],
    )


def _paths():
    return [
        "system/controlDict",
        "0/battery/T",
        "0/heater/T",
        "system/battery/fvSchemes",
        "system/battery/fvSolution",
        "system/heater/fvSchemes",
        "system/heater/fvSolution",
        "system/battery/blockMeshDict",
        "system/heater/blockMeshDict",
    ]


def test_region_required_dictionary_contract():
    assert native_command_region(["-region", "battery"]) == "battery"
    assert required_dictionary("blockMesh", ["-region", "battery"]) == "system/battery/blockMeshDict"
    with pytest.raises(ValueError):
        native_command_region(["-region", "../battery"])


def test_two_region_case_compiles_regional_mesh_and_validation_pipeline():
    paths = _paths()
    graph = compile_case_build_graph(
        _execution(paths, regions=["battery", "heater"]),
        {path: "dummy" for path in paths},
    )
    assert graph.valid, graph.failures
    assert [(x.command, x.arguments) for x in graph.native_pipeline] == [
        ("blockMesh", ["-region", "battery"]),
        ("blockMesh", ["-region", "heater"]),
        ("checkMesh", ["-region", "battery"]),
        ("checkMesh", ["-region", "heater"]),
    ]


def test_root_checkmesh_hint_is_not_authoritative_for_named_regions():
    paths = _paths()
    graph = compile_case_build_graph(
        _execution(
            paths,
            regions=["battery", "heater"],
            pipeline=[NativeOpenFOAMCommand(command="checkMesh", role="mesh_validation")],
        ),
        {path: "dummy" for path in paths},
    )
    checks = [x for x in graph.native_pipeline if x.command == "checkMesh"]
    assert [x.arguments for x in checks] == [["-region", "battery"], ["-region", "heater"]]
    assert any("<root>" in warning for warning in graph.warnings)
