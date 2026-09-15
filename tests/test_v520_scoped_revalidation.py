from __future__ import annotations

from conftest import foam_header, make_plan, make_state
from openfoam_agent.engineering.case_delta_graph import compile_case_delta_graph
from openfoam_agent.engineering.revalidation import record_scoped_revalidation
from openfoam_agent.schemas.engineering import ExecutionScope, NativeOpenFOAMCommand, OpenFOAMExecutionSpec
from openfoam_agent.tools.workspace import CaseWorkspace


def _two_region_plan(state):
    plan = make_plan(state.intake, solver="foamMultiRun")
    return plan.model_copy(update={
        "solver": "foamMultiRun",
        "solver_provider_id": "execution.foamMultiRun",
        "execution": OpenFOAMExecutionSpec(
            driver="foamMultiRun",
            driver_provider_id="execution.foamMultiRun",
            scopes=[
                ExecutionScope(name="battery", solver_module="solid", solver_provider_id="solver.solid"),
                ExecutionScope(name="heater", solver_module="solid", solver_provider_id="solver.solid"),
            ],
        ),
        "required_case_files": [
            "system/controlDict",
            "system/battery/fvSolution",
            "system/heater/fvSolution",
            "system/battery/blockMeshDict",
            "system/heater/blockMeshDict",
        ],
    })


def _workspace(tmp_path):
    ws = CaseWorkspace(tmp_path)
    for path in (
        "system/controlDict",
        "system/battery/fvSolution",
        "system/heater/fvSolution",
        "system/battery/blockMeshDict",
        "system/heater/blockMeshDict",
    ):
        ws.write_text(path, foam_header(path) + "testValue 1;\n")
    return ws


def test_v520_numerical_delta_reuses_all_mesh_scope_evidence(tmp_path):
    state = make_state()
    plan = _two_region_plan(state)
    ws = _workspace(tmp_path)
    path = "system/battery/fvSolution"
    graph = compile_case_delta_graph(
        ws,
        plan,
        replacements=[(path, foam_header(path) + "testValue 2;\n")],
        validate_pre_solve=True,
        phase="runtime_repair",
    )
    assert graph.valid, graph.failures
    assert graph.affected_mesh_scope_keys == ()
    assert graph.reused_mesh_scope_keys == ("region:battery", "region:heater")
    assert [item.command for item in graph.native_pipeline] == []
    assert graph.revalidation_domains == ("dictionary", "pre_solve")

    record = record_scoped_revalidation(
        state, graph, phase="runtime_repair", solver_retry_required=True
    )
    assert record.mesh_validation_scopes_avoided == 2
    assert record.solver_retry_required


def test_v520_single_region_mesh_delta_revalidates_only_affected_scope(tmp_path):
    state = make_state()
    plan = _two_region_plan(state)
    ws = _workspace(tmp_path)
    path = "system/battery/blockMeshDict"
    graph = compile_case_delta_graph(
        ws,
        plan,
        replacements=[(path, foam_header(path) + "testValue 2;\n")],
        validate_pre_solve=True,
        phase="repair",
    )
    assert graph.valid, graph.failures
    assert graph.affected_mesh_scope_keys == ("region:battery",)
    assert graph.reused_mesh_scope_keys == ("region:heater",)
    commands = [(item.command, item.arguments) for item in graph.native_pipeline]
    assert commands == [
        ("blockMesh", ["-region", "battery"]),
        ("checkMesh", ["-region", "battery"]),
    ]
    assert not any("heater" in args for _, args in commands)
    assert graph.revalidation_domains == ("dictionary", "mesh", "pre_solve")


def test_v520_delta_native_pipeline_uses_dependency_order(tmp_path):
    state = make_state()
    plan = make_plan(state.intake)
    plan.required_case_files = [
        "system/controlDict",
        "system/blockMeshDict",
        "system/surfaceFeaturesDict",
        "system/snappyHexMeshDict",
    ]
    ws = CaseWorkspace(tmp_path)
    for path in plan.required_case_files:
        ws.write_text(path, foam_header(path) + "testValue 1;\n")
    graph = compile_case_delta_graph(
        ws,
        plan,
        replacements=[
            ("system/blockMeshDict", foam_header("system/blockMeshDict") + "testValue 2;\n"),
            ("system/surfaceFeaturesDict", foam_header("system/surfaceFeaturesDict") + "testValue 2;\n"),
            ("system/snappyHexMeshDict", foam_header("system/snappyHexMeshDict") + "testValue 2;\n"),
        ],
        native_hints=[
            NativeOpenFOAMCommand(command="snappyHexMesh", arguments=["-overwrite"], role="mesh"),
            NativeOpenFOAMCommand(command="surfaceFeatures", role="mesh"),
            NativeOpenFOAMCommand(command="blockMesh", role="mesh"),
        ],
        validate_pre_solve=True,
        phase="repair",
    )
    assert graph.valid, graph.failures
    commands = [item.command for item in graph.native_pipeline]
    assert commands.index("blockMesh") < commands.index("snappyHexMesh")
    assert commands.index("surfaceFeatures") < commands.index("snappyHexMesh")
    assert commands[-1] == "checkMesh"
