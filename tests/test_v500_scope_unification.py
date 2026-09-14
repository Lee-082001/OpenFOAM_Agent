from __future__ import annotations

from types import SimpleNamespace

from conftest import make_state
from openfoam_agent.contracts.execution_scopes import (
    ROOT_SCOPE_KEY,
    current_mesh_evidence_failures,
    execution_scopes,
    scope_key,
    scope_mesh_digest,
)
from openfoam_agent.contracts.mesh_dependencies import MeshDependencyGraph
from openfoam_agent.schemas.engineering import (
    ExecutionScope,
    OpenFOAMExecutionSpec,
    RegionSolverAssignment,
)
from openfoam_agent.tools.execution_contracts import (
    control_dict_binding,
    runtime_arguments,
    sealed_solver_mirror,
    validate_execution_contract,
)
from openfoam_agent.tools.workspace import CaseWorkspace


def _multi_execution():
    return OpenFOAMExecutionSpec(
        driver="foamMultiRun",
        driver_provider_id="execution.foamMultiRun",
        scopes=[
            ExecutionScope(name="battery", solver_module="solid", solver_provider_id="solver.solid"),
            ExecutionScope(name="heater", solver_module="solid", solver_provider_id="solver.solid"),
        ],
    )


def test_v500_execution_schema_exposes_scopes_as_the_agent_topology_contract():
    schema = OpenFOAMExecutionSpec.model_json_schema()["properties"]
    assert "scopes" in schema
    assert "regions" not in schema
    assert "solver_module" not in schema
    assert "solver_provider_id" not in schema


def test_v500_legacy_execution_shapes_normalize_to_scopes_without_downstream_mode_branching():
    single = OpenFOAMExecutionSpec(
        driver="foamRun",
        driver_provider_id="execution.foamRun",
        solver_module="incompressibleFluid",
        solver_provider_id="solver.incompressibleFluid",
    )
    assert [(scope.name, scope.solver_module) for scope in single.scopes] == [(None, "incompressibleFluid")]

    multi = OpenFOAMExecutionSpec(
        driver="foamMultiRun",
        driver_provider_id="execution.foamMultiRun",
        regions=[
            RegionSolverAssignment(region="battery", solver_module="solid", provider_id="solver.solid"),
            RegionSolverAssignment(region="heater", solver_module="solid", provider_id="solver.solid"),
        ],
    )
    assert [(scope.name, scope.solver_module) for scope in multi.scopes] == [
        ("battery", "solid"), ("heater", "solid")
    ]


def test_v500_driver_contract_not_controller_branch_validates_scope_shape_and_runtime_binding():
    execution = OpenFOAMExecutionSpec(
        driver="foamRun",
        driver_provider_id="execution.foamRun",
        scopes=[ExecutionScope(
            name=None,
            solver_module="incompressibleFluid",
            solver_provider_id="solver.incompressibleFluid",
        )],
    )
    assert validate_execution_contract(execution) == []
    assert runtime_arguments(execution) == ["-solver", "incompressibleFluid"]
    assert control_dict_binding(execution) == ("scalar", "solver", "incompressibleFluid")
    assert sealed_solver_mirror(execution) == ("incompressibleFluid", "solver.incompressibleFluid")

    multi = _multi_execution()
    assert validate_execution_contract(multi) == []
    assert runtime_arguments(multi) == []
    assert control_dict_binding(multi) == (
        "named_map", "regionSolvers", {"battery": "solid", "heater": "solid"}
    )
    assert sealed_solver_mirror(multi) == ("foamMultiRun", "execution.foamMultiRun")


def test_v500_scopes_are_uniform_root_or_named_not_single_multi_modes():
    root = OpenFOAMExecutionSpec(
        driver="foamRun", driver_provider_id="execution.foamRun",
        scopes=[ExecutionScope(name=None, solver_module="solid", solver_provider_id="solver.solid")],
    )
    assert [scope_key(item) for item in execution_scopes(root)] == [ROOT_SCOPE_KEY]
    assert [scope_key(item) for item in execution_scopes(_multi_execution())] == [
        "region:battery", "region:heater"
    ]


def test_v500_material_only_repair_does_not_invalidate_mesh_digest(tmp_path):
    ws = CaseWorkspace(tmp_path)
    ws.write_text("system/battery/blockMeshDict", "FoamFile{}\nvertices();\n")
    ws.write_text("constant/battery/physicalProperties", "FoamFile{}\ntransport constIsoSolid;\n")
    plan = SimpleNamespace(execution=_multi_execution())
    graph = MeshDependencyGraph(ws)
    graph.record_native("blockMesh", ["-region", "battery"], plan)
    scope = execution_scopes(plan)[0]
    before = scope_mesh_digest(ws, scope)
    ws.write_text("constant/battery/physicalProperties", "FoamFile{}\ntransport polynomialSolid;\n")
    after_material = scope_mesh_digest(ws, scope)
    assert after_material == before
    ws.write_text("system/battery/blockMeshDict", "FoamFile{}\nvertices((0 0 0));\n")
    assert scope_mesh_digest(ws, scope) != before


def test_v500_mesh_evidence_gate_iterates_execution_scopes_uniformly(tmp_path):
    ws = CaseWorkspace(tmp_path)
    ws.write_text("system/battery/blockMeshDict", "FoamFile{}\n")
    ws.write_text("system/heater/blockMeshDict", "FoamFile{}\n")
    plan = SimpleNamespace(execution=_multi_execution())
    graph = MeshDependencyGraph(ws)
    graph.record_native("blockMesh", ["-region", "battery"], plan)
    graph.record_native("blockMesh", ["-region", "heater"], plan)
    scopes = execution_scopes(plan)
    evidence = SimpleNamespace(passed=True, cell_count=128)
    state = SimpleNamespace(
        mesh_evidence_by_scope={scope_key(scope): evidence for scope in scopes},
        mesh_manifest_by_scope={scope_key(scope): scope_mesh_digest(ws, scope) for scope in scopes},
    )
    assert current_mesh_evidence_failures(state, plan, ws, max_mesh_cells=5_000_000) == []

    ws.write_text("system/battery/blockMeshDict", "FoamFile{}\n// changed\n")
    failures = current_mesh_evidence_failures(state, plan, ws, max_mesh_cells=5_000_000)
    assert len(failures) == 1
    assert "battery" in failures[0]


def test_v500_state_has_one_scope_keyed_mesh_evidence_authority():
    state = make_state()
    assert hasattr(state, "mesh_evidence_by_scope")
    assert hasattr(state, "mesh_manifest_by_scope")


def test_v500_case_build_graph_compiles_all_scopes_through_one_generic_loop():
    from openfoam_agent.engineering.case_build_graph import compile_case_build_graph

    plan = SimpleNamespace(
        execution=_multi_execution(),
        required_case_files=[
            "system/battery/blockMeshDict",
            "system/heater/blockMeshDict",
        ],
    )
    action = SimpleNamespace(
        plan=plan,
        native_pipeline=[],
        mesh_commands=[],
        validate_dictionaries=[],
        surface_checks=[],
    )
    bundle = {
        "system/battery/blockMeshDict": "FoamFile{}\n",
        "system/heater/blockMeshDict": "FoamFile{}\n",
    }
    graph = compile_case_build_graph(action, bundle)
    assert graph.valid
    invocations = [(item.command, item.arguments) for item in graph.native_pipeline]
    assert invocations == [
        ("blockMesh", ["-region", "battery"]),
        ("blockMesh", ["-region", "heater"]),
        ("checkMesh", ["-region", "battery"]),
        ("checkMesh", ["-region", "heater"]),
    ]


def test_v500_case_delta_revalidates_only_the_mesh_scope_whose_dependency_changed(tmp_path):
    from openfoam_agent.engineering.case_delta_graph import compile_case_delta_graph

    ws = CaseWorkspace(tmp_path)
    ws.write_text("system/battery/blockMeshDict", "FoamFile{}\nvertices();\n")
    ws.write_text("system/heater/blockMeshDict", "FoamFile{}\nvertices();\n")
    plan = SimpleNamespace(
        execution=_multi_execution(),
        required_case_files=[
            "system/battery/blockMeshDict",
            "system/heater/blockMeshDict",
        ],
    )
    dependency = MeshDependencyGraph(ws)
    dependency.record_native("blockMesh", ["-region", "battery"], plan)
    dependency.record_native("blockMesh", ["-region", "heater"], plan)

    graph = compile_case_delta_graph(
        ws,
        plan,
        replacements=[(
            "system/battery/blockMeshDict",
            "FoamFile{}\nvertices((0 0 0));\n",
        )],
        validate_pre_solve=False,
    )
    assert graph.valid
    invocations = [(item.command, item.arguments) for item in graph.native_pipeline]
    assert ("blockMesh", ["-region", "battery"]) in invocations
    assert ("checkMesh", ["-region", "battery"]) in invocations
    assert ("blockMesh", ["-region", "heater"]) not in invocations
    assert ("checkMesh", ["-region", "heater"]) not in invocations
