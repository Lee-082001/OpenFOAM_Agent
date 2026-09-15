from __future__ import annotations

from types import SimpleNamespace

from conftest import make_state
from openfoam_agent.engineering.case_build_graph import compile_case_build_graph
from openfoam_agent.engineering.candidate_replan_context import build_partitioned_candidate_replan_prompt
from openfoam_agent.schemas.engineering import ExecutionScope, OpenFOAMExecutionSpec


def _two_solid_plan():
    return SimpleNamespace(
        execution=OpenFOAMExecutionSpec(
            driver="foamMultiRun",
            driver_provider_id="execution.foamMultiRun",
            scopes=[
                ExecutionScope(name="battery", solver_module="solid", solver_provider_id="solver.solid"),
                ExecutionScope(name="heater", solver_module="solid", solver_provider_id="solver.solid"),
            ],
        ),
        required_case_files=[
            "system/controlDict",
            "system/battery/blockMeshDict",
            "system/heater/blockMeshDict",
            "constant/battery/polyMesh/points",
            "constant/battery/polyMesh/faces",
            "constant/battery/polyMesh/owner",
            "constant/battery/polyMesh/neighbour",
            "constant/battery/polyMesh/boundary",
            "constant/heater/polyMesh/points",
            "constant/heater/polyMesh/faces",
            "constant/heater/polyMesh/owner",
            "constant/heater/polyMesh/neighbour",
            "constant/heater/polyMesh/boundary",
        ],
    )


def _action(plan):
    return SimpleNamespace(
        plan=plan,
        native_pipeline=[],
        mesh_commands=[],
        validate_dictionaries=[],
        surface_checks=[],
    )


def test_v505_two_region_blockmesh_outputs_are_deferred_native_requirements():
    plan = _two_solid_plan()
    bundle = {
        "system/controlDict": "FoamFile{}\n",
        "system/battery/blockMeshDict": "FoamFile{}\n",
        "system/heater/blockMeshDict": "FoamFile{}\n",
    }
    graph = compile_case_build_graph(_action(plan), bundle)
    assert graph.valid
    assert graph.missing_required_paths == ()
    assert len(graph.deferred_native_required_paths) == 10
    assert set(graph.authored_required_paths) == set(bundle)
    assert [(x.command, x.arguments) for x in graph.native_pipeline] == [
        ("blockMesh", ["-region", "battery"]),
        ("blockMesh", ["-region", "heater"]),
        ("checkMesh", ["-region", "battery"]),
        ("checkMesh", ["-region", "heater"]),
    ]


def test_v505_poly_mesh_is_not_deferred_without_a_compiled_native_producer():
    plan = _two_solid_plan()
    bundle = {"system/controlDict": "FoamFile{}\n"}
    graph = compile_case_build_graph(_action(plan), bundle)
    assert not graph.valid
    assert "constant/battery/polyMesh/points" in graph.missing_required_paths
    assert graph.deferred_native_required_paths == ()


def test_v505_candidate_replan_partition_stays_bounded_and_preserves_native_deferred_files():
    retained = {
        "goal": "repair retained candidate",
        "failed_paths": ["system/fvSolution"],
        "missing_required_files": ["system/fvSolution"],
        "deferred_native_required_files": [f"constant/battery/polyMesh/file{i}" for i in range(20)],
        "manifest": [{"path": f"system/file{i}", "kind": "raw", "chars": 10000} for i in range(120)],
        "failed_artifacts": [
            {"path": "system/fvSolution", "kind": "raw", "content": "x" * 60000}
        ],
        "controller_build_policy": {"manifest_authority": "EngineeringPlan.required_case_files"},
        "plan_capsule": {
            "solver": "foamMultiRun",
            "solver_provider_id": "execution.foamMultiRun",
            "confirmed_intake_sha256": "a" * 64,
            "required_case_files": ["system/fvSolution"] + [f"constant/battery/polyMesh/file{i}" for i in range(20)],
        },
    }
    payload = {
        "phase": "prepare",
        "step": 4,
        "retained_candidate": retained,
        "deterministic_bindings": {"confirmed_intake": {"sha256": "a" * 64}},
        "budget": {"steps_remaining_in_current_window": 4},
    }
    built, capsule, metrics = build_partitioned_candidate_replan_prompt(
        "Repair retained candidate:\n", payload, max_chars=18_000
    )
    assert len(built.prompt) <= 18_000
    assert metrics["candidateReplanPartitioned"] == 1
    assert capsule["retained_candidate"]["missing_required_files"] == ["system/fvSolution"]
    assert capsule["retained_candidate"]["deferred_native_required_files"]


def test_v505_retained_candidate_never_requests_native_poly_mesh_outputs_from_model(tmp_path, graph_path):
    from conftest import ScriptedLLM, make_plan, make_state
    from openfoam_agent.engineering import CFDEngineeringAgent
    from openfoam_agent.schemas.engineering import CaseBundleFile, ExecuteCasePlanAction

    state = make_state()
    base = make_plan(state.intake, solver="foamMultiRun")
    plan = base.model_copy(update={
        "solver": "foamMultiRun",
        "solver_provider_id": "execution.foamMultiRun",
        "execution": _two_solid_plan().execution,
        "required_case_files": _two_solid_plan().required_case_files + ["system/fvSolution"],
    })
    candidate = ExecuteCasePlanAction(
        type="execute_case_plan",
        goal="two solid blockMesh candidate",
        files=[
            CaseBundleFile(path="system/controlDict", content="x"),
            CaseBundleFile(path="system/battery/blockMeshDict", content="x"),
            CaseBundleFile(path="system/heater/blockMeshDict", content="x"),
        ],
        plan=plan,
    )
    agent = CFDEngineeringAgent(ScriptedLLM([]), workspace=tmp_path, capability_db=graph_path)
    agent._pending_candidate_execution = candidate
    agent._pending_candidate_failed_paths = ("system/fvSolution",)
    capsule = agent._candidate_repair_context()
    assert capsule is not None
    assert capsule["missing_required_files"] == ["system/fvSolution"]
    assert len(capsule["deferred_native_required_files"]) == 10
    assert not any("polyMesh" in path for path in capsule["missing_required_files"])
