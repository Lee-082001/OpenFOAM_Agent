from __future__ import annotations

from types import SimpleNamespace

from conftest import ScriptedLLM, make_plan, make_state
from openfoam_agent.engineering import CFDEngineeringAgent, EngineeringPolicy
from openfoam_agent.engineering.case_build_graph import compile_case_build_graph
from openfoam_agent.schemas.engineering import (
    CaseBundleFile,
    ExecuteCasePlanAction,
    ExecutionScope,
    NativeOpenFOAMCommand,
    OpenFOAMExecutionSpec,
    RetrySolverAction,
)
from openfoam_agent.tools.native_contracts import command_permitted, required_dictionary
from openfoam_agent.tools.safe_runner import SafeRunner
from openfoam_agent.tools.workspace import CaseWorkspace
from openfoam_agent.workflow.states import State


def _root_execution(plan, pipeline):
    files = [
        CaseBundleFile(path="system/controlDict", content="FoamFile{}\n"),
        CaseBundleFile(path="system/blockMeshDict", content="FoamFile{}\n"),
        CaseBundleFile(path="system/surfaceFeatureExtractDict", content="FoamFile{}\n"),
        CaseBundleFile(path="system/snappyHexMeshDict", content="FoamFile{}\n"),
        CaseBundleFile(path="constant/triSurface/body.stl", content="solid x\nendsolid x\n"),
    ]
    return ExecuteCasePlanAction(
        type="execute_case_plan",
        goal="mesh",
        files=files,
        native_pipeline=pipeline,
        plan=plan,
    )


def test_v506_dependency_dag_preserves_feature_extraction_before_snappy():
    state = make_state()
    plan = make_plan(state.intake)
    plan.required_case_files = ["system/controlDict", "system/blockMeshDict", "system/snappyHexMeshDict"]
    action = _root_execution(
        plan,
        [
            NativeOpenFOAMCommand(command="blockMesh", role="mesh"),
            NativeOpenFOAMCommand(command="surfaceFeatureExtract", role="mesh"),
            NativeOpenFOAMCommand(command="snappyHexMesh", arguments=["-overwrite"], role="mesh"),
        ],
    )
    graph = compile_case_build_graph(action, {item.path: item.content for item in action.files})
    assert graph.valid, graph.failures
    commands = [item.command for item in graph.native_pipeline]
    assert commands[:3] == ["blockMesh", "surfaceFeatureExtract", "snappyHexMesh"]
    assert commands[-1] == "checkMesh"


def test_v506_dependency_dag_repairs_agent_order_from_declared_producers():
    state = make_state()
    plan = make_plan(state.intake)
    plan.required_case_files = ["system/controlDict", "system/blockMeshDict", "system/snappyHexMeshDict"]
    action = _root_execution(
        plan,
        [
            NativeOpenFOAMCommand(command="snappyHexMesh", arguments=["-overwrite"], role="mesh"),
            NativeOpenFOAMCommand(command="surfaceFeatureExtract", role="mesh"),
            NativeOpenFOAMCommand(command="blockMesh", role="mesh"),
        ],
    )
    graph = compile_case_build_graph(action, {item.path: item.content for item in action.files})
    assert graph.valid, graph.failures
    commands = [item.command for item in graph.native_pipeline]
    snappy = commands.index("snappyHexMesh")
    assert commands.index("blockMesh") < snappy
    assert commands.index("surfaceFeatureExtract") < snappy


def test_v506_foundation_surface_features_is_a_first_class_native_contract(tmp_path):
    assert command_permitted("surfaceFeatures", "authoring")
    assert required_dictionary("surfaceFeatures", []) == "system/surfaceFeaturesDict"
    assert "surfaceFeatures" in SafeRunner.OFFLINE_FALLBACK_ALLOWED
    assert CaseWorkspace(tmp_path).is_mesh_affecting_path("system/surfaceFeaturesDict")

    state = make_state()
    plan = make_plan(state.intake)
    plan.required_case_files = ["system/controlDict", "system/blockMeshDict", "system/snappyHexMeshDict"]
    files = [
        CaseBundleFile(path="system/controlDict", content="FoamFile{}\n"),
        CaseBundleFile(path="system/blockMeshDict", content="FoamFile{}\n"),
        CaseBundleFile(path="system/surfaceFeaturesDict", content="FoamFile{}\n"),
        CaseBundleFile(path="system/snappyHexMeshDict", content="FoamFile{}\n"),
        CaseBundleFile(path="constant/triSurface/body.stl", content="solid x\nendsolid x\n"),
    ]
    action = ExecuteCasePlanAction(
        type="execute_case_plan",
        goal="Foundation feature mesh",
        files=files,
        native_pipeline=[
            NativeOpenFOAMCommand(command="snappyHexMesh", arguments=["-overwrite"], role="mesh"),
            NativeOpenFOAMCommand(command="surfaceFeatures", role="mesh"),
            NativeOpenFOAMCommand(command="blockMesh", role="mesh"),
        ],
        plan=plan,
    )
    graph = compile_case_build_graph(action, {item.path: item.content for item in files})
    assert graph.valid, graph.failures
    commands = [item.command for item in graph.native_pipeline]
    snappy = commands.index("snappyHexMesh")
    assert commands.index("surfaceFeatures") < snappy
    assert commands.index("blockMesh") < snappy


def test_v506_multiregion_retry_does_not_require_legacy_root_mesh_evidence(tmp_path, graph_path, monkeypatch):
    state = make_state()
    base = make_plan(state.intake, solver="foamMultiRun")
    plan = base.model_copy(update={
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
    })
    state.mesh_evidence = None  # legacy root mirror is intentionally absent
    state.current_state = State.ENGINEERING
    agent = CFDEngineeringAgent(
        ScriptedLLM([]), workspace=tmp_path, capability_db=graph_path,
        policy=EngineeringPolicy(require_solve_ready_gate=False),
    )

    monkeypatch.setattr(agent.safety, "validate_plan", lambda *_: SimpleNamespace(valid=True, failures=[]))
    monkeypatch.setattr(agent.safety, "validate_native_inputs", lambda: SimpleNamespace(valid=True, failures=[]))
    monkeypatch.setattr(agent, "_mesh_evidence_failures", lambda *_: [])
    monkeypatch.setattr(agent, "_validate_observed_provenance", lambda *_: [])
    monkeypatch.setattr(agent, "_validate_engineering_defaults", lambda *_: [])
    monkeypatch.setattr(agent.workspace, "seal", lambda *_: SimpleNamespace())

    event, outcome = agent._dispatch_retry_solver(
        state,
        RetrySolverAction(type="retry_solver", plan=plan, rationale="same approved solver"),
        approved_solver="foamMultiRun",
        step=1,
        native_execution=True,
    )
    assert event.success
    assert outcome is not None and outcome.retry
    assert state.current_state == State.SIMULATION
