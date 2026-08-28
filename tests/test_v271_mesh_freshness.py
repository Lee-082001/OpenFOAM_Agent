from __future__ import annotations

from openfoam_agent.engineering import CFDEngineeringAgent, EngineeringPolicy
from openfoam_agent.schemas.engineering import (
    BlockAction,
    FinishPreviewAction,
    RetrySolverAction,
    RunMeshCommandAction,
    SearchCapabilitiesAction,
    WriteCaseFileAction,
)
from openfoam_agent.tools.workspace import CaseWorkspace
from openfoam_agent.workflow.states import State

from conftest import FakeOpenFOAMTools, ScriptedLLM, control_dict, make_plan, make_state, mesh_ok_log, tool_result


def test_mesh_manifest_ignores_solver_inputs_but_tracks_mesh_artifacts(tmp_path):
    workspace = CaseWorkspace(tmp_path)
    baseline = workspace.mesh_manifest_digest()

    workspace.write_text("system/fvSchemes", "ddtSchemes {};\n")
    workspace.write_text("system/fvSolution", "solvers {};\n")
    workspace.write_text("0/U", "dimensions [0 1 -1 0 0 0 0];\n")
    workspace.write_text("system/controlDict", control_dict())
    assert workspace.mesh_manifest_digest() == baseline

    workspace.write_text(
        "system/blockMeshDict",
        "FoamFile { version 2.0; format ascii; class dictionary; object blockMeshDict; }\n",
    )
    after_block_mesh = workspace.mesh_manifest_digest()
    assert after_block_mesh != baseline

    workspace.write_text("constant/triSurface/cylinder.stl", "solid cylinder\nendsolid cylinder\n")
    assert workspace.mesh_manifest_digest() != after_block_mesh


def test_mesh_path_classifier_is_narrow_and_explicit(tmp_path):
    workspace = CaseWorkspace(tmp_path)
    assert workspace.is_mesh_affecting_path("system/blockMeshDict")
    assert workspace.is_mesh_affecting_path("system/snappyHexMeshDict")
    assert workspace.is_mesh_affecting_path("system/surfaceFeatureExtractDict")
    assert workspace.is_mesh_affecting_path("system/createPatchDict")
    assert workspace.is_mesh_affecting_path("constant/polyMesh/boundary")
    assert workspace.is_mesh_affecting_path("constant/triSurface/cylinder.stl")
    assert workspace.is_mesh_affecting_path("constant/geometry/cylinder.obj")

    assert not workspace.is_mesh_affecting_path("system/fvSchemes")
    assert not workspace.is_mesh_affecting_path("system/fvSolution")
    assert not workspace.is_mesh_affecting_path("system/controlDict")
    assert not workspace.is_mesh_affecting_path("0/U")
    assert not workspace.is_mesh_affecting_path("0/p")
    assert not workspace.is_mesh_affecting_path("constant/physicalProperties")


def _prepared_agent(tmp_path, graph_path, tools, repair_actions):
    state = make_state()
    plan = make_plan(state.intake)
    llm = ScriptedLLM(
        [
            SearchCapabilitiesAction(
                type="search_capabilities",
                query="incompressibleFluid",
                rationale="Observe solver capability evidence.",
            ),
            WriteCaseFileAction(
                type="write_case_file",
                path="system/controlDict",
                content=control_dict(),
                rationale="control",
            ),
            RunMeshCommandAction(type="run_mesh_command", command="checkMesh", rationale="mesh evidence"),
            FinishPreviewAction(type="finish_preview", plan=plan, rationale="seal"),
            *repair_actions,
        ]
    )
    agent = CFDEngineeringAgent(
        llm,
        workspace=tmp_path,
        capability_db=graph_path,
        tools=tools,
        policy=EngineeringPolicy(max_agent_steps=12),
    )
    agent.prepare(state, native_execution=True)
    assert state.current_state == State.MESH_READY
    return state, plan, agent


def test_runtime_solver_input_repair_does_not_require_checkmesh_rerun(tmp_path, graph_path):
    state_seed = make_state()
    plan = make_plan(state_seed.intake)
    tools = FakeOpenFOAMTools(
        mesh_results={"checkMesh": [tool_result("checkMesh", success=True, stdout=mesh_ok_log())]}
    )
    state, _, agent = _prepared_agent(
        tmp_path,
        graph_path,
        tools,
        [
            WriteCaseFileAction(
                type="write_case_file",
                path="system/fvSolution",
                content="solvers {};\n",
                rationale="repair solver dictionary",
            ),
            RetrySolverAction(type="retry_solver", plan=plan, rationale="retry without redundant checkMesh"),
        ],
    )

    outcome = agent.repair_runtime(
        state,
        runtime_log="--> FOAM FATAL ERROR: bad fvSolution\n",
        attempt=1,
        native_execution=True,
    )

    assert outcome.retry
    assert tools.mesh_calls == ["checkMesh"]
    assert state.mesh_evidence is not None and state.mesh_evidence.passed


def test_runtime_mesh_repair_still_requires_fresh_checkmesh(tmp_path, graph_path):
    state_seed = make_state()
    plan = make_plan(state_seed.intake)
    tools = FakeOpenFOAMTools(
        mesh_results={"checkMesh": [tool_result("checkMesh", success=True, stdout=mesh_ok_log())]}
    )
    state, _, agent = _prepared_agent(
        tmp_path,
        graph_path,
        tools,
        [
            WriteCaseFileAction(
                type="write_case_file",
                path="system/blockMeshDict",
                content="FoamFile { version 2.0; format ascii; class dictionary; object blockMeshDict; }\n",
                rationale="repair mesh input",
            ),
            RetrySolverAction(type="retry_solver", plan=plan, rationale="attempt retry with stale mesh evidence"),
            BlockAction(
                type="block",
                reason="Fresh checkMesh is required after a mesh-affecting edit.",
                needs_user_input=False,
                rationale="Do not bypass mesh freshness validation.",
            ),
        ],
    )

    outcome = agent.repair_runtime(
        state,
        runtime_log="--> FOAM FATAL ERROR: mesh-related failure\n",
        attempt=1,
        native_execution=True,
    )

    assert not outcome.retry
    rejected = [event for event in state.engineering_events if event.action_type == "retry_solver"]
    assert rejected and not rejected[-1].success
    assert "passing checkMesh" in rejected[-1].output_excerpt


def test_rehydrated_runtime_agent_restores_mesh_freshness_from_sealed_case(tmp_path, graph_path):
    state_seed = make_state()
    plan = make_plan(state_seed.intake)
    tools = FakeOpenFOAMTools(
        mesh_results={"checkMesh": [tool_result("checkMesh", success=True, stdout=mesh_ok_log())]}
    )
    state, _, original_agent = _prepared_agent(tmp_path, graph_path, tools, [])
    assert original_agent is not None

    rehydrated = CFDEngineeringAgent(
        ScriptedLLM(
            [
                WriteCaseFileAction(
                    type="write_case_file",
                    path="system/fvSolution",
                    content="solvers {};\n",
                    rationale="repair only solver input",
                ),
                RetrySolverAction(type="retry_solver", plan=plan, rationale="retry with persisted mesh evidence"),
            ]
        ),
        workspace=tmp_path,
        capability_db=graph_path,
        tools=tools,
        policy=EngineeringPolicy(max_agent_steps=12),
    )

    outcome = rehydrated.repair_runtime(
        state,
        runtime_log="--> FOAM FATAL ERROR: bad fvSolution\n",
        attempt=1,
        native_execution=True,
    )

    assert outcome.retry
    assert tools.mesh_calls == ["checkMesh"]
