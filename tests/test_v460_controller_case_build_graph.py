from __future__ import annotations

from conftest import FakeOpenFOAMTools, ScriptedLLM, make_plan, make_state
from openfoam_agent.engineering import CFDEngineeringAgent
from openfoam_agent.engineering.case_build_graph import compile_case_build_graph
from openfoam_agent.schemas.engineering import (
    CaseAuthoringAction,
    CaseBundleFile,
    ExecuteCasePlanAction,
    NativeOpenFOAMCommand,
)


def _execution(plan, paths, *, validate=None, pipeline=None):
    return ExecuteCasePlanAction(
        type="execute_case_plan",
        goal="author frozen case",
        files=[CaseBundleFile(path=path, content=f"content for {path}") for path in paths],
        validate_dictionaries=validate or [],
        native_pipeline=pipeline or [],
        plan=plan,
    )


def test_author_case_no_longer_requires_model_owned_native_pipeline():
    action = CaseAuthoringAction(
        type="author_case",
        goal="author assigned case artifacts",
        files=[CaseBundleFile(path="system/controlDict", content="control")],
    )
    assert action.native_pipeline == []
    assert action.mesh_commands == []


def test_build_graph_rejects_missing_required_files_before_any_validation_action():
    state = make_state(syntax_evidence=False)
    plan = make_plan(state.intake)
    plan.required_case_files = [
        "system/controlDict",
        "system/fvSchemes",
        "system/fvSolution",
        "0/U",
        "0/p",
    ]
    execution = _execution(
        plan,
        ["system/controlDict", "system/fvSchemes", "system/blockMeshDict"],
        validate=["system/controlDict", "system/fvSolution", "0/U"],
    )
    bundle = {item.path: item.content for item in execution.files}
    graph = compile_case_build_graph(execution, bundle)
    assert not graph.valid
    assert graph.missing_required_paths == ("system/fvSolution", "0/U", "0/p")
    assert any("system/fvSolution" in item for item in graph.failures)
    assert any("Ignored stale validation hint" in item for item in graph.warnings)


def test_build_graph_derives_blockmesh_and_final_checkmesh_without_model_pipeline():
    state = make_state(syntax_evidence=False)
    plan = make_plan(state.intake)
    plan.required_case_files = ["system/controlDict", "system/blockMeshDict"]
    execution = _execution(plan, plan.required_case_files)
    bundle = {item.path: item.content for item in execution.files}
    graph = compile_case_build_graph(execution, bundle)
    assert graph.valid
    assert [item.command for item in graph.native_pipeline] == ["blockMesh", "checkMesh"]


def test_build_graph_preserves_multiregion_checkmesh_hints_at_the_end():
    state = make_state(syntax_evidence=False)
    plan = make_plan(state.intake)
    plan.required_case_files = ["system/controlDict", "system/blockMeshDict"]
    execution = _execution(
        plan,
        plan.required_case_files,
        pipeline=[
            NativeOpenFOAMCommand(command="checkMesh", arguments=["-region", "fluid"], role="mesh_validation"),
            NativeOpenFOAMCommand(command="blockMesh", role="mesh"),
            NativeOpenFOAMCommand(command="checkMesh", arguments=["-region", "solid"], role="mesh_validation"),
        ],
    )
    bundle = {item.path: item.content for item in execution.files}
    graph = compile_case_build_graph(execution, bundle)
    assert [item.command for item in graph.native_pipeline] == ["blockMesh", "checkMesh", "checkMesh"]
    assert graph.native_pipeline[-2].arguments == ["-region", "fluid"]
    assert graph.native_pipeline[-1].arguments == ["-region", "solid"]


def test_build_graph_drops_stale_native_utility_with_missing_strong_prerequisite():
    state = make_state(syntax_evidence=False)
    plan = make_plan(state.intake)
    plan.required_case_files = ["system/controlDict"]
    execution = _execution(
        plan,
        ["system/controlDict"],
        pipeline=[NativeOpenFOAMCommand(command="topoSet", role="mesh")],
    )
    bundle = {item.path: item.content for item in execution.files}
    graph = compile_case_build_graph(execution, bundle)
    assert graph.valid
    assert [item.command for item in graph.native_pipeline] == ["checkMesh"]
    assert any("system/topoSetDict" in item for item in graph.warnings)


def test_execute_case_plan_missing_manifest_is_transactional_and_repairable(tmp_path, graph_path):
    state = make_state(syntax_evidence=False)
    plan = make_plan(state.intake)
    plan.required_case_files = [
        "system/controlDict",
        "system/fvSchemes",
        "system/fvSolution",
        "0/U",
        "0/p",
    ]
    execution = _execution(
        plan,
        ["system/controlDict", "system/fvSchemes", "system/blockMeshDict"],
        validate=["system/fvSolution"],
    )
    agent = CFDEngineeringAgent(
        ScriptedLLM([]), workspace=tmp_path, capability_db=graph_path, tools=FakeOpenFOAMTools()
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
    assert not terminal
    assert state.engineering_events[-1].action_type == "case_build_graph"
    assert state.engineering_events[-1].failure_category == "case"
    assert agent._case_plan_retry_required(state)
    assert set(agent._pending_candidate_failed_paths) == {"system/fvSolution", "0/U", "0/p"}
    assert agent.workspace.list_authored() == []
    capsule = agent._candidate_repair_context()
    assert capsule is not None
    assert capsule["missing_required_files"] == ["system/fvSolution", "0/U", "0/p"]


def test_stale_validation_hint_never_becomes_an_executable_action(tmp_path, graph_path):
    state = make_state(syntax_evidence=False)
    plan = make_plan(state.intake)
    plan.required_case_files = ["system/controlDict"]
    execution = _execution(
        plan,
        ["system/controlDict"],
        validate=["system/fvSolution"],
    )
    # We only need to prove graph compilation has no target for the stale mirror.
    graph = compile_case_build_graph(execution, {"system/controlDict": "x"})
    assert graph.valid
    assert "system/fvSolution" not in graph.dictionary_paths
    assert any("system/fvSolution" in warning for warning in graph.warnings)
