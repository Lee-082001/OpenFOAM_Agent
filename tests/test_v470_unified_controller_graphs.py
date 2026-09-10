from __future__ import annotations

from pathlib import Path

import pytest

from conftest import make_plan, make_state, foam_header
from openfoam_agent.engineering.case_delta_graph import compile_case_delta_graph
from openfoam_agent.postprocessing.build_graph import compile_postprocess_graph
from openfoam_agent.schemas.engineering import (
    CaseBundleFile,
    CaseFilePatch,
    EngineeringDefaultAssumption,
    EngineeringPlan,
    NativeOpenFOAMCommand,
    RepairCasePlanAction,
    RuntimeCaseRepairAction,
    StrategyRevisionAction,
)
from openfoam_agent.schemas.postprocessing import (
    PostProcessConfigFile,
    PostProcessRunSpec,
    PostProcessingExecutionPlanAction,
)
from openfoam_agent.tools.execution_policy import COMMAND_EFFECTS
from openfoam_agent.tools.native_contracts import native_tool_contract, registered_effects
from openfoam_agent.tools.workspace import CaseWorkspace, WorkspaceSafetyError
from openfoam_agent.runtime.restart import prepare_parallel_restart
from rc2_helpers import restart_case


def _write_minimal_required(ws: CaseWorkspace, plan) -> None:
    plan.required_case_files = ["system/controlDict", "system/fvSchemes"]
    ws.write_text("system/controlDict", foam_header("system/controlDict") + "startFrom startTime;\nstartTime 0;\nendTime 10;\n")
    ws.write_text("system/fvSchemes", foam_header("system/fvSchemes") + "ddtSchemes { default Euler; }\n")


def test_repair_stale_validation_hint_never_becomes_action_authority(tmp_path):
    ws = CaseWorkspace(tmp_path)
    state = make_state(); plan = make_plan(state.intake)
    _write_minimal_required(ws, plan)
    graph = compile_case_delta_graph(
        ws, plan,
        replacements=[("system/controlDict", foam_header("system/controlDict") + "startFrom startTime;\nstartTime 0;\nendTime 20;\n")],
        validate_dictionaries=["system/missingNeverAuthored"],
        validate_pre_solve=True,
    )
    assert graph.valid
    assert "system/missingNeverAuthored" not in graph.dictionary_paths
    assert any("Ignored stale validation hint" in item for item in graph.warnings)


def test_runtime_repair_exact_duplicate_payload_is_normalized_before_semantics():
    payload = {
        "type": "repair_runtime_case",
        "diagnosis": "same edit echoed twice",
        "replacement_files": [
            {"path": "system/controlDict", "content": "same"},
            {"path": "system/controlDict", "content": "same"},
        ],
    }
    action = RuntimeCaseRepairAction.model_validate(payload)
    assert len(action.replacement_files) == 1


def test_conflicting_repair_replacements_reach_case_delta_conflict_not_pydantic(tmp_path):
    action = RepairCasePlanAction.model_validate({
        "type": "repair_case_plan",
        "diagnosis": "conflict",
        "replacement_files": [
            {"path": "system/controlDict", "content": "A"},
            {"path": "system/controlDict", "content": "B"},
        ],
    })
    assert len(action.replacement_files) == 2
    ws = CaseWorkspace(tmp_path); state = make_state(); plan = make_plan(state.intake)
    _write_minimal_required(ws, plan)
    graph = compile_case_delta_graph(ws, plan, replacements=[(x.path, x.content) for x in action.replacement_files])
    assert not graph.valid
    assert any("Conflicting replacement" in item for item in graph.failures)


def test_strategy_revision_does_not_require_model_authored_checkmesh():
    action = StrategyRevisionAction.model_validate({
        "type": "revise_mesh_strategy",
        "diagnosis": "replace mesh dictionary",
        "replacement_files": [{"path": "system/blockMeshDict", "content": "x"}],
        "mesh_commands": ["blockMesh"],
    })
    assert action.mesh_commands == ["blockMesh"]


def test_case_delta_controller_adds_final_checkmesh_for_mesh_change(tmp_path):
    ws = CaseWorkspace(tmp_path); state = make_state(); plan = make_plan(state.intake)
    plan.required_case_files = ["system/blockMeshDict"]
    content = foam_header("system/blockMeshDict") + "convertToMeters 1;\n"
    ws.write_text("system/blockMeshDict", content)
    graph = compile_case_delta_graph(
        ws, plan,
        replacements=[("system/blockMeshDict", content + "// revised\n")],
        mesh_commands=["blockMesh"],
        validate_pre_solve=True,
    )
    assert graph.valid
    commands = [x.command for x in graph.native_pipeline]
    assert commands[-1] == "checkMesh"
    assert commands.count("checkMesh") == 1


def test_postprocess_missing_run_config_is_rejected_before_mutation(tmp_path):
    ws = CaseWorkspace(tmp_path)
    plan = PostProcessingExecutionPlanAction(
        type="execute_postprocessing_plan",
        goal="test",
        configs=[PostProcessConfigFile(path="postprocessConfig/a", content="abc")],
        runs=[PostProcessRunSpec(dictionary_path="postprocessConfig/missing")],
        summary="test",
    )
    graph = compile_postprocess_graph(ws, plan)
    assert not graph.valid
    assert not ws.resolve_case_path("postprocessConfig/a").exists()
    assert any("missing config" in x for x in graph.failures)


def test_postprocess_exact_duplicate_config_is_harmless(tmp_path):
    ws = CaseWorkspace(tmp_path)
    plan = PostProcessingExecutionPlanAction.model_validate({
        "type": "execute_postprocessing_plan",
        "goal": "test",
        "configs": [
            {"path": "postprocessConfig/a", "content": "abc"},
            {"path": "postprocessConfig/a", "content": "abc"},
        ],
        "summary": "test",
    })
    graph = compile_postprocess_graph(ws, plan)
    assert graph.valid
    assert graph.config_files == {"postprocessConfig/a": "abc"}


def test_postprocess_conflicting_duplicate_config_is_controller_failure(tmp_path):
    ws = CaseWorkspace(tmp_path)
    plan = PostProcessingExecutionPlanAction.model_validate({
        "type": "execute_postprocessing_plan",
        "goal": "test",
        "configs": [
            {"path": "postprocessConfig/a", "content": "abc"},
            {"path": "postprocessConfig/a", "content": "xyz"},
        ],
        "summary": "test",
    })
    graph = compile_postprocess_graph(ws, plan)
    assert not graph.valid
    assert any("Conflicting" in x for x in graph.failures)


def test_restart_uses_compiled_runtime_interval_when_legacy_completion_missing(tmp_path):
    ws = CaseWorkspace(tmp_path); _, plan = restart_case(ws)
    plan.completion = None
    new, _ = prepare_parallel_restart(ws, plan, time_name="2")
    assert new.execution.parallel.restart.time_name == "2"


def test_conflicting_engineering_defaults_are_not_silently_first_wins():
    state = make_state(); base = make_plan(state.intake).model_dump(mode="python")
    base["engineering_defaults"] = [
        EngineeringDefaultAssumption(parameter="inlet_velocity", value="0.1", unit="m/s", rationale="a").model_dump(mode="python"),
        EngineeringDefaultAssumption(parameter="inlet_velocity", value="1.0", unit="m/s", rationale="b").model_dump(mode="python"),
    ]
    plan = EngineeringPlan.model_validate(base)
    assert len(plan.engineering_defaults) == 2
    assert any("engineering_defaults" in item for item in plan.plan_conflicts)


def test_native_effect_registry_is_single_source_for_execution_policy():
    assert COMMAND_EFFECTS == registered_effects()
    assert native_tool_contract("blockMesh").required_dictionary == "system/blockMeshDict"
    assert native_tool_contract("checkMesh").controller_finalizer


def test_workspace_text_transaction_rolls_back_on_mid_commit_failure(tmp_path, monkeypatch):
    ws = CaseWorkspace(tmp_path)
    ws.write_text("system/a", "old-a")
    ws.write_text("system/b", "old-b")
    original = ws._atomic_write
    calls = {"n": 0}
    def flaky(path: Path, content: str):
        calls["n"] += 1
        if calls["n"] == 2:
            raise OSError("injected commit failure")
        return original(path, content)
    monkeypatch.setattr(ws, "_atomic_write", flaky)
    with pytest.raises(OSError, match="injected"):
        ws.commit_text_transaction({"system/a": "new-a", "system/b": "new-b"})
    assert ws.read_text("system/a") == "old-a"
    assert ws.read_text("system/b") == "old-b"
