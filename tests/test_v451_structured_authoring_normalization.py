from __future__ import annotations

from conftest import FakeOpenFOAMTools, ScriptedLLM, make_plan, make_state
from openfoam_agent.engineering import CFDEngineeringAgent
from openfoam_agent.schemas.engineering import (
    CaseAuthoringAction,
    CaseAuthoringTurn,
    ExecuteCasePlanAction,
)


def _checkmesh():
    return {"command": "checkMesh", "role": "mesh_validation"}


def _typed(path: str, entries: list[tuple[str, str]], foam_class: str | None = None):
    data = {
        "path": path,
        "entries": [{"path": key, "value": value} for key, value in entries],
    }
    if foam_class is not None:
        data["foam_class"] = foam_class
    return data


def test_v451_identical_duplicate_entry_is_normalized_before_nested_pydantic_failure():
    turn = CaseAuthoringTurn.model_validate({
        "action": {
            "type": "author_case",
            "goal": "author fvSchemes",
            "typed_dictionaries": [
                _typed("system/fvSchemes", [
                    ("ddtSchemes.default", "steadyState"),
                    ("ddtSchemes.default", "steadyState;"),
                ])
            ],
            "native_pipeline": [_checkmesh()],
        }
    })
    spec = turn.action.typed_dictionaries[0]
    assert len(spec.entries) == 1
    assert spec.entries[0].path == "ddtSchemes.default"
    assert spec.entry_conflicts == []


def test_v451_conflicting_duplicate_entry_is_carried_not_structured_output_rejected():
    turn = CaseAuthoringTurn.model_validate({
        "action": {
            "type": "author_case",
            "goal": "author fvSchemes",
            "typed_dictionaries": [
                _typed("system/fvSchemes", [
                    ("ddtSchemes.default", "steadyState"),
                    ("ddtSchemes.default", "Euler"),
                ])
            ],
            "native_pipeline": [_checkmesh()],
        }
    })
    spec = turn.action.typed_dictionaries[0]
    assert len(spec.entries) == 1
    assert spec.entries[0].value == "steadyState"
    assert spec.entry_conflicts == ["ddtSchemes.default"]


def test_v451_duplicate_typed_files_merge_before_file_path_validation():
    action = CaseAuthoringAction.model_validate({
        "type": "author_case",
        "goal": "author fvSchemes",
        "typed_dictionaries": [
            _typed("system/fvSchemes", [("ddtSchemes.default", "steadyState")]),
            _typed("system/fvSchemes", [("gradSchemes.default", "Gauss linear")]),
        ],
        "native_pipeline": [_checkmesh()],
    })
    assert len(action.typed_dictionaries) == 1
    entries = {entry.path: entry.value for entry in action.typed_dictionaries[0].entries}
    assert entries == {
        "ddtSchemes.default": "steadyState",
        "gradSchemes.default": "Gauss linear",
    }
    assert action.authoring_conflicts == []


def test_v451_duplicate_raw_file_echo_is_deduplicated():
    action = CaseAuthoringAction.model_validate({
        "type": "author_case",
        "goal": "author raw file",
        "files": [
            {"path": "system/controlDict", "content": "same"},
            {"path": "system/controlDict", "content": "same"},
        ],
        "native_pipeline": [_checkmesh()],
    })
    assert len(action.files) == 1
    assert action.authoring_conflicts == []


def test_v451_conflicting_raw_file_echo_is_carried_for_deterministic_repair():
    action = CaseAuthoringAction.model_validate({
        "type": "author_case",
        "goal": "author raw file",
        "files": [
            {"path": "system/controlDict", "content": "first"},
            {"path": "system/controlDict", "content": "second"},
        ],
        "native_pipeline": [_checkmesh()],
    })
    assert len(action.files) == 1
    assert action.authoring_conflicts == ["raw-file:system/controlDict"]


def test_v451_controller_conflict_fields_are_hidden_from_llm_json_schema():
    schema_text = str(CaseAuthoringTurn.model_json_schema())
    assert "entry_conflicts" not in schema_text
    assert "authoring_conflicts" not in schema_text


def test_v451_conflict_reaches_retained_candidate_repair_without_writing_case(tmp_path, graph_path):
    state = make_state()
    plan = make_plan(state.intake)
    plan.required_case_files = ["system/fvSchemes"]
    execution = ExecuteCasePlanAction.model_validate({
        "type": "execute_case_plan",
        "goal": "author conflicting fvSchemes",
        "typed_dictionaries": [
            _typed("system/fvSchemes", [
                ("ddtSchemes.default", "steadyState"),
                ("ddtSchemes.default", "Euler"),
            ])
        ],
        "native_pipeline": [_checkmesh()],
        "required_case_files": ["system/fvSchemes"],
        "plan": plan.model_dump(mode="python"),
    })
    agent = CFDEngineeringAgent(
        ScriptedLLM([]), workspace=tmp_path, capability_db=graph_path, tools=FakeOpenFOAMTools()
    )
    terminal = agent._execute_case_plan(
        state,
        execution,
        llm_step=1,
        progress_phase="engineering",
        progress_step=1,
        progress_limit=12,
        native_execution=True,
    )
    assert terminal is False
    assert agent._pending_candidate_execution is execution
    assert agent._pending_candidate_failed_paths == ("system/fvSchemes",)
    assert state.engineering_events[-1].action_type == "authoring_semantic_conflict"
    assert agent._case_plan_retry_required(state)
    assert not (agent.workspace.case_dir / "system/fvSchemes").exists()
