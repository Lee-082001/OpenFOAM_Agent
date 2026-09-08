"""v4.2.3 regressions for controller-owned authoring manifests."""
from copy import deepcopy

from conftest import make_state, make_plan
from openfoam_agent.engineering.authoring_tasks import accept_task, compile_tasks
from openfoam_agent.schemas.engineering import (
    CaseAuthoringAction,
    CaseBundleFile,
    ExecuteCasePlanAction,
    NativeOpenFOAMCommand,
)


def _checkmesh():
    return NativeOpenFOAMCommand(command="checkMesh", role="mesh_validation")


def test_execute_case_plan_uses_frozen_plan_manifest_not_model_echo():
    state = make_state(syntax_evidence=False)
    plan = make_plan(state.intake)
    plan.required_case_files = ["0/U", "0/p", "system/controlDict"]
    action = ExecuteCasePlanAction(
        type="execute_case_plan",
        goal="author case",
        files=[CaseBundleFile(path="0/U", content="U"), CaseBundleFile(path="0/p", content="p")],
        required_case_files=["0/U"],  # stale/incomplete model mirror
        native_pipeline=[_checkmesh()],
        plan=plan,
    )
    assert action.required_case_files == plan.required_case_files


def test_author_case_required_manifest_is_optional_compatibility_metadata():
    action = CaseAuthoringAction(
        type="author_case",
        goal="author one file",
        files=[CaseBundleFile(path="0/U", content="U")],
        native_pipeline=[_checkmesh()],
    )
    assert action.required_case_files == []


def test_author_case_harmless_duplicate_hints_are_normalized():
    action = CaseAuthoringAction.model_validate({
        "type": "author_case",
        "goal": "author one file",
        "files": [{"path": "0/U", "content": "U"}],
        "required_case_files": ["0/U", "0/U"],
        "validate_dictionaries": ["0/U", "0/U"],
        "mesh_commands": ["checkMesh", "checkMesh"],
    })
    assert action.required_case_files == ["0/U"]
    assert action.validate_dictionaries == ["0/U"]
    assert action.mesh_commands == ["checkMesh"]


def test_intermediate_partition_drops_redundant_native_commands_instead_of_failing():
    action = CaseAuthoringAction(
        type="author_case",
        task_id="task:1",
        defer_native=True,
        goal="author assigned files",
        files=[CaseBundleFile(path="0/U", content="U")],
        native_pipeline=[_checkmesh()],
        mesh_commands=["checkMesh"],
    )
    assert action.native_pipeline == []
    assert action.mesh_commands == []


def test_partition_acceptance_ignores_stale_required_file_echo_but_keeps_path_coverage():
    state = make_state(syntax_evidence=False)
    plan = make_plan(state.intake)
    plan.required_case_files = ["system/controlDict"]
    payload = {
        "frozen_engineering_plan": plan.model_dump(mode="json"),
        "confirmed_intake": state.intake.model_dump(mode="json"),
        "implementation_evidence_pack": {"records": [], "file_coverage": [], "complete": False},
        "assets": [],
    }
    queue = compile_tasks("", payload, 16000)
    meta = queue["tasks"][0]["authoring_task"]
    action = CaseAuthoringAction(
        type="author_case",
        task_id=meta["id"],
        defer_native=False,
        goal="author controlDict",
        files=[CaseBundleFile(path="system/controlDict", content="control")],
        required_case_files=["0/not-the-plan"],
        native_pipeline=[_checkmesh()],
    )
    result = accept_task(queue, action, plan)
    assert result is not None
    assert result.required_case_files == ["system/controlDict"]
    assert result.plan == plan


def test_partition_still_rejects_wrong_actual_file_coverage():
    state = make_state(syntax_evidence=False)
    plan = make_plan(state.intake)
    plan.required_case_files = ["system/controlDict"]
    payload = {
        "frozen_engineering_plan": plan.model_dump(mode="json"),
        "confirmed_intake": state.intake.model_dump(mode="json"),
        "implementation_evidence_pack": {"records": [], "file_coverage": [], "complete": False},
        "assets": [],
    }
    queue = compile_tasks("", payload, 16000)
    meta = queue["tasks"][0]["authoring_task"]
    before = deepcopy(queue)
    action = CaseAuthoringAction(
        type="author_case",
        task_id=meta["id"],
        defer_native=False,
        goal="wrong file",
        files=[CaseBundleFile(path="0/U", content="U")],
        native_pipeline=[_checkmesh()],
    )
    import pytest
    with pytest.raises(ValueError, match="file coverage mismatch"):
        accept_task(queue, action, plan)
    assert queue == before


def test_authoring_normalizes_duplicate_native_validation_and_moves_it_last():
    action = CaseAuthoringAction.model_validate({
        "type": "author_case",
        "goal": "mesh then validate",
        "files": [{"path": "system/controlDict", "content": "control"}],
        "native_pipeline": [
            {"command": "checkMesh", "role": "mesh_validation"},
            {"command": "blockMesh", "role": "mesh"},
            {"command": "checkMesh", "role": "mesh_validation"},
        ],
    })
    assert [x.command for x in action.native_pipeline] == ["blockMesh", "checkMesh"]
