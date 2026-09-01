from __future__ import annotations

import json

import pytest

from openfoam_agent.engineering import CFDEngineeringAgent, EngineeringPolicy
from openfoam_agent.schemas.engineering import (
    BlockAction,
    EngineeringSequenceAction,
    FinishPreviewAction,
    RetrySolverAction,
    RunMeshCommandAction,
    SearchCapabilitiesAction,
    ValidateDictionaryAction,
    ValidatePreSolveAction,
    WriteCaseFileAction,
)
from openfoam_agent.workflow.states import State

from conftest import (
    FakeOpenFOAMTools,
    ScriptedLLM,
    control_dict,
    make_plan,
    make_state,
    mesh_ok_log,
    tool_result,
)


def _boundary_file() -> str:
    return """FoamFile
{
    version 2.0;
    format ascii;
    class polyBoundaryMesh;
    object boundary;
}
2
(
    inlet
    {
        type patch;
        nFaces 1;
        startFace 0;
    }
    farField
    {
        type patch;
        nFaces 1;
        startFace 1;
    }
)
"""


def _field_u() -> str:
    return """FoamFile
{
    version 2.0;
    format ascii;
    class volVectorField;
    object U;
}
dimensions [0 1 -1 0 0 0 0];
internalField uniform (1 0 0);
boundaryField
{
    inlet { type fixedValue; value uniform (1 0 0); }
    farField { type zeroGradient; }
}
"""


def _solver_dict(name: str) -> str:
    return f"FoamFile {{ version 2.0; format ascii; class dictionary; object {name}; }}\n"


def test_sequence_schema_rejects_repeat_write_without_validation():
    with pytest.raises(ValueError, match="rewrites system/blockMeshDict"):
        EngineeringSequenceAction(
            type="sequence",
            goal="Repair a blockMesh dictionary",
            rationale="Avoid unvalidated repair thrashing.",
            actions=[
                WriteCaseFileAction(
                    type="write_case_file",
                    path="system/blockMeshDict",
                    content="vertices ();\n",
                    rationale="first edit",
                ),
                WriteCaseFileAction(
                    type="write_case_file",
                    path="system/blockMeshDict",
                    content="vertices ((0 0 0));\n",
                    rationale="second edit",
                ),
            ],
        )


def test_prepare_sequence_reduces_llm_turns_and_preserves_raw_events(tmp_path, graph_path):
    state = make_state()
    plan = make_plan(state.intake)
    sequence = EngineeringSequenceAction(
        type="sequence",
        goal="Create and validate a basic mesh, then seal the plan",
        rationale="The deterministic progression is predictable on success.",
        actions=[
            WriteCaseFileAction(
                type="write_case_file",
                path="system/controlDict",
                content=control_dict(),
                rationale="write control",
            ),
            WriteCaseFileAction(
                type="write_case_file",
                path="system/blockMeshDict",
                content=_solver_dict("blockMeshDict"),
                rationale="write mesh input",
            ),
            RunMeshCommandAction(
                type="run_mesh_command",
                command="blockMesh",
                rationale="generate mesh",
            ),
            RunMeshCommandAction(
                type="run_mesh_command",
                command="checkMesh",
                rationale="validate mesh",
            ),
            FinishPreviewAction(
                type="finish_preview",
                plan=plan,
                rationale="seal the validated case",
            ),
        ],
    )
    llm = ScriptedLLM(
        [
            SearchCapabilitiesAction(
                type="search_capabilities",
                query="incompressibleFluid",
                rationale="observe solver capability",
            ),
            sequence,
        ]
    )
    tools = FakeOpenFOAMTools(
        mesh_results={
            "blockMesh": [tool_result("blockMesh", success=True, stdout="blockMesh complete\n")],
            "checkMesh": [tool_result("checkMesh", success=True, stdout=mesh_ok_log())],
        }
    )
    agent = CFDEngineeringAgent(
        llm,
        workspace=tmp_path,
        capability_db=graph_path,
        tools=tools,
        policy=EngineeringPolicy(max_agent_steps=4, hard_max_agent_steps=4),
    )

    agent.prepare(state, native_execution=True)

    assert state.current_state == State.MESH_READY
    assert len(llm.prompts) == 2
    assert len(state.engineering_events) == 6  # 1 singleton + 5 deterministic sequence members
    sequence_events = [event for event in state.engineering_events if event.sequence_id]
    assert len(sequence_events) == 5
    assert {event.step for event in sequence_events} == {2}
    assert tools.mesh_calls == ["blockMesh", "checkMesh"]


def test_sequence_stops_on_first_native_failure_and_compacts_next_prompt(tmp_path, graph_path):
    state = make_state()
    sequence = EngineeringSequenceAction(
        type="sequence",
        goal="Attempt and validate blockMesh",
        rationale="Stop immediately if generation fails.",
        actions=[
            WriteCaseFileAction(
                type="write_case_file",
                path="system/blockMeshDict",
                content=_solver_dict("blockMeshDict"),
                rationale="write mesh input",
            ),
            RunMeshCommandAction(
                type="run_mesh_command",
                command="blockMesh",
                rationale="generate mesh",
            ),
            RunMeshCommandAction(
                type="run_mesh_command",
                command="checkMesh",
                rationale="must be skipped after failure",
            ),
        ],
    )
    llm = ScriptedLLM(
        [
            sequence,
            BlockAction(
                type="block",
                reason="Stop after observing the compact sequence failure.",
                needs_user_input=False,
                rationale="test checkpoint",
            ),
        ]
    )
    tools = FakeOpenFOAMTools(
        mesh_results={
            "blockMesh": [
                tool_result(
                    "blockMesh",
                    success=False,
                    stderr="--> FOAM FATAL ERROR:\nbad vertex topology\nFOAM exiting\n",
                )
            ]
        }
    )
    agent = CFDEngineeringAgent(
        llm,
        workspace=tmp_path,
        capability_db=graph_path,
        tools=tools,
        policy=EngineeringPolicy(max_agent_steps=3, hard_max_agent_steps=3),
    )

    agent.prepare(state, native_execution=True)

    assert tools.mesh_calls == ["blockMesh"]
    assert len([event for event in state.engineering_events if event.sequence_id]) == 2
    assert '"kind": "engineering_sequence_summary"' in llm.prompts[1]
    assert "bad vertex topology" in llm.prompts[1]
    assert '"sequence_index"' not in llm.prompts[1]


def test_solver_input_sequence_runs_presolve_readiness_without_checkmesh(tmp_path, graph_path):
    state = make_state()
    sequence = EngineeringSequenceAction(
        type="sequence",
        goal="Construct solver inputs and validate solve readiness",
        rationale="These edits do not change the mesh.",
        actions=[
            WriteCaseFileAction(
                type="write_case_file",
                path="system/fvSchemes",
                content=_solver_dict("fvSchemes"),
                rationale="write schemes",
            ),
            ValidateDictionaryAction(
                type="validate_dictionary",
                path="system/fvSchemes",
                rationale="validate schemes",
            ),
            WriteCaseFileAction(
                type="write_case_file",
                path="system/fvSolution",
                content=_solver_dict("fvSolution"),
                rationale="write solution controls",
            ),
            ValidateDictionaryAction(
                type="validate_dictionary",
                path="system/fvSolution",
                rationale="validate solution controls",
            ),
            WriteCaseFileAction(
                type="write_case_file",
                path="0/U",
                content=_field_u(),
                rationale="write initial velocity field",
            ),
            ValidatePreSolveAction(
                type="validate_pre_solve",
                required_case_files=["0/U"],
                rationale="check all declared solve inputs and patch coverage",
            ),
        ],
    )
    llm = ScriptedLLM(
        [
            sequence,
            BlockAction(
                type="block",
                reason="Pre-solve sequence completed for the test.",
                needs_user_input=False,
                rationale="test terminal",
            ),
        ]
    )
    tools = FakeOpenFOAMTools()
    agent = CFDEngineeringAgent(
        llm,
        workspace=tmp_path,
        capability_db=graph_path,
        tools=tools,
        policy=EngineeringPolicy(max_agent_steps=3, hard_max_agent_steps=3),
    )
    agent.workspace.write_text("system/controlDict", control_dict())
    agent.workspace.write_text("constant/polyMesh/boundary", _boundary_file())

    agent.prepare(state, native_execution=True)

    presolve = next(
        event for event in state.engineering_events if event.action_type == "validate_pre_solve"
    )
    assert presolve.success
    assert tools.mesh_calls == []


def test_runtime_repair_sequence_validates_then_retries_in_one_llm_turn(tmp_path, graph_path):
    state = make_state()
    plan = make_plan(state.intake).model_copy(update={"required_case_files": ["0/U"]})
    repair_sequence = EngineeringSequenceAction(
        type="sequence",
        goal="Repair solver controls and retry only after solve-readiness validation",
        rationale="No new engineering decision is needed if every validation passes.",
        actions=[
            WriteCaseFileAction(
                type="write_case_file",
                path="system/fvSolution",
                content=_solver_dict("fvSolution") + "solvers {};\n",
                rationale="repair fvSolution",
            ),
            ValidateDictionaryAction(
                type="validate_dictionary",
                path="system/fvSolution",
                rationale="validate repair",
            ),
            ValidatePreSolveAction(
                type="validate_pre_solve",
                required_case_files=["0/U"],
                rationale="ensure solver inputs are complete",
            ),
            RetrySolverAction(
                type="retry_solver",
                plan=plan,
                rationale="retry after deterministic readiness passes",
            ),
        ],
    )
    llm = ScriptedLLM(
        [
            SearchCapabilitiesAction(
                type="search_capabilities",
                query="incompressibleFluid",
                rationale="observe solver capability",
            ),
            RunMeshCommandAction(
                type="run_mesh_command",
                command="checkMesh",
                rationale="establish current mesh evidence",
            ),
            FinishPreviewAction(
                type="finish_preview",
                plan=plan,
                rationale="seal solve-ready baseline",
            ),
            repair_sequence,
        ]
    )
    tools = FakeOpenFOAMTools(
        mesh_results={
            "checkMesh": [tool_result("checkMesh", success=True, stdout=mesh_ok_log())]
        }
    )
    agent = CFDEngineeringAgent(
        llm,
        workspace=tmp_path,
        capability_db=graph_path,
        tools=tools,
        policy=EngineeringPolicy(
            max_agent_steps=5,
            hard_max_agent_steps=5,
            require_solve_ready_gate=True,
        ),
    )
    agent.workspace.write_text("system/controlDict", control_dict())
    agent.workspace.write_text("system/fvSchemes", _solver_dict("fvSchemes"))
    agent.workspace.write_text("system/fvSolution", _solver_dict("fvSolution"))
    agent.workspace.write_text("0/U", _field_u())
    agent.workspace.write_text("constant/polyMesh/boundary", _boundary_file())

    agent.prepare(state, native_execution=True)
    assert state.current_state == State.SOLVE_READY
    prompts_before_repair = len(llm.prompts)

    outcome = agent.repair_runtime(
        state,
        runtime_log="--> FOAM FATAL ERROR:\nbad solver setup\nFOAM exiting\n",
        attempt=1,
        native_execution=True,
    )

    assert outcome.retry
    assert len(llm.prompts) == prompts_before_repair + 1
    repair_events = [event for event in state.engineering_events if event.sequence_goal and "Repair solver" in event.sequence_goal]
    assert [event.action_type for event in repair_events] == [
        "write_case_file",
        "validate_dictionary",
        "validate_pre_solve",
        "retry_solver",
    ]
    assert all(event.success for event in repair_events)
    assert tools.mesh_calls == ["checkMesh"]


def test_engineering_sequence_schema_remains_strict_output_compatible():
    from openfoam_agent.llm.openai_client import validate_structured_output_schema
    from openfoam_agent.schemas.engineering import EngineeringTurn

    validate_structured_output_schema(EngineeringTurn)
    schema = EngineeringTurn.model_json_schema()
    payload = json.dumps(schema)
    assert '"sequence"' in payload
    assert '"validate_pre_solve"' in payload
