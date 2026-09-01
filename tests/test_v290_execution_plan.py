from __future__ import annotations

from openfoam_agent.engineering.agent import CFDEngineeringAgent, EngineeringPolicy
from openfoam_agent.schemas.engineering import CaseBundleFile, ExecuteCasePlanAction
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


def _solver_dict(name: str) -> str:
    return f"FoamFile {{ object {name}; }}\n"


def _field_u() -> str:
    return """FoamFile { object U; }
dimensions [0 1 -1 0 0 0 0];
internalField uniform (1 0 0);
boundaryField
{
    inlet { type fixedValue; value uniform (1 0 0); }
    outlet { type zeroGradient; }
}
"""


def _boundary_file() -> str:
    return """FoamFile { object boundary; }
2
(
inlet
{
    type patch;
    nFaces 10;
    startFace 0;
}
outlet
{
    type patch;
    nFaces 10;
    startFace 10;
}
)
"""


def _full_execution_plan(state, *, block_mesh_text: str = "FoamFile { object blockMeshDict; }\n"):
    plan = make_plan(state.intake).model_copy(update={"required_case_files": ["0/U"]})
    return ExecuteCasePlanAction(
        type="execute_case_plan",
        goal="Build, validate, mesh, and seal the complete exploratory case",
        files=[
            CaseBundleFile(path="system/controlDict", content=control_dict()),
            CaseBundleFile(path="system/fvSchemes", content=_solver_dict("fvSchemes")),
            CaseBundleFile(path="system/fvSolution", content=_solver_dict("fvSolution")),
            CaseBundleFile(path="system/blockMeshDict", content=block_mesh_text),
            CaseBundleFile(path="0/U", content=_field_u()),
        ],
        validate_dictionaries=[
            "system/controlDict",
            "system/fvSchemes",
            "system/fvSolution",
            "system/blockMeshDict",
            "0/U",
        ],
        mesh_commands=["blockMesh", "checkMesh"],
        required_case_files=["0/U"],
        plan=plan,
        rationale="The complete case can be executed deterministically until a real failure occurs.",
    )


def test_execute_case_plan_reaches_solve_ready_in_one_llm_turn(tmp_path, graph_path):
    state = make_state()
    action = _full_execution_plan(state)
    llm = ScriptedLLM([action])
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
        policy=EngineeringPolicy(
            max_agent_steps=4,
            hard_max_agent_steps=4,
            require_solve_ready_gate=True,
            preload_capabilities=True,
        ),
    )
    # Fake blockMesh does not create polyMesh, so provide the deterministic mesh
    # boundary artifact that real blockMesh would create.
    agent.workspace.write_text("constant/polyMesh/boundary", _boundary_file())

    agent.prepare(state, native_execution=True)

    assert state.current_state == State.SOLVE_READY
    assert len(llm.prompts) == 1
    assert tools.mesh_calls == ["blockMesh", "checkMesh"]
    assert state.case_seal is not None
    plan_events = [e for e in state.engineering_events if e.sequence_id]
    assert plan_events
    assert all(e.step == 1 for e in plan_events)
    assert plan_events[-1].action_type == "finish_preview"
    assert plan_events[-1].success
    # The explicit pre-solve event exists once; finish_preview reuses its manifest binding.
    assert sum(e.action_type == "validate_pre_solve" for e in plan_events) == 1


def test_execute_case_plan_failure_returns_to_llm_then_repairs(tmp_path, graph_path):
    state = make_state()
    first = _full_execution_plan(state, block_mesh_text="FoamFile { object blockMeshDict; }\n// bad topology\n")
    plan = first.plan
    repair = ExecuteCasePlanAction(
        type="execute_case_plan",
        goal="Repair the failed mesh input and continue to solve-ready",
        files=[
            CaseBundleFile(
                path="system/blockMeshDict",
                content="FoamFile { object blockMeshDict; }\n// corrected topology\n",
            )
        ],
        validate_dictionaries=["system/blockMeshDict"],
        mesh_commands=["blockMesh", "checkMesh"],
        required_case_files=["0/U"],
        plan=plan,
        rationale="Use the real blockMesh failure from the previous execution plan.",
    )
    llm = ScriptedLLM([first, repair])
    tools = FakeOpenFOAMTools(
        mesh_results={
            "blockMesh": [
                tool_result(
                    "blockMesh",
                    success=False,
                    stderr="--> FOAM FATAL ERROR:\nbad vertex topology\nFOAM exiting\n",
                ),
                tool_result("blockMesh", success=True, stdout="blockMesh complete\n"),
            ],
            "checkMesh": [tool_result("checkMesh", success=True, stdout=mesh_ok_log())],
        }
    )
    agent = CFDEngineeringAgent(
        llm,
        workspace=tmp_path,
        capability_db=graph_path,
        tools=tools,
        policy=EngineeringPolicy(
            max_agent_steps=4,
            hard_max_agent_steps=4,
            require_solve_ready_gate=True,
            preload_capabilities=True,
        ),
    )
    agent.workspace.write_text("constant/polyMesh/boundary", _boundary_file())

    agent.prepare(state, native_execution=True)

    assert state.current_state == State.SOLVE_READY
    assert len(llm.prompts) == 2
    assert "bad vertex topology" in llm.prompts[1]
    assert tools.mesh_calls == ["blockMesh", "blockMesh", "checkMesh"]
    first_plan_events = [
        e for e in state.engineering_events if e.sequence_id == "engineering:execution-plan:0001"
    ]
    assert first_plan_events[-1].action_type == "run_mesh_command"
    assert not first_plan_events[-1].success
    assert all(e.action_type != "finish_preview" for e in first_plan_events)


def test_execute_case_plan_schema_is_strict_output_compatible():
    from openfoam_agent.llm.openai_client import validate_structured_output_schema
    from openfoam_agent.schemas.engineering import EngineeringTurn

    validate_structured_output_schema(EngineeringTurn)
