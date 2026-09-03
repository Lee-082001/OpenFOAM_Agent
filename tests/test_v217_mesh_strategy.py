from __future__ import annotations

from collections import deque

from openfoam_agent.engineering.agent import CFDEngineeringAgent, EngineeringPolicy
from openfoam_agent.llm.openai_client import validate_structured_output_schema
from openfoam_agent.schemas.engineering import (
    BlockAction,
    BlockMeshBlock,
    BlockMeshBoundaryPatch,
    BlockMeshVertex,
    CaseBundleFile,
    ExecuteCasePlanAction,
    PrepareTurn,
    StrategyRevisionAction,
    StrategyRevisionTurn,
    TypedBlockMeshFile,
)
from openfoam_agent.tools.foam_serializer import serialize_block_mesh
from openfoam_agent.tools.openfoam import OpenFOAMTools
from openfoam_agent.workflow.states import State

from conftest import FakeOpenFOAMTools, make_plan, make_state, mesh_ok_log, tool_result


class ScriptedLLM:
    def __init__(self, actions):
        self.actions = deque(actions)
        self.schemas = []
        self.prompts = []
        self.store = False

    def generate(self, schema, prompt, **kwargs):
        del kwargs
        self.schemas.append(schema)
        self.prompts.append(prompt)
        return schema(action=self.actions.popleft())


class ContractFakeTools(FakeOpenFOAMTools):
    def mesh_tool_contracts(self):
        return OpenFOAMTools.mesh_tool_contracts()

    def mesh_command_precondition(self, command, case_dir):
        if command == "snappyHexMesh":
            return False, "snappyHexMesh requires a fully 3D base mesh; test mesh has an empty patch."
        return True, ""


def _block_mesh() -> TypedBlockMeshFile:
    vertices = [
        (-1.0, -1.0, -0.05),
        (1.0, -1.0, -0.05),
        (1.0, 1.0, -0.05),
        (-1.0, 1.0, -0.05),
        (-1.0, -1.0, 0.05),
        (1.0, -1.0, 0.05),
        (1.0, 1.0, 0.05),
        (-1.0, 1.0, 0.05),
    ]
    return TypedBlockMeshFile(
        vertices=[BlockMeshVertex(coordinates=value) for value in vertices],
        blocks=[
            BlockMeshBlock(
                vertices=(0, 1, 2, 3, 4, 5, 6, 7),
                cells=(20, 20, 1),
                grading="simpleGrading (1 1 1)",
            )
        ],
        boundary=[
            BlockMeshBoundaryPatch(name="inlet", type="patch", faces=[(0, 4, 7, 3)]),
            BlockMeshBoundaryPatch(name="outlet", type="patch", faces=[(1, 2, 6, 5)]),
            BlockMeshBoundaryPatch(name="walls", type="wall", faces=[(0, 1, 5, 4), (3, 7, 6, 2)]),
            BlockMeshBoundaryPatch(name="frontBack", type="empty", faces=[(0, 3, 2, 1), (4, 5, 6, 7)]),
        ],
    )


def _plan_with_snappy(state) -> ExecuteCasePlanAction:
    plan = make_plan(state.intake).model_copy(update={"required_case_files": ["0/U"]})
    return ExecuteCasePlanAction(
        type="execute_case_plan",
        goal="2D mesh strategy test",
        files=[
            CaseBundleFile(path="0/U", content="dimensions [0 1 -1 0 0 0 0];\ninternalField uniform (1 0 0);\nboundaryField\n{\n    inlet { type fixedValue; value uniform (1 0 0); }\n}\n"),
            CaseBundleFile(path="system/controlDict", content="solver incompressibleFluid;\n"),
            CaseBundleFile(path="system/fvSchemes", content="ddtSchemes { default Euler; }\n"),
            CaseBundleFile(path="system/fvSolution", content="PIMPLE { nCorrectors 2; }\n"),
            CaseBundleFile(path="system/snappyHexMeshDict", content="FoamFile { object snappyHexMeshDict; }\n"),
        ],
        block_mesh=_block_mesh(),
        validate_dictionaries=["system/blockMeshDict", "system/snappyHexMeshDict", "system/controlDict", "system/fvSchemes", "system/fvSolution", "0/U"],
        mesh_commands=["blockMesh", "snappyHexMesh", "checkMesh"],
        required_case_files=["0/U"],
        plan=plan,
    )


def test_block_mesh_serializer_renders_boundary_as_list_not_nested_mapping():
    text = serialize_block_mesh(_block_mesh())
    assert "boundary\n(" in text
    assert "frontBack\n    {" in text
    assert "type empty;" in text
    assert "blocks\n(" in text
    assert "hex (0 1 2 3 4 5 6 7) (20 20 1) simpleGrading (1 1 1)" in text
    assert "boundary.frontBack" not in text


def test_snappy_precondition_detects_empty_patch_without_native_execution(tmp_path):
    boundary = tmp_path / "constant" / "polyMesh" / "boundary"
    boundary.parent.mkdir(parents=True)
    boundary.write_text("frontBack { type empty; }\n", encoding="utf-8")
    ok, reason = OpenFOAMTools.mesh_command_precondition("snappyHexMesh", tmp_path)
    assert not ok
    assert "fully 3D" in reason
    assert "empty" in reason


def test_tool_precondition_escalates_to_strategy_revision_and_reaches_solve_ready(tmp_path, graph_path):
    state = make_state()
    initial = _plan_with_snappy(state)
    revision = StrategyRevisionAction(
        type="revise_mesh_strategy",
        diagnosis="snappyHexMesh is incompatible with the strict 2D base mesh; keep the blockMesh result and remove snappy from the pipeline.",
        drop_paths=["system/snappyHexMeshDict"],
        mesh_commands=["checkMesh"],
        validate_pre_solve=True,
    )
    llm = ScriptedLLM([initial, revision])
    tools = ContractFakeTools(
        mesh_results={
            "blockMesh": [tool_result("blockMesh", success=True, stdout="ok\n")],
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
            compact_phase_schemas=True,
            state_delta_context=True,
        ),
    )
    # Pre-solve test fixtures expect a generated boundary file. The fake blockMesh
    # does not create one, so provide the native artifact fixture explicitly.
    boundary = agent.workspace.case_dir / "constant" / "polyMesh" / "boundary"
    boundary.parent.mkdir(parents=True, exist_ok=True)
    boundary.write_text(
        "FoamFile { object boundary; }\n1\n(\ninlet\n{\n type patch;\n nFaces 1;\n startFace 0;\n}\n)\n",
        encoding="utf-8",
    )
    agent.prepare(state, native_execution=True)

    assert llm.schemas == [PrepareTurn, StrategyRevisionTurn]
    assert tools.mesh_calls == ["blockMesh", "checkMesh"]
    assert not agent.workspace.resolve_case_path("system/snappyHexMeshDict").exists()
    assert any(event.action_type == "mesh_tool_precondition" for event in state.engineering_events)
    assert state.current_state == State.SOLVE_READY


def test_repeated_identical_native_mesh_failure_requests_strategy_revision(tmp_path, graph_path):
    state = make_state()
    initial = _plan_with_snappy(state).model_copy(update={"mesh_commands": ["blockMesh", "checkMesh"]})
    # First local repair repeats blockMesh and receives the same normalized diagnostic.
    from openfoam_agent.schemas.engineering import CaseFilePatch, RepairCasePlanAction
    repair = RepairCasePlanAction(
        type="repair_case_plan",
        diagnosis="try one local topology correction",
        patches=[CaseFilePatch(path="system/blockMeshDict", old="scale 1;", new="scale 1.0;")],
        mesh_commands=["blockMesh"],
        validate_pre_solve=False,
    )
    llm = ScriptedLLM([
        initial,
        repair,
        BlockAction(type="block", reason="strategy revision requested after repeated identical failure"),
    ])
    tools = FakeOpenFOAMTools(
        mesh_results={
            "blockMesh": [
                tool_result("blockMesh", success=False, stderr="--> FOAM FATAL ERROR:\nidentical topology failure\nfile: a line 1\n"),
                tool_result("blockMesh", success=False, stderr="--> FOAM FATAL ERROR:\nidentical topology failure\nfile: b line 9\n"),
            ]
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
            preload_capabilities=True,
            compact_phase_schemas=True,
        ),
    )
    agent.prepare(state, native_execution=True)
    assert llm.schemas[0] is PrepareTurn
    assert llm.schemas[1].__name__ == "RepairTurn"
    assert llm.schemas[2] is StrategyRevisionTurn


def test_strategy_revision_schema_is_strict_output_compatible():
    validate_structured_output_schema(StrategyRevisionTurn)
