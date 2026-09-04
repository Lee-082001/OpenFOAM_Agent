from __future__ import annotations

import json

import pytest

from conftest import FakeOpenFOAMTools, make_plan, make_state
from openfoam_agent.engineering.agent import CFDEngineeringAgent, EngineeringPolicy
from openfoam_agent.llm.structured_schema import compile_transport_schema, validate_structured_output_schema
from openfoam_agent.schemas.engineering import (
    BlockMeshBlock,
    BlockMeshBoundaryPatch,
    BlockMeshVertex,
    CandidateBlockMeshRepairAction,
    CandidateBlockMeshRepairTurn,
    BlockMeshRepairTurn,
    BlockMeshRepairAction,
    ExecuteCasePlanAction,
    TypedBlockMeshFile,
)
from openfoam_agent.tools.block_mesh_topology import validate_block_mesh_topology
from openfoam_agent.tools.foam_serializer import FoamSerializationError, serialize_block_mesh


def _two_block_mesh(*, internal_as_boundary: bool = False) -> TypedBlockMeshFile:
    # Two conformal hex blocks joined in +x. The shared face is (1, 2, 6, 5).
    vertices = [
        (0.0, 0.0, 0.0),  # 0
        (1.0, 0.0, 0.0),  # 1
        (1.0, 1.0, 0.0),  # 2
        (0.0, 1.0, 0.0),  # 3
        (0.0, 0.0, 0.1),  # 4
        (1.0, 0.0, 0.1),  # 5
        (1.0, 1.0, 0.1),  # 6
        (0.0, 1.0, 0.1),  # 7
        (2.0, 0.0, 0.0),  # 8
        (2.0, 1.0, 0.0),  # 9
        (2.0, 0.0, 0.1),  # 10
        (2.0, 1.0, 0.1),  # 11
    ]
    boundary_face = (1, 2, 6, 5) if internal_as_boundary else (0, 4, 7, 3)
    return TypedBlockMeshFile(
        vertices=[BlockMeshVertex(coordinates=item) for item in vertices],
        blocks=[
            BlockMeshBlock(vertices=(0, 1, 2, 3, 4, 5, 6, 7), cells=(10, 10, 1)),
            BlockMeshBlock(vertices=(1, 8, 9, 2, 5, 10, 11, 6), cells=(10, 10, 1)),
        ],
        boundary=[
            BlockMeshBoundaryPatch(name="testPatch", type="wall", faces=[boundary_face]),
        ],
    )


def test_topology_ir_rejects_internal_shared_face_as_boundary():
    report = validate_block_mesh_topology(_two_block_mesh(internal_as_boundary=True))
    assert not report.valid
    assert any(item.code == "boundary_face_is_internal" for item in report.issues)
    assert "2 block owners [0, 1]" in report.render()


def test_serializer_never_emits_topologically_invalid_block_mesh():
    with pytest.raises(FoamSerializationError, match="boundary_face_is_internal"):
        serialize_block_mesh(_two_block_mesh(internal_as_boundary=True))


def test_valid_external_face_passes_topology_contract():
    mesh = _two_block_mesh(internal_as_boundary=False)
    report = validate_block_mesh_topology(mesh)
    assert report.valid, report.render()
    text = serialize_block_mesh(mesh)
    assert "testPatch" in text
    assert "hex (0 1 2 3 4 5 6 7)" in text


def test_candidate_repair_context_preserves_failed_structured_block_mesh(tmp_path, graph_path):
    state = make_state()
    plan = make_plan(state.intake).model_copy(update={"required_case_files": ["0/U"]})
    candidate = ExecuteCasePlanAction(
        type="execute_case_plan",
        goal="candidate topology repair",
        block_mesh=_two_block_mesh(internal_as_boundary=True),
        mesh_commands=["blockMesh", "checkMesh"],
        required_case_files=["0/U"],
        plan=plan,
    )
    agent = CFDEngineeringAgent(
        object(),
        workspace=tmp_path,
        capability_db=graph_path,
        tools=FakeOpenFOAMTools(),
    )
    agent._pending_candidate_execution = candidate
    agent._pending_candidate_failed_paths = ("system/blockMeshDict",)

    context = agent._candidate_repair_context()
    assert context is not None
    artifact = context["failed_artifacts"][0]
    assert artifact["kind"] == "block_mesh"
    assert artifact["spec"]["blocks"][1]["vertices"] == [1, 8, 9, 2, 5, 10, 11, 6]


def test_candidate_repair_can_replace_block_mesh_semantically(tmp_path, graph_path):
    state = make_state()
    plan = make_plan(state.intake).model_copy(update={"required_case_files": ["0/U"]})
    candidate = ExecuteCasePlanAction(
        type="execute_case_plan",
        goal="candidate topology repair",
        block_mesh=_two_block_mesh(internal_as_boundary=True),
        mesh_commands=["blockMesh", "checkMesh"],
        required_case_files=["0/U"],
        plan=plan,
    )
    agent = CFDEngineeringAgent(
        object(),
        workspace=tmp_path,
        capability_db=graph_path,
        tools=FakeOpenFOAMTools(),
    )
    agent._pending_candidate_execution = candidate
    repair = CandidateBlockMeshRepairAction(
        type="repair_candidate_block_mesh",
        diagnosis="replace an internal boundary face with a true exterior face",
        block_mesh=_two_block_mesh(internal_as_boundary=False),
    )
    # Native execution is intentionally disabled: this test proves semantic replacement
    # and deterministic serialization occur before a native mesh command is attempted.
    agent._execute_candidate_block_mesh_repair(
        state,
        repair,
        llm_step=1,
        progress_phase="engineering",
        progress_step=1,
        progress_limit=4,
        native_execution=False,
    )
    assert agent._structured_block_mesh is not None
    assert validate_block_mesh_topology(agent._structured_block_mesh).valid
    assert "testPatch" in agent.workspace.read_text("system/blockMeshDict")


def test_committed_mesh_repair_contract_accepts_full_block_mesh_replacement(tmp_path, graph_path):
    state = make_state()
    plan = make_plan(state.intake).model_copy(update={"required_case_files": ["0/U"]})
    agent = CFDEngineeringAgent(
        object(),
        workspace=tmp_path,
        capability_db=graph_path,
        tools=FakeOpenFOAMTools(),
    )
    agent._pending_execution_plan = plan
    repair = BlockMeshRepairAction(
        type="repair_block_mesh",
        diagnosis="replace structured block topology",
        block_mesh=_two_block_mesh(internal_as_boundary=False),
    )
    agent._execute_block_mesh_repair(
        state,
        repair,
        llm_step=2,
        progress_phase="engineering",
        native_execution=False,
    )
    assert agent._structured_block_mesh is not None
    assert validate_block_mesh_topology(agent._structured_block_mesh).valid
    assert "boundary\n(" in agent.workspace.read_text("system/blockMeshDict")


def test_block_mesh_serialization_failure_enters_candidate_replan_route(tmp_path, graph_path):
    state = make_state()
    plan = make_plan(state.intake).model_copy(update={"required_case_files": ["0/U"]})
    candidate = ExecuteCasePlanAction(
        type="execute_case_plan",
        goal="candidate topology repair",
        block_mesh=_two_block_mesh(internal_as_boundary=True),
        mesh_commands=["blockMesh", "checkMesh"],
        required_case_files=["0/U"],
        plan=plan,
    )
    agent = CFDEngineeringAgent(
        object(),
        workspace=tmp_path,
        capability_db=graph_path,
        tools=FakeOpenFOAMTools(),
        policy=EngineeringPolicy(compact_phase_schemas=True),
    )
    state.engineering_round_start_index = 0
    agent._pending_candidate_execution = candidate
    agent._pending_candidate_failed_paths = ("system/blockMeshDict",)
    state.engineering_events.append(
        agent._event(1, "block_mesh_serialize", False, "topology contract failed")
    )
    schema, _, phase = agent._phase_contract(state, "prepare")
    assert phase == "block_mesh_replan"
    assert schema is CandidateBlockMeshRepairTurn


def test_specialized_block_mesh_repair_schemas_are_backend_portable_and_compact():
    for schema in (CandidateBlockMeshRepairTurn, BlockMeshRepairTurn):
        validate_structured_output_schema(schema)
        claude = compile_transport_schema(schema, backend="claude")
        codex = compile_transport_schema(schema, backend="codex")
        assert "prefixItems" not in json.dumps(claude, sort_keys=True)
        assert "prefixItems" not in json.dumps(codex, sort_keys=True)
        # Dedicated repair schemas avoid reintroducing the giant generic RepairTurn
        # union on every local mesh-topology correction.
        assert len(json.dumps(schema.model_json_schema(), sort_keys=True)) < 6000


def test_block_mesh_failure_signature_ignores_transient_face_vertex_labels():
    first = "Trying to specify a boundary face 4(3 4 20 19) on the face on cell 3 which is either an internal face or already belongs to some other patch. This is face 3 of patch 4 named cylinder."
    second = "Trying to specify a boundary face 4(3 19 20 4) on the face on cell 3 which is either an internal face or already belongs to some other patch. This is face 3 of patch 4 named cylinder."
    sig1 = CFDEngineeringAgent._native_failure_signature("blockMesh", "foam_fatal_error", first)
    sig2 = CFDEngineeringAgent._native_failure_signature("blockMesh", "foam_fatal_error", second)
    assert sig1 == sig2
