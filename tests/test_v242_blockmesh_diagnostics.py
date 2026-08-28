from __future__ import annotations

from io import StringIO

from openfoam_agent.engineering import CFDEngineeringAgent, EngineeringPolicy
from openfoam_agent.progress import CLIProgressReporter
from openfoam_agent.schemas.engineering import BlockAction, RunMeshCommandAction
from openfoam_agent.tools.diagnostics import extract_openfoam_failure_diagnostic

from conftest import FakeOpenFOAMTools, ScriptedLLM, make_state, tool_result


FATAL_LOG = """Create time\n\nCreating block mesh from \"system/blockMeshDict\"\n\n--> FOAM FATAL ERROR:\nBlock hex (0 1 2 3 4 5 6 7) has inward-pointing faces\n    4(0 4 7 3)\n\n    From function Foam::blockDescriptor::check()\n    in file blockMesh/blockDescriptor/blockDescriptor.C at line 105.\n\nFOAM aborting\n"""


def test_blockmesh_diagnostic_extracts_fatal_block_and_bounds_noise():
    stdout = "\n".join(f"noise {index}" for index in range(100)) + "\n" + FATAL_LOG
    diagnostic = extract_openfoam_failure_diagnostic(stdout, "Aborted (core dumped)\n")

    assert "FOAM FATAL ERROR" in diagnostic
    assert "inward-pointing faces" in diagnostic
    assert "blockDescriptor.C" in diagnostic
    assert "noise 0" not in diagnostic


def test_blockmesh_failure_is_shown_to_user_and_next_agent_turn(tmp_path, graph_path):
    state = make_state()
    llm = ScriptedLLM(
        [
            RunMeshCommandAction(
                type="run_mesh_command",
                command="blockMesh",
                rationale="Validate the current mesh topology.",
            ),
            BlockAction(
                type="block",
                reason="Stop after observing the diagnostic in this test.",
                needs_user_input=False,
                rationale="Test terminal action.",
            ),
        ]
    )
    raw_stdout = "banner\n" + FATAL_LOG + "\npost-fatal tail\n"
    raw_stderr = "Aborted (core dumped)\nprivate path: /tmp/private/system/blockMeshDict\n"
    tools = FakeOpenFOAMTools(
        mesh_results={
            "blockMesh": [
                tool_result(
                    "blockMesh",
                    success=False,
                    stdout=raw_stdout,
                    stderr=raw_stderr,
                )
            ]
        }
    )
    stream = StringIO()
    agent = CFDEngineeringAgent(
        llm,
        workspace=tmp_path,
        capability_db=graph_path,
        tools=tools,
        policy=EngineeringPolicy(max_agent_steps=3, hard_max_agent_steps=3),
        progress=CLIProgressReporter("normal", stream=stream),
    )

    agent.prepare(state, native_execution=True)

    failed = state.engineering_events[0]
    assert failed.action_type == "run_mesh_command"
    assert not failed.success
    assert "fatal diagnostic captured" in failed.summary
    assert "FOAM FATAL ERROR" in failed.output_excerpt
    assert "inward-pointing faces" in failed.output_excerpt

    # The next model turn must see the useful fatal block, not merely returnCode=1.
    assert len(llm.prompts) >= 2
    assert "FOAM FATAL ERROR" in llm.prompts[1]
    assert "inward-pointing faces" in llm.prompts[1]

    progress = stream.getvalue()
    assert "blockMesh returned status 1; fatal diagnostic captured." in progress
    assert "reason:" in progress
    assert "FOAM FATAL ERROR" in progress
    assert "inward-pointing faces" in progress
    assert "/tmp/private" not in progress

    # Complete raw stdout/stderr remains available in the local workspace log.
    log = (agent.workspace.log_dir / "001.blockMesh.log").read_text(encoding="utf-8")
    assert raw_stdout in log
    assert raw_stderr in log
