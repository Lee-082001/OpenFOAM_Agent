from __future__ import annotations

import time

import pytest

from conftest import FakeOpenFOAMTools, ScriptedLLM, make_state
from openfoam_agent.contracts.models import ResourceLimits
from openfoam_agent.engineering import CFDEngineeringAgent, EngineeringPolicy
from openfoam_agent.schemas.common import ToolResult
from openfoam_agent.schemas.engineering import OpenFOAMExecutionSpec, ValidateDictionaryAction
from openfoam_agent.tools.diagnostics import classify_native_validation
from openfoam_agent.tools.openfoam import OpenFOAMTools
from openfoam_agent.workflow.engine import CFDWorkflow
from openfoam_agent.workflow.states import State
from test_v400_execution_contracts import synthetic_runner


def _dict_text(obj: str = "testDict") -> str:
    return f"""FoamFile
{{
    version 2.0;
    format ascii;
    class dictionary;
    object {obj};
}}
value 1;
"""


def _control_dict() -> str:
    return """FoamFile
{
    version 2.0;
    format ascii;
    class dictionary;
    object controlDict;
}
application foamRun;
solver incompressibleFluid;
startFrom latestTime;
startTime 3;
stopAt endTime;
endTime 10;
writeControl timeStep;
writeInterval 10;
purgeWrite 0;
runTimeModifiable true;
functions
{
}
"""


def test_v430_timeout_probe_is_inconclusive_not_case_failure():
    result = ToolResult(
        success=False,
        command=["foamDictionary", "-keywords", "system/controlDict"],
        return_code=124,
        stdout="application\nsolver\nstartFrom\nendTime\nfunctions\n",
        termination_reason="timeout",
    )
    assessment = classify_native_validation(result, command_name="foamDictionary", probe=True)
    assert assessment.status == "inconclusive"
    assert assessment.category == "tool"
    assert assessment.workflow_success


def test_v430_explicit_foam_fatal_is_case_failure_even_for_probe():
    result = ToolResult(
        success=False,
        command=["foamDictionary"],
        return_code=1,
        stderr="FOAM FATAL IO ERROR\nbad dictionary token\n",
    )
    assessment = classify_native_validation(result, command_name="foamDictionary", probe=True)
    assert assessment.status == "fail"
    assert assessment.category == "case"
    assert not assessment.workflow_success


def test_v430_runner_does_not_timeout_after_primary_process_exits(tmp_path):
    runner, ws = synthetic_runner(
        tmp_path,
        {"foamDictionary": "python3 -c 'import os,time; p=os.fork(); (time.sleep(10) if p==0 else os.write(1,b\"application\\n\")); os._exit(0)'\n"},
        limits=ResourceLimits(wall_seconds=6, total_wall_seconds=6),
    )
    started = time.monotonic()
    result = runner.run(["foamDictionary"], cwd=ws.case_dir, timeout=4)
    elapsed = time.monotonic() - started
    assert result.success
    assert result.return_code == 0
    assert result.termination_reason == "exited"
    assert "application" in result.stdout
    assert result.process_group_terminated
    assert elapsed < 4


class _TimeoutDictionaryTools(FakeOpenFOAMTools):
    def foam_dictionary_validate(self, file_path, cwd=None):
        del cwd
        self.dictionary_calls.append(str(file_path))
        return ToolResult(
            success=False,
            command=["foamDictionary", "-keywords", str(file_path)],
            return_code=124,
            stdout="application\nsolver\nendTime\n",
            termination_reason="timeout",
        )


def test_v430_optional_dictionary_probe_timeout_does_not_fail_authoring(tmp_path, graph_path):
    tools = _TimeoutDictionaryTools()
    agent = CFDEngineeringAgent(
        ScriptedLLM([]), workspace=tmp_path, capability_db=graph_path, tools=tools,
        policy=EngineeringPolicy(foam_dictionary_probe=True),
    )
    agent.workspace.write_text("system/testDict", _dict_text())
    event = agent._dispatch_tool_action(
        ValidateDictionaryAction(type="validate_dictionary", path="system/testDict", rationale=""),
        step=1, native_execution=True, phase="prepare", state=None,
    )
    assert event.success
    assert event.validation_status == "inconclusive"
    assert event.failure_category == "tool"
    assert "continuing" in event.summary


def test_v430_dictionary_probe_is_skipped_by_default(tmp_path, graph_path):
    tools = FakeOpenFOAMTools()
    agent = CFDEngineeringAgent(
        ScriptedLLM([]), workspace=tmp_path, capability_db=graph_path, tools=tools
    )
    agent.workspace.write_text("system/testDict", _dict_text())
    event = agent._dispatch_tool_action(
        ValidateDictionaryAction(type="validate_dictionary", path="system/testDict", rationale=""),
        step=1, native_execution=True, phase="prepare", state=None,
    )
    assert event.success
    assert event.validation_status == "pass"
    assert tools.dictionary_calls == []


def test_v430_safety_preflight_no_longer_spawns_foam_dictionary(tmp_path, graph_path):
    tools = FakeOpenFOAMTools()
    agent = CFDEngineeringAgent(
        ScriptedLLM([]), workspace=tmp_path, capability_db=graph_path, tools=tools
    )
    agent.workspace.write_text("system/controlDict", _dict_text("controlDict"))
    result = agent.safety.validate_native_inputs()
    assert result.valid
    assert result.tool_results == []
    assert tools.dictionary_calls == []


def test_v430_zero_step_consumer_uses_shadow_case_and_leaves_original_unchanged(tmp_path):
    runner, ws = synthetic_runner(
        tmp_path,
        {
            "foamRun": (
                "grep -q '^endTime    0;' system/controlDict || exit 21\n"
                "grep -q '^startTime    0;' system/controlDict || exit 22\n"
                "echo initialized\n"
            )
        },
    )
    for folder in ("system", "0", "constant"):
        (ws.case_dir / folder).mkdir(parents=True, exist_ok=True)
    original = _control_dict()
    (ws.case_dir / "system/controlDict").write_text(original, encoding="utf-8")
    tools = OpenFOAMTools(runner)
    execution = OpenFOAMExecutionSpec(
        driver="foamRun", driver_provider_id="driver.foamRun",
        solver_module="incompressibleFluid", solver_provider_id="solver.incompressibleFluid",
    )
    result, note = tools.zero_step_consumer_validate(ws.case_dir, execution, timeout=5)
    assert result is not None and result.success
    assert "initialized" in result.stdout
    assert "shadow" in note.lower()
    assert (ws.case_dir / "system/controlDict").read_text(encoding="utf-8") == original
    assert not (ws.root / ".validation-shadow").exists()


def test_v430_tool_failure_routes_to_controller_not_llm_repair(tmp_path, graph_path):
    state = make_state(); state.current_state = State.ENGINEERING
    agent = CFDEngineeringAgent(
        ScriptedLLM([]), workspace=tmp_path, capability_db=graph_path, tools=FakeOpenFOAMTools()
    )
    event = agent._event(
        1, "validate_dictionary", False, "validator timed out",
        validation_status="inconclusive", failure_category="tool",
    )
    terminal = agent._route_failed_event(state, event, default_terminal=False)
    assert terminal
    assert state.current_state == State.ENGINEERING_BLOCKED
    assert state.primary_failure is not None
    assert state.primary_failure["category"] == "tool"


def test_v430_case_failure_remains_llm_repair_eligible(tmp_path, graph_path):
    state = make_state(); state.current_state = State.ENGINEERING
    agent = CFDEngineeringAgent(
        ScriptedLLM([]), workspace=tmp_path, capability_db=graph_path, tools=FakeOpenFOAMTools()
    )
    event = agent._event(
        1, "run_mesh_command", False, "FOAM FATAL IO ERROR in blockMeshDict",
        validation_status="fail", failure_category="case",
    )
    terminal = agent._route_failed_event(state, event, default_terminal=False)
    assert not terminal
    assert state.current_state == State.ENGINEERING
    assert state.primary_failure is not None


def test_v430_secondary_recovery_failure_cannot_hide_primary_failure():
    state = make_state(); state.current_state = State.ENGINEERING
    state.primary_failure = {
        "category": "case", "action_type": "blockMesh",
        "summary": "blockMesh reported a real topology error",
    }
    workflow = object.__new__(CFDWorkflow)

    def boom(_state):
        raise RuntimeError("secondary context budget failure")

    workflow.step = boom
    CFDWorkflow.run(workflow, state, max_steps=1)
    assert state.current_state == State.FAILED
    note = state.history[-1]["note"]
    assert "Primary engineering failure preserved" in note
    assert "blockMesh reported a real topology error" in note
    assert "secondary context budget failure" in note
    assert state.secondary_failures


def test_v430_runtime_timeout_is_not_sent_to_cfd_repair(tmp_path, graph_path):
    from conftest import configure_mock_completed_run, mesh_ok_log, tool_result
    from openfoam_agent.runtime import RuntimeOrchestrator
    from openfoam_agent.schemas.simulation import RuntimePolicy
    from test_v2_runtime import _prepared_agent

    tools = FakeOpenFOAMTools(
        mesh_results={"checkMesh": [tool_result("checkMesh", success=True, stdout=mesh_ok_log())]},
        foam_runs=[
            ToolResult(
                success=False,
                command=["foamRun"],
                return_code=124,
                stdout="Time = 0.1\n",
                termination_reason="timeout",
            )
        ],
    )
    state, _, llm, agent = _prepared_agent(tmp_path, graph_path, tools, [])
    configure_mock_completed_run(state, agent, end_time=0.2)
    state.approve_solve()
    prompt_count = len(llm.prompts)

    runtime = RuntimeOrchestrator(
        tools, agent, RuntimePolicy(max_attempts=2, solver_timeout_seconds=30)
    )
    runtime.run(state)

    assert state.current_state == State.ENGINEERING_BLOCKED
    assert len(llm.prompts) == prompt_count
    assert state.primary_failure is not None
    assert state.primary_failure["category"] == "tool"
    assert "inconclusive" in state.primary_failure["summary"].lower()
