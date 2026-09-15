from __future__ import annotations

from types import SimpleNamespace

from conftest import FakeOpenFOAMTools, ScriptedLLM, foam_header, make_plan, make_state
from openfoam_agent.contracts.models import RegionCaseLayout, RegionInterface
from openfoam_agent.engineering import CFDEngineeringAgent, EngineeringPolicy
from openfoam_agent.engineering.runtime_repair_context import build_partitioned_runtime_repair_prompt
from openfoam_agent.schemas.common import ToolResult
from openfoam_agent.schemas.engineering import (
    BlockAction,
    ExecutionScope,
    OpenFOAMExecutionSpec,
    RuntimeRepairTurn,
    ValidatePreSolveAction,
)
from openfoam_agent.tools.openfoam import OpenFOAMTools
from openfoam_agent.workflow.states import State
from test_v400_execution_contracts import synthetic_runner


def _solid_control_dict() -> str:
    return """FoamFile
{
    version 2.0;
    format ascii;
    class dictionary;
    object controlDict;
}
application foamMultiRun;
startFrom startTime;
startTime 0;
stopAt endTime;
endTime 300;
deltaT 0.1;
writeControl timeStep;
writeInterval 10;
purgeWrite 0;
adjustTimeStep false;
runTimeModifiable false;
functions
{
}
"""


def _solid_execution() -> OpenFOAMExecutionSpec:
    return OpenFOAMExecutionSpec(
        driver="foamMultiRun",
        driver_provider_id="execution.foamMultiRun",
        scopes=[
            ExecutionScope(name="battery", solver_module="solid", solver_provider_id="solver.solid"),
            ExecutionScope(name="heater", solver_module="solid", solver_provider_id="solver.solid"),
        ],
    )


def test_v504_one_step_shadow_catches_first_equation_solver_lookup(tmp_path):
    runner, ws = synthetic_runner(
        tmp_path,
        {
            "foamMultiRun": (
                "if grep -q '^endTime    0;' system/controlDict; then echo initialized; exit 0; fi\n"
                "grep -q '^endTime    0.10000000000000001;' system/controlDict || exit 31\n"
                "echo 'Time = 0.1'\n"
                "echo '--> FOAM FATAL IO ERROR:' 1>&2\n"
                "echo 'keyword e is undefined in dictionary solvers' 1>&2\n"
                "exit 1\n"
            )
        },
    )
    for folder in ("system", "0/battery", "0/heater", "constant/battery", "constant/heater"):
        (ws.case_dir / folder).mkdir(parents=True, exist_ok=True)
    original = _solid_control_dict()
    (ws.case_dir / "system/controlDict").write_text(original, encoding="utf-8")
    tools = OpenFOAMTools(runner)
    execution = _solid_execution()

    zero, zero_note = tools.zero_step_consumer_validate(ws.case_dir, execution, timeout=5)
    assert zero is not None and zero.success
    assert "zero-step" in zero_note.lower()

    one, one_note = tools.one_step_consumer_validate(ws.case_dir, execution, timeout=5)
    assert one is not None and not one.success
    assert "keyword e is undefined" in (one.stderr + one.stdout)
    assert "one-step" in one_note.lower()
    assert (ws.case_dir / "system/controlDict").read_text(encoding="utf-8") == original
    assert not (ws.root / ".validation-shadow").exists()


class _SolidProbeTools(FakeOpenFOAMTools):
    def zero_step_consumer_validate(self, case_dir, execution, timeout=60):
        del case_dir, execution, timeout
        return ToolResult(
            success=True,
            command=["foamMultiRun"],
            return_code=0,
            stdout="zero-step initialization OK\n",
        ), "Zero-step consumer validation executed in a temporary shadow case."

    def one_step_consumer_validate(self, case_dir, execution, timeout=60):
        del case_dir, execution, timeout
        return ToolResult(
            success=False,
            command=["foamMultiRun"],
            return_code=1,
            stderr=(
                "--> FOAM FATAL IO ERROR:\n"
                "keyword e is undefined in dictionary solvers\n"
                "FOAM exiting\n"
            ),
        ), "One-step consumer validation executed in a temporary shadow case."


def test_v504_presolve_rejects_two_solid_case_before_user_solve_when_e_solver_missing(tmp_path, graph_path):
    state = make_state()
    base = make_plan(state.intake, solver="foamMultiRun")
    required = ["system/controlDict", "system/fvSolution", "0/battery/T", "0/heater/T"]
    plan = base.model_copy(update={
        "solver": "foamMultiRun",
        "solver_provider_id": "execution.foamMultiRun",
        "execution": _solid_execution(),
        "region_layouts": [
            RegionCaseLayout(region="battery", required_fields=["T"]),
            RegionCaseLayout(region="heater", required_fields=["T"]),
        ],
        "interfaces": [
            RegionInterface(
                region="battery",
                patch="battery_to_heater",
                neighbour_region="heater",
                neighbour_patch="heater_to_battery",
                fields=["T"],
            )
        ],
        "required_case_files": required,
    })
    state.engineering_plan = plan
    tools = _SolidProbeTools()
    agent = CFDEngineeringAgent(
        ScriptedLLM([]),
        workspace=tmp_path,
        capability_db=graph_path,
        tools=tools,
        policy=EngineeringPolicy(
            zero_step_consumer_validation=True,
            one_step_consumer_validation=True,
        ),
    )
    # This regression is about native shadow coverage, not the separate static
    # completeness checker; make the latter a known-good prerequisite.
    agent.presolve.validate = lambda _plan: SimpleNamespace(
        valid=True,
        failures=[],
        checked_files=required,
        mesh_patches=["battery_to_heater", "heater_to_battery"],
        warnings=[],
    )
    action = ValidatePreSolveAction(
        type="validate_pre_solve",
        required_case_files=required,
        rationale="exercise first solid equation solve before approval",
    )

    event = agent._dispatch_tool_action(
        action,
        step=1,
        native_execution=True,
        phase="prepare",
        state=state,
    )

    assert not event.success
    assert event.failure_category == "case"
    assert event.validation_status == "fail"
    assert "One-step OpenFOAM consumer execution rejected" in event.summary
    assert "keyword e is undefined" in event.output_excerpt


def test_v504_runtime_repair_partition_fits_large_multiregion_solid_failure_under_18k():
    required = ["system/controlDict", "system/fvSolution", "system/fvSchemes"]
    required += [f"constant/battery/property{i}" for i in range(35)]
    required += [f"constant/heater/property{i}" for i in range(35)]
    approved_plan = {
        "solver": "foamMultiRun",
        "solver_provider_id": "execution.foamMultiRun",
        "execution": {
            "driver": "foamMultiRun",
            "driver_provider_id": "execution.foamMultiRun",
            "scopes": [
                {"name": "battery", "solver_module": "solid", "solver_provider_id": "solver.solid"},
                {"name": "heater", "solver_module": "solid", "solver_provider_id": "solver.solid"},
            ],
            "arguments": [],
            "parallel": {"mode": "serial", "ranks": 1},
        },
        "region_layouts": [
            {"region": "battery", "required_fields": ["T", "e"]},
            {"region": "heater", "required_fields": ["T", "e"]},
        ],
        "interfaces": [
            {
                "region": "battery",
                "patch": "battery_to_heater",
                "neighbour_region": "heater",
                "neighbour_patch": "heater_to_battery",
                "fields": ["T"],
            }
        ],
        "temporal_behavior": "transient",
        "motion_kind": "static",
        "mesh_motion_requirement": "static",
        "required_case_files": required,
        "confirmed_fact_bindings": [
            {
                "fact_id": f"physics.fact{i}",
                "case_files": [required[i % len(required)]],
                "plan_fields": ["problem_interpretation"],
                "explanation": "x" * 800,
            }
            for i in range(80)
        ],
    }
    payload = {
        "phase": "runtime_repair",
        "step": 1,
        "confirmed_facts": [
            {"id": f"fact.{i}", "value": "confirmed physical requirement " + ("v" * 1200), "source": "user"}
            for i in range(12)
        ],
        "approved_plan": approved_plan,
        "approved_plan_sha256": "a" * 64,
        "native_failure": (
            "--> FOAM FATAL IO ERROR:\n"
            "keyword e is undefined in dictionary <CASE_DIR><LOCAL_PATH:solvers>\n"
            + ("native diagnostic detail\n" * 250)
        ),
        "case_file_contract_scan": {
            "checked_count": 74,
            "invalid_count": 0,
            "invalid": [],
        },
        "relevant_case_files": [
            {"path": "system/fvSolution", "sha256": "b" * 64, "content": "solvers\n{\n" + ("    T { solver smoothSolver; }\n" * 300) + "}\n", "truncated": False},
            {"path": "system/controlDict", "sha256": "c" * 64, "content": _solid_control_dict() * 12, "truncated": False},
            {"path": "0/battery/T", "sha256": "d" * 64, "content": "boundaryField {}\n" * 300, "truncated": False},
            {"path": "0/heater/T", "sha256": "e" * 64, "content": "boundaryField {}\n" * 300, "truncated": False},
        ],
        "mesh_evidence": {"passed": True, "cell_count": 10000, "manifest_current": True},
        "available_evidence": [
            {"evidence_id": f"E{i}", "kind": "reference", "detail": "reference text " + ("r" * 1200)}
            for i in range(20)
        ],
        "evidence_gap_status": [],
        "bindings": {
            "intake_sha256": "1" * 64,
            "plan_sha256": "a" * 64,
            "manifest_sha256": "2" * 64,
            "check_mesh_passed": True,
            "mesh_scope_evidence": {
                "region:battery": {"passed": True, "cell_count": 5000, "raw_log_sha256": "3" * 64},
                "region:heater": {"passed": True, "cell_count": 5000, "raw_log_sha256": "4" * 64},
            },
        },
        "engineering_assumption_policy": {"authorized": True},
        "evidence_retrieval_policy": {"available": True},
        "budget": {"steps_remaining_in_current_window": 4},
    }

    built, capsule, metrics = build_partitioned_runtime_repair_prompt(
        "Repair the runtime failure:\n",
        payload,
        max_chars=18_000,
    )

    assert len(built.prompt) <= 18_000
    assert metrics["runtimeRepairPartitioned"] == 1
    assert metrics["runtimeRepairFocusedFiles"] >= 1
    assert capsule["approved_plan_sha256"] == "a" * 64
    assert capsule["context_partition"]["active"] is True
    assert set(capsule["context_partition"]["selected_regions"]) == {"battery", "heater"}
    projected_required = capsule["approved_plan"]["required_case_files"]
    assert "system/fvSolution" in projected_required
    assert len(projected_required) < len(required)
    assert "keyword e is undefined" in capsule["native_failure"]


class _RuntimeRepairCaptureLLM:
    def __init__(self):
        self.prompts: list[str] = []

    def generate(self, schema, prompt: str, *, system_prompt: str | None = None):
        del system_prompt
        assert schema is RuntimeRepairTurn
        self.prompts.append(prompt)
        return RuntimeRepairTurn(
            action=BlockAction(
                type="block",
                reason="capture only",
                needs_user_input=False,
                rationale="test",
            )
        )


def test_v504_context_controller_uses_runtime_partition_before_model_call(tmp_path, graph_path):
    state = make_state()
    base = make_plan(state.intake, solver="foamMultiRun")
    required = ["system/controlDict", "system/fvSolution", "system/fvSchemes"] + [
        f"constant/battery/p{i}" for i in range(35)
    ] + [f"constant/heater/p{i}" for i in range(35)]
    plan = base.model_copy(update={
        "solver": "foamMultiRun",
        "solver_provider_id": "execution.foamMultiRun",
        "execution": _solid_execution(),
        "region_layouts": [
            RegionCaseLayout(region="battery", required_fields=["T"]),
            RegionCaseLayout(region="heater", required_fields=["T"]),
        ],
        "interfaces": [
            RegionInterface(
                region="battery", patch="battery_to_heater",
                neighbour_region="heater", neighbour_patch="heater_to_battery", fields=["T"],
            )
        ],
        "required_case_files": required,
        "assumptions": [("historical runtime context " + "x" * 1800) for _ in range(50)],
    })
    state.engineering_plan = plan
    llm = _RuntimeRepairCaptureLLM()
    agent = CFDEngineeringAgent(
        llm,
        workspace=tmp_path,
        capability_db=graph_path,
        tools=FakeOpenFOAMTools(),
        policy=EngineeringPolicy(
            compact_phase_schemas=True,
            max_model_prompt_chars=18_000,
            bounded_evidence_context=True,
        ),
    )
    agent.workspace.write_text("system/fvSolution", "solvers\n{\n    T { solver smoothSolver; }\n}\n" + ("// x\n" * 2500))
    agent.workspace.write_text("system/controlDict", _solid_control_dict())

    turn = agent._generate_turn(
        state,
        step=1,
        local_step=1,
        current_step_limit=4,
        phase="runtime_repair",
        runtime_log=(
            "--> FOAM FATAL IO ERROR:\n"
            "keyword e is undefined in dictionary <CASE_DIR><LOCAL_PATH:solvers>\n"
        ),
        native_execution=True,
    )

    assert isinstance(turn, RuntimeRepairTurn)
    assert len(llm.prompts) == 1
    prompt = llm.prompts[0]
    assert len(prompt) <= 18_000
    assert '"state_mode": "runtime_failure_partition_v1"' in prompt
    assert "keyword e is undefined" in prompt
    assert "historical runtime context" not in prompt


def test_v504_repair_runtime_enters_model_with_large_two_solid_plan_without_context_error(tmp_path, graph_path):
    state = make_state()
    base = make_plan(state.intake, solver="foamMultiRun")
    required = ["system/controlDict", "system/fvSolution", "system/fvSchemes"] + [
        f"constant/battery/p{i}" for i in range(35)
    ] + [f"constant/heater/p{i}" for i in range(35)]
    plan = base.model_copy(update={
        "solver": "foamMultiRun",
        "solver_provider_id": "execution.foamMultiRun",
        "execution": _solid_execution(),
        "region_layouts": [
            RegionCaseLayout(region="battery", required_fields=["T"]),
            RegionCaseLayout(region="heater", required_fields=["T"]),
        ],
        "interfaces": [
            RegionInterface(
                region="battery", patch="battery_to_heater",
                neighbour_region="heater", neighbour_patch="heater_to_battery", fields=["T"],
            )
        ],
        "required_case_files": required,
        "assumptions": [("archived plan detail " + "z" * 2000) for _ in range(60)],
    })
    state.engineering_plan = plan
    llm = _RuntimeRepairCaptureLLM()
    agent = CFDEngineeringAgent(
        llm,
        workspace=tmp_path,
        capability_db=graph_path,
        tools=FakeOpenFOAMTools(),
        policy=EngineeringPolicy(
            compact_phase_schemas=True,
            max_model_prompt_chars=18_000,
            bounded_evidence_context=True,
            max_runtime_repair_steps=2,
        ),
    )
    agent.workspace.write_text(
        "system/controlDict", foam_header("system/controlDict") + "endTime 300;\ndeltaT 0.1;\n"
    )
    agent.workspace.write_text(
        "system/fvSolution",
        foam_header("system/fvSolution") + "solvers\n{\n    T { solver smoothSolver; }\n}\n" + ("// x\n" * 2500),
    )
    agent.workspace.write_text("system/fvSchemes", foam_header("system/fvSchemes") + "ddtSchemes {}\n")
    state.case_seal = agent.workspace.seal(plan)
    state.current_state = State.SOLVE_READY
    state.approve_solve()

    outcome = agent.repair_runtime(
        state,
        runtime_log=(
            "--> FOAM FATAL IO ERROR:\n"
            "keyword e is undefined in dictionary <CASE_DIR><LOCAL_PATH:solvers>\n"
        ),
        attempt=1,
        native_execution=False,
    )

    assert not outcome.retry
    assert state.current_state == State.ENGINEERING_BLOCKED
    assert len(llm.prompts) == 1
    assert len(llm.prompts[0]) <= 18_000
    assert '"state_mode": "runtime_failure_partition_v1"' in llm.prompts[0]
    assert "keyword e is undefined" in llm.prompts[0]
