from __future__ import annotations

import pytest

from conftest import FakeOpenFOAMTools, foam_header, make_plan, make_state
from openfoam_agent.engineering import CFDEngineeringAgent, EngineeringPolicy
from openfoam_agent.schemas.engineering import (
    BlockAction,
    CaseBundleFile,
    FinishPreviewAction,
    RepairCasePlanAction,
    RepairTurn,
    RetrySolverAction,
    RunMeshCommandAction,
)
from openfoam_agent.workflow.states import State


def _fv_schemes(*, with_laplacian: bool = False) -> str:
    text = foam_header("system/fvSchemes") + """
ddtSchemes
{
    default Euler;
}
gradSchemes
{
    default Gauss linear;
}
divSchemes
{
    default none;
    div(phi,U) Gauss linearUpwind grad(U);
}
interpolationSchemes
{
    default linear;
}
snGradSchemes
{
    default corrected;
}
"""
    if with_laplacian:
        text += """
laplacianSchemes
{
    default Gauss linear corrected;
}
"""
    return text


def _seed_core(agent: CFDEngineeringAgent) -> None:
    agent.workspace.write_text(
        "system/controlDict",
        foam_header("system/controlDict") + "solver incompressibleFluid;\nendTime 1;\ndeltaT 0.01;\n",
    )
    agent.workspace.write_text("system/fvSchemes", _fv_schemes(with_laplacian=False))
    agent.workspace.write_text(
        "system/fvSolution",
        foam_header("system/fvSolution") + "solvers {}\nPIMPLE {}\n",
    )


class _CaptureRepairLLM:
    store = False
    model = "synthetic-repair"

    def __init__(self):
        self.prompts: list[str] = []
        self.schemas: list[type] = []

    def generate(self, schema, prompt, **kwargs):
        del kwargs
        self.schemas.append(schema)
        self.prompts.append(prompt)
        return RepairTurn(
            action=BlockAction(
                type="block",
                reason="capture only",
                block_kind="other",
                needs_user_input=False,
            )
        )


def _make_repair_agent(tmp_path, graph_path, llm=None):
    return CFDEngineeringAgent(
        llm or _CaptureRepairLLM(),
        workspace=tmp_path,
        capability_db=graph_path,
        tools=FakeOpenFOAMTools(),
        policy=EngineeringPolicy(
            max_model_prompt_chars=18_000,
            max_authoring_prompt_chars=32_000,
            compact_phase_schemas=True,
            state_delta_context=True,
            bounded_evidence_context=True,
            staged_case_authoring=True,
        ),
    )


def _record_laplacian_failure(agent, state):
    event = agent._event(
        2,
        "validate_pre_solve",
        False,
        "Zero-step OpenFOAM consumer initialization rejected the case.",
        """checkedFiles=11
meshPatches=4
nativeCommand: foamRun
returnCode: 1
diagnosticKind: foam_fatal_io_error
--> FOAM FATAL IO ERROR:
keyword laplacianSchemes is undefined in dictionary \"<WORKSPACE><LOCAL_PATH:fvSchemes>\"
file: <WORKSPACE><LOCAL_PATH:fvSchemes> from line 10 to 21.
FOAM exiting
""",
        validation_status="fail",
        failure_category="case",
    )
    state.engineering_events.append(event)
    agent._record_unresolved_failure(state, event)
    return event


def test_failure_local_repair_context_fits_18k_with_bloated_plan_and_case(tmp_path, graph_path):
    llm = _CaptureRepairLLM()
    agent = _make_repair_agent(tmp_path, graph_path, llm)
    state = make_state(); state.transition(State.ENGINEERING, "test")
    plan = make_plan(state.intake)
    plan.required_case_files = ["system/controlDict", "system/fvSchemes", "system/fvSolution"]
    # This makes the old full pending-plan repair prompt far larger than the CLI 18k
    # envelope while the new repair projection intentionally omits the audit prose.
    plan.assumptions = [(f"large historical assumption {i}: " + "x" * 2200) for i in range(80)]
    agent._pending_execution_plan = plan
    _seed_core(agent)
    for i in range(20):
        agent.workspace.write_text(
            f"system/unrelated{i}.dict",
            foam_header(f"system/unrelated{i}.dict") + ("entry value;\n" * 120),
        )
    _record_laplacian_failure(agent, state)

    turn = agent._generate_turn(
        state,
        step=3,
        local_step=3,
        current_step_limit=12,
        phase="prepare",
        native_execution=True,
    )

    assert isinstance(turn, RepairTurn)
    assert llm.schemas[-1] is RepairTurn
    prompt = llm.prompts[-1]
    assert len(prompt) <= 18_000
    assert '"state_mode": "case_validation_repair_v1"' in prompt
    assert "laplacianSchemes" in prompt
    assert '"path": "system/fvSchemes"' in prompt
    assert "div(phi,U)" in prompt
    assert "large historical assumption" not in prompt
    assert "capability_graph_hint" not in prompt
    assert "pending_engineering_plan" not in prompt
    assert "current_engineering_plan" not in prompt


def test_redacted_basename_selects_exact_failed_file_first(tmp_path, graph_path):
    agent = _make_repair_agent(tmp_path, graph_path)
    state = make_state(); plan = make_plan(state.intake)
    _seed_core(agent)
    files = agent._validation_relevant_case_files(
        state,
        plan,
        'keyword laplacianSchemes is undefined in dictionary "<WORKSPACE><LOCAL_PATH:fvSchemes>"',
    )
    assert files
    assert files[0]["path"] == "system/fvSchemes"
    assert "div(phi,U)" in files[0]["content"]


def test_huge_failed_dictionary_is_bounded_in_repair_context_not_rejected(tmp_path, graph_path):
    llm = _CaptureRepairLLM(); agent = _make_repair_agent(tmp_path, graph_path, llm)
    state = make_state(); state.transition(State.ENGINEERING, "test")
    plan = make_plan(state.intake)
    plan.required_case_files = ["system/controlDict", "system/fvSchemes", "system/fvSolution"]
    agent._pending_execution_plan = plan
    _seed_core(agent)
    huge = foam_header("system/fvSchemes") + "ddtSchemes {}\n" + ("// filler\n" * 18000)
    agent.workspace.write_text("system/fvSchemes", huge)
    _record_laplacian_failure(agent, state)

    agent._generate_turn(
        state, step=3, local_step=3, current_step_limit=12,
        phase="prepare", native_execution=True,
    )
    prompt = llm.prompts[-1]
    assert len(prompt) <= 18_000
    assert '"path": "system/fvSchemes"' in prompt
    assert '"truncated": true' in prompt


@pytest.mark.parametrize(
    ("runtime", "retry_hint", "terminal_type"),
    [
        (False, False, FinishPreviewAction),
        (False, True, FinishPreviewAction),
        (True, False, RetrySolverAction),
        (True, True, RetrySolverAction),
    ],
)
def test_repair_terminal_action_is_owned_by_controller_phase_not_retry_hint(
    tmp_path, graph_path, runtime, retry_hint, terminal_type
):
    agent = _make_repair_agent(tmp_path, graph_path)
    state = make_state(); state.transition(State.ENGINEERING, "test")
    plan = make_plan(state.intake)
    plan.required_case_files = ["system/controlDict", "system/fvSchemes", "system/fvSolution"]
    agent._pending_execution_plan = plan
    _seed_core(agent)

    repair = RepairCasePlanAction(
        type="repair_case_plan",
        diagnosis="Repair the failed dictionary only.",
        replacement_files=[
            CaseBundleFile(path="system/fvSchemes", content=_fv_schemes(with_laplacian=True))
        ],
        validate_pre_solve=True,
        retry_solver=retry_hint,
    )
    actions, _, graph = agent._repair_actions(state, repair, runtime=runtime)

    assert graph.valid, graph.failures
    assert not any(isinstance(action, RunMeshCommandAction) for action in actions)
    assert isinstance(actions[-1], terminal_type)
    if runtime:
        assert not any(isinstance(action, FinishPreviewAction) for action in actions)
    else:
        assert not any(isinstance(action, RetrySolverAction) for action in actions)


def test_dictionary_only_repair_does_not_rebuild_or_recheck_mesh(tmp_path, graph_path):
    agent = _make_repair_agent(tmp_path, graph_path)
    state = make_state(); state.transition(State.ENGINEERING, "test")
    plan = make_plan(state.intake)
    plan.required_case_files = ["system/controlDict", "system/fvSchemes", "system/fvSolution"]
    agent._pending_execution_plan = plan
    _seed_core(agent)

    repair = RepairCasePlanAction(
        type="repair_case_plan",
        diagnosis="Add the missing fvSchemes laplacian section selected by the Engineering Agent.",
        replacement_files=[
            CaseBundleFile(path="system/fvSchemes", content=_fv_schemes(with_laplacian=True))
        ],
        validate_pre_solve=True,
    )
    actions, _, graph = agent._repair_actions(state, repair, runtime=False)
    assert graph.valid, graph.failures
    assert set(graph.changed_files) == {"system/fvSchemes"}
    assert graph.native_pipeline == ()
    assert not any(isinstance(action, RunMeshCommandAction) for action in actions)
    assert [action.type for action in actions] == [
        "write_case_file", "validate_pre_solve", "finish_preview"
    ]


def test_plan_metadata_validation_failure_uses_same_bounded_repair_path(tmp_path, graph_path):
    llm = _CaptureRepairLLM(); agent = _make_repair_agent(tmp_path, graph_path, llm)
    state = make_state(); state.transition(State.ENGINEERING, "test")
    plan = make_plan(state.intake)
    plan.assumptions = [("audit prose " + "q" * 1800) for _ in range(80)]
    agent._pending_execution_plan = plan
    _seed_core(agent)
    event = agent._event(
        2, "finish_preview", False,
        "Engineering plan rejected by deterministic safety/evidence gate.",
        "Confirmed classification fact classification.problem_type requires at least one case semantic assertion.",
        validation_status="fail", failure_category="case",
    )
    state.engineering_events.append(event); agent._record_unresolved_failure(state, event)

    agent._generate_turn(
        state, step=3, local_step=3, current_step_limit=12,
        phase="prepare", native_execution=True,
    )
    prompt = llm.prompts[-1]
    assert len(prompt) <= 18_000
    assert "classification.problem_type" in prompt
    assert "audit prose" not in prompt
    assert '"state_mode": "case_validation_repair_v1"' in prompt


from openfoam_agent.schemas.common import ToolResult
from openfoam_agent.schemas.engineering import (
    OpenFOAMExecutionSpec,
    SearchCapabilitiesAction,
    ValidatePreSolveAction,
)
from openfoam_agent.verification.safety import parse_check_mesh_evidence
from conftest import mesh_ok_log, tool_result


class _OneFailureZeroStepTools(FakeOpenFOAMTools):
    def __init__(self):
        super().__init__()
        self.zero_step_calls = 0

    def zero_step_consumer_validate(self, case_dir, execution, timeout=30):
        del case_dir, execution, timeout
        self.zero_step_calls += 1
        if self.zero_step_calls == 1:
            return (
                ToolResult(
                    success=False,
                    command=["foamRun"],
                    return_code=1,
                    stderr=(
                        "--> FOAM FATAL IO ERROR:\n"
                        'keyword laplacianSchemes is undefined in dictionary "<WORKSPACE><LOCAL_PATH:fvSchemes>"\n'
                        "FOAM exiting\n"
                    ),
                ),
                "Zero-step consumer validation executed in a temporary shadow case.",
            )
        return (
            ToolResult(
                success=True,
                command=["foamRun"],
                return_code=0,
                stdout="zero-step initialization OK\n",
            ),
            "Zero-step consumer validation executed in a temporary shadow case.",
        )


class _SingleRepairLLM:
    store = False
    model = "synthetic-repair"

    def __init__(self, action):
        self.action = action
        self.prompts: list[str] = []
        self.schemas: list[type] = []

    def generate(self, schema, prompt, **kwargs):
        del kwargs
        self.schemas.append(schema)
        self.prompts.append(prompt)
        assert schema is RepairTurn
        return RepairTurn(action=self.action)


def _field_u() -> str:
    return foam_header("0/U", "volVectorField") + """
dimensions [0 1 -1 0 0 0 0];
internalField uniform (0 0 0);
boundaryField
{
    inlet { type fixedValue; value uniform (1 0 0); }
    outlet { type zeroGradient; }
}
"""


def _mesh_boundary() -> str:
    return """FoamFile
{
    version 2.0;
    format ascii;
    class polyBoundaryMesh;
    object boundary;
}
2
(
    inlet { type patch; nFaces 1; startFace 0; }
    outlet { type patch; nFaces 1; startFace 1; }
)
"""


@pytest.mark.parametrize("retry_hint", [False, True])
def test_native_presolve_failure_repairs_only_failed_dictionary_and_reaches_solve_ready(tmp_path, graph_path, retry_hint):
    state = make_state(); state.transition(State.ENGINEERING, "test")
    plan = make_plan(state.intake)
    plan.required_case_files = [
        "system/controlDict", "system/fvSchemes", "system/fvSolution", "0/U"
    ]
    plan.execution = OpenFOAMExecutionSpec(
        driver="foamRun",
        driver_provider_id="execution.foamRun",
        solver_module="incompressibleFluid",
        solver_provider_id="solver.incompressibleFluid",
    )
    repair = RepairCasePlanAction(
        type="repair_case_plan",
        diagnosis="Repair the OpenFOAM-reported fvSchemes omission only.",
        replacement_files=[
            CaseBundleFile(path="system/fvSchemes", content=_fv_schemes(with_laplacian=True))
        ],
        validate_pre_solve=True,
        retry_solver=retry_hint,
    )
    llm = _SingleRepairLLM(repair)
    tools = _OneFailureZeroStepTools()
    agent = CFDEngineeringAgent(
        llm,
        workspace=tmp_path,
        capability_db=graph_path,
        tools=tools,
        policy=EngineeringPolicy(
            max_model_prompt_chars=18_000,
            compact_phase_schemas=True,
            bounded_evidence_context=True,
            staged_case_authoring=True,
            require_solve_ready_gate=True,
        ),
    )
    agent.workspace.write_text(
        "system/controlDict",
        foam_header("system/controlDict") + "solver incompressibleFluid;\nendTime 1;\ndeltaT 0.01;\n",
    )
    agent.workspace.write_text("system/fvSchemes", _fv_schemes(with_laplacian=False))
    agent.workspace.write_text(
        "system/fvSolution", foam_header("system/fvSolution") + "solvers {}\nPIMPLE {}\n"
    )
    agent.workspace.write_text("0/U", _field_u())
    agent.workspace.write_text("constant/polyMesh/boundary", _mesh_boundary())
    agent._pending_execution_plan = plan

    for query in ("incompressibleFluid", "foamRun"):
        observed = agent._dispatch_tool_action(
            SearchCapabilitiesAction(
                type="search_capabilities", query=query, rationale="test provenance"
            ),
            step=1,
            native_execution=False,
            phase="prepare",
            state=state,
        )
        state.engineering_events.append(observed)

    state.mesh_evidence = parse_check_mesh_evidence(
        tool_result("checkMesh", success=True, stdout=mesh_ok_log())
    )
    agent._checkmesh_mesh_manifest = agent.workspace.mesh_manifest_digest()

    failed, terminal = agent._dispatch_prepare(
        state,
        ValidatePreSolveAction(
            type="validate_pre_solve",
            required_case_files=plan.required_case_files,
            rationale="controller pre-solve",
        ),
        step=2,
        native_execution=True,
    )
    assert not failed.success and not terminal
    state.engineering_events.append(failed)
    agent._record_unresolved_failure(state, failed)
    assert tools.zero_step_calls == 1

    turn = agent._generate_turn(
        state, step=3, local_step=3, current_step_limit=12,
        phase="prepare", native_execution=True,
    )
    assert isinstance(turn, RepairTurn)
    assert len(llm.prompts[-1]) <= 18_000
    assert '"state_mode": "case_validation_repair_v1"' in llm.prompts[-1]

    terminal = agent._execute_prepare_decision(
        state,
        turn.action,
        llm_step=3,
        progress_phase="engineering",
        progress_step=3,
        progress_limit=12,
        native_execution=True,
    )
    assert terminal
    assert state.current_state == State.SOLVE_READY
    assert tools.zero_step_calls == 2
    assert tools.mesh_calls == []
    assert "laplacianSchemes" in agent.workspace.read_text("system/fvSchemes")
    assert state.primary_failure is None


def test_successful_repair_support_read_is_returned_in_next_failure_local_turn(tmp_path, graph_path):
    llm = _CaptureRepairLLM(); agent = _make_repair_agent(tmp_path, graph_path, llm)
    state = make_state(); state.transition(State.ENGINEERING, "test")
    plan = make_plan(state.intake)
    plan.required_case_files = ["system/controlDict", "system/fvSchemes", "system/fvSolution"]
    agent._pending_execution_plan = plan
    _seed_core(agent)
    _record_laplacian_failure(agent, state)
    state.engineering_events.append(
        agent._event(
            3,
            "read_case_file",
            True,
            "Read case file system/fvSolution.",
            "system/fvSolution current content: solvers { p { solver GAMG; } }",
        )
    )

    agent._generate_turn(
        state, step=4, local_step=4, current_step_limit=12,
        phase="prepare", native_execution=True,
    )
    prompt = llm.prompts[-1]
    assert "supporting_observations" in prompt
    assert "solver GAMG" in prompt
    assert len(prompt) <= 18_000
