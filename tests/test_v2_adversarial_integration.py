from __future__ import annotations

from pathlib import Path

from openfoam_agent.engineering import CFDEngineeringAgent, EngineeringPolicy
from openfoam_agent.runtime import RuntimeOrchestrator
from openfoam_agent.schemas.common import ToolResult
from openfoam_agent.schemas.engineering import (
    EngineeringDecision,
    EngineeringEvidence,
    canonical_engineering_evidence_id,
    EngineeringPlan,
    FinishPreviewAction,
    InspectEnvironmentAction,
    RetrySolverAction,
    RunMeshCommandAction,
    SearchCapabilitiesAction,
    SearchReferencesAction,
    WriteCaseFileAction,
)
from openfoam_agent.schemas.intake import CFDIntakeSpec, IntakeFact
from openfoam_agent.schemas.request import UserRequest
from openfoam_agent.schemas.simulation import RuntimePolicy
from openfoam_agent.workflow.state import CFDState
from openfoam_agent.workflow.states import State

from conftest import FakeOpenFOAMTools, ScriptedLLM, mesh_ok_log, tool_result


class NativeMeshFakeTools(FakeOpenFOAMTools):
    """Fake native tools that also create OpenFOAM-like native mesh artifacts."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mesh_generation = 0

    def run_mesh_command(self, command, case_dir):
        result = super().run_mesh_command(command, case_dir)
        if command == "blockMesh" and result.success:
            self.mesh_generation += 1
            poly_mesh = Path(case_dir) / "constant" / "polyMesh"
            poly_mesh.mkdir(parents=True, exist_ok=True)
            (poly_mesh / "boundary").write_text(
                f"native boundary generation {self.mesh_generation}\n", encoding="utf-8"
            )
            (poly_mesh / "points").write_text(
                f"native points generation {self.mesh_generation}\n", encoding="utf-8"
            )
        return result


def _dynamic_state() -> CFDState:
    prompt = (
        "2D 원형 실린더가 y방향으로 0.2D sin(2*pi*f*t) 강제진동하는 외부유동. "
        "Re=200, 비압축성 transient, 나머지는 탐색용으로 정해줘."
    )
    intake = CFDIntakeSpec(
        title="Oscillating-cylinder wake",
        facts=[
            IntakeFact(
                id="request.summary",
                category="context",
                label="Request",
                value=prompt,
                source="derived",
                reason="Normalized user request.",
            ),
            IntakeFact(
                id="classification.problem_type",
                category="classification",
                label="Problem class",
                value="external_flow",
                source="derived",
                reason="Flow is around an immersed cylinder.",
                depends_on=["request.summary"],
            ),
            IntakeFact(
                id="objective.primary",
                category="objective",
                label="Objective",
                value="unsteady wake around an oscillating cylinder",
                source="user",
                evidence="강제진동하는 외부유동",
            ),
            IntakeFact(
                id="operating.reynolds_number",
                category="scale",
                label="Reynolds number",
                value="200",
                source="user",
                evidence="Re=200",
            ),
            IntakeFact(
                id="temporal.behavior",
                category="temporal",
                label="Temporal behavior",
                value="transient",
                source="user",
                evidence="transient",
            ),
            IntakeFact(
                id="motion.primary",
                category="motion",
                label="Cylinder motion",
                value="y=0.2D sin(2*pi*f*t)",
                source="user",
                evidence="y방향으로 0.2D sin(2*pi*f*t) 강제진동",
            ),
        ],
        status="ready_for_review",
    )
    request = UserRequest(prompt=prompt, exploratory_completion_authorized=True)
    state = CFDState(run_id="hard-dynamic-run", user_request=request, intake=intake)
    state.confirm_intake()
    return state


def _dynamic_plan(intake: CFDIntakeSpec) -> EngineeringPlan:
    return EngineeringPlan(
        case_name="oscillatingCylinder",
        solver="incompressibleFluid",
        solver_provider_id="solver.incompressibleFluid",
        openfoam_version="14",
        problem_interpretation=(
            "Transient incompressible external flow around a prescribed transversely "
            "oscillating cylinder at Re=200."
        ),
        temporal_behavior="transient",
        motion_kind="prescribed_deformation",
        mesh_motion_requirement="deforming",
        mesh_strategy=(
            "Agent-selected body-fitted exploratory mesh with a deforming mesh region; "
            "implementation chosen from observed Foundation-v14 capabilities/references."
        ),
        decisions=[
            EngineeringDecision(
                area="solver",
                choice="incompressibleFluid",
                rationale="Capability evidence covers transient incompressible flow and mesh motion.",
            ),
            EngineeringDecision(
                area="mesh_motion",
                choice="deforming mesh",
                rationale="Prescribed cylinder displacement requires moving boundary support.",
            ),
        ],
        assumptions=["Exploratory far-field dimensions are agent-selected."],
        confirmed_fact_ids=[fact.id for fact in intake.facts if fact.category != "context"],
        evidence=[
            EngineeringEvidence(
                evidence_id=canonical_engineering_evidence_id(
                    "capability", "solver.incompressibleFluid"
                ),
                note="Selected from deterministic available_evidence.",
            ),
            EngineeringEvidence(
                evidence_id=canonical_engineering_evidence_id(
                    "openfoam_reference",
                    "source:dynamicMesh/displacementLaplacian.C",
                ),
                note="Selected from deterministic available_evidence.",
            ),
        ],
        postprocess_strategy=["Inspect lift/drag and wake phase relative to imposed motion."],
        confirmed_intake_sha256=intake.digest(),
    )


def _control_dict() -> str:
    return (
        "solver incompressibleFluid;\n"
        "startFrom startTime;\n"
        "startTime 0;\n"
        "endTime 20;\n"
        "deltaT 0.002;\n"
    )


def test_hard_dynamic_mesh_failure_repair_runtime_repair_and_native_seal(
    tmp_path, graph_path, monkeypatch
):
    # Provide an installed-reference tree so the agent can perform real bounded search.
    source_root = tmp_path / "official-source"
    reference_file = source_root / "dynamicMesh" / "displacementLaplacian.C"
    reference_file.parent.mkdir(parents=True)
    reference_file.write_text(
        "// Foundation reference fixture\nclass displacementLaplacian {};\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("WM_PROJECT_DIR", str(tmp_path))
    monkeypatch.setenv("FOAM_SRC", str(source_root))

    state = _dynamic_state()
    plan = _dynamic_plan(state.intake)

    unsafe_control = (
        "solver incompressibleFluid;\n"
        "#codeStream { code #{ system(\\\"touch /tmp/pwned\\\"); #}; };\n"
    )
    actions = [
        InspectEnvironmentAction(type="inspect_environment", rationale="Inspect Foundation runtime."),
        SearchCapabilitiesAction(
            type="search_capabilities",
            query="incompressible transient mesh motion",
            rationale="Find capability evidence without deterministic solver routing.",
        ),
        SearchReferencesAction(
            type="search_references",
            query="displacementLaplacian",
            scope="source",
            rationale="Check the installed Foundation source before writing motion dictionaries.",
        ),
        WriteCaseFileAction(
            type="write_case_file",
            path="system/controlDict",
            content=unsafe_control,
            rationale="Adversarial first attempt: safety gate must reject executable directives.",
        ),
        WriteCaseFileAction(
            type="write_case_file",
            path="system/controlDict",
            content=_control_dict(),
            rationale="Write a safe solver control dictionary after observing rejection.",
        ),
        WriteCaseFileAction(
            type="write_case_file",
            path="constant/dynamicMeshDict",
            content=(
                'libs ("libfvMotionSolvers.so");\n'
                "motionSolver displacementLaplacian;\n"
                "diffusivity quadratic inverseDistance 1(cylinder);\n"
            ),
            rationale="Author an agent-selected dynamic-mesh implementation.",
        ),
        WriteCaseFileAction(
            type="write_case_file",
            path="0/pointDisplacement",
            content="dimensions [0 1 0 0 0 0 0];\ninternalField uniform (0 0 0);\n",
            rationale="Write moving-mesh field data.",
        ),
        WriteCaseFileAction(
            type="write_case_file",
            path="system/blockMeshDict",
            content="vertices (); // generation-1 exploratory mesh\n",
            rationale="Create the first body-fitted background mesh attempt.",
        ),
        RunMeshCommandAction(type="run_mesh_command", command="blockMesh", rationale="Generate mesh."),
        RunMeshCommandAction(type="run_mesh_command", command="checkMesh", rationale="Validate first mesh."),
        WriteCaseFileAction(
            type="write_case_file",
            path="system/blockMeshDict",
            content="vertices (); // generation-2 repaired mesh\n",
            rationale="Repair mesh after observing the actual checkMesh failure.",
        ),
        RunMeshCommandAction(type="run_mesh_command", command="blockMesh", rationale="Regenerate repaired mesh."),
        RunMeshCommandAction(type="run_mesh_command", command="checkMesh", rationale="Validate repaired mesh."),
        FinishPreviewAction(type="finish_preview", plan=plan, rationale="Seal the validated dynamic case."),
        # Runtime-repair actions begin after the first foamRun failure.
        WriteCaseFileAction(
            type="write_case_file",
            path="system/fvSolution",
            content="solvers { p { solver PCG; tolerance 1e-7; relTol 0.05; } }\n",
            rationale="Repair numerics after observing the real SIGFPE runtime log.",
        ),
        RunMeshCommandAction(
            type="run_mesh_command",
            command="checkMesh",
            rationale="Refresh mesh evidence because an execution input changed.",
        ),
        RetrySolverAction(type="retry_solver", plan=plan, rationale="Retry the same approved solver."),
    ]
    llm = ScriptedLLM(actions)

    first_mesh_failure = (
        "cells: 34000\n"
        "Mesh non-orthogonality Max: 87 average: 11\n"
        "Max skewness = 6.1\n"
        "cells with negative volume: 3\n"
        "Failed 3 mesh checks.\n"
    )
    tools = NativeMeshFakeTools(
        mesh_results={
            "blockMesh": [
                tool_result("blockMesh", success=True, stdout="blockMesh generation 1 complete\n"),
                tool_result("blockMesh", success=True, stdout="blockMesh generation 2 complete\n"),
            ],
            "checkMesh": [
                tool_result("checkMesh", success=False, stderr=first_mesh_failure),
                tool_result("checkMesh", success=True, stdout=mesh_ok_log(cells=36000)),
                tool_result("checkMesh", success=True, stdout=mesh_ok_log(cells=36000)),
            ],
        },
        foam_runs=[
            ToolResult(
                success=False,
                command=["foamRun"],
                return_code=1,
                stdout="Time = 0.014\nCourant Number mean: 0.3 max: 1.7\n",
                stderr="Floating point exception (core dumped)\n",
            ),
            ToolResult(
                success=True,
                command=["foamRun"],
                return_code=0,
                stdout=(
                    "Time = 0.020\n"
                    "Solving for Ux, Initial residual = 0.01, Final residual = 1e-06\n"
                    "Courant Number mean: 0.08 max: 0.31\n"
                    "time step continuity errors : sum local = 1e-8, global = 2e-9, cumulative = 3e-9\n"
                    "End\n"
                ),
            ),
        ],
    )

    agent = CFDEngineeringAgent(
        llm,
        workspace=tmp_path / "workspace",
        capability_db=graph_path,
        tools=tools,
        policy=EngineeringPolicy(max_agent_steps=24, max_mesh_cells=100_000),
    )

    agent.prepare(state, native_execution=True)
    assert state.current_state == State.MESH_READY
    assert state.mesh_evidence is not None and state.mesh_evidence.passed
    assert state.mesh_evidence.cell_count == 36000

    # The unsafe executable-content attempt must be an observed rejection, not a crash.
    rejected = [
        event
        for event in state.engineering_events
        if event.action_type == "write_case_file" and not event.success
    ]
    assert rejected
    assert "executable/unsafe directives" in rejected[0].summary
    assert any("executable/unsafe directives" in prompt for prompt in llm.prompts[4:])

    # Capability and installed-source observations must actually return to the same agent loop.
    assert any(
        event.action_type == "search_capabilities"
        and "solver.incompressibleFluid" in event.output_excerpt
        for event in state.engineering_events
    )
    assert any(
        event.action_type == "search_references"
        and "source:dynamicMesh/displacementLaplacian.C" in event.output_excerpt
        for event in state.engineering_events
    )

    # The failed checkMesh evidence must be visible before the mesh repair action.
    assert any("negative volume: 3" in prompt for prompt in llm.prompts[10:])
    assert tools.mesh_calls[:4] == ["blockMesh", "checkMesh", "blockMesh", "checkMesh"]

    # Native OpenFOAM-generated mesh inputs must be inside the immutable pre-solve seal.
    assert state.case_seal is not None
    sealed = {item.path: item for item in state.case_seal.files}
    assert sealed["constant/polyMesh/boundary"].origin == "native"
    assert sealed["constant/polyMesh/points"].origin == "native"
    assert sealed["constant/dynamicMeshDict"].origin == "agent"
    pre_runtime_manifest = state.case_seal.manifest_sha256

    state.approve_solve()

    # Mirror the interactive /solve path: reconstruct the engineering agent from the
    # sealed workspace rather than relying on in-memory authored-path/checkMesh state.
    rehydrated_agent = CFDEngineeringAgent(
        llm,
        workspace=tmp_path / "workspace",
        capability_db=graph_path,
        tools=tools,
        policy=EngineeringPolicy(max_agent_steps=24, max_mesh_cells=100_000),
    )
    runtime = RuntimeOrchestrator(
        tools,
        rehydrated_agent,
        RuntimePolicy(max_attempts=2, solver_timeout_seconds=30),
    )
    runtime.run(state)

    assert state.current_state == State.EXECUTION_DONE
    assert state.runtime_report is not None and state.runtime_report.success
    assert len(state.runtime_report.attempts) == 2
    assert state.runtime_report.attempts[0].repair_requested is True
    assert tools.foam_run_solvers == ["incompressibleFluid", "incompressibleFluid"]
    assert state.last_runtime_log_excerpt is not None
    assert "Floating point exception" in state.last_runtime_log_excerpt

    # Runtime edit must force a fresh checkMesh and produce a new integrity seal.
    assert tools.mesh_calls[-1] == "checkMesh"
    assert state.case_seal is not None
    assert state.case_seal.manifest_sha256 != pre_runtime_manifest
    repaired_sealed = {item.path: item for item in state.case_seal.files}
    assert repaired_sealed["system/fvSolution"].origin == "agent"
    assert repaired_sealed["constant/polyMesh/boundary"].origin == "native"

    # Both native mesh/tool logs and solver-attempt logs are retained outside case inputs.
    log_names = {path.name for path in rehydrated_agent.workspace.log_dir.iterdir()}
    assert any(name.endswith(".checkMesh.log") for name in log_names)
    assert "foamRun.attempt-001.log" in log_names
    assert "foamRun.attempt-002.log" in log_names
