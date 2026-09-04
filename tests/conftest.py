from __future__ import annotations

from collections import defaultdict, deque
from pathlib import Path

import pytest

from openfoam_agent.schemas.common import ToolResult
from openfoam_agent.schemas.engineering import ConfirmedFactBinding, EngineeringPlan, EngineeringTurn
from openfoam_agent.schemas.postprocessing import PostProcessingTurn
from openfoam_agent.schemas.feedback import FeedbackAssessment
from openfoam_agent.schemas.intake import CFDIntakeSpec, IntakeFact
from openfoam_agent.schemas.request import UserRequest
from openfoam_agent.workflow.state import CFDState


ROOT = Path(__file__).resolve().parents[1]
GRAPH = ROOT / "config" / "openfoam14_capability_graph.json"


def make_intake(*, reynolds: str = "1000") -> CFDIntakeSpec:
    user_text = f"사각형 장애물 주위 vortex shedding Re={reynolds} 나머지는 탐색용으로 정해줘"
    return CFDIntakeSpec(
        title="Square-obstacle vortex shedding",
        facts=[
            IntakeFact(
                id="request.summary",
                category="context",
                label="Request",
                value=user_text,
                source="derived",
                reason="Normalized the user request.",
            ),
            IntakeFact(
                id="classification.problem_type",
                category="classification",
                label="Problem class",
                value="external_flow",
                source="derived",
                reason="The user requested flow around an obstacle.",
                depends_on=["request.summary"],
            ),
            IntakeFact(
                id="objective.primary",
                category="objective",
                label="Objective",
                value="vortex shedding around a square obstacle",
                source="user",
                evidence="사각형 장애물 주위 vortex shedding",
            ),
            IntakeFact(
                id="operating.reynolds_number",
                category="scale",
                label="Reynolds number",
                value=reynolds,
                source="user",
                evidence=f"Re={reynolds}",
            ),
        ],
        status="ready_for_review",
    )


def make_state(*, reynolds: str = "1000") -> CFDState:
    request = UserRequest(
        prompt=f"사각형 장애물 주위 vortex shedding Re={reynolds} 나머지는 탐색용으로 정해줘",
        exploratory_completion_authorized=True,
    )
    state = CFDState(run_id="test-run", user_request=request, intake=make_intake(reynolds=reynolds))
    state.confirm_intake()
    return state


def make_plan(intake: CFDIntakeSpec, *, solver: str = "incompressibleFluid") -> EngineeringPlan:
    return EngineeringPlan(
        case_name="squareWake",
        solver=solver,
        solver_provider_id=f"solver.{solver}",
        openfoam_version="14",
        problem_interpretation="Transient external flow around a stationary square obstacle.",
        temporal_behavior="transient",
        motion_kind="static",
        mesh_motion_requirement="static",
        mesh_strategy="Agent-selected exploratory 2D mesh strategy.",
        decisions=[],
        assumptions=["Exploratory domain dimensions selected by the engineering agent."],
        confirmed_fact_ids=[
            fact.id for fact in intake.facts if fact.category != "context"
        ],
        confirmed_fact_bindings=[
            ConfirmedFactBinding(
                fact_id=fact.id,
                plan_fields=["problem_interpretation"],
                explanation="Test fixture binds the confirmed fact to the engineering interpretation.",
            )
            for fact in intake.facts if fact.category != "context"
        ],
        evidence=[],
        postprocess_strategy=["Inspect wake unsteadiness and force history."],
        confirmed_intake_sha256=intake.digest(),
    )


def foam_header(path: str, class_name: str = "dictionary") -> str:
    location, object_name = path.rsplit("/", 1)
    return (
        "FoamFile\n"
        "{\n"
        "    version 2.0;\n"
        "    format ascii;\n"
        f"    class {class_name};\n"
        f'    location "{location}";\n'
        f"    object {object_name};\n"
        "}\n"
    )


def control_dict(solver: str = "incompressibleFluid") -> str:
    return foam_header("system/controlDict") + (
        f"solver {solver};\nstartFrom startTime;\nstartTime 0;\nendTime 10;\ndeltaT 0.01;\n"
    )


def mesh_ok_log(cells: int = 1200) -> str:
    return (
        f"cells: {cells}\n"
        "Mesh non-orthogonality Max: 21.5 average: 4.0\n"
        "Max skewness = 1.2\n"
        "cells with negative volume: 0\n"
        "Mesh OK.\n"
    )


class ScriptedLLM:
    def __init__(self, actions):
        self.turns = deque(actions)
        self.prompts: list[str] = []
        self.schemas: list[type] = []

    def generate(self, schema, prompt: str, *, system_prompt: str | None = None):
        del system_prompt
        self.prompts.append(prompt)
        self.schemas.append(schema)
        if not self.turns:
            raise AssertionError("ScriptedLLM exhausted")
        action = self.turns.popleft()
        if schema is EngineeringTurn:
            return action if isinstance(action, EngineeringTurn) else EngineeringTurn(action=action)
        if schema is PostProcessingTurn:
            return (
                action
                if isinstance(action, PostProcessingTurn)
                else PostProcessingTurn(action=action)
            )
        if schema is FeedbackAssessment:
            return action if isinstance(action, FeedbackAssessment) else FeedbackAssessment.model_validate(action)
        raise AssertionError(f"Unexpected schema requested: {schema}")


class FakeOpenFOAMTools:
    def __init__(
        self,
        *,
        mesh_results=None,
        foam_runs=None,
        postprocess_runs=None,
        version: str | None = "14",
        check_mesh_available: bool = True,
    ):
        self.version = version
        self.check_mesh_available = check_mesh_available
        self.mesh_results = defaultdict(deque)
        for command, results in (mesh_results or {}).items():
            self.mesh_results[command].extend(results)
        self.foam_runs = deque(foam_runs or [])
        self.postprocess_runs = deque(postprocess_runs or [])
        self.mesh_calls: list[str] = []
        self.dictionary_calls: list[str] = []
        self.foam_run_solvers: list[str] = []
        self.postprocess_calls: list[dict[str, object]] = []

    def detected_foundation_version(self):
        return self.version

    def check_mesh_preflight(self):
        if self.check_mesh_available:
            return {"name": "checkMesh", "available": True, "trusted": True, "path_redacted": "checkMesh"}
        return {
            "name": "checkMesh",
            "available": False,
            "trusted": False,
            "reason": "Allowlisted OpenFOAM executable is unavailable: checkMesh",
        }

    def environment_snapshot(self):
        return {
            "wm_project": "OpenFOAM",
            "wm_project_version": self.version or "",
            "foundation_version": self.version,
            "commands": [],
            "foam_tutorials": "",
            "foam_src": "",
            "foam_etc": "",
        }

    def foam_dictionary_validate(self, file_path, cwd=None):
        del cwd
        self.dictionary_calls.append(str(file_path))
        return ToolResult(
            success=True,
            command=["foamDictionary", "-keywords", str(file_path)],
            return_code=0,
            stdout="dictionary OK\n",
        )

    def surface_check(self, geometry_path, cwd=None):
        del cwd
        return ToolResult(
            success=True,
            command=["surfaceCheck", str(geometry_path)],
            return_code=0,
            stdout="Surface OK\n",
        )

    def run_mesh_command(self, command, case_dir):
        del case_dir
        self.mesh_calls.append(command)
        queue = self.mesh_results[command]
        if queue:
            return queue.popleft()
        return ToolResult(success=True, command=[command], return_code=0, stdout="OK\n")

    def foam_run(
        self,
        case_dir,
        solver,
        *,
        stream_output=False,
        timeout=3600,
        output_callback=None,
    ):
        del case_dir, stream_output, timeout
        self.foam_run_solvers.append(solver)
        if not self.foam_runs:
            raise AssertionError("No scripted foamRun result")
        result = self.foam_runs.popleft()
        if output_callback is not None:
            for line in result.stdout.splitlines(keepends=True):
                output_callback(line)
            for line in result.stderr.splitlines(keepends=True):
                output_callback(line)
        return result

    def foam_post_process(
        self,
        case_dir,
        dictionary_path,
        *,
        solver=None,
        latest_time=False,
        timeout=900,
    ):
        del case_dir, timeout
        self.postprocess_calls.append(
            {
                "dictionary_path": str(dictionary_path),
                "solver": solver,
                "latest_time": latest_time,
            }
        )
        if self.postprocess_runs:
            return self.postprocess_runs.popleft()
        return ToolResult(
            success=True,
            command=["foamPostProcess"],
            return_code=0,
            stdout="Time = 1\nEnd\n",
        )


def tool_result(command: str, *, success: bool, stdout: str = "", stderr: str = "") -> ToolResult:
    return ToolResult(
        success=success,
        command=[command],
        return_code=0 if success else 1,
        stdout=stdout,
        stderr=stderr,
    )


@pytest.fixture
def graph_path() -> Path:
    return GRAPH
