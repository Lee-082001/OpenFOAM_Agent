from __future__ import annotations

from pathlib import Path

from openfoam_agent.agents.intake import IntakeAgent
from openfoam_agent.engineering import CFDEngineeringAgent, EngineeringPolicy
from openfoam_agent.llm import WorkflowLLMs
from openfoam_agent.runtime import RuntimeOrchestrator
from openfoam_agent.review import CFDFeedbackReviewAgent
from openfoam_agent.postprocessing import CFDPostProcessingAgent, PostProcessingPolicy
from openfoam_agent.progress import NullProgressReporter, ProgressEvent, ProgressReporter
from openfoam_agent.schemas.simulation import RuntimePolicy
from openfoam_agent.tools.openfoam import OpenFOAMTools
from openfoam_agent.workflow.state import CFDState
from openfoam_agent.workflow.states import State


class CFDWorkflow:
    """v2 workflow: intake/confirmation -> one engineering agent -> safety gates."""

    def __init__(
        self,
        llm,
        capability_db: str | Path,
        workspace: str | Path,
        openfoam_tools: OpenFOAMTools | None = None,
        runtime_policy: RuntimePolicy | None = None,
        postprocessing_policy: PostProcessingPolicy | None = None,
        engineering_policy: EngineeringPolicy | None = None,
        native_execution: bool = True,
        stream_solver_output: bool = False,
        postprocessing_enabled: bool = True,
        progress: ProgressReporter | None = None,
    ) -> None:
        self.progress = progress or NullProgressReporter()
        self.llms = WorkflowLLMs.coerce(llm)
        self.intake = IntakeAgent(self.llms.intake)
        self.tools = openfoam_tools or OpenFOAMTools.for_workspace(workspace)
        self.engineering = CFDEngineeringAgent(
            self.llms.engineering,
            workspace=workspace,
            capability_db=capability_db,
            tools=self.tools,
            policy=engineering_policy,
            progress=self.progress,
        )
        self.runtime = RuntimeOrchestrator(
            self.tools,
            self.engineering,
            runtime_policy,
            stream_output=stream_solver_output,
            progress=self.progress,
        )
        self.review = CFDFeedbackReviewAgent(self.llms.review, progress=self.progress)
        self.postprocess = CFDPostProcessingAgent(
            self.llms.postprocessing,
            workspace=workspace,
            tools=self.tools,
            policy=postprocessing_policy,
            progress=self.progress,
        )
        self.native_execution = native_execution
        self.postprocessing_enabled = postprocessing_enabled

    def step(self, state: CFDState) -> CFDState:
        match state.current_state:
            case State.INIT:
                state.transition(State.INTAKE_ANALYSIS, "Workflow started.")
            case State.INTAKE_ANALYSIS:
                self.progress.emit(
                    ProgressEvent(phase="intake", message="CFD intake 분석", status="start")
                )
                self.intake.run(state)
                self.progress.emit(
                    ProgressEvent(
                        phase="intake",
                        message=f"intake 분석 완료: state={state.current_state.value}",
                        status="success",
                    )
                )
            case State.ENGINEERING:
                self.engineering.prepare(state, native_execution=self.native_execution)
            case State.SIMULATION:
                self.runtime.run(state)
                if (
                    state.current_state == State.EXECUTION_DONE
                    and state.runtime_report is not None
                    and state.runtime_report.success
                ):
                    if self.postprocessing_enabled:
                        self.postprocess.run(state)
                    else:
                        state.transition(
                            State.RESULT_REVIEW_REQUIRED,
                            "OpenFOAM runtime completed; automatic post-processing was skipped. Human result review is required.",
                        )
                    if state.current_state == State.RESULT_REVIEW_REQUIRED:
                        state.mark_feedback_awaiting_review()
                        self.progress.emit(
                            ProgressEvent(
                                phase="review",
                                message="runtime/post-processing 완료; human result review 대기",
                                status="success",
                            )
                        )
            case State.RUNTIME_REPAIR:
                # RUNTIME_REPAIR is an internal transient state owned by RuntimeOrchestrator.
                # Reaching the top-level workflow means an internal repair exit failed to close
                # its state transition. Block deterministically instead of exposing a generic
                # "No v2 handler" failure that hides the orchestration bug.
                state.solve_approved = False
                state.transition(
                    State.ENGINEERING_BLOCKED,
                    "Internal invariant violation: RUNTIME_REPAIR escaped RuntimeOrchestrator without an explicit repair decision.",
                )
            case _:
                state.transition(State.FAILED, f"No v2 handler for {state.current_state.value}.")
        return state

    def run(self, state: CFDState, max_steps: int = 12) -> CFDState:
        terminal = {
            State.INTAKE_REVIEW_REQUIRED,
            State.NEEDS_CLARIFICATION,
            State.CASE_PREVIEW_READY,
            State.MESH_READY,
            State.SOLVE_READY,
            State.ENGINEERING_REVIEW_REQUIRED,
            State.ENGINEERING_BLOCKED,
            State.RESULT_REVIEW_REQUIRED,
            State.REVISION_READY,
            State.COMPLETE,
            State.DONE,
            State.FAILED,
        }
        for _ in range(max_steps):
            if state.current_state in terminal:
                return state
            failed_stage = state.current_state
            try:
                self.step(state)
            except Exception as exc:
                state.transition(
                    State.FAILED,
                    f"{type(exc).__name__} during {failed_stage.value}: {str(exc) or repr(exc)}",
                )
                return state
        state.transition(State.FAILED, "Workflow max_steps exceeded.")
        return state
