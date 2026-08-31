from __future__ import annotations

from pathlib import Path

from openfoam_agent.engineering import CFDEngineeringAgent
from openfoam_agent.progress import (
    NullProgressReporter,
    ProgressEvent,
    ProgressReporter,
    SolverProgressTracker,
)
from openfoam_agent.schemas.simulation import RuntimePolicy, RuntimeReport, SimulationAttempt
from openfoam_agent.tools.diagnostics import diagnose_openfoam_failure
from openfoam_agent.tools.openfoam import OpenFOAMTools
from openfoam_agent.tools.parsers import parse_runtime_log
from openfoam_agent.tools.workspace import WorkspaceSafetyError
from openfoam_agent.workflow.state import CFDState
from openfoam_agent.workflow.states import State


class RuntimeOrchestrator:
    """Run the approved solver and return real failures to the same engineering agent."""

    def __init__(
        self,
        tools: OpenFOAMTools,
        engineering: CFDEngineeringAgent,
        policy: RuntimePolicy | None = None,
        *,
        stream_output: bool = False,
        progress: ProgressReporter | None = None,
    ) -> None:
        self.tools = tools
        self.engineering = engineering
        self.policy = policy or RuntimePolicy()
        self.stream_output = stream_output
        self.progress = progress or NullProgressReporter()

    def run(self, state: CFDState) -> CFDState:
        if not state.solve_approved:
            state.transition(State.FAILED, "Solver execution blocked: user /solve approval is missing.")
            return state
        if state.engineering_plan is None or state.case_seal is None or state.case_dir is None:
            state.transition(State.FAILED, "Solver execution blocked: sealed engineering case is missing.")
            return state
        try:
            self.engineering.workspace.adopt_seal(state.case_seal)
            self.engineering.safety.verify_seal(state.engineering_plan, state.case_seal)
        except WorkspaceSafetyError as exc:
            state.transition(State.FAILED, f"Solver execution blocked by integrity gate: {exc}")
            return state

        attempts: list[SimulationAttempt] = []
        self.progress.emit(
            ProgressEvent(
                phase="runtime",
                message=f"foamRun 시작: solver={state.engineering_plan.solver}",
                status="start",
                metrics={"maxAttempts": self.policy.max_attempts},
            )
        )
        for attempt_number in range(1, self.policy.max_attempts + 1):
            plan = state.engineering_plan
            assert plan is not None
            try:
                self.engineering.safety.verify_seal(plan, state.case_seal)
            except WorkspaceSafetyError as exc:
                state.transition(State.FAILED, f"Runtime input integrity failure: {exc}")
                return state

            self.progress.emit(
                ProgressEvent(
                    phase="runtime",
                    message=f"foamRun attempt {attempt_number}/{self.policy.max_attempts}",
                    status="start",
                )
            )
            tracker = SolverProgressTracker(
                self.progress,
                attempt=attempt_number,
                attempt_limit=self.policy.max_attempts,
            )
            run = self.tools.foam_run(
                state.case_dir,
                solver=plan.solver,
                stream_output=self.stream_output,
                timeout=self.policy.solver_timeout_seconds,
                output_callback=(tracker.feed if self.progress.enabled() else None),
            )
            log = "\n".join(part for part in (run.stdout, run.stderr) if part)
            self.engineering.workspace.write_log(
                f"foamRun.attempt-{attempt_number:03d}.log", log
            )
            result = parse_runtime_log(log, return_code=run.return_code)
            state.simulation = result
            state.simulation_attempts = attempt_number
            attempt = SimulationAttempt(attempt=attempt_number, result=result)
            attempts.append(attempt)

            diagnostic = None if result.success else diagnose_openfoam_failure(
                run, command_name="foamRun"
            )
            diagnostic_text = diagnostic.render() if diagnostic is not None else ""
            self.progress.emit(
                ProgressEvent(
                    phase="runtime",
                    message=(
                        f"foamRun attempt {attempt_number} 완료"
                        if result.success
                        else f"foamRun attempt {attempt_number} 실패; native diagnostic captured; repair 판단으로 이동"
                    ),
                    status="success" if result.success else "failure",
                    metrics={
                        "lastTime": result.last_time,
                        "maxCo": result.courant_max,
                        "returnCode": result.return_code,
                    },
                    details=(
                        tuple(
                            self.engineering.redact_native_observation(line)[:800]
                            for line in diagnostic_text.splitlines()
                            if line.strip()
                        )[:24]
                        if diagnostic_text
                        else ()
                    ),
                )
            )

            if result.success:
                state.runtime_report = RuntimeReport(
                    success=True,
                    attempts=attempts,
                    final_result=result,
                )
                state.transition(
                    State.EXECUTION_DONE,
                    "foamRun completed with finite, fatal-error-free execution evidence. "
                    "Result review remains required.",
                )
                return state

            if attempt_number >= self.policy.max_attempts:
                state.runtime_report = RuntimeReport(
                    success=False,
                    attempts=attempts,
                    final_result=result,
                )
                state.transition(
                    State.ENGINEERING_BLOCKED,
                    "Runtime retry budget exhausted; latest OpenFOAM log is preserved for review.",
                )
                return state

            self.progress.emit(
                ProgressEvent(
                    phase="runtime-repair",
                    message=f"실패 로그를 CFDEngineeringAgent에 반환: attempt={attempt_number}",
                    status="start",
                )
            )
            outcome = self.engineering.repair_runtime(
                state,
                runtime_log=(diagnostic_text or log[-8_000:]),
                attempt=attempt_number,
                native_execution=True,
            )
            attempt.repair_requested = outcome.retry
            if not outcome.retry:
                state.runtime_report = RuntimeReport(
                    success=False,
                    attempts=attempts,
                    final_result=result,
                )
                return state

        raise RuntimeError("Bounded runtime loop exited unexpectedly.")
