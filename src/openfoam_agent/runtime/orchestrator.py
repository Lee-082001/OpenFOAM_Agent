from __future__ import annotations

from pathlib import Path
from contextlib import nullcontext
from openfoam_agent.tools.safe_runner import SafeRunner
from openfoam_agent.tools.execution_policy import ExecutionContext, ExecutionPolicyError
from openfoam_agent.tools.parsers import parse_runtime_stream
from openfoam_agent.runtime.completion import completion_contract, verify_result_outputs, snapshot_result_outputs
from openfoam_agent.runtime.parallel import prepare_parallel, verify_parallel_inputs, reconstruct_parallel

from openfoam_agent.engineering import CFDEngineeringAgent
from openfoam_agent.progress import (
    NullProgressReporter,
    ProgressEvent,
    ProgressReporter,
    SolverProgressTracker,
)
from openfoam_agent.schemas.simulation import (
    RuntimePolicy,
    RuntimeRepairDecision,
    RuntimeReport,
    SimulationAttempt,
)
from openfoam_agent.tools.diagnostics import diagnose_openfoam_failure, classify_native_validation
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
        self.engineering.bind_checkpoint(state)
        self.engineering.checkpoint(state, "runtime-entry")
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

        if state.execution_approval is None:
            state.transition(State.FAILED, "Durable execution approval is missing; a boolean is not authority.")
            return state
        try:
            state.assert_confirmed_intake()
            state.execution_approval.check_plan(state.engineering_plan)
            state.execution_approval.check_repair_files(state.case_seal)
            contract = completion_contract(state.engineering_plan, self.engineering.workspace)
        except (ValueError, OSError) as exc:
            state.transition(State.ENGINEERING_REVIEW_REQUIRED, f"Runtime contract requires review: {exc}")
            return state
        attempts: list[SimulationAttempt] = []
        parallel_manifest = None
        execution = state.engineering_plan.execution
        runtime_driver = execution.driver if execution is not None else "foamRun"
        target_summary = (
            ",".join(f"{item.region}:{item.solver_module}" for item in execution.regions)
            if execution is not None and execution.regions
            else state.engineering_plan.solver
        )
        self.progress.emit(
            ProgressEvent(
                phase="runtime",
                message=f"{runtime_driver} 시작: target={target_summary}",
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
                    message=f"{runtime_driver} attempt {attempt_number}/{self.policy.max_attempts}",
                    status="start",
                )
            )
            tracker = SolverProgressTracker(
                self.progress,
                attempt=attempt_number,
                attempt_limit=self.policy.max_attempts,
                runtime_label=runtime_driver,
            )
            state.pending_action = {"status": "intent", "kind": "runtime", "attempt": attempt_number}
            self.engineering.checkpoint(state, "solver-start-intent")
            runner = getattr(self.tools, "runner", None)
            context = ExecutionContext(state.execution_approval, plan, state.case_seal, self.engineering.workspace)
            scope = runner.approved_execution(context) if isinstance(runner, SafeRunner) else nullcontext()
            try:
                state.execution_approval.check_plan(plan)
                state.execution_approval.check_repair_files(state.case_seal)
                before_outputs = snapshot_result_outputs(self.engineering.workspace, contract)
                with scope:
                    if plan.execution is not None and plan.execution.parallel.mode == "local_mpi":
                        if parallel_manifest is None:
                            parallel_manifest = prepare_parallel(self.tools, self.engineering.workspace, plan)
                            state.parallel_evidence = parallel_manifest
                        verify_parallel_inputs(self.engineering.workspace, plan, parallel_manifest)
                    if plan.execution is not None:
                        run = self.tools.run_execution(
                            state.case_dir, plan.execution, stream_output=self.stream_output,
                            timeout=self.policy.solver_timeout_seconds,
                            output_callback=(tracker.feed if self.progress.enabled() else None),
                        )
                    else:
                        run = self.tools.foam_run(
                            state.case_dir, solver=plan.solver, stream_output=self.stream_output,
                            timeout=self.policy.solver_timeout_seconds,
                            output_callback=(tracker.feed if self.progress.enabled() else None),
                        )
            except (ValueError, OSError, ExecutionPolicyError) as exc:
                state.transition(State.ENGINEERING_BLOCKED, f"Execution policy blocked process creation/retry: {exc}")
                return state
            log = "\n".join(part for part in (run.stdout, run.stderr) if part)
            self.engineering.workspace.write_log(
                f"{runtime_driver}.attempt-{attempt_number:03d}.log", log
            )
            parse_options = dict(return_code=run.return_code, runtime_driver=runtime_driver,
                                 contract=contract, termination_reason=run.termination_reason)
            if run.log_path:
                # Read the complete disk stream, not the bounded UI tail.
                with Path(run.log_path).open("rb") as stream:
                    result = parse_runtime_stream(stream, **parse_options)
                if run.log_sha256 and result.log_sha256 != run.log_sha256:
                    result.success = result.completed = False
                    result.evidence_failures.append("Native disk log hash differs from the captured stream.")
            else:
                result = parse_runtime_log(log, **parse_options)
            if result.success:
                try:
                    if plan.execution is not None and plan.execution.parallel.mode == "local_mpi":
                        scope = runner.approved_execution(context) if isinstance(runner, SafeRunner) else nullcontext()
                        with scope:
                            reconstruct_parallel(self.tools, self.engineering.workspace, plan, result.last_time)
                    verified, failures, output_records = verify_result_outputs(
                        self.engineering.workspace, plan, contract, result.last_time, before=before_outputs)
                    state.result_output_evidence = output_records
                except (ValueError, OSError, ExecutionPolicyError) as exc:
                    verified, failures = False, [str(exc)]
                result.outputs_verified = verified
                if not verified:
                    result.success = False
                    result.evidence_failures.extend(failures)
            if isinstance(runner, SafeRunner):
                state.native_process_records = list(runner.budget.records)
            state.simulation = result
            state.simulation_attempts = attempt_number
            state.pending_action = {"status": "completed", "kind": "runtime", "attempt": attempt_number}
            self.engineering.checkpoint(state, "solver-result-captured")
            state.pending_action = None
            attempt = SimulationAttempt(attempt=attempt_number, result=result)
            attempts.append(attempt)

            assessment = None if result.success else classify_native_validation(
                run, command_name=runtime_driver, probe=False
            )
            diagnostic = None if assessment is None else assessment.diagnostic
            diagnostic_text = diagnostic.render() if diagnostic is not None else ""
            if result.success:
                runtime_message = f"{runtime_driver} attempt {attempt_number} 완료"
                runtime_status = "success"
            elif result.process_success:
                runtime_message = (
                    f"{runtime_driver} attempt {attempt_number} completed but result/completion evidence is incomplete"
                )
                runtime_status = "failure"
            elif assessment is not None and assessment.status == "inconclusive":
                runtime_message = (
                    f"{runtime_driver} attempt {attempt_number} infrastructure/tool result is inconclusive; "
                    "CFD repair is not invoked"
                )
                runtime_status = "failure"
            else:
                runtime_message = (
                    f"{runtime_driver} attempt {attempt_number} 실패; native diagnostic captured; case-level failure로 분류되어 repair 판단으로 이동"
                )
                runtime_status = "failure"
            self.progress.emit(
                ProgressEvent(
                    phase="runtime",
                    message=runtime_message,
                    status=runtime_status,
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
                    f"{runtime_driver} met its termination contract and bounded output checks. "
                    "Numerical accuracy and physical goal achievement remain unverified pending result review.",
                )
                return state

            if result.process_success:
                state.runtime_report = RuntimeReport(success=False, attempts=attempts, final_result=result)
                state.solve_approved = False
                state.transition(State.RESULT_REVIEW_REQUIRED,
                    "Process exited normally but requested completion/output evidence is incomplete; user review required.")
                return state

            if assessment is not None and assessment.status == "inconclusive":
                event = self.engineering._event(
                    attempt_number,
                    "runtime_execution",
                    False,
                    assessment.reason,
                    diagnostic_text or log[-4000:],
                    native_command_executed=True,
                    validation_status="inconclusive",
                    failure_category=assessment.category or "infra",
                )
                state.engineering_events.append(event)
                self.engineering._record_unresolved_failure(state, event)
                state.runtime_report = RuntimeReport(success=False, attempts=attempts, final_result=result)
                state.solve_approved = False
                state.transition(
                    State.ENGINEERING_BLOCKED,
                    "Runtime execution was inconclusive because of tool/infrastructure failure; "
                    "the CFD case was not rewritten and the native log was preserved.",
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
            if outcome.decision != RuntimeRepairDecision.RETRY_SOLVER:
                state.runtime_report = RuntimeReport(
                    success=False,
                    attempts=attempts,
                    final_result=result,
                )
                if state.current_state == State.RUNTIME_REPAIR:
                    # Defensive state invariant: the transient repair state is internal
                    # and must never escape to the top-level workflow.
                    state.solve_approved = False
                    state.transition(
                        State.ENGINEERING_BLOCKED,
                        "Runtime repair returned without an explicit terminal/retry state; "
                        f"decision={outcome.decision.value}. {outcome.reason}",
                    )
                return state

            if state.current_state != State.SIMULATION:
                state.runtime_report = RuntimeReport(
                    success=False,
                    attempts=attempts,
                    final_result=result,
                )
                state.solve_approved = False
                state.transition(
                    State.ENGINEERING_BLOCKED,
                    "Runtime repair requested RETRY_SOLVER without restoring SIMULATION state.",
                )
                return state

        raise RuntimeError("Bounded runtime loop exited unexpectedly.")
