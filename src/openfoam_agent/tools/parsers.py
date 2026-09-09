from __future__ import annotations

import hashlib
import math
import re

from openfoam_agent.schemas.simulation import ResidualSample, SimulationResult
from openfoam_agent.contracts.models import RuntimeContract


_NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
RESIDUAL_RE = re.compile(
    rf"Solving for (?P<field>[^,]+),\s*Initial residual = (?P<initial>{_NUMBER}),\s*Final residual = (?P<final>{_NUMBER})",
    re.IGNORECASE,
)
TIME_RE = re.compile(rf"^Time\s*=\s*(?P<time>{_NUMBER})\s*(?P<unit>s)?\s*$", re.IGNORECASE)
COURANT_RE = re.compile(
    rf"Courant Number\s+mean:\s*(?P<mean>{_NUMBER})\s+max:\s*(?P<max>{_NUMBER})",
    re.IGNORECASE,
)
CONTINUITY_RE = re.compile(
    rf"time step continuity errors\s*:\s*sum local\s*=\s*(?P<local>{_NUMBER}),\s*global\s*=\s*(?P<global>{_NUMBER}),\s*cumulative\s*=\s*(?P<cumulative>{_NUMBER})",
    re.IGNORECASE,
)
END_RE = re.compile(r"^End\s*$")
FATAL_RE = re.compile(r"--> FOAM FATAL(?: IO)? ERROR:", re.IGNORECASE)
NON_FINITE_RE = re.compile(
    r"(?<![A-Za-z])(?:nan|[-+]?inf(?:inity)?)(?![A-Za-z])|floating point exception|sigfpe",
    re.IGNORECASE,
)


class RuntimeLogAccumulator:
    """Incremental, bounded log evidence. A successful exit is not CFD validity."""

    def __init__(self, contract=None, *, residual_limit=256):
        from collections import deque
        self.contract = contract
        self.residuals = deque(maxlen=residual_limit)
        self.residual_count = 0
        self.current_time = None
        self.first_time = None
        self.progress_steps = 0
        self.courant_max = None
        self.continuity_max = None
        self.end_marker = False
        self.invalid = False
        self.non_finite = False
        self.fatal_lines = []
        self.time_reversed = False
        self._last_residuals = {}
        self._consecutive = {}
        self.hash = hashlib.sha256()

    def _flush_step(self):
        if self.current_time is None or self.contract is None:
            return
        thresholds = (
            self.contract.result_acceptance.residual_thresholds
            if isinstance(self.contract, RuntimeContract)
            else self.contract.residual_thresholds
        )
        for threshold in thresholds:
            sample = self._last_residuals.get(threshold.field)
            value = sample.initial_residual if sample is not None else None
            previous = self._consecutive.get(threshold.field, 0)
            self._consecutive[threshold.field] = previous + 1 if value is not None and value <= threshold.value else 0
        self._last_residuals.clear()

    def feed(self, raw: bytes):
        self.hash.update(raw)
        if len(raw) > 65536:
            self.invalid = True
            return
        line = raw.decode("utf-8", errors="replace")
        stripped = line.strip()
        if self.fatal_lines and len(self.fatal_lines) < 8:
            self.fatal_lines.append(line[:250])
        elif not self.fatal_lines and FATAL_RE.search(line):
            self.fatal_lines.append(line[:250])
        if "enabling floating point exception trapping" not in line.casefold() and NON_FINITE_RE.search(line):
            self.non_finite = True
        tm = TIME_RE.fullmatch(stripped)
        if tm:
            value = _finite_float(tm.group("time"))
            if value is None:
                self.invalid = True
            elif self.current_time != value:
                self._flush_step()
                if self.first_time is None:
                    self.first_time = value
                start = (
                    self.contract.execution_bound.start_time
                    if isinstance(self.contract, RuntimeContract)
                    else (self.contract.start_time if self.contract else 0.0)
                )
                previous = self.current_time if self.current_time is not None else start
                if value < previous:
                    self.time_reversed = True
                if value > previous:
                    self.progress_steps += 1
                self.current_time = value
        rm = RESIDUAL_RE.search(line)
        if rm:
            initial, final = _finite_float(rm.group("initial")), _finite_float(rm.group("final"))
            if initial is None or final is None or initial < 0 or final < 0:
                self.invalid = True
            else:
                sample = ResidualSample(time=self.current_time or 0.0, field=rm.group("field").strip(),
                                        initial_residual=initial, final_residual=final)
                self.residuals.append(sample); self.residual_count += 1
                if len(self._last_residuals) >= 256 and sample.field not in self._last_residuals:
                    self.invalid = True
                else:
                    self._last_residuals[sample.field] = sample
        cm = COURANT_RE.search(line)
        if cm:
            value = _finite_float(cm.group("max"))
            if value is None or value < 0:
                self.invalid = True
            else:
                self.courant_max = max(value, self.courant_max or 0.0)
        cont = CONTINUITY_RE.search(line)
        if cont:
            value = _finite_float(cont.group("cumulative"))
            if value is None:
                self.invalid = True
            else:
                self.continuity_max = max(abs(value), self.continuity_max or 0.0)
        if END_RE.fullmatch(stripped):
            self.end_marker = True

    def finish(self, *, return_code, runtime_driver="foamRun", termination_reason="exited"):
        self._flush_step()
        failures = []
        if return_code != 0:
            failures.append(f"{runtime_driver} returned non-zero status {return_code}.")
        if termination_reason != "exited":
            failures.append(f"Process termination reason: {termination_reason}.")
        if not self.end_marker:
            failures.append("OpenFOAM End marker is missing.")
        if self.fatal_lines:
            failures.append("OpenFOAM reported a fatal error.")
        if self.non_finite:
            failures.append("Runtime log contains non-finite/SIGFPE evidence.")
        if self.invalid or self.time_reversed:
            failures.append("Runtime log contains malformed, overlong, or reversed-time evidence.")
        process_success = not failures
        progressed = self.progress_steps > 0
        if not progressed:
            failures.append("Runtime log contains no positive Time progress evidence.")
        termination = False
        acceptance_verified = False
        acceptance_warnings = []
        contract = self.contract
        if contract is None:
            failures.append("Runtime contract is missing; process exit does not prove bounded execution completion.")
        elif isinstance(contract, RuntimeContract):
            bound = contract.execution_bound
            acceptance = contract.result_acceptance
            acceptance_warnings.extend(bound.warnings)
            acceptance_warnings.extend(acceptance.warnings)
            if self.progress_steps < bound.minimum_steps:
                failures.append("Insufficient progress steps for the execution-bound contract.")
            elif bound.mode == "transient":
                if bound.end_time is None:
                    failures.append("Transient execution ended without a machine-verifiable end-time bound.")
                else:
                    tolerance = max(1e-10, abs(bound.end_time) * 1e-9)
                    termination = self.current_time is not None and self.current_time >= bound.end_time - tolerance
                    if not termination:
                        failures.append(f"Requested execution end time {bound.end_time} was not reached.")
            else:
                # For steady/custom execution the process is bounded by max-iteration
                # and/or wall-time policy. A clean OpenFOAM End after positive progress
                # proves execution completion; convergence is a separate result claim.
                termination = process_success and progressed and self.end_marker
                if not termination:
                    failures.append("Steady/custom bounded execution did not end cleanly.")

            thresholds = acceptance.residual_thresholds
            if thresholds:
                residual_subset_verified = all(
                    self._consecutive.get(x.field, 0) >= acceptance.consecutive_samples
                    for x in thresholds
                )
                acceptance_verified = residual_subset_verified and acceptance.criteria_complete
                if not residual_subset_verified:
                    acceptance_warnings.append("Configured steady residual acceptance thresholds were not satisfied by the captured consecutive samples.")
                elif not acceptance.criteria_complete:
                    acceptance_warnings.append("Machine-compiled residual criteria were satisfied, but additional result-acceptance criteria remain advisory/unimplemented.")
            elif acceptance.criteria_complete:
                acceptance_verified = True
            else:
                acceptance_verified = False
        elif self.progress_steps < contract.minimum_steps:
            failures.append("Insufficient progress steps for the completion contract.")
        elif contract.mode == "transient":
            tolerance = max(1e-10, abs(contract.end_time) * 1e-9)
            termination = self.current_time is not None and self.current_time >= contract.end_time - tolerance
            if not termination:
                failures.append(f"Requested end time {contract.end_time} was not reached.")
        elif contract.mode == "steady":
            termination = all(self._consecutive.get(x.field, 0) >= contract.consecutive_samples
                              for x in contract.residual_thresholds)
            if not termination:
                failures.append("Steady outer-iteration initial-residual convergence criterion was not met.")
            acceptance_verified = termination
        else:
            failures.append("Custom completion requires an implemented deterministic evaluator; a model claim is insufficient.")
        completed = process_success and progressed and termination and not failures
        return SimulationResult(success=completed, completed=completed, return_code=return_code,
            process_success=process_success, progressed=progressed, termination_verified=termination,
            termination_reason=termination_reason, first_time=self.first_time, last_time=self.current_time,
            progress_steps=self.progress_steps, residuals=list(self.residuals), residual_sample_count=self.residual_count,
            residuals_truncated=self.residual_count > len(self.residuals), courant_max=self.courant_max,
            continuity_error=self.continuity_max, numerical_quality_verified=acceptance_verified,
            acceptance_warnings=acceptance_warnings, fatal_error="".join(self.fatal_lines) or None,
            non_finite_detected=self.non_finite, end_marker_found=self.end_marker,
            log_sha256=self.hash.hexdigest(), evidence_failures=failures)


def parse_runtime_log(text: str, *, return_code: int, runtime_driver="foamRun", contract=None,
                      termination_reason="exited") -> SimulationResult:
    import io
    return parse_runtime_stream(io.BytesIO(text.encode("utf-8")), return_code=return_code,
            runtime_driver=runtime_driver, contract=contract, termination_reason=termination_reason)


def parse_runtime_stream(stream, *, return_code: int, runtime_driver="foamRun", contract=None,
                         termination_reason="exited") -> SimulationResult:
    accumulator = RuntimeLogAccumulator(contract)
    while True:
        raw = stream.readline(65537)
        if not raw:
            break
        accumulator.feed(raw if isinstance(raw, bytes) else raw.encode())
    return accumulator.finish(return_code=return_code, runtime_driver=runtime_driver,
                              termination_reason=termination_reason)


def _finite_float(value: str) -> float | None:
    try:
        parsed = float(value)
    except ValueError:
        return None
    return parsed if math.isfinite(parsed) else None
