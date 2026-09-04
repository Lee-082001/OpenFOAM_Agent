from __future__ import annotations

import hashlib
import math
import re

from openfoam_agent.schemas.simulation import ResidualSample, SimulationResult


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


def parse_runtime_log(
    text: str, *, return_code: int, runtime_driver: str = "foamRun"
) -> SimulationResult:
    current_time = 0.0
    times: list[float] = []
    residuals: list[ResidualSample] = []
    courant: list[float] = []
    continuity: list[float] = []
    end_marker = False
    invalid_metric = False

    lines = text.splitlines()
    for line in lines:
        stripped = line.strip()
        tm = TIME_RE.fullmatch(stripped)
        if tm:
            value = _finite_float(tm.group("time"))
            if value is None:
                invalid_metric = True
            else:
                current_time = value
                times.append(value)
        rm = RESIDUAL_RE.search(line)
        if rm:
            initial = _finite_float(rm.group("initial"))
            final = _finite_float(rm.group("final"))
            if initial is None or final is None or initial < 0 or final < 0:
                invalid_metric = True
            else:
                residuals.append(
                    ResidualSample(
                        time=current_time,
                        field=rm.group("field").strip(),
                        initial_residual=initial,
                        final_residual=final,
                    )
                )
        cm = COURANT_RE.search(line)
        if cm:
            value = _finite_float(cm.group("max"))
            if value is None:
                invalid_metric = True
            else:
                courant.append(value)
        cont = CONTINUITY_RE.search(line)
        if cont:
            value = _finite_float(cont.group("cumulative"))
            if value is None:
                invalid_metric = True
            else:
                continuity.append(value)
        if END_RE.fullmatch(stripped):
            end_marker = True

    fatal = _extract_fatal(lines)
    filtered = "\n".join(
        line
        for line in lines
        if "enabling floating point exception trapping" not in line.casefold()
    )
    non_finite = NON_FINITE_RE.search(filtered) is not None
    failures: list[str] = []
    if return_code != 0:
        failures.append(f"{runtime_driver} returned non-zero status {return_code}.")
    if not end_marker:
        failures.append("OpenFOAM End marker is missing.")
    if not times:
        failures.append("Runtime log contains no Time progress evidence.")
    if fatal:
        failures.append("OpenFOAM reported a fatal error.")
    if non_finite:
        failures.append("Runtime log contains non-finite/SIGFPE evidence.")
    if invalid_metric:
        failures.append("Runtime log contains malformed/non-finite numeric evidence.")

    success = not failures
    return SimulationResult(
        success=success,
        completed=success,
        return_code=return_code,
        last_time=max(times) if times else None,
        residuals=residuals,
        courant_max=max(courant) if courant else None,
        continuity_error=max((abs(value) for value in continuity), default=None),
        fatal_error=fatal,
        non_finite_detected=non_finite,
        end_marker_found=end_marker,
        log_sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
        evidence_failures=failures,
    )


def _finite_float(value: str) -> float | None:
    try:
        parsed = float(value)
    except ValueError:
        return None
    return parsed if math.isfinite(parsed) else None


def _extract_fatal(lines: list[str]) -> str | None:
    for index, line in enumerate(lines):
        if FATAL_RE.search(line):
            return "\n".join(lines[index:index + 8])[:2000]
    return None
