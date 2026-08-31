from __future__ import annotations

import hashlib
import math
import re
import statistics

from openfoam_agent.schemas.postprocessing import ForceCoefficientAnalysis


_NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
_MAG_U_INF = re.compile(rf"(?m)^\s*magUInf\s+(?P<value>{_NUMBER})\s*;")
_L_REF = re.compile(rf"(?m)^\s*lRef\s+(?P<value>{_NUMBER})\s*;")


def analyze_force_coefficients(
    coefficient_text: str,
    dictionary_text: str,
    *,
    source_path: str,
    dictionary_path: str,
    discard_fraction: float = 0.25,
) -> ForceCoefficientAnalysis:
    if not 0.0 <= discard_fraction < 0.9:
        raise ValueError("discard_fraction must be in [0, 0.9).")

    header: list[str] | None = None
    rows: list[list[float]] = []
    for raw in coefficient_text.splitlines():
        stripped = raw.strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            tokens = stripped.lstrip("#").strip().split()
            if "Time" in tokens and "Cd" in tokens and "Cl" in tokens:
                header = tokens
            continue
        if header is None:
            continue
        tokens = stripped.split()
        if len(tokens) < len(header):
            continue
        try:
            values = [float(token) for token in tokens[: len(header)]]
        except ValueError:
            continue
        if all(math.isfinite(value) for value in values):
            rows.append(values)

    if header is None:
        raise ValueError("forceCoeffs data does not contain a '# Time ... Cd ... Cl ...' header.")
    if not rows:
        raise ValueError("forceCoeffs data contains no finite numeric samples.")

    time_i = header.index("Time")
    cd_i = header.index("Cd")
    cl_i = header.index("Cl")
    rows.sort(key=lambda row: row[time_i])

    samples_total = len(rows)
    start = min(samples_total - 1, int(samples_total * discard_fraction))
    used = rows[start:]
    times = [row[time_i] for row in used]
    cds = [row[cd_i] for row in used]
    cls = [row[cl_i] for row in used]

    limitations: list[str] = []
    mean_cd = statistics.fmean(cds) if cds else None
    mean_cl = statistics.fmean(cls) if cls else None
    rms_cl = None
    if cls and mean_cl is not None:
        rms_cl = math.sqrt(statistics.fmean((value - mean_cl) ** 2 for value in cls))

    frequency = None
    periods_observed = 0
    period_cv = None
    if len(times) >= 5 and mean_cl is not None:
        crossings = _upward_zero_crossings(times, [value - mean_cl for value in cls])
        if len(crossings) >= 3:
            periods = [b - a for a, b in zip(crossings, crossings[1:]) if b > a]
            if periods:
                mean_period = statistics.fmean(periods)
                if mean_period > 0:
                    frequency = 1.0 / mean_period
                    periods_observed = len(periods)
                    if len(periods) >= 2:
                        period_cv = statistics.pstdev(periods) / mean_period
        else:
            limitations.append(
                "Too few complete lift-coefficient oscillations were observed after transient discard to estimate shedding frequency."
            )
    else:
        limitations.append("Too few force-coefficient samples were available for shedding-frequency estimation.")

    ref_velocity = _dictionary_scalar(_MAG_U_INF, dictionary_text)
    ref_length = _dictionary_scalar(_L_REF, dictionary_text)
    strouhal = None
    if frequency is not None and ref_velocity and ref_length:
        strouhal = frequency * ref_length / ref_velocity
    elif frequency is not None:
        limitations.append(
            "Shedding frequency was estimated, but Strouhal number was not computed because magUInf/lRef were not found in the executed post-processing dictionary."
        )

    if periods_observed < 4 and frequency is not None:
        limitations.append(
            "Frequency estimate is based on fewer than four observed lift periods and should be treated as exploratory."
        )
    if period_cv is not None and period_cv > 0.15:
        limitations.append(
            "Lift-cycle period variability is high; a single shedding frequency may not represent the retained interval well."
        )

    return ForceCoefficientAnalysis(
        source_path=source_path,
        dictionary_path=dictionary_path,
        evidence_sha256=hashlib.sha256(coefficient_text.encode("utf-8")).hexdigest(),
        dictionary_sha256=hashlib.sha256(dictionary_text.encode("utf-8")).hexdigest(),
        samples_total=samples_total,
        samples_used=len(used),
        discard_fraction=discard_fraction,
        start_time=times[0] if times else None,
        end_time=times[-1] if times else None,
        mean_cd=mean_cd,
        mean_cl=mean_cl,
        rms_cl=rms_cl,
        shedding_frequency=frequency,
        reference_velocity=ref_velocity,
        reference_length=ref_length,
        strouhal_number=strouhal,
        periods_observed=periods_observed,
        period_cv=period_cv,
        limitations=limitations,
    )


def _upward_zero_crossings(times: list[float], signal: list[float]) -> list[float]:
    crossings: list[float] = []
    for t0, t1, y0, y1 in zip(times, times[1:], signal, signal[1:]):
        if t1 <= t0:
            continue
        if y0 <= 0.0 < y1:
            denom = y1 - y0
            fraction = 0.0 if denom == 0 else -y0 / denom
            crossings.append(t0 + fraction * (t1 - t0))
    return crossings


def _dictionary_scalar(pattern: re.Pattern[str], text: str) -> float | None:
    match = pattern.search(text)
    if not match:
        return None
    try:
        value = float(match.group("value"))
    except ValueError:
        return None
    return value if math.isfinite(value) and value > 0 else None
