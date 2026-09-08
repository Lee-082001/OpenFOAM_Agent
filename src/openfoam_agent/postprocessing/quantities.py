"""Bounded scalar-table analysis with explicit provenance and declared semantics.

This verifies arithmetic and source bytes, not the CFD meaning of a column. Units,
region and selection remain declared unless separately checked against native output.
"""
from __future__ import annotations
import bisect
import csv
import hashlib
import math
from pathlib import Path
from openfoam_agent.contracts.models import QuantityOfInterest


def clean_series(rows, *, time_column=0, value_column=1):
    retained = []
    duplicates = restarts = 0
    for row in rows:
        if max(time_column, value_column) >= len(row):
            raise ValueError("Requested scalar column is missing.")
        t, y = float(row[time_column]), float(row[value_column])
        if not math.isfinite(t) or not math.isfinite(y):
            raise ValueError("Non-finite scalar sample; refusing to silently discard it.")
        if retained and t <= retained[-1][0]:
            if t == retained[-1][0]:
                duplicates += 1
            else:
                restarts += 1
            # A restarted trajectory replaces ALL old samples at/after its restart.
            while retained and retained[-1][0] >= t:
                retained.pop()
        retained.append((t, y))
    return retained, {"duplicate_times": duplicates, "restart_segments": restarts}


def scalar_rows(path: Path, *, max_bytes=16_000_000, max_rows=500_000):
    if path.stat().st_size > max_bytes:
        raise ValueError("Scalar source exceeds the bounded analysis size limit.")
    digest = hashlib.sha256()
    rows = []
    with path.open("rb") as stream:
        for raw in stream:
            digest.update(raw)
            if len(raw) > 65536:
                raise ValueError("Scalar source line is too long.")
            text = raw.decode("utf-8").strip()
            if not text or text.startswith(("#", "//")):
                continue
            tokens = next(csv.reader([text])) if "," in text else text.split()
            try:
                values = [float(token.strip()) for token in tokens]
            except ValueError:
                if not rows and any(token.casefold() in {"time", "t"} for token in tokens):
                    continue
                raise ValueError("Only scalar CSV/whitespace tables are supported; malformed/vector data rejected.") from None
            if not all(math.isfinite(v) for v in values):
                raise ValueError("Source contains non-finite values.")
            rows.append(values)
            if len(rows) > max_rows:
                raise ValueError("Scalar sample count exceeds the analysis limit.")
    return rows, digest.hexdigest()


def interpolate(series, time):
    index = bisect.bisect_left(series, time, key=lambda row: row[0])
    if index < len(series) and series[index][0] == time:
        return series[index][1]
    if index == 0 or index == len(series):
        raise ValueError("Requested time lies outside observed data; extrapolation is forbidden.")
    t0, y0 = series[index-1]
    t1, y1 = series[index]
    return y0 + (y1-y0) * (time-t0) / (t1-t0)


def window(series, start=None, end=None):
    if not series:
        raise ValueError("No scalar samples.")
    start = series[0][0] if start is None else start
    end = series[-1][0] if end is None else end
    if not math.isfinite(start) or not math.isfinite(end) or start > end:
        raise ValueError("Invalid analysis time interval.")
    first, last = interpolate(series, start), interpolate(series, end)
    return [(start, first), *[(t,y) for t,y in series if start < t < end], *(([(end,last)]) if end > start else [])]


def time_statistics(series):
    if len(series) < 2 or series[-1][0] <= series[0][0]:
        raise ValueError("Time statistics require a positive observed interval.")
    duration = series[-1][0]-series[0][0]
    integral = square = 0.0
    deltas = []
    for (t0,y0),(t1,y1) in zip(series, series[1:]):
        dt = t1-t0
        if dt <= 0:
            raise ValueError("Time samples must be strictly increasing after restart reconciliation.")
        deltas.append(dt)
        integral += dt * (y0+y1)/2
        # Exact integral of the square of a piecewise-linear interpolant.
        square += dt * (y0*y0+y0*y1+y1*y1)/3
    mean = integral/duration
    metrics = {"duration": duration, "integral": integral, "time_mean": mean,
        "rms": math.sqrt(max(0.0,square/duration)),
        "fluctuation_rms": math.sqrt(max(0.0,square/duration-mean*mean)),
        "uniform_time_spacing": all(math.isclose(dt,deltas[0],rel_tol=1e-6,abs_tol=1e-12) for dt in deltas)}
    if any(not math.isfinite(v) for v in metrics.values() if isinstance(v, (int,float))):
        raise ValueError("Scalar integration overflowed; no finite result.")
    return metrics


def analyze_quantity(workspace, spec: QuantityOfInterest):
    if spec.native_field is not None:
        return analyze_native_quantity(workspace,spec)
    source = workspace.resolve_result_path(spec.source_path, must_exist=True)
    rows, digest = scalar_rows(source)
    series, cleanup = clean_series(rows, time_column=spec.time_column, value_column=spec.value_column)
    sources = [{"path": spec.source_path, "sha256": digest, "size_bytes": source.stat().st_size}]
    if len(series) < spec.minimum_samples:
        raise ValueError("Insufficient retained samples for the declared quantity.")
    if spec.operation in {"difference", "balance"}:
        if spec.other_source_path is None:
            raise ValueError("Difference/balance requires a second scalar source with the same declared units.")
        other_path = workspace.resolve_result_path(spec.other_source_path, must_exist=True)
        other_rows, other_digest = scalar_rows(other_path)
        other, other_cleanup = clean_series(other_rows, time_column=spec.time_column, value_column=spec.other_value_column)
        if len(other) < spec.minimum_samples:
            raise ValueError("Insufficient second-source samples.")
        start = spec.start_time if spec.start_time is not None else max(series[0][0],other[0][0])
        end = spec.end_time if spec.end_time is not None else min(series[-1][0],other[-1][0])
        a, b = window(series,start,end), window(other,start,end)
        times = sorted({t for t,y in a} | {t for t,y in b})
        series = [(t, interpolate(a,t)-interpolate(b,t)) for t in times]
        sources.append({"path": spec.other_source_path, "sha256": other_digest, "size_bytes": other_path.stat().st_size})
        cleanup["other_source"] = other_cleanup
    else:
        series = window(series, spec.start_time, spec.end_time)
    if len(series) < spec.minimum_samples:
        raise ValueError("Declared time window has insufficient observed samples.")
    stats = time_statistics(series) if len(series) > 1 else {}
    op = spec.operation
    value = (min(y for t,y in series) if op == "minimum" else
             max(y for t,y in series) if op == "maximum" else
             series[-1][1] if op == "last" else
             stats["time_mean"] if op in {"difference", "balance"} else stats[op])
    bounds = ((spec.expected_min is None or value >= spec.expected_min) and
              (spec.expected_max is None or value <= spec.expected_max))
    return {"id": spec.id, "quantity_contract": spec.model_dump(mode="json"), "quantity": spec.quantity, "region": spec.region, "selection": spec.selection,
        "operation": op, "value": value, "unit": spec.unit+"*s" if op == "integral" else spec.unit,
        "samples_used": len(series), "start_time": series[0][0], "end_time": series[-1][0],
        "sources": sources, "statistics": stats, "cleanup": cleanup, "arithmetic_verified": True,
        "declared_bounds_satisfied": bounds, "physical_semantics_verified": False,
        "limitations": ["Scalar columns, units, region and selection are declared by the approved plan; their physical meaning is not automatically verified.",
                        "Time integration uses piecewise-linear interpolation; no extrapolation."]}


def analyze_native_quantity(workspace,spec):
    from .native_fields import reduce_field
    series,sources,semantic=reduce_field(workspace,spec)
    series=window(series,spec.start_time,spec.end_time)
    if len(series)<spec.minimum_samples:raise ValueError("Insufficient physical observation samples.")
    stats=time_statistics(series) if len(series)>1 else {}
    op=spec.operation
    value=(min(y for t,y in series) if op=="minimum" else max(y for t,y in series) if op=="maximum" else series[-1][1] if op=="last" else stats[op])
    return {"id":spec.id,"quantity_contract":spec.model_dump(mode="json"),"quantity":spec.quantity,"region":spec.region,"selection":spec.selection,"operation":op,"value":value,
        "unit":spec.unit+"*s" if op=="integral" else spec.unit,"sources":sources,"statistics":stats,"samples_used":len(series),
        "start_time":series[0][0],"end_time":series[-1][0],"arithmetic_verified":True,"physical_semantics_verified":True,
        "semantics":semantic,"declared_bounds_satisfied":(spec.expected_min is None or value>=spec.expected_min) and (spec.expected_max is None or value<=spec.expected_max),
        "limitations":["Dimensions, region/patch, geometric reduction and saved values are verified; solver physics and quantity naming are not independently certified."]}
