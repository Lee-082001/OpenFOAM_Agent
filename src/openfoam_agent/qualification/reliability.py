from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


_ACCEPTED_CASE_STATES = {
    "SOLVE_READY",
    "SIMULATION",
    "POSTPROCESSING",
    "EXECUTION_DONE",
    "RESULT_REVIEW_REQUIRED",
    "COMPLETE",
    "DONE",
}
_REQUIREMENT_FAILURE_PATTERNS = (
    "critical requirement assurance",
    "numeric semantic assertion",
    "numeric semantic evidence",
    "semantic assertion for",
    "confirmed fact binding",
    "frozen requirement",
    "protected entry",
)


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class BenchmarkTrialSpec(_Model):
    trial_id: str = Field(pattern=r"^[A-Za-z0-9_.-]+$", max_length=80)
    category: str = Field(min_length=1, max_length=120)
    prompt: str = Field(min_length=1, max_length=4000)
    expected_outcome: Literal["accept", "reject", "repair_then_accept"]
    expected_requirement_violation_detection: bool = False
    repair_success_stage: Literal["execution", "pre_solve"] = "execution"
    fault_injection: str | None = Field(default=None, max_length=1000)
    notes: str = Field(default="", max_length=1200)


class BenchmarkManifest(_Model):
    schema_version: Literal["1.0"] = "1.0"
    release: str = Field(min_length=1, max_length=40)
    status: Literal["NOT_RUN", "PARTIAL", "COMPLETE"] = "NOT_RUN"
    variants: list[str] = Field(min_length=1, max_length=8)
    result_file_template: str = "{trial_id}__{variant}.json"
    trials: list[BenchmarkTrialSpec] = Field(min_length=1, max_length=200)

    @model_validator(mode="after")
    def unique_ids(self):
        ids = [item.trial_id for item in self.trials]
        if len(ids) != len(set(ids)):
            raise ValueError("Benchmark trial IDs must be unique.")
        if len(self.variants) != len(set(self.variants)):
            raise ValueError("Benchmark variants must be unique.")
        if "{trial_id}" not in self.result_file_template or "{variant}" not in self.result_file_template:
            raise ValueError("result_file_template must contain {trial_id} and {variant}.")
        return self


class TrialEvaluation(_Model):
    trial_id: str
    variant: str
    expected_outcome: Literal["accept", "reject", "repair_then_accept"]
    final_state: str
    case_accepted: bool
    execution_success: bool
    requirement_violation_detected: bool
    repair_attempted: bool
    false_acceptance: bool
    unnecessary_blocking: bool
    repair_success: bool | None
    engineering_total_tokens: int | None = None
    native_commands_total: int | None = None
    mesh_validation_scopes_avoided: int = 0


def _flatten_failure_text(report: dict[str, Any]) -> str:
    chunks: list[str] = []
    for key in ("message", "primary_failure"):
        value = report.get(key)
        if value is not None:
            chunks.append(json.dumps(value, ensure_ascii=True, sort_keys=True, default=str))
    for event in report.get("engineering_events", []) or []:
        if not isinstance(event, dict):
            continue
        if event.get("failure_category") == "user_contract":
            chunks.append("user_contract")
        if not event.get("success", True):
            chunks.append(str(event.get("summary", "")))
            chunks.append(str(event.get("output_excerpt", "")))
    return "\n".join(chunks).casefold()


def _requirement_violation_detected(report: dict[str, Any]) -> bool:
    text = _flatten_failure_text(report)
    if "user_contract" in text:
        return True
    return any(pattern in text for pattern in _REQUIREMENT_FAILURE_PATTERNS)


def _repair_attempted(report: dict[str, Any]) -> bool:
    runtime = report.get("runtime_report") or {}
    attempts = runtime.get("attempts") if isinstance(runtime, dict) else None
    if isinstance(attempts, list) and len(attempts) > 1:
        return True
    attempts_count = (report.get("reliability_observables") or {}).get("runtime_repair_attempts", 0)
    if isinstance(attempts_count, int) and not isinstance(attempts_count, bool) and attempts_count > 0:
        return True
    for event in report.get("engineering_events", []) or []:
        if not isinstance(event, dict) or event.get("success") is not True:
            continue
        sequence = str(event.get("sequence_id") or "")
        if sequence.startswith(("runtime-repair:", "repair-plan:")):
            return True

    return False


def _engineering_tokens(report: dict[str, Any]) -> int | None:
    observables = report.get("reliability_observables") or {}
    usage = observables.get("engineering_llm_usage") if isinstance(observables, dict) else None
    if isinstance(usage, dict) and isinstance(usage.get("totalTokens"), (int, float)):
        return int(usage["totalTokens"])
    records = report.get("engineering_llm_usage_records") or []
    values = [
        int(item.get("totalTokens", 0) or 0)
        for item in records
        if isinstance(item, dict) and isinstance(item.get("totalTokens", 0), (int, float))
    ]
    return sum(values) if values else None


def evaluate_report(
    spec: BenchmarkTrialSpec,
    variant: str,
    report: dict[str, Any],
) -> TrialEvaluation:
    final_state = str(report.get("final_state") or "UNKNOWN")
    accepted = final_state in _ACCEPTED_CASE_STATES
    runtime = report.get("runtime_report") or {}
    execution_success = runtime.get("success") is True if isinstance(runtime, dict) else False
    requirement_detected = _requirement_violation_detected(report)
    repair_attempted = _repair_attempted(report)

    expected_accept = spec.expected_outcome in {"accept", "repair_then_accept"}
    false_acceptance = spec.expected_outcome == "reject" and accepted
    unnecessary_blocking = expected_accept and not accepted
    repair_success = None
    if spec.expected_outcome == "repair_then_accept":
        stage_success = execution_success if spec.repair_success_stage == "execution" else accepted
        repair_success = bool(repair_attempted and accepted and stage_success)

    budget = report.get("budget") or {}
    native_commands = None
    if isinstance(budget, dict):
        measured = budget.get("actual_process_spawn_attempts")
        legacy = budget.get("native_commands_total")
        if isinstance(measured, (int, float)):
            native_commands = measured
        elif isinstance(legacy, (int, float)):
            native_commands = legacy
    observables = report.get("reliability_observables") or {}
    avoided = observables.get("mesh_validation_scopes_avoided", 0) if isinstance(observables, dict) else 0
    if not isinstance(avoided, (int, float)):
        avoided = 0

    return TrialEvaluation(
        trial_id=spec.trial_id,
        variant=variant,
        expected_outcome=spec.expected_outcome,
        final_state=final_state,
        case_accepted=accepted,
        execution_success=execution_success,
        requirement_violation_detected=requirement_detected,
        repair_attempted=repair_attempted,
        false_acceptance=false_acceptance,
        unnecessary_blocking=unnecessary_blocking,
        repair_success=repair_success,
        engineering_total_tokens=_engineering_tokens(report),
        native_commands_total=(int(native_commands) if native_commands is not None else None),
        mesh_validation_scopes_avoided=int(avoided),
    )


def _rate(numerator: int, denominator: int) -> float | None:
    return round(numerator / denominator, 6) if denominator else None


def _mean(values: list[int]) -> float | None:
    return round(sum(values) / len(values), 3) if values else None


def aggregate_reliability(evaluations: list[TrialEvaluation]) -> dict[str, Any]:
    by_variant: dict[str, list[TrialEvaluation]] = {}
    for item in evaluations:
        by_variant.setdefault(item.variant, []).append(item)

    summary: dict[str, Any] = {}
    for variant, rows in sorted(by_variant.items()):
        reject_rows = [item for item in rows if item.expected_outcome == "reject"]
        accept_rows = [item for item in rows if item.expected_outcome in {"accept", "repair_then_accept"}]
        repair_rows = [item for item in rows if item.expected_outcome == "repair_then_accept"]
        req_rows = [
            item for item in rows
            if item.trial_id and item.requirement_violation_detected
        ]
        # The requirement-detection denominator must come from the manifest expectation,
        # so callers should use aggregate_with_manifest when they need that metric. The
        # plain aggregate still exposes observed counts without inventing ground truth.
        token_values = [item.engineering_total_tokens for item in rows if item.engineering_total_tokens is not None]
        native_values = [item.native_commands_total for item in rows if item.native_commands_total is not None]
        summary[variant] = {
            "attempted": len(rows),
            "case_acceptance_rate": _rate(sum(item.case_accepted for item in rows), len(rows)),
            "execution_success_rate": _rate(sum(item.execution_success for item in accept_rows), len(accept_rows)),
            "false_acceptance_rate": _rate(sum(item.false_acceptance for item in reject_rows), len(reject_rows)),
            "unnecessary_blocking_rate": _rate(sum(item.unnecessary_blocking for item in accept_rows), len(accept_rows)),
            "repair_success_rate": _rate(sum(bool(item.repair_success) for item in repair_rows), len(repair_rows)),
            "requirement_violation_detections_observed": len(req_rows),
            "average_engineering_total_tokens": _mean([int(v) for v in token_values]),
            "average_native_commands": _mean([int(v) for v in native_values]),
            "mesh_validation_scopes_avoided": sum(item.mesh_validation_scopes_avoided for item in rows),
        }
    return summary


def aggregate_with_manifest(
    manifest: BenchmarkManifest,
    evaluations: list[TrialEvaluation],
) -> dict[str, Any]:
    summary = aggregate_reliability(evaluations)
    spec_by_id = {item.trial_id: item for item in manifest.trials}
    by_variant: dict[str, list[TrialEvaluation]] = {}
    for item in evaluations:
        by_variant.setdefault(item.variant, []).append(item)
    for variant, rows in by_variant.items():
        expected = [
            item for item in rows
            if spec_by_id[item.trial_id].expected_requirement_violation_detection
        ]
        summary[variant]["requirement_violation_detection_rate"] = _rate(
            sum(item.requirement_violation_detected for item in expected), len(expected)
        )
    for variant in manifest.variants:
        rows = by_variant.get(variant, [])
        expected_accept = sum(spec.expected_outcome in {"accept", "repair_then_accept"} for spec in manifest.trials)
        expected_repair = sum(spec.expected_outcome == "repair_then_accept" for spec in manifest.trials)
        metrics = summary.setdefault(variant, {"attempted": 0})
        metrics.update({
            "expected": len(manifest.trials),
            "missing": len(manifest.trials) - len(rows),
            "coverage_rate": _rate(len(rows), len(manifest.trials)),
            "execution_success_over_expected": _rate(sum(row.execution_success for row in rows if row.expected_outcome in {"accept", "repair_then_accept"}), expected_accept),
            "repair_success_over_expected": _rate(sum(row.repair_success is True for row in rows), expected_repair),
            "observed_rate_denominator": "observed_reports_only",
        })
    return summary


def evaluate_results_directory(
    manifest: BenchmarkManifest,
    results_dir: Path,
) -> dict[str, Any]:
    evaluations: list[TrialEvaluation] = []
    missing: list[str] = []
    for spec in manifest.trials:
        for variant in manifest.variants:
            relative = manifest.result_file_template.format(trial_id=spec.trial_id, variant=variant)
            path = results_dir / relative
            if not path.is_file():
                missing.append(relative)
                continue
            report = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(report, dict):
                raise ValueError(f"Benchmark result must be a JSON object: {path}")
            evaluations.append(evaluate_report(spec, variant, report))
    return {
        "release": manifest.release,
        "manifest_status": manifest.status,
        "expected_reports": len(manifest.trials) * len(manifest.variants),
        "observed_reports": len(evaluations),
        "missing_reports": missing,
        "complete": not missing,
        "summary_by_variant": aggregate_with_manifest(manifest, evaluations),
        "trials": [item.model_dump(mode="json") for item in evaluations],
    }


def _load_manifest(path: Path) -> BenchmarkManifest:
    return BenchmarkManifest.model_validate_json(path.read_text(encoding="utf-8"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Aggregate OpenFOAM Agent reliability benchmark reports.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    result = evaluate_results_directory(_load_manifest(args.manifest), args.results_dir)
    text = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True)
    if args.output:
        args.output.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
