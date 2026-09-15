from __future__ import annotations

import json

from openfoam_agent.qualification.reliability import (
    BenchmarkManifest,
    BenchmarkTrialSpec,
    aggregate_with_manifest,
    evaluate_report,
    evaluate_results_directory,
)


def _report(state, *, requirement_failure=False, repair=False, tokens=1000, native=4, avoided=0):
    events = []
    if requirement_failure:
        events.append({
            "success": False,
            "action_type": "finish_preview",
            "summary": "Engineering plan rejected by deterministic safety/evidence gate.",
            "output_excerpt": "Numeric semantic assertion for operating.reynolds_number recomputes to 500, not target 1000.",
            "failure_category": None,
        })
    if repair:
        events.append({
            "success": True,
            "action_type": "write_case_file",
            "sequence_id": "runtime-repair:repair-plan:0001",
            "summary": "repair",
            "output_excerpt": "",
        })
    return {
        "final_state": state,
        "engineering_events": events,
        "runtime_report": {"success": state == "RESULT_REVIEW_REQUIRED", "attempts": [{}, {}] if repair else [{}]},
        "budget": {"native_commands_total": native},
        "reliability_observables": {
            "engineering_llm_usage": {"totalTokens": tokens},
            "runtime_repair_attempts": 1 if repair else 0,
            "mesh_validation_scopes_avoided": avoided,
        },
    }


def test_v520_reliability_metrics_distinguish_false_acceptance_and_repair_success():
    reject = BenchmarkTrialSpec(
        trial_id="bad_re", category="requirement", prompt="Re=1000",
        expected_outcome="reject", expected_requirement_violation_detection=True,
    )
    repair = BenchmarkTrialSpec(
        trial_id="repair", category="runtime", prompt="heat",
        expected_outcome="repair_then_accept",
    )
    ev1 = evaluate_report(reject, "ours", _report("ENGINEERING_BLOCKED", requirement_failure=True))
    ev2 = evaluate_report(repair, "ours", _report("RESULT_REVIEW_REQUIRED", repair=True, avoided=2))
    assert ev1.requirement_violation_detected
    assert not ev1.false_acceptance
    assert ev2.repair_success is True
    assert ev2.mesh_validation_scopes_avoided == 2

    manifest = BenchmarkManifest(release="5.2.0", variants=["ours"], trials=[reject, repair])
    summary = aggregate_with_manifest(manifest, [ev1, ev2])["ours"]
    assert summary["false_acceptance_rate"] == 0.0
    assert summary["repair_success_rate"] == 1.0
    assert summary["requirement_violation_detection_rate"] == 1.0


def test_v520_benchmark_directory_keeps_missing_trials_explicit(tmp_path):
    spec = BenchmarkTrialSpec(
        trial_id="case1", category="baseline", prompt="pipe", expected_outcome="accept"
    )
    manifest = BenchmarkManifest(release="5.2.0", variants=["ours", "plain_llm"], trials=[spec])
    (tmp_path / "case1__ours.json").write_text(json.dumps(_report("SOLVE_READY")), encoding="utf-8")
    result = evaluate_results_directory(manifest, tmp_path)
    assert result["observed_reports"] == 1
    assert result["expected_reports"] == 2
    assert result["missing_reports"] == ["case1__plain_llm.json"]
    assert result["complete"] is False
