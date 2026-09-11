from __future__ import annotations

import json

from openfoam_agent.engineering.repair_context import (
    REPAIR_HISTORY_ITEMS,
    REPAIR_SUPPORT_CHARS,
    REPAIR_SUPPORT_ITEMS,
    build_partitioned_validation_repair_prompt,
    diagnostic_has_explicit_alternatives,
    project_repair_episode,
    repair_episode_requires_direct_attempt,
)


def _native_failure() -> dict[str, object]:
    return {
        "action_type": "run_openfoam_command",
        "success": False,
        "summary": "foamMultiRun zero-step consumer failed",
        "output_excerpt": (
            "Unknown transport type constIso\n\n"
            "Supported transport types:\n"
            "constIsoSolid\npolynomialSolid\nexponentialSolid\nconstAnisoSolid\ntabulatedSolid\n"
        ),
        "failure_signature": "native:foamMultiRun:unknown_transport",
        "failure_category": "case",
    }


def _episode(*, repaired_generation: bool = False) -> dict[str, object]:
    previous = [
        {"generation": 0, "diagnosis": "older repair " + ("x" * 900), "changed_files": ["system/fvSchemes"]}
        for _ in range(12)
    ]
    if repaired_generation:
        previous.append({
            "generation": 3,
            "diagnosis": "direct transport repair attempted",
            "changed_files": ["constant/battery/physicalProperties"],
        })
    return {
        "episode_id": "repair-0001",
        "root_failure": _native_failure(),
        "current_failure": _native_failure(),
        "current_implicated_files": [
            "constant/battery/physicalProperties",
            "constant/heater/physicalProperties",
        ],
        "previous_repairs": previous,
        "supporting_observations": [
            {
                "action_type": "search_references",
                "success": True,
                "summary": f"reference search {index}",
                "output_excerpt": f"result-{index}: " + ("reference-data-" * 200),
            }
            for index in range(12)
        ],
        "validation_generation": 3,
    }


def test_native_supported_values_are_sufficiency_evidence_not_a_python_choice():
    assert diagnostic_has_explicit_alternatives(_native_failure())
    assert not diagnostic_has_explicit_alternatives("Unknown transport type constIso")


def test_model_visible_repair_episode_is_a_bounded_window_not_the_archive():
    projection = project_repair_episode(_episode())
    assert projection is not None
    assert projection["archive_counts"]["supporting_observations"] == 12
    assert len(projection["support_evidence_window"]) <= REPAIR_SUPPORT_ITEMS
    assert len(projection["previous_repairs_summary"]) <= REPAIR_HISTORY_ITEMS
    assert len(json.dumps(projection["support_evidence_window"], ensure_ascii=False)) <= REPAIR_SUPPORT_CHARS + 256
    assert "supporting_observations" not in projection
    assert projection["root_failure_digest"]


def test_direct_repair_first_reopens_after_one_repair_in_same_generation():
    assert repair_episode_requires_direct_attempt(_episode(repaired_generation=False))
    assert not repair_episode_requires_direct_attempt(_episode(repaired_generation=True))


def test_partitioned_prompt_does_not_reinject_raw_support_archive():
    episode = _episode()
    payload = {
        "state_mode": "case_validation_repair_v2",
        "repair_episode_projection": project_repair_episode(episode),
        "validation_failure": _native_failure(),
        "relevant_case_files": [
            {"path": "constant/battery/physicalProperties", "sha256": "a", "content": "x" * 9000},
            {"path": "constant/heater/physicalProperties", "sha256": "b", "content": "y" * 9000},
        ],
        "supporting_evidence": [
            {"id": f"E{index}", "detail": "evidence-" * 300} for index in range(20)
        ],
        "supporting_observations": episode["supporting_observations"],
        "repair_contract": {"mode": "failure_local_delta_only"},
    }
    built, capsule, metrics = build_partitioned_validation_repair_prompt(
        "Repair this native validation failure:\n",
        payload,
        max_chars=18_000,
    )
    assert len(built.prompt) <= 18_000
    assert "supporting_observations" not in capsule
    assert len(capsule["repair_episode"]["support_evidence_window"]) <= REPAIR_SUPPORT_ITEMS
    assert metrics["repairSupportVisible"] <= REPAIR_SUPPORT_ITEMS * 2
    assert metrics["repairFileContentChars"] <= 7000
