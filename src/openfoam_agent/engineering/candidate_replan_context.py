from __future__ import annotations

from copy import deepcopy
import json

from openfoam_agent.llm.context import (
    ContextBudgetError,
    PromptBuildResult,
    build_bounded_json_prompt,
    compact_text,
)


def _bounded_artifact(item: dict[str, object], *, content_chars: int) -> dict[str, object]:
    kind = str(item.get("kind") or "")
    out: dict[str, object] = {
        "path": item.get("path"),
        "kind": kind,
    }
    if kind == "raw":
        content = str(item.get("content") or "")
        out["content"] = compact_text(content, max(256, content_chars)) if content else ""
        out["truncated"] = len(str(out["content"])) < len(content)
        return out
    if kind == "typed_dictionary":
        out["foam_class"] = item.get("foam_class")
        entries = list(item.get("entries") or [])
        out["entries"] = deepcopy(entries[:24])
        out["entries_truncated"] = len(entries) > 24
        return out
    if kind == "block_mesh":
        # Candidate block-mesh topology has its own semantic repair phase.  Replan
        # only needs a compact identity so this generic path cannot be dominated by
        # thousands of vertices/faces.
        spec = item.get("spec")
        text = json.dumps(spec, ensure_ascii=False, separators=(",", ":")) if spec is not None else ""
        out["spec_excerpt"] = compact_text(text, max(512, content_chars)) if text else ""
        return out
    return out


def project_candidate_replan_capsule(
    payload: dict[str, object],
    *,
    manifest_limit: int,
    artifact_limit: int,
    total_artifact_chars: int,
) -> dict[str, object]:
    retained = payload.get("retained_candidate")
    retained = retained if isinstance(retained, dict) else {}
    failed_artifacts = [item for item in list(retained.get("failed_artifacts") or []) if isinstance(item, dict)]
    selected_artifacts = failed_artifacts[:artifact_limit]
    bounded_artifacts: list[dict[str, object]] = []
    remaining = total_artifact_chars
    for item in selected_artifacts:
        slots = max(1, len(selected_artifacts) - len(bounded_artifacts))
        allowance = max(512, remaining // slots)
        bounded = _bounded_artifact(item, content_chars=min(allowance, 6000))
        remaining = max(0, remaining - len(json.dumps(bounded, ensure_ascii=False)))
        bounded_artifacts.append(bounded)

    manifest = [item for item in list(retained.get("manifest") or []) if isinstance(item, dict)]
    missing = [str(path) for path in list(retained.get("missing_required_files") or []) if str(path).strip()]
    deferred = [str(path) for path in list(retained.get("deferred_native_required_files") or []) if str(path).strip()]
    failed_paths = [str(path) for path in list(retained.get("failed_paths") or []) if str(path).strip()]

    failure = retained.get("deterministic_failure")
    failure = failure if isinstance(failure, dict) else {}
    bounded_failure = {
        "action_type": failure.get("action_type"),
        "category": failure.get("category"),
        "validation_status": failure.get("validation_status"),
        "summary": compact_text(str(failure.get("summary") or ""), 2000) if failure.get("summary") else "",
        "diagnostic": compact_text(str(failure.get("diagnostic") or ""), 6000) if failure.get("diagnostic") else "",
        "failed_paths": [
            str(path) for path in list(failure.get("failed_paths") or []) if str(path).strip()
        ][:20],
    }

    plan_capsule = retained.get("plan_capsule") if isinstance(retained.get("plan_capsule"), dict) else {}
    required = [str(path) for path in list(plan_capsule.get("required_case_files") or []) if str(path).strip()]
    focus = set(missing) | set(deferred) | set(failed_paths)
    required_focus = [path for path in required if path in focus]
    if not required_focus:
        required_focus = required[:24]

    deterministic = payload.get("deterministic_bindings")
    deterministic = deterministic if isinstance(deterministic, dict) else {}
    intake_binding = deterministic.get("confirmed_intake")
    intake_binding = intake_binding if isinstance(intake_binding, dict) else {}

    return {
        "state_mode": "candidate_replan_partition_v1",
        "phase": payload.get("phase"),
        "step": payload.get("step"),
        "confirmed_intake_sha256": intake_binding.get("sha256"),
        "retained_candidate": {
            "goal": retained.get("goal"),
            "failed_paths": failed_paths[:20],
            "deterministic_failure": bounded_failure,
            "missing_required_files": missing[:40],
            "deferred_native_required_files": deferred[:40],
            "manifest": deepcopy(manifest[:manifest_limit]),
            "failed_artifacts": bounded_artifacts,
            "controller_build_policy": deepcopy(retained.get("controller_build_policy")),
            "pipeline": {
                "required_case_files": required_focus,
            },
            "plan_capsule": {
                "solver": plan_capsule.get("solver"),
                "solver_provider_id": plan_capsule.get("solver_provider_id"),
                "confirmed_intake_sha256": plan_capsule.get("confirmed_intake_sha256"),
                "required_case_files": required_focus,
            },
        },
        "budget": deepcopy(payload.get("budget")),
        "context_partition": {
            "active": True,
            "authority": "full retained candidate and EngineeringPlan remain Python-held; this is a failed-artifact-local projection only",
            "manifest_total": len(manifest),
            "failed_artifacts_total": len(failed_artifacts),
        },
    }


def build_partitioned_candidate_replan_prompt(
    instruction: str,
    payload: dict[str, object],
    *,
    max_chars: int,
) -> tuple[PromptBuildResult, dict[str, object], dict[str, int]]:
    """Build a bounded retained-candidate repair prompt.

    Replan never needs the whole generic engineering context.  Python keeps the full
    candidate and plan as authority while the model receives only the deterministic
    failure paths, genuinely missing authored files, deferred native outputs, and the
    implicated retained artifacts needed to return a small delta.
    """
    attempts = [
        (80, 6, 10_000),
        (40, 4, 7_000),
        (20, 2, 4_500),
        (12, 1, 2_500),
    ]
    last_error: ContextBudgetError | None = None
    for index, (manifest_limit, artifact_limit, artifact_chars) in enumerate(attempts, start=1):
        capsule = project_candidate_replan_capsule(
            payload,
            manifest_limit=manifest_limit,
            artifact_limit=artifact_limit,
            total_artifact_chars=artifact_chars,
        )
        try:
            built = build_bounded_json_prompt(instruction, capsule, max_chars=max_chars)
            retained = capsule.get("retained_candidate") or {}
            artifacts = retained.get("failed_artifacts") if isinstance(retained, dict) else []
            manifest = retained.get("manifest") if isinstance(retained, dict) else []
            return built, capsule, {
                "candidateReplanPartitioned": 1,
                "candidateReplanPartitionAttempts": index,
                "candidateReplanManifestVisible": len(list(manifest or [])),
                "candidateReplanFocusedArtifacts": len(list(artifacts or [])),
                "candidateReplanDiagnosticChars": len(
                    str((retained.get("deterministic_failure") or {}).get("diagnostic") or "")
                ) if isinstance(retained, dict) else 0,
            }
        except ContextBudgetError as exc:
            last_error = exc
    raise ContextBudgetError(
        "Retained-candidate failure-local projection still exceeds the deterministic model-context budget. "
        "The full candidate remains controller-held and no case mutation was authorized."
    ) from last_error
