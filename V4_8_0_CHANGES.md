# OpenFOAM Agent v4.8.0 — Phase-Controller Architecture Hardening

v4.8.0 is an architectural release built from the live failures observed through v4.7.x. It keeps the validated CaseBuildGraph/CaseDeltaGraph/native/runtime gates, but removes the remaining broad-context and phase-coupling failure modes.

## 1. Persistent multi-turn RepairEpisode

Committed-case/native validation repair no longer depends on whichever Engineering event happened most recently. Python stores a controller-owned `RepairEpisode` with:

- root failure,
- current native/deterministic failure,
- currently implicated files,
- previous repair deltas,
- bounded supporting reads/searches,
- validation generation.

Support reads/searches, including failed support lookups, cannot overwrite the current native failure. A later real validation failure advances `current_failure` while preserving `root_failure`. This fixes the thermal multi-region sequence where `constIso` was repaired, `hConst` became the next actual OpenFOAM error, and subsequent reference activity previously caused the active failure to disappear.

OpenFOAM internal source/template paths such as `<OPENFOAM_ROOT><LOCAL_PATH:solidThermo>` or absolute `/opt/openfoam...` paths are diagnostic provenance, not reference IDs. `read_reference` requires an indexed reference returned through the reference system.

## 2. Unicode application boundary

CLI user text now passes one normalization boundary before intake/history/hash/model use:

- unpaired UTF-16 surrogate code points are removed,
- remaining text is normalized to Unicode NFC,
- ordinary multilingual text is preserved.

This prevents terminal/paste artifacts from causing `UnicodeEncodeError: surrogates not allowed` before the first LLM request.

## 3. Stateless-safe postprocessing context

Postprocessing uses previous-response delta context only when the backend actually supports persisted state and `store=True`. Stateless Codex/Claude/Ollama-style adapters always receive a self-contained bounded postprocess capsule on every turn. A second stateless turn therefore never receives `delta_from_previous_response` with `use_previous_response=False`.

## 4. Phase-specific bounded context capsules

`llm/context_capsules.py` provides common projections for confirmed intake, EngineeringPlan, feedback history and revision history. Full authoritative objects stay in Python state; model-facing capsules carry the values required for that phase plus immutable digests.

Major phases now use phase-specific plan projections instead of copying the full EngineeringPlan into protected prompt subtrees:

- staged authoring,
- human revision decision,
- human revision authoring,
- strategy revision,
- committed validation repair,
- runtime repair,
- postprocessing,
- feedback/result review,
- generic Engineering fallback.

Authoring partitioning retains a strict distinction between model context and controller authority: if a large authoring prompt is split into file tasks, the task compiler is rebound internally to the complete Python-held frozen plan while each task sent to the model remains compact. Task acceptance still checks the full frozen-plan digest.

## 5. Human revision is decision -> authoring

Confirmed result/mesh feedback no longer asks one LLM turn to reconsider the EngineeringPlan and author file changes simultaneously.

```text
REVISION_READY
  -> RevisionDecisionTurn
       -> decide_revision + EngineeringPlanPatch + target files
  -> Python validates/stages the revised plan
  -> RevisionAuthoringTurn
       -> file-only RepairCasePlanAction
  -> CaseDeltaGraph
  -> archive prior solved output immediately before mutation
  -> validate/reseal
```

The first phase cannot author case files. The second phase cannot modify the plan again. Pre-mutation context/provider failures leave the previous solved outputs intact and restore the proposal to a retryable revision state.

## 6. Physical Engineering controller decomposition

The previous ~6.4k-line `engineering/agent.py` accumulated context, lifecycle, decision routing, authoring, repair and runtime-repair behavior in one class body. v4.8.0 keeps `CFDEngineeringAgent` as a compatibility facade but moves phase behavior into physical modules:

- `engineering/phases/context_controller.py`
- `engineering/phases/lifecycle_controller.py`
- `engineering/phases/decision_controller.py`
- `engineering/phases/authoring_controller.py`
- `engineering/phases/repair_controller.py`
- `engineering/phases/repair_episode.py`

The facade is approximately 3.5k lines and delegates to these controllers. Existing public/internal call shapes used by regression tests are preserved through thin wrappers.

## 7. Authority model retained

The refactor does not move CFD decisions into Python.

- LLM Agent: physics/solver/mesh/BC/numerics/default engineering choices and repair content.
- Python controller: context/state boundaries, immutable digests, validation target compilation, native action authorization, dependency/freshness checks, transactional mutation, CaseSeal and solve approval.

Existing hard gates for immutable user assets, unsafe content, native toolchain readiness, FOAM FATAL diagnostics, mesh/checkMesh, pre-solve consumers, CaseSeal, explicit solve approval and result provenance remain strict.

## Regression qualification

The final source test set contains 67 `tests/test_*.py` files. It is executed in four disjoint groups to avoid monolithic environment-timeout artifacts. Final counts are recorded in `OpenFOAM_Agent_v4.8.0_verification.json`. A clean v4.7.4 baseline is also patched independently and receives the same regression qualification before release packaging.
