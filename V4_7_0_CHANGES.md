# OpenFOAM Agent v4.7.0 — Unified Controller Graphs

v4.7.0 extends the controller-owned graph boundary from initial authoring to every major mutation/execution-adjacent phase.

## Unified authority model

The frozen `EngineeringPlan` remains the CFD design source of truth. LLM outputs are classified conceptually as CONTENT (model-owned authored values/files), HINT (model suggestions such as native strategy), or AUTHORITY (controller-compiled execution ordering, required manifest closure, final checkMesh, runtime bounds, restart interval, and postprocess dependency graph). Hints cannot independently create stale validation or native actions.

## CaseDeltaGraph

Committed-case repair, runtime repair, mesh strategy revision, and human-feedback revision now compile a `CaseDeltaGraph` before workspace mutation. The graph builds an effective candidate from the controller-tracked case plus the proposed delta, verifies required-file closure, path/content safety, changed-file semantics, native prerequisites and phase permissions, then deterministically appends exactly the required mesh consumers and final `checkMesh`. Stale validator/surface/native hints are advisory only.

## Transactional mutation

Multi-file text changes use a rollback-capable workspace transaction. The controller validates the entire delta and deterministic action budget before mutation. If a later file write/delete raises, all touched files, modes and authored-path bookkeeping are restored.

## Structured repair normalization

Harmless exact duplicate repair/revision paths, replacement payloads, patch objects and native hints are normalized before nested Pydantic validation. Conflicting duplicates are retained for deterministic graph/candidate conflict routing instead of becoming expensive structured-output retries or silent first-wins choices.

## PostProcessGraph

Postprocessing configuration content and run intent are compiled into a controller-owned `PostProcessGraph`. Every run/force-analysis dictionary dependency must exist in the candidate or current workspace before any new config is committed. Exact duplicates are deduplicated; conflicting config content blocks before mutation.

## Restart/runtime contract convergence

Parallel restart now uses the same compiled `RuntimeContract.execution_bound` as normal solve execution. It no longer independently requires legacy `EngineeringPlan.completion` when a bounded transient interval is already deterministically available from the sealed plan/controlDict.

## NativeToolContractRegistry

Native command effects, strong dictionary prerequisites and permitted controller phases are centralized in `tools/native_contracts.py`. `CaseBuildGraph`, `CaseDeltaGraph` and execution policy consume the same registry instead of maintaining overlapping command maps.

## EngineeringPlan conflict handling

Conflicting duplicate engineering defaults/bindings/evidence are no longer silently resolved by keeping the first item. Exact duplicates normalize; substantive duplicate conflicts are surfaced as deterministic plan conflicts for repair/review.

## Compatibility and hard gates

Confirmed intake closure, unsafe paths/content, immutable user assets, native provider availability, real OpenFOAM fatal diagnostics, blockMesh/checkMesh failures, mesh freshness, CaseSeal integrity, explicit solve approval, process/resource safety and result provenance remain strict.
