# OpenFOAM Agent v4.5.0

## Runtime contract compiler

A live v4.4.0 case successfully completed authoring, `blockMesh`, `checkMesh`, pre-solve validation and case sealing, reaching `SOLVE_READY`. The Engineering Agent had already selected `Maximum 2000 iterations`, residual targets for `U` and `p`, outlet-split stability, and conservation criteria as `engineering_defaults`. Nevertheless `/solve` returned `ENGINEERING_REVIEW_REQUIRED` because the same information was not duplicated into `EngineeringPlan.completion`.

v4.5.0 removes that duplicate-ownership requirement. `compile_runtime_contract()` derives a controller-owned runtime view from values that already exist. It never chooses new CFD values.

### ExecutionBoundContract

Controls safe native execution only:

- transient start/end time when literal or explicitly planned;
- steady iteration span when literal in `controlDict` or already selected by the Agent;
- approved wall-time limit;
- minimum progress;
- required result fields.

A steady/custom run may proceed when it is safely bounded even if result acceptance is incomplete.

### ResultAcceptanceContract

Controls post-execution claims only:

- explicit plan residual thresholds when available;
- machine-compilable residual subsets from Agent-owned steady defaults;
- remaining convergence/QoI/conservation prose as advisory criteria.

Incomplete or unimplemented result acceptance does not invalidate bounded execution. It prevents automatic numerical-quality claims and remains visible in result review. Explicit complete `CompletionContract` criteria remain supported and can still be machine-verified.

## Native-pipeline consolidation

Staged authoring and repair normalize out `foamDictionary` from `native_pipeline`/legacy command lists. The controller already performs deterministic dictionary/header validation, while native mesh/solver consumers provide stronger implementation evidence. This removes repeated per-file subprocesses and reduces native budget/latency without weakening `blockMesh`, `checkMesh`, pre-solve, case-seal, or solver gates.

## Message lifecycle

Reports no longer blindly render `state.history[-1]`. The displayed message is the latest transition note that actually produced `current_state`, preventing stale superseded failures from being attached to a later healthy state.

## Runtime semantics

For the new compiled `RuntimeContract`, a clean steady/custom OpenFOAM `End` after positive progress proves bounded execution completion. Residual/result acceptance is evaluated separately and recorded in `numerical_quality_verified` plus `acceptance_warnings`. Output freshness remains independently verified.

## Compatibility

The legacy `completion_contract()` adapter remains for restart/tests and explicit completion users, but production runtime orchestration uses `compile_runtime_contract()`. Explicit transient completion still must agree with literal `controlDict` settings.
