# OpenFOAM Agent v4.8.1 — Prepare Repair Terminal Ownership

## Motivation

A live Foundation OpenFOAM v14, Re=1000, 3D transient lid-driven cubic cavity run exposed a controller phase-order bug. The initial execution plan reached `validate_pre_solve`, the zero-step consumer rejected the case, and a subsequent `system/fvSchemes` repair plus revalidation succeeded. However, the LLM repair object also carried `retry_solver=true`. The shared `_repair_actions(..., runtime=False)` path interpreted that hint as authority and appended `RetrySolverAction`, which the prepare dispatcher correctly rejected with `retry_solver is only valid after a solver failure.` The next LLM turn then blocked.

This was not a CFD/numerics failure after the repair. It was a controller ownership error between prepare recovery and runtime recovery.

## Fix

The terminal repair action is now determined only by the controller phase:

- `runtime=False` (prepare/pre-solve/dictionary/native case repair) -> `FinishPreviewAction`
- `runtime=True` (solver runtime repair) -> `RetrySolverAction`

`RepairCasePlanAction.retry_solver` remains a compatibility/model hint in the shared schema but cannot promote prepare recovery into solver execution. Prepare repair therefore completes the existing validation -> seal -> `SOLVE_READY` -> explicit `/solve` approval path. Runtime repair preserves the existing retry behavior.

No failure records are cleared early. Successful `finish_preview` follows the established final-validation/seal path, which resolves the prior failure only after the repaired case has actually passed the gates.

## Regression coverage

`tests/test_v474_failure_local_repair.py` now includes:

1. A four-way controller matrix over `runtime` x `retry_solver` hint:
   - prepare + false -> finish_preview
   - prepare + true -> finish_preview
   - runtime + false -> retry_solver
   - runtime + true -> retry_solver
2. The existing synthetic zero-step failure -> `fvSchemes` repair -> zero-step revalidation -> `SOLVE_READY` integration flow is run for both `retry_solver=false` and `retry_solver=true`.
3. Existing assertions retain proof that zero-step validation runs twice, mesh is not unnecessarily rebuilt, the repaired dictionary is committed, and the original primary failure is cleared only after successful completion.

The fix intentionally does not hard-code `laplacianSchemes` or any OpenFOAM numerical choice; the synthetic missing-laplacian condition remains only a regression trigger.
