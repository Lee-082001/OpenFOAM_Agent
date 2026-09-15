# OpenFOAM Agent v5.0.2

## Multi-region controller regression fix

v5.0.2 is a maintenance release over v5.0.1. It does not change CFD decision ownership or introduce a new solver/mesh policy. It fixes two controller protocol regressions observed in staged multi-region authoring.

### Fixed

- Restores the full controller-held `EngineeringPlan` as the input to deterministic pre-solve validation. `execution.scopes` and declared interfaces are no longer discarded by projecting the case down to `required_case_files` alone.
- Prevents named-region cases such as `battery` + `heater` from being falsely validated as a root-layout case requiring `0/*` and `constant/polyMesh/boundary`.
- Verifies that `ValidatePreSolveAction.required_case_files` exactly matches the frozen plan manifest before validation. A mismatch is classified as controller/infrastructure failure rather than a CFD case failure, so it is not routed into case repair.
- Treats `task_id` and `defer_native` as controller-owned partition metadata. When no authoring partition queue exists, harmless model echoes of those fields are normalized away instead of rejecting an otherwise complete authoring bundle.
- Tightens the authoring prompt so unpartitioned turns explicitly leave `task_id=null` and `defer_native=false` unless an `authoring_task` was supplied by the controller.

### Regression coverage

- `ValidatePreSolveAction` dispatch must call `PreSolveCompletenessGate.validate(plan)` and must not use the scope-losing legacy required-files-only projection.
- Manifest drift between the validation action and frozen plan is rejected as an infrastructure/controller invariant failure.
- Unpartitioned staged authoring accepts a complete bundle even if the model emits stale partition metadata, while the controller resets that metadata before execution.
- Package/project version consistency remains checked across maintenance releases.

### Intentionally unchanged

- `execution.scopes` remains the Agent-selected execution-topology authority.
- The Agent still owns CFD physics, solver, mesh, boundary/material/numerics decisions and ordinary engineering defaults.
- Python/controller still owns validation, workspace safety, native command authorization, evidence freshness, sealing and execution gates.
- `validate_required_case_files()` remains available as a narrow compatibility helper, but the staged execution path no longer uses it as a substitute for the frozen plan.
