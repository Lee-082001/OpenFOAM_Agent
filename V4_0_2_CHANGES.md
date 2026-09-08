# OpenFOAM Agent v4.0.2

This release fixes prepare-design context exhaustion observed with evidence-heavy multiphysics requests.

## Design/evidence context partitioning
- `prepare_design` and `prepare_design_decide` now recover from prompt-budget overflow by projecting a deterministic design capsule instead of immediately failing.
- Confirmed intake, intake digest, assumption policy, execution bindings, and budgets remain authoritative and are not truncated.
- Older evidence is retained durably in state but projected as identity/reference/summary; only a small newest slice carries bounded detail.
- Environment/capability inventories are projected to compact identity hints instead of monopolizing design context.
- Evidence-gap and recent-observation windows are bounded independently.
- If mandatory confirmed requirements alone exceed the configured prompt budget, the workflow still fails closed and tells the operator to raise the budget or reduce authoritative input.

## Regression
- Added 18k-budget regressions reproducing the evidence-heavy failure shape.
- No actual OpenFOAM, MPI, or live Codex call is claimed by these tests.
