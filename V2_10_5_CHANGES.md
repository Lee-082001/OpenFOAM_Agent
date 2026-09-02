# v2.10.5 — Plan-Only Repair and Solver Metadata Consistency

- Allows `repair_case_plan` to carry only `updated_plan` when deterministic artifacts do not need modification.
- Still rejects true no-op repairs that contain neither artifact changes nor an updated plan.
- Adds compact prompt guidance requiring the exact observed capability provider solver name to be copied into both `EngineeringPlan.solver` and `system/controlDict`.
- Explicitly forbids solver placeholders such as `foamRunNameHere`, `solverName`, `TBD`, `TODO`, and `placeholder`.
- Keeps deterministic solver/provider and controlDict/EngineeringPlan exact-match validation unchanged.
- Adds a regression test for the real failure mode: successful case creation, `blockMesh`, `checkMesh`, and pre-solve validation followed by a solver-metadata mismatch that is repaired without touching case files.
