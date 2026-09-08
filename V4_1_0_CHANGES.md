# OpenFOAM Agent v4.1.0

## Risk- and stage-aware evidence policy

This release refactors evidence from a universal design-time hard gate into three classes:

- **MANDATORY**: fail-closed at the named verification stage (confirmed requirements, trusted execution provider, approval, mesh freshness, case seal/completion).
- **DEFERRED**: design can proceed; resolve before use or replace with deterministic/native validation (version-specific syntax and specialized implementation details).
- **ADVISORY**: absence never blocks by itself; record engineering judgement/default provenance and validate outcomes.

### Structural changes
- `EngineeringDecision` records `risk_level`, `evidence_policy`, and `verification_stage`.
- Design acceptance no longer demands explicit source excerpts for every future case file.
- Trusted installed capability providers can satisfy design-stage provider provenance directly from the deterministic installation catalog.
- The design prompt receives a compact `verified_execution_candidates` inventory.
- Raw free-form OpenFOAM authoring remains explicit-evidence gated.
- Typed dictionaries and structured `blockMesh` may proceed without one source excerpt per file because Python owns serialization and downstream parser/native validation.
- Context partitioning preserves the evidence policy and verified execution candidates.

### Safety retained
User-confirmed requirements, native execution approval, case sealing, current checkMesh/mesh freshness, command safety, resource budgets, runtime completion and result validation remain fail-closed.

Actual OpenFOAM/MPI/live Codex end-to-end execution is not claimed by this release unless separately run in the target environment.
