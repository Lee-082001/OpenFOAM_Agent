# OpenFOAM Agent v5.0.1

## Scope-authority maintenance release

v5.0.1 is a maintenance release over the v5.0.0 authority/evidence + execution-scope architecture. It does not introduce a new CFD decision policy. Its purpose is to make the v5 authority boundary executable and internally consistent after the scope-unification migration.

### Fixed

- Removes the stale `design_seal._canonical_region_layouts` call that could survive the v5.0.0 scope migration and raise `NameError` on the first engineering design seal.
- Keeps `execution.scopes` as the only new-plan execution-topology authority. `region_layouts` remains an empty compatibility mirror at design sealing rather than a second topology source.
- Promotes both package metadata locations from `5.0.0` to `5.0.1` and adds a release regression that verifies version consistency.
- Adds a regression that fails if `_canonical_region_layouts` returns to `materialize_engineering_plan`.
- Verifies the scope-unification artifacts and execution/native contract JSON before mutation.

### Intentionally unchanged

- Agent still owns solver, execution driver, scopes, mesh strategy, boundary conditions, material choices, numerics and ordinary engineering defaults.
- The controller still owns provider/evidence validation, execution contracts, workspace safety, native command authorization, evidence freshness, case sealing and solve approval.
- Root and named-region cases remain one `execution.scopes` abstraction; v5.0.1 does not reintroduce single-region/multi-region controller modes.

### Upgrade base

This maintenance migration expects an already scope-unified OpenFOAM Agent v5.0.0 tree. It accepts both known states:

1. the v5.0.0 scope tree with the stale `_canonical_region_layouts` assignment, and
2. the same tree after the standalone design-seal hotfix.

The migration is idempotent with respect to the design-seal hotfix and refuses ambiguous local variants rather than guessing.
