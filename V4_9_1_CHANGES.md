# OpenFOAM Agent v4.9.1 — Repository Cleanup

v4.9.1 is a behavior-preserving hygiene release after the v4.9.0 controller-sealed design boundary.

## Removed

- unused imports left behind when the v4.8 monolithic engineering controller was physically split into phase controllers;
- generated historical `verification/` logs/XML/reports that belong in Git history or CI artifacts rather than the active source tree;
- old top-level release-note files prior to v4.9.0 (their history remains in Git);
- obsolete v4 RC2 example artifacts and their checker script;
- the old V2 evaluation-plan artifact;
- historical audit snapshots under `docs/` that are superseded by the live code/tests;
- local Python/test caches plus `.orig`/`.rej` patch leftovers;
- stale `v2` branding in the current CLI/workflow text.

## Intentionally retained

Some old-looking structures are **not dead code yet** and are therefore not deleted in this release:

- legacy all-phase/non-staged LLM contracts such as `EngineeringTurn` / `PrepareTurn`, because existing regression tests and direct library callers still reference them;
- v2.x `implementation_refs`/digest migration adapters, because removing them changes persisted checkpoint/case-seal compatibility;
- `State.DONE`, because it remains a backward checkpoint compatibility value;
- duplicated capability graph files under `config/` and package data, because they currently serve source-tree development and installed-package lookup paths respectively.

Removing those requires an explicit breaking migration of tests/persisted-state formats rather than deleting code merely because it is old.

## No CFD authority change

No solver, mesh, boundary-condition, numerical-method, repair, safety-gate, evidence, or execution policy is changed. The Engineering Agent still owns CFD decisions; Python still owns validation, sandboxing, provenance, execution authorization and case sealing.
