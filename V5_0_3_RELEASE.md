# OpenFOAM Agent v5.0.3

## Scope

v5.0.3 is a maintenance release over v5.0.2. It preserves the v5.0.2 multi-region controller fixes and repairs the pre-commit mesh-strategy revision recovery path exposed by bounded generated-surface authoring.

## Fixes

- Fix `project_strategy_plan()` passing `max_chars=60` to `compact_text()`, whose public contract requires at least 64 characters. Normal engineering-default units such as `m/s` can now be projected into a strategy-revision prompt without raising `ValueError: max_chars must be >= 64`.
- Keep strategy replacement Agent-owned. An infeasible generated STL authoring attempt can now reach the existing `strategy_revision` turn, where the Agent may choose a different self-contained mesh representation; Python still validates and executes the chosen strategy rather than hard-coding a Y-pipe/blockMesh policy.
- Add a source-level regression guard for literal `compact_text()` limits below the helper contract minimum.
- Make the v5.0.2 release consistency test maintenance-release-safe so future version bumps do not fail solely because the historical test hard-coded `5.0.2`.

## Regression coverage

New v5.0.3 tests verify:

1. a strategy-plan projection containing an engineering default with `unit="m/s"` succeeds;
2. a pre-commit `authoring_feasibility:precommit_strategy` failure can compile a bounded `strategy_revision` context and reach the strategy-revision model contract;
3. no literal `compact_text(..., N)` call under `src/openfoam_agent` violates the `N >= 64` helper contract;
4. package and project metadata report v5.0.3 consistently.
