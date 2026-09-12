# OpenFOAM Agent v4.9.2 - Required manifest contract fix + repository cleanup

v4.9.2 fixes the staged-design failure where a controller-sealed plan could carry an empty `required_case_files` manifest into authoring and then crash while constructing `ValidatePreSolveAction`.

## Required-case manifest contract

- `EngineeringDesign.required_case_files` is now a non-empty Agent-owned solve-input contract.
- The staged-design prompt explicitly requires a concrete, complete manifest under `0/`, `constant/`, and `system/` for the design selected by the Agent.
- The controller sealer defensively rejects an empty manifest.
- `CaseBuildGraph` rejects an empty durable manifest fail-closed, so malformed/legacy state cannot reach pre-solve construction with `required_case_files=[]`.
- Python does not choose CFD-required files; the Agent declares them and Python validates authored coverage and execution safety.

## Consolidated repository cleanup

This ZIP also consolidates the earlier v4.9.1 hygiene work without requiring its temporary two-stage carrier script:

- prunes top-level imports proven unused in the split engineering phase controllers;
- removes superseded historical verification bundles, early release-note files, obsolete RC2 examples and old audit snapshots;
- removes local cache/reject artifacts;
- removes stale v2 branding from current CLI/workflow text;
- retains compatibility-sensitive schemas/state adapters that are still referenced by tests or persisted checkpoints.

## Delivery

The migration script operates on the existing checkout in place, never edits `.git`, performs a no-write preflight, and creates a tar.gz backup of every file it will modify/delete before applying changes.
