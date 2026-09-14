# OpenFOAM Agent v5.0.0 Scope-Unified Authority/Evidence Hardening

This follow-up hardening keeps the public version at 5.0.0 while removing controller-level
`single-region` / `multi-region` modes from the v5 authority model.

## Architectural invariant

**The Agent selects CFD/execution topology. Python validates and executes the selected contracts.**

Python must not select `foamRun` vs `foamMultiRun` from scope count, nor treat one root scope and
multiple named scopes as different CFD workflow modes.

## Canonical topology

`OpenFOAMExecutionSpec.scopes` is the Agent-owned topology authority.

- root namespace: `ExecutionScope(name=None)` -> scope key `root`
- named OpenFOAM region: `ExecutionScope(name="battery")` -> `region:battery`

Legacy `solver_module`, `solver_provider_id`, `regions`, and `region_layouts` remain only as
load/compatibility projections and are hidden from new structured-output schemas where applicable.

## Contract-driven driver behavior

`data/execution_driver_contracts.json` describes driver syntax/shape for known providers.
`foamRun` and `foamMultiRun` are not selected by controller branches. The Agent selects the
provider; the controller validates its declared scopes and renders control/runtime syntax from data.

## Unified mesh evidence

Canonical state:

- `mesh_evidence_by_scope`
- `mesh_manifest_by_scope`

Legacy global/per-region fields are compatibility mirrors only. Finalization iterates the scopes
of the Agent-selected execution topology uniformly.

## Dependency-scoped freshness

Mesh evidence freshness is based on the observed mesh dependency DAG. Material, thermo, BC or
numerical-file edits do not stale `checkMesh` evidence unless an observed mesh producer actually
depends on them. A battery-only mesh change invalidates/revalidates only the battery scope.

Native tool dependency/scope metadata is package data in `native_tool_contracts.json`.

## Generic build and delta graphs

Initial authoring and repair both use the same scope abstraction:

1. infer only controller-authorized native consumers from actually authored tool dictionaries;
2. bind invocations to declared scopes through native-tool contracts;
3. compile final mesh validators by iterating scopes;
4. on repair, replay/revalidate only affected scopes.

There is no controller-level `if single_region / elif multi_region` workflow branch.

## Failure lifecycle

A later successful `validate_pre_solve` resolves stale prior failures of the same action type and
moves them into `resolved_failures`, preventing repaired native diagnostics from remaining the
reported primary failure.

## Regression target from the battery/heater case

Expected sequence after this hardening:

- battery/heater `blockMesh` pass;
- battery/heater `checkMesh` pass;
- `constIso` native diagnostic -> focused material repair;
- `hConst` native diagnostic -> focused material repair;
- zero-step consumer passes;
- previous mesh evidence remains current because material files are not mesh dependencies;
- final evidence gate accepts both declared scopes without requiring redundant `checkMesh` runs.
