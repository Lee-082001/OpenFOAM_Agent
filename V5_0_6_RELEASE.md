# OpenFOAM Agent v5.0.6

## Scope

v5.0.6 is the native-execution-correctness maintenance release on top of v5.0.5.

## Fixes

- Replaces inferred-command prepending with a stable dependency-DAG ordering pass. Agent order remains the tie-breaker for independent commands; declared producer/consumer relationships alone force reordering.
- Adds controller-owned ordering artifacts so `blockMesh` and feature extraction must precede `snappyHexMesh` whenever those producers are present. Cyclic native dependency contracts fail closed.
- Adds OpenFOAM Foundation `surfaceFeatures`/`system/surfaceFeaturesDict` as a first-class native tool contract, including trusted-runner compatibility and mesh-freshness tracking. OpenFOAM Foundation v13/v14 installations can therefore use the native utility instead of being rejected as effect=unknown.
- Completes v5 scope unification in `retry_solver`: the scope-keyed `mesh_evidence_by_scope`/`mesh_manifest_by_scope` gate is authoritative; the legacy root `state.mesh_evidence` and global mesh-manifest mirror no longer reject valid named-region retries.
- Fixes the retry dispatch typo that called the nonexistent `_region_mesh_failures()` instead of the implemented scope-aware `_mesh_evidence_failures()`.

## Regression coverage

The v5.0.6 tests cover:

1. preserving `blockMesh -> surfaceFeatureExtract -> snappyHexMesh` when already valid;
2. correcting an Agent-proposed order where `snappyHexMesh` appears before its declared base-mesh/feature producers;
3. Foundation `surfaceFeatures` contract authorization and ordering before `snappyHexMesh`;
4. named-region solver retry succeeding without a legacy root mesh-evidence mirror.

The dependency DAG is execution metadata only. It does not select a mesh strategy or substitute Python CFD heuristics for Agent engineering decisions.
