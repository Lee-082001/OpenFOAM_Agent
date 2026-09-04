# OpenFOAM Agent v3.1.0 changes

## Semantic blockMesh topology contract

The v3.0.2 run that motivated this release produced syntactically valid OpenFOAM files but failed `blockMesh` because a face declared on the `cylinder` patch was actually an internal/shared block face. Local repair then changed only face ordering and fell back to exact text patches, which consumed LLM turns without changing the topology class. v3.1.0 fixes the representation and repair model rather than adding a cylinder/O-grid template.

- Added `tools/block_mesh_topology.py` with a generic `BlockMeshTopologyIR` and deterministic topology report.
- Canonical block faces are derived from every `hex` and indexed by owner block. A declared boundary face must match a real block face and have exactly one owner.
- Rejects internal/shared boundary faces, nonexistent faces, non-manifold (>2-owner) faces, duplicate patch assignment, degenerate block/boundary faces, explicit edges that are not block edges, duplicate explicit edges, and invalid `mergePatchPairs` references.
- `serialize_block_mesh()` now refuses an invalid topology before any workspace mutation or native command. OpenFOAM remains authoritative for geometric orientation and mesh quality.

## Representation-aware repair

- Added compact `CandidateBlockMeshRepairTurn` / `repair_candidate_block_mesh`. When pre-commit topology validation fails, the exact failed structured `block_mesh` remains in memory and the model returns only one corrected semantic replacement.
- Added compact `BlockMeshRepairTurn` / `repair_block_mesh` for local native blockMesh failures after commit. Python preserves the exact structured representation that generated `system/blockMeshDict`; repair no longer depends on whitespace-sensitive exact patches.
- The committed local repair pipeline is deterministic: structured replacement -> `foamDictionary`/header validation -> `blockMesh` -> `checkMesh` -> PreSolve -> finish preview.
- Generic `RepairTurn` and `CasePlanRetryTurn` remain compact; the blockMesh schema is loaded only for the dedicated mesh-repair phase. Current schema sizes are about 4.3-4.5k characters for the two dedicated turns rather than inflating every generic repair turn.
- `block_mesh_serialize` failures now participate in the transactional candidate-repair retry budget/routing.

## No-progress / escalation hardening

- blockMesh native failure signatures normalize transient numeric face/cell/vertex labels. Diagnostics that differ only as `4(3 4 20 19)` vs `4(3 19 20 4)` now have the same failure signature.
- Mesh repair-cycle accounting now counts `patch_case_file` as an actual repair mutation, closing a loophole where repeated mesh text patches were not counted as repair cycles.
- Repeated equivalent native mesh failure after a structured local repair escalates to strategy revision instead of continuing local topology tweaks.

## Regression coverage

Added tests for internal/shared boundary-face rejection, valid external-face acceptance, pre-serialization rejection, retained structured candidate context, semantic candidate repair, semantic committed repair, specialized phase routing, Claude/Codex schema portability, compact schema size, and normalized repeated failure signatures.

Full suite: **261 passed**.
