# OpenFOAM Agent v2.17.0

## Dedicated blockMesh DSL

- Adds `TypedBlockMeshFile` with explicit vertices, hex blocks, edges, boundary patches and merge-patch pairs.
- Deterministic Python renders blockMesh list/dictionary syntax and owns punctuation only.
- `execute_case_plan` rejects `system/blockMeshDict` inside generic `typed_dictionaries`; use `block_mesh`.
- Raw `blockMeshDict` remains readable/patchable for compatibility.

## Deterministic mesh tool contracts

- Exposes narrow executable prerequisites in the Engineering context.
- `snappyHexMesh` is preflighted against the current base `polyMesh`; an `empty` patch blocks execution because the snapping stage requires a fully 3D base mesh.
- A precondition rejection consumes no native command and does not select an alternative meshing strategy.
- Custom/fake tool backends that do not implement the optional contract methods retain legacy behavior.

## Failure novelty and strategy escalation

- Native mesh failures carry a normalized failure signature.
- A strategy-scoped tool-contract failure immediately switches the next compact contract to `StrategyRevisionTurn`.
- The same normalized native mesh failure recurring twice also escalates from local repair to strategy revision.
- `revise_mesh_strategy` is delta-only: it may patch/replace/drop meshing artifacts, optionally provide a new typed blockMesh, update the plan, and execute a new mesh pipeline ending in `checkMesh`.
- The Engineering Agent chooses the replacement strategy; Python only enforces safety, tool contracts, execution and evidence freshness.

## Regression

- 212 tests pass.
- Added end-to-end state-machine tests for tool-precondition escalation and repeated-native-failure escalation.
- Added serializer regression proving blockMesh `boundary` is rendered as a patch list rather than a generic nested mapping.
- `StrategyRevisionTurn` passes strict Structured Output schema validation.
