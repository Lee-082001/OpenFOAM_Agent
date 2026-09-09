# OpenFOAM Agent v4.6.0 — Controller-Owned CaseBuildGraph

## Live failure that motivated the release

A v4.5.1 staged-authoring response returned only `system/controlDict`, `system/fvSchemes`, and `system/blockMeshDict`, while its independent validation mirror still included `system/fvSolution`. The old executor trusted both lists: it committed the three returned files and then executed `ValidateDictionaryAction(system/fvSolution)`, producing `FileNotFoundError` and incorrectly routing the failure as infrastructure.

## Architectural change

`EngineeringPlan.required_case_files` is now the single source of truth for authored solve inputs. The controller renders the candidate and compiles a `CaseBuildGraph` before any write. The graph owns:

- required-manifest coverage,
- static/header targets derived from actual artifacts,
- surface validation derived from actual surface files,
- filtering of stale documentary/validation hints,
- strong native-utility prerequisite handling,
- obvious blockMesh/snappy consumer inference, and
- final checkMesh ordering.

Model fields such as `validate_dictionaries`, `surface_checks`, `mesh_commands`, and `native_pipeline` are compatibility/strategy hints only. They cannot independently schedule a validator against an unauthored path.

## Transactional missing-file repair

If any frozen required path is absent, the controller emits `case_build_graph` before workspace mutation, retains the complete partial candidate in memory, records the exact missing paths, and switches the next model call to the compact retained-candidate repair contract. The repair prompt asks only for missing/conflicting artifacts. Python never invents their CFD content.

For a later repair after a complete case was already committed and a native consumer failed, the graph may reuse only paths tracked by `CaseWorkspace.list_authored()` as the baseline. This preserves delta repair without allowing a fresh incomplete candidate to pass coverage by accident.

## Validation philosophy

This release removes another duplicated contract rather than weakening validation. Unsafe paths/content, false semantic claims, serialization failures, native mesh errors, checkMesh freshness, pre-solve completeness, CaseSeal, runtime bounds and solve approval remain strict. Missing authored inputs are now caught earlier and categorized as case/authoring contract failures instead of surfacing later as infrastructure `FileNotFoundError`.
