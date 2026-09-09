# OpenFOAM Agent v4.3.1

## Live failure addressed

A live v4.3.0 run accepted the Engineering design for a fully filled two-outlet water bifurcation, then staged authoring returned:

```text
Case authoring cannot be completed in this environment without fabricating the required surface artifact.
```

No OpenFOAM tool had failed. The Agent had treated a missing external surface as an integrity problem even though the user had supplied only conceptual topology and explicitly delegated ordinary engineering choices.

## Geometry ownership policy

v4.3.1 separates exact user-owned geometry from representative Agent-owned geometry.

- User assets remain immutable.
- Missing external CAD/STL is blocking only when exact external geometry is itself a confirmed requirement.
- Concept geometry plus delegated ordinary dimensions may be implemented as representative procedural geometry.
- Simple generic pipe/channel/obstacle cases should prefer self-contained typed `blockMesh`.
- If a surface workflow is genuinely useful, the Agent may author a bounded case-local ASCII STL/OBJ under `constant/triSurface` and validate it before meshing.
- Creating that representative geometry is an authorized engineering choice, not evidence fabrication.
- Selected representative dimensions are recorded as `engineering_default` provenance and never promoted to user facts or exact fidelity.

## Block policy consistency

The progress-first assumption policy has authorized ordinary engineering defaults since v4.2. The older `BlockAction` path still depended on the legacy `exploratory_completion_authorized` flag. v4.3.1 removes that inconsistency: `engineering_choice_missing` is rejected as a terminal block and must be resolved by Agent-owned engineering defaults. Genuine physics/routing ambiguity, unavailable required tools, security/integrity failures and exact missing user assets remain valid blocking conditions.

## Context preservation

The full geometry policy is present in design/authoring context. File-scoped compact authoring tasks receive a concise equivalent contract so an 8k forced partition remains viable.

## Verification

The release adds geometry-ownership regressions and reruns the full suite. The package is generated from a clean v4.3.0 baseline and independently reapplied/compared before release.
