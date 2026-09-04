
## v3.2.0 — Engineering evidence / assumption contract

Structured evidence storage is separated from event logging, delegated engineering defaults receive explicit provenance, and repeated evidence-infrastructure failures deterministically close retrieval instead of consuming LLM turns. See `V3_2_CHANGES.md`.
# OpenFOAM Agent v3.0.0 changes

## Semantic PreSolve foundation

v3.0 replaces literal set-difference boundary validation with a reusable OpenFOAM semantic interpretation layer under `src/openfoam_agent/verification/foam_semantics/`.

- Added syntax-preserving `FoamKey` / `FoamEntry` models with raw token form, quoted-vs-word information, pattern state, and dictionary order.
- Added `MeshIR` / `MeshPatch` with patch name, mesh patch type, `inGroups`, and source order.
- Added `BoundarySelector` and `BoundaryFieldInterpreter` with OpenFOAM v13 effective-selection semantics: exact patch -> patchGroup -> automatic `empty` -> regex/wildcard.
- Same-tier overlapping group/regex matches preserve ordering; later matching entries win.
- Quoted keys are not automatically treated as regex. Quoted strings without pattern metacharacters remain literal selectors.
- Dynamic/unsupported selectors are retained as `INDETERMINATE` evidence. They produce a limitation warning rather than a false `missing patchField` failure.
- `BoundaryResolution` records status, match kind, effective field type, certainty, and a resolution trace.

## PreSolve integration

- Boundary coverage and mesh/field constraint checks now consume the same effective boundary resolution.
- `frontAndBack`/other `empty` mesh patches honor OpenFOAM's automatic-empty behavior before wildcard fallback.
- Existing missing-patch failure wording remains compatible while adding semantic-resolution context.
- PreSolve results expose semantic warnings and per-field patch resolution kinds.
- Successful pre-solve progress output surfaces semantic warnings instead of silently dropping indeterminate evidence.

## Regression coverage

Added v3 regression tests for:

- `"(walls|obstacle)"` regex coverage.
- `"wall.*"` and `".*"` wildcard coverage.
- exact selector precedence over regex.
- patchGroup precedence over regex.
- later regex precedence within the regex tier.
- automatic `empty` precedence before regex fallback.
- quoted literal semantics.
- one non-pattern selector acting as exact for one patch and group selector for others.
- dynamic selector -> `INDETERMINATE`, not false missing.
- PreSolve integration for coverage and constraint validation.

Full suite: **239 passed**.

## v3.0.1 — structured-output backend compatibility

v3.0.1 adds a backend-specific Structured Schema Compiler between canonical Pydantic contracts and CLI transports. Claude receives tuple schemas without `prefixItems`; Codex receives OpenAI-strict object schemas with all properties required, `additionalProperties: false`, and transport-inapplicable defaults removed. Final output remains validated by the original Pydantic model. See `V3_0_1_CHANGES.md`.


## v3.0.2 — canonical FoamFile contract and solve-input header integrity

v3.0.2 closes the gap between dictionary-syntax acceptance and OpenFOAM `regIOobject` readability. Generic typed files and `blockMeshDict` now share one canonical `FoamFile` header renderer. Header `object` and `location` are derived from the case path, text serialization owns `format ascii`/`version 2.0`, and initial fields use a proven field class from `internalField` or an explicit bounded `foam_class` when the shape is indeterminate. Matching legacy `FoamFile.*` typed entries are consumed as compatibility metadata; conflicting object/location/class metadata is rejected.

The complete-case transactional preflight now validates solve-critical raw/typed artifacts before the first workspace mutation. `validate_dictionary` checks the IOobject-facing header before calling `foamDictionary`, and PreSolve checks header presence, required metadata, object/path consistency, system `class dictionary`, field-class validity, and unambiguous `internalField` class consistency. A runtime-repair context also includes a deterministic batch scan of core system files, current initial fields, and Agent-declared required solve inputs so systematic header defects are repaired together.

Regression suite: **252 passed**. See `V3_0_2_CHANGES.md`.

## v3.1.0 — semantic blockMesh topology and representation-aware repair

v3.1.0 interprets `TypedBlockMeshFile` as a deterministic face-ownership graph before serialization. Boundary faces must be actual exterior one-owner block faces; internal/shared, nonexistent, duplicate, non-manifold, degenerate, and invalid explicit edge/merge-patch references are rejected before native `blockMesh`. The Agent still owns geometry and mesh design; Python proves generic topology invariants only.

Pre-commit topology failures use a dedicated compact `CandidateBlockMeshRepairTurn` against the retained structured candidate. Native blockMesh failures use a dedicated `BlockMeshRepairTurn` against the exact structured mesh that generated the current file, eliminating whitespace-sensitive text patching as the primary mesh-topology repair path. Repeated blockMesh signatures normalize transient numeric labels before strategy escalation, and patch-based mesh mutations now count toward repair-cycle budgets.

Regression suite: **261 passed**. See `V3_1_CHANGES.md`.
