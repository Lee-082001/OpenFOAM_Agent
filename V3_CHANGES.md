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
