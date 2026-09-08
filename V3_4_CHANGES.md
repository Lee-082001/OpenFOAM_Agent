# OpenFOAM Agent v3.4.0 changes

## Evidence-gap ledger authority

- Evidence-gap IDs are now treated strictly as opaque workflow metadata. The Engineering Agent decides what evidence is missing; Python owns the effective gap ID and refinement lifecycle.
- A self-refining model response such as `gap_id=G1086, refines_gap_id=G1086` no longer aborts Structured Output validation. If `G1086` already exists, Python preserves it as the parent and deterministically issues a fresh child ID.
- Different evidence requests that accidentally reuse the same ID in one batch are preserved as separate requests with reissued IDs instead of being semantically merged.
- Unknown refinement parents are cleared as protocol noise. The single-retrieval/stagnation hard fuse still applies to the effective ledger IDs.

## Branch-local structured validation

- Phase action unions remain transport-compatible plain `anyOf` schemas for Codex/Claude, but a Python pre-validator routes a payload to the branch named by `action.type` before normal union validation.
- A malformed `gather_evidence` action now reports only the actual `GatherEvidenceAction` error instead of additional `ReadCaseFileAction`, `ExecuteCasePlanAction`, and `BlockAction` noise.
- Backend schema compilation is unchanged: canonical Pydantic models still compile for Codex strict schemas and Claude tuple compatibility, then final output is revalidated by the canonical model.

## Installed-vs-documented OpenFOAM provenance

- `InstalledOpenFOAMIR` now accepts only `installed_source` components. Documented Foundation profiles can no longer be promoted to `verification_level=installed` accidentally.
- OpenFOAM source discovery now inspects bounded `addToRunTimeSelectionTable(...)` and `addNamedToRunTimeSelectionTable(...)` registrations under `$FOAM_SRC/fvModels` and `$FOAM_SRC/functionObjects`, in addition to coarse source-library discovery.
- This allows runtime-selectable types actually present in a sourced Foundation 13/14 tree to become installed evidence without adding Python type enums.
- The v13 capability graph had several copied v14 evidence-note labels; those notes are corrected to v13.

## Foundation 13/14 phase-change fallback evidence

- Both bundled capability graphs include documented `solidificationMelting` and `VoFSolidificationMelting` fvModels for solid/liquid melting and solidification workflows.
- Documented `heatTransferLimitedPhaseChange` and `coefficientPhaseChange` providers are also represented for fluid-fluid phase-change evidence.
- Documented fallback remains distinct from actual installation evidence and never grants executable authority.

## More semantic PreSolve field validation

- The remaining raw substring checks for `dimensions` and `internalField` were removed from PreSolve.
- A bounded top-level OpenFOAM entry projection now ignores comments, skips nested dictionaries, and treats dynamic directives/expansions as indeterminate rather than proving a required field entry missing.
- Native `foamDictionary` validation remains part of the solve-input gate; Python does not try to replace OpenFOAM's full dictionary evaluator.

## Model-call observability and bounded waiting

- Codex CLI and Claude Code model calls now emit a progress heartbeat while the blocking subprocess is still running, instead of appearing frozen for up to the full timeout.
- Added `--codex-timeout` and `--claude-timeout` CLI controls; defaults remain 900 seconds for backward compatibility.
- Heartbeat reporting is observational only and cannot change model-call success/failure semantics.

## Context-size hardening

- `environment_snapshot()` no longer serializes a status dictionary for every discovered OpenFOAM executable into every LLM turn.
- The normal model capsule carries the installation fingerprint, executable count, execution drivers, installed solver/fvModel semantic inventory, and a flag indicating that the full capability inventory is queryable through evidence retrieval.
- Arbitrary installed utilities remain discoverable through `CapabilityCatalog`; this reduces prompt growth without restoring a feature allowlist.

## Diagnostics and regression coverage

- A one-shot evidence-retrieval infrastructure failure now includes its bounded exception cause in the visible event summary before the phase retrieval fuse closes.
- Regression tests cover the original `G1086 -> G1086` crash, duplicate/unknown gap IDs, branch-local validation, OF13/14 runtime-selection discovery, installed/documented provenance separation, phase-change fallback, comment/directive-aware field semantics, and Codex/Claude wait heartbeats.
