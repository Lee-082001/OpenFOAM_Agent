# v2.13.0 changes

## Evidence-gap-driven retrieval

- Added `gather_evidence` with 1-4 explicit evidence gaps per LLM turn.
- Each gap declares a stable `gap_id`, the missing external evidence, why it is required, bounded capability/reference queries, reference scope, and optional bounded top-reference reads.
- Prepare no longer exposes free-form capability/reference search actions in the compact production contract.
- Python batches deterministic retrieval, issues canonical evidence IDs, tracks per-gap novelty, and marks zero-information-gain gaps stagnant.
- Repeated stagnant gaps are refused without another filesystem search.
- `gather_evidence` counts as progress only when it actually issues new observed evidence.
- After the configurable prepare retrieval hard fuse (default 3 batches), the phase schema becomes `execute_case_plan | block`.

## Runtime repair schema hardening

- Added compact `repair_runtime_case`.
- Added `file_patches[].edits[]` so several ordered exact edits may target the same file in one runtime turn.
- Python expands grouped edits into the existing exact `patch_case_file` primitives; every `old` fragment must still match exactly once, sequentially.
- Runtime repair no longer needs to repeat the EngineeringPlan in its Structured Output; the approved solver/plan remains Python state.
- Runtime model context is now a bounded failure-focused slice: confirmed facts, approved-plan capsule, native diagnostic, mesh freshness, evidence-gap ledger, and relevant current case files.
- Runtime retrieval, when genuinely needed for release/tool syntax, uses the same evidence-gap batch mechanism.

## Compatibility and safety

- Legacy `repair_case_plan` now allows multiple sequential exact patches to the same file, but still rejects patch+replacement conflicts.
- Retained-candidate repair gets the same sequential-patch behavior.
- Transactional authoring, workspace sandboxing, exact capability/provider validation, pre-solve completeness, checkMesh freshness, CaseSeal and explicit `/solve` approval are unchanged.

## Tests

The release tree passes 183 regression tests, including new tests for batch evidence retrieval, novelty/stagnation cutoff, decision-only prepare schema after the retrieval fuse, same-file grouped runtime edits, failure-focused runtime context, and legacy same-file exact patches.
