# OpenFOAM Agent v2.11.0 changes

## Why this release exists

A complete `execute_case_plan` was previously expanded directly into sequential file writes. If a deterministic workspace rule rejected file N, files 1..N-1 remained in the active case while files N+1..end were never written. The next RepairTurn then treated that partial workspace as the baseline, causing missing-file cascades, unnecessary reference searches, repeated LLM calls, and large token growth.

## Transactional authoring preflight

- Raw case files and typed dictionaries are fully rendered before the first write.
- `CaseWorkspace.validate_candidate_bundle()` applies the same sandbox/content/library/per-file-size rules as real writes and checks the aggregate authored-size budget without mutating the workspace.
- If any candidate fails, a `case_bundle_preflight` observation is emitted and **no candidate file is written**.
- Typed-dictionary serialization failures follow the same pre-commit semantics.
- Once preflight passes, all case-file writes precede dictionary validation and native mesh commands, so native failures always operate on a complete authored bundle.

## Dedicated complete-plan retry contract

After a pre-commit authoring failure, compact Engineering switches to `CasePlanRetryTurn`. It permits only:

- a corrected complete `execute_case_plan`, or
- `block`.

Search/reference/read/delta-repair actions are deliberately excluded because there is no partial case to repair. The retry prompt tells the Engineering Agent to preserve unchanged engineering choices and correct only the diagnosed authoring defect while resubmitting the complete plan.

## Bounded authoring retries

Complete-plan serialization/preflight failures have a separate `max_case_plan_authoring_retries` policy (default 3). Repeated invalid bundles therefore cannot consume the full 12/24-turn Engineering budget or earn progress extensions through unrelated searches.

## Safety boundary

This release does not make CFD choices in Python and does not automatically relax the library/content allowlist. Python only verifies whether the Agent-authored complete bundle satisfies deterministic execution policy before committing it.

## Regression coverage

The release tree passes 177 tests, including regressions that verify:

- an unsafe `system/controlDict` in a later bundle position does not leave earlier files behind;
- the next model contract is `CasePlanRetryTurn`, not delta RepairTurn;
- a corrected complete second plan reaches `SOLVE_READY` with the normal mesh pipeline; and
- three repeated unsafe complete plans block without writing any candidate case file.
