# OpenFOAM Agent v4.3.0

## Why this release exists

A live v4.2.3 run reached actual OpenFOAM case authoring successfully: eight required files were written and the deterministic execution plan advanced into validation. The first `foamDictionary` probe printed the `controlDict` keyword list (`application`, `solver`, `startFrom`, `endTime`, etc.) but the process wrapper returned timeout status 124. The workflow treated that tool outcome as proof that the CFD case was invalid, entered LLM repair, and a later Engineering context-budget failure became the final visible error.

The problem was architectural: validation-tool uncertainty, CFD case invalidity, infrastructure failure, security failure, and user-contract violation shared one boolean failure path. v4.3.0 separates those responsibilities.

## 1. Tri-state validation

Native validation observations are classified as:

- `pass`: the requested validator/consumer established acceptance;
- `fail`: explicit case-level evidence establishes invalidity, such as `FOAM FATAL ERROR`, `FOAM FATAL IO ERROR`, deterministic mesh/semantic rejection, or a real consumer failure classified as case-level;
- `inconclusive`: timeout, output/runner/infrastructure termination, or advisory validator non-zero status does not establish case invalidity.

`EngineeringEvent` stores the validation status and failure category.

## 2. Failure categories and routing

Failures are classified as:

- `case`: CFD/OpenFOAM/mesh semantic failure; bounded LLM repair remains possible;
- `tool`: validator/utility uncertainty; do not rewrite CFD files merely to satisfy a broken probe;
- `infra`: workflow/process/context/filesystem infrastructure; controller handles or blocks;
- `security`: execution/workspace policy violation; fail closed;
- `user_contract`: confirmed requirement violation; fail closed or require review.

Only `case` failures automatically remain repair-eligible. Tool/infra/security/user-contract failures do not enter blind CFD repair.

## 3. foamDictionary is advisory, not authority

`foamDictionary` is no longer a mandatory authoring or PreSolve permission gate. The default authoring path performs deterministic workspace/content and `FoamFile` checks without spawning a dictionary probe. An optional probe can still be enabled for diagnostics; timeout/nonfatal non-zero outcomes become `inconclusive/tool` and do not prove the case invalid. Explicit OpenFOAM fatal output is still a case failure.

## 4. Consumer-oriented PreSolve validation

PreSolve continues deterministic required-file and semantic validation, then can perform a bounded zero-step consumer initialization when safe:

1. verify serial topology and a bounded, symlink-free case input tree;
2. copy only `0/`, `constant/`, and `system/` into a temporary workspace shadow;
3. modify only the copied `controlDict` to `startFrom startTime`, `startTime 0`, `stopAt endTime`, `endTime 0`, bounded write settings, and `runTimeModifiable false`;
4. run the frozen selected execution driver/solver in a Python-owned validation execution context;
5. delete the shadow tree.

The real case is never modified and production solve approval is not granted. Non-empty function objects, parallel topology, symlinks, oversized input trees, or ambiguous controlDict literals cause the consumer probe to be skipped/inconclusive rather than guessed.

## 5. Mesh validation remains strict

`blockMesh`, `checkMesh`, mesh freshness, dependency invalidation, required-file coverage, case seals, execution approval, resource limits, completion and result freshness remain strict. Critical mesh commands that are inconclusive stop deterministically instead of allowing the workflow to assume mesh validity.

## 6. SafeRunner lifecycle fix

The streaming runner now treats primary process exit as authoritative. After exit it drains queued output for a bounded grace period rather than requiring stdout EOF indefinitely. This prevents a descendant that inherited stdout from making an already-exited parent look like a timeout. Reader shutdown is also bounded and tolerant of pipe close during cleanup.

## 7. Primary failure preservation

`CFDState` now keeps `primary_failure` and bounded `secondary_failures`. If a case-level failure is followed by a context/repair exception, the workflow terminal message explicitly preserves the primary failure and records the later exception as secondary. This prevents `ContextBudgetError` from hiding the real OpenFOAM diagnostic.

## 8. Runtime routing

Runtime applies the same classification. Explicit OpenFOAM case diagnostics remain repair-eligible. Timeout/tool/infrastructure uncertainty blocks with logs preserved and does not ask the Engineering LLM to change CFD files. Completion/output-evidence incompleteness remains a result-review path rather than case rewrite.

## Verification scope

The release adds v4.3 regression tests for tri-state native classification, optional dictionary-probe timeout behavior, SafeRunner descendant-pipe handling, zero-step shadow consumer validation, tool/infra routing, case repair eligibility, and primary/secondary failure preservation. Full regression and clean-patch/package equivalence are recorded in the v4.3.0 verification artifacts.

Actual end-to-end OpenFOAM 13 + Codex CLI execution in the user's sourced environment remains a live qualification step; this packaging environment does not claim that native CFD run.
