# OpenFOAM Agent v2.3 Architecture

## 1. Authority invariant

v2.3 separates four kinds of authority:

```text
Agent engineering authority      -> what CFD design/repair/analysis is appropriate?
Deterministic execution authority -> may this exact file/command/case execute?
Human revision authority          -> may a reviewed sealed case be changed?
Human result authority            -> is this reviewed result accepted as COMPLETE?
```

Python must not turn semantic CFD labels into solver/mesh/BC recipes. The model must not turn its own prose into execution or scientific-success evidence.

## 2. State flow

```text
INIT -> INTAKE_ANALYSIS -> INTAKE_REVIEW_REQUIRED
                         user /confirm
                              |
                              v
                         ENGINEERING
                              |
                              v
                         MESH_READY
                   /feedback /       \ /solve
                           v           v
                    FEEDBACK_RECEIVED  SIMULATION
                           |             | failure -> RUNTIME_REPAIR -> SIMULATION
                           v             | success
                    REVISION_READY       v
                         /confirm    EXECUTION_DONE
                           |             |
                           v             v
                      ENGINEERING   POSTPROCESSING (optional)
                           |             |
                           +-> MESH_READY+-> RESULT_REVIEW_REQUIRED
                                             |             |
                                      /feedback             /accept
                                             v             v
                                      FEEDBACK_RECEIVED   COMPLETE
```

`EXECUTION_DONE` means deterministic runtime evidence passed. `RESULT_REVIEW_REQUIRED` means the result is available for engineering/human review. `COMPLETE` requires explicit `/accept`; it is not an automatic scientific-validation claim.

## 3. Intake and immutable user facts

`IntakeAgent` extracts supported facts and explicit exploratory authority. After the first `/confirm`, the intake digest is immutable. Every EngineeringPlan must cover confirmed non-context fact IDs and carry the exact intake digest. Human feedback cannot silently change a confirmed fact; feedback that explicitly changes Re, geometry, requested physics, or another confirmed user fact is routed back to intake review.

## 4. CFDEngineeringAgent

One production engineering agent owns solver, mesh, BC, normalization, numerics, motion, repair and case implementation. It uses capability/reference/file/native-tool actions. Preparation has a 120-action soft budget, progress-aware +20 extensions, and 200 hard cap by default, plus separate native-command, mesh-repair and runtime-repair budgets.

Human-confirmed revisions use the same engineering tools and safety gates but start a **fresh resource-accounting round**. Older EngineeringEvents remain available as provenance/evidence; they do not consume the new revision's action/native/mesh-repair budget.

## 5. Solve-input sealing

Before `MESH_READY`, all execution inputs under `0/`, `constant/`, and `system/` are SHA-256 sealed, including native-generated `constant/polyMesh`. `EngineeringPlan` is separately hashed. Any file write/delete invalidates current checkMesh evidence. `/solve` is legal only for the current sealed case.

## 6. Runtime and post-processing

`foamRun` failures return real stdout/stderr to the same engineering agent for bounded repair. Automatic repair cannot switch the already approved solver and must re-establish checkMesh evidence after input edits.

After runtime success, `CFDPostProcessingAgent` may write only `postprocessConfig/`, execute trusted `foamPostProcess`, read only numeric time directories and `postProcessing/`, and request deterministic analysis. Cd/Cl/f/St values come from parsed native files and executed dictionary hashes, never model prose. The Agent may additionally report advisory scientific confidence/review focus; this is explicitly non-deterministic and cannot transition to COMPLETE.

## 7. Human-feedback review boundary

`/feedback` at `MESH_READY` or `RESULT_REVIEW_REQUIRED` creates `HumanFeedback` with:

- immutable human statement;
- scope (`mesh` or `result`);
- run/state provenance;
- SHA-256 of the evidence snapshot that was visible when feedback was submitted.

`CFDFeedbackReviewAgent` receives that feedback plus confirmed intake, plan, mesh/runtime/post-processing evidence and case manifest. It returns hypotheses and an advisory `RevisionProposal`; it has **no write or native execution tools**.

A proposal is cryptographically bound to the exact baseline plan digest and case-manifest digest. If either changes before confirmation, revision is rejected. `REVISION_READY` therefore means "proposal ready, case still unchanged." `/reject` can return to the unchanged prior review state; the proposal remains in audit history.

## 8. Human revision confirmation

A second `/confirm` on `REVISION_READY` authorizes a new engineering round. Before any active-case edit, the previous results are moved out of the active case into:

```text
revision-history/rev-XXXX/
  baseline_inputs/
    0/
    constant/
    system/
  case_outputs/
    <numeric time dirs>/
    postProcessing/
    postprocessConfig/
  logs/
```

The archive is private. `baseline_inputs/` preserves the exact pre-revision solver inputs for rollback/comparison, while moved outputs prevent stale evidence from contaminating the new run. The old solve/post-processing reports are removed from the active state before re-engineering.

If the proposal requires a case revision, `finish_preview` is deterministically rejected when the current solver-input manifest is unchanged. A successful revised seal creates `RevisionRecord` with before/after plan hashes, before/after manifest hashes, archive path, linked feedback IDs, and added/removed/modified file hashes. Linked feedback moves to `awaiting_rerun`.

The revised case returns to `MESH_READY`; a fresh `/solve` is mandatory. When the new result reaches `RESULT_REVIEW_REQUIRED`, linked feedback becomes `awaiting_review`. Only `/accept` marks it resolved and transitions to `COMPLETE`; another `/feedback` starts another revision cycle.

## 9. Deterministic safety responsibilities

Python owns only generic authority/integrity/resource rules:

- sandbox/path validation and private workspace permissions;
- executable directive/dynamic-code/library blocking;
- trusted OpenFOAM executable provenance and sanitized subprocess environment;
- confirmed-fact and capability/reference/tool provenance checks;
- solver/controlDict consistency;
- native syntax/mesh evidence;
- current input SHA-256 sealing;
- feedback proposal baseline binding and revision diffing;
- stale-result archive isolation;
- step/native/retry/time/file/cell limits;
- `/confirm`, `/solve`, `/feedback`, `/accept` state permissions.

It does not decide that a particular physical problem requires a particular solver, mesh strategy, discretization, turbulence model, or corrective action.

## 10. Success semantics

```text
MESH_READY              current sealed case + current passing checkMesh
EXECUTION_DONE          bounded foamRun evidence passed
RESULT_REVIEW_REQUIRED  result/post-processing evidence ready for review
COMPLETE                human explicitly accepted the reviewed result
```

None of these states, including COMPLETE, is a universal proof of mesh independence, time-step independence, reference agreement, experimental validation, or physical correctness.

## Progress event boundary (v2.3)

Long-running work emits observational `ProgressEvent` objects through a shared reporter injected into `CFDWorkflow`, `CFDEngineeringAgent`, `RuntimeOrchestrator`, `CFDPostProcessingAgent`, and `CFDFeedbackReviewAgent`. This is deliberately outside the engineering decision authority.

The progress channel may expose only deterministic/action-level observations such as phase, bounded step number, action type, relative case file, allowlisted command, tool success/failure, `checkMesh` metrics, solver `Time`/Courant summaries, retry counts, and post-processing metrics already derived from native evidence. It must not expose the model `rationale`, hidden reasoning, API credentials, or new unredacted host-file contents.

`CLIProgressReporter` writes to stderr. `quiet` drops events, `normal` filters low-value read/list actions and throttles solver time markers, and `verbose` renders all action events and enables raw `foamRun` stdout. Reporter/callback exceptions are observational failures only: `SafeRunner` catches output-callback exceptions so a renderer cannot terminate or alter a native OpenFOAM process.
