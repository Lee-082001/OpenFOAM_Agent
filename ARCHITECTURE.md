# OpenFOAM Agent v2.4 Architecture

## 1. Authority invariant

v2.4 separates four kinds of authority:

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

Before `MESH_READY`, all execution inputs under `0/`, `constant/`, and `system/` are SHA-256 sealed, including native-generated `constant/polyMesh`. `EngineeringPlan` is separately hashed. `checkMesh` freshness uses a narrower mesh-only manifest: mesh-generation dictionaries, geometry inputs, and generated `constant/polyMesh`. Solver-control or initial-field edits remain covered by the full case seal but do not invalidate mesh evidence. `/solve` is legal only for the current sealed case.

## 6. Runtime and post-processing

`foamRun` failures return real stdout/stderr to the same engineering agent for bounded repair. Automatic repair cannot switch the already approved solver. It must re-establish `checkMesh` evidence only after mesh-affecting edits; solver-only changes use their own deterministic/native validation and preserve current mesh evidence.

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
- confirmed-fact provenance plus canonical capability/reference evidence-ID checks;
- Python-owned deterministic bindings for confirmed intake, current `checkMesh`, and case manifest (these are never LLM-authored evidence claims);
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

## Token-aware remote working context (v2.4)

The durable local audit state and the remote model working context are intentionally different representations.

- `CFDState.engineering_events`, runtime reports, logs, result files, and revision history remain complete local provenance subject to their existing storage bounds.
- The engineering model receives only 12 recent event projections with bounded excerpts plus compact cumulative capability/reference provenance.
- Post-processing does not receive the full runtime residual array. It receives total residual sample count, latest residual evidence per recent field, and a short recent tail.
- Result inventories are summarized and bounded before transmission; the full result tree remains available through explicit result-list/read tools.
- Feedback review receives a compact runtime report and bounded recent feedback history.
- Every model prompt passes a deterministic pre-API character budget. If the domain-specific projection is still unexpectedly large, a generic marked compaction pass reduces long strings/lists before an API call is allowed.

This compaction never authorizes an engineering claim. Final plan/evidence validation continues to inspect the original local EngineeringEvents, case hashes, native evidence, and confirmed intake rather than trusting compact summaries.

The CLI progress stream exposes request-size telemetry (`promptChars`, structured-schema characters, and a conservative tokenizer-free `approxTokens` estimate). The estimate is diagnostic only. When the provider returns usage metadata, a separate `LLM-USAGE` event exposes exact input/output/total token counts. The OpenAI adapter also receives an explicit CLI-default `max_output_tokens=16000`, preventing a small structured action from leaving the model's full output window uncapped.

## Role-based model routing (v2.6.0)

The workflow accepts either one legacy `StructuredLLM` or a `WorkflowLLMs` bundle. A single LLM is expanded uniformly for backward compatibility. Routed workflows bind intake, engineering, post-processing, and human-feedback review to independent model adapters. Runtime repair and confirmed case revision are intentionally not separate model roles: both continue through the same `CFDEngineeringAgent` and therefore inherit the engineering model.

Model routing changes reasoning capacity/cost allocation only. It does not grant additional tools, relax deterministic gates, change confirmed facts, or make model output execution evidence. The CLI resolves model names before workflow construction and records the resulting role map in reports/progress for auditability.

## Unified native failure observation (v2.5.0)

All failed allowlisted OpenFOAM executions now pass through one `NativeFailureDiagnostic` observation path rather than command-specific error handling. This covers engineering utilities (`foamDictionary`, `surfaceCheck`, `blockMesh`, `surfaceFeatureExtract`, `snappyHexMesh`, `createPatch`, `checkMesh`), runtime `foamRun`, and post-processing `foamPostProcess`.

The command layer preserves complete stdout/stderr in the private workspace log. A deterministic extractor then identifies the first OpenFOAM fatal/IO marker or common process-failure marker (abort, segmentation fault, floating-point exception); if no explicit marker exists, it returns a bounded output tail. The diagnostic records only the logical command name, return code, observed diagnostic kind, and bounded native excerpt. Local absolute paths are redacted before model/user display.

The same bounded diagnostic becomes the next relevant Agent observation and is rendered in normal CLI progress under the failed native action. It is descriptive evidence only: Python does not infer a CFD cause, choose a repair, change a mesh strategy, or authorize success from the diagnostic.


## Solve-readiness contract (v2.5.1)

A mesh-valid case is not necessarily solver-ready. The CLI path now separates `MESH_READY`, `PRE_SOLVE_VALIDATION`, and `SOLVE_READY`. The Engineering Agent declares solver-specific required inputs in `EngineeringPlan.required_case_files`. The deterministic layer checks file presence, dictionary syntax, and boundary-patch coverage without selecting fields, boundary conditions, numerics, or solver implementation. Only `SOLVE_READY` is eligible for interactive `/solve` approval.

## Local LLM provider boundary (v2.7.0)

All reasoning providers implement the same `StructuredLLM.generate(schema, prompt, system_prompt=...)` protocol. `CFDWorkflow`, Intake, Engineering, Runtime repair, PostProcessing, and Review do not branch on OpenAI versus Ollama. Provider selection and client construction are confined to the CLI/backend layer.

```text
OpenFOAM Agent roles
        |
        +-- StructuredLLM protocol
               |
               +-- OpenAILLM -> OpenAI Responses API
               |
               +-- OllamaLLM -> http://localhost:11434/v1/chat/completions
                                      |
                                      +-- user-created SSH local tunnel
                                      +-- mlfm4.knu.ac.kr:127.0.0.1:11434
```

The Ollama provider is not an Agent and does not acquire tool authority. All filesystem mutation, native OpenFOAM execution, evidence validation, safety gates, retries, and state transitions remain in the existing deterministic/agent orchestration layers.

Ollama endpoint configuration is loopback-only by construction. A startup `/v1/models` probe verifies the tunnel/service and requested models before workflow execution. Provider failure is terminal for that backend invocation; there is no implicit OpenAI fallback.
