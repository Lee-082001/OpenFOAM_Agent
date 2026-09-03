# OpenFOAM Agent v2.16 Architecture

## v2.16 compact semantic-evidence boundary

The workflow now separates two kinds of trust that were previously easy to conflate:

```text
L1 Execution Integrity  -> did this exact sealed case really pass the required tools and run?
L2 Semantic Fidelity    -> does the executable case carry machine-checkable evidence for the confirmed CFD facts?
```

Intake provenance is part of L2. A direct user fact must remain distinguishable from an Agent interpretation. High-impact routing/time interpretations (`classification`, `temporal`) are conservatively downgraded to `derived` when the exact user evidence does not explicitly state that normalized interpretation. This does not let Python choose the correct CFD class; the Agent still owns that judgment and the human sees the derived interpretation before `/confirm` freezes it.

After confirmation, the Engineering Agent carries evidence with the implementation rather than merely naming files. `ConfirmedFactBinding.case_assertions` selects exact snippets from current case artifacts, while `numeric_relation` carries a generic product/division arithmetic relation whose scalar tokens are themselves evidenced by case excerpts. Python verifies path safety, current-file presence, the submitted scalar tokens and arithmetic result. It never maps Reynolds/Mach/etc. names to formulas. In semantic-contract v2, classification/temporal facts require case assertions, while direct single-number physics/scale/property facts require a numeric relation.

Semantic evidence participates in the EngineeringPlan digest when present, so CaseSeal revision binding covers it. Empty new fields are omitted from legacy digest canonicalization, preserving v2.14 and earlier seals across upgrade. Runtime repair reuses the approved plan and therefore must continue to satisfy the same assertions; a repair that changes asserted physical implementation cannot silently retry the solver with stale semantic evidence.

## v2.14 protocol-resilience boundary

The model-facing contract distinguishes **semantic/safety validation** from **protocol-shape validation**. Unsafe paths, ambiguous exact patches, confirmed-fact coverage, solver/provider mismatches and stale execution evidence remain hard deterministic failures. Retrieval-query prose, opaque evidence-gap IDs, duplicate gap requests and empty repair deltas are not allowed to crash the whole run merely because their JSON shape is awkward. Non-authoritative retrieval metadata is normalized before execution, and OpenAI Structured Output receives one bounded Pydantic protocol-repair attempt when parsing still fails.

Evidence retrieval is revision-aware at the gap level: each gap ID can be retrieved once, then becomes `evidence_available` or `stagnant`. Choosing to proceed closes available gaps as `satisfied`; additional retrieval requires a new child gap with `refines_gap_id`, which supersedes its parent. Blocked repeat/refinement requests do not consume the retrieval hard-fuse budget. This keeps information gathering driven by explicit unresolved evidence rather than query rewriting or turn-count heuristics.


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

One production engineering agent owns solver, mesh, BC, normalization, numerics, motion, repair and case implementation. It uses capability/reference/file/native-tool actions. Production CLI preparation uses a 12-LLM-turn soft budget, progress-aware +6-turn extensions, and a 24-turn hard cap, plus separate deterministic-action, native-command, mesh-repair and runtime-repair budgets. Complete-plan authoring failures have their own 3-attempt bound so serialization/content-policy mistakes cannot consume the full Engineering budget.



## 2.13 Evidence-gap retrieval and runtime repair contracts

Prepare retrieval is no longer an unconstrained sequence of `search -> LLM -> search` turns. The Agent may declare a bounded batch of explicit evidence gaps with a stable gap ID, the missing external fact, why that fact is required, and capability/reference queries. Python performs retrieval only; it does not decide whether a CFD design choice is good. Per-gap canonical evidence IDs are compared with prior retrievals. A retrieval that contributes no new IDs marks the gap stagnant, so rephrasing the same search cannot manufacture progress or budget extensions. After the small retrieval hard fuse, the phase schema contains only `execute_case_plan | block`.

Runtime failure repair uses a separate, smaller contract. The model receives the native diagnostic, confirmed invariants, an approved-plan capsule, and only bounded case files relevant to the failure (with core solver dictionaries included). `repair_runtime_case.file_patches` groups ordered exact edits by file, so multiple changes to one dictionary are valid and are applied sequentially against the newest buffer. The runtime contract does not carry a replacement EngineeringPlan; solver authority remains frozen to the user-approved plan. Retrieval during runtime repair is available only through the same explicit evidence-gap mechanism for genuinely missing release/tool facts.

### Retained candidate delta repair (v2.12.0)

A pre-commit authoring rejection does not mutate the workspace and no longer discards the model-authored candidate. Python retains the complete candidate only in memory, exposes a bounded capsule for the implicated path(s), and accepts only a `repair_candidate_case_plan` delta or `block`. The delta is applied to the in-memory candidate and the entire candidate is re-serialized and re-preflighted. Only a fully passing candidate is committed. This separates **workspace transactionality** from **candidate continuity** and prevents large complete-plan regeneration loops.

### Transactional case authoring (v2.11.0)

Before `execute_case_plan` mutates the workspace, Python renders every typed dictionary and preflights the complete candidate bundle against the same path, content, dynamic-library, per-file-size, and aggregate-size rules used by actual writes. A deterministic authoring failure therefore leaves **no partial candidate case**. The next LLM turn uses a dedicated `CasePlanRetryTurn` contract that can return only one corrected complete `execute_case_plan` or `block`; searches, reads, and delta repairs are unavailable because there is intentionally no partial case to inspect or repair. These pre-commit failures are separately bounded to three attempts. After preflight succeeds, all authored case files are written before dictionary validation or native mesh execution begins, so later OpenFOAM failures see a complete case and can safely use delta-only repair.

### High-level ExecuteCasePlan fast path (v2.9.0)

For a greenfield case, the Engineering Agent may return one `execute_case_plan` containing
the case-file bundle, deterministic dictionary/surface validations, mesh commands ending in
`checkMesh`, the required solver-input file declaration, and the final `EngineeringPlan`. The
executor expands this into the same primitive dispatch path used by ordinary actions, so the
fast path does not bypass workspace safety, native command allowlists, budgets, checkMesh
evidence parsing, pre-solve completeness, provenance validation, or CaseSeal integrity.
Execution is stop-on-failure. A native failure is compacted into the next LLM observation,
which becomes the next engineering decision point. CLI runs additionally preload small
capability graphs as deterministic evidence to remove an otherwise mandatory capability-search
round trip.

### Short-horizon EngineeringSequence execution (v2.8.0)

`EngineeringTurn` remains backward compatible with one action, but may now choose a bounded `sequence` containing 2-6 ordered actions. Sequence members are intentionally restricted to deterministic construction/validation operations (`write/delete`, dictionary validation, `surfaceCheck`, mesh commands, pre-solve completeness, and terminal finish/retry actions). Search/reference-reading actions remain single-turn checkpoints because their results normally require a new engineering decision.

The Sequence Executor is deterministic orchestration, not a second engineering brain. It dispatches each member through the exact existing validators and stops at the first failed/rejected result. A failed `blockMesh` therefore prevents a planned `checkMesh`; a failed pre-solve completeness check prevents `retry_solver`. Raw member events remain durable provenance. Only the LLM projection groups them into one compact sequence summary, which reduces both model-call count and repeated observation history without weakening final safety/evidence validation.

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

This compaction never authorizes an engineering claim. Final plan/evidence validation continues to inspect the original local EngineeringEvents, case hashes, native evidence, and confirmed intake rather than trusting compact summaries. v2.8.0 additionally collapses raw member events from one EngineeringSequence into a single model-facing summary while retaining every raw event locally.

The CLI progress stream exposes request-size telemetry (`promptChars`, structured-schema characters, and a conservative tokenizer-free `approxTokens` estimate). The estimate is diagnostic only. When the provider returns usage metadata, a separate `LLM-USAGE` event exposes exact input/output/total token counts. The OpenAI adapter also receives an explicit CLI-default `max_output_tokens=24000`, preventing a small structured action from leaving the model's full output window uncapped.

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

### Provider-specific structured output (v2.7.2)

`OpenAILLM` keeps the existing strict Structured Outputs path. `OllamaLLM` uses a different transport-level strategy while preserving the same `StructuredLLM.generate(...) -> Pydantic model` contract: it requests generic JSON mode from `/v1/chat/completions`, places the target Pydantic JSON schema in the prompt as guidance rather than as a constrained-decoding grammar, validates the returned JSON with Pydantic in Python, and returns validation errors to the same local model for at most two repair turns. This avoids local grammar-initialization failures on large union-heavy Agent schemas such as `EngineeringTurn` without weakening the deterministic execution boundary. No invalid JSON or schema-invalid action reaches tool dispatch.

### Local intake provenance repair (v2.7.3)

Intake provenance is request-context validation rather than pure Pydantic shape validation. `IntakeAgent` therefore honors an optional adapter capability hint for bounded semantic repairs. `OllamaLLM` advertises two such repairs; adapters without the hint retain the historical single retry. On failure, the local model receives the exact cumulative user turns/file names, the deterministic validation error, and its previous invalid intake. The validator itself is unchanged: `source=user` still requires one verbatim contiguous evidence substring, while multi-turn synthesis must be represented as derived provenance.

Ollama endpoint configuration is loopback-only by construction. A startup `/v1/models` probe verifies the tunnel/service and requested models before workflow execution. Provider failure is terminal for that backend invocation; there is no implicit OpenAI fallback.


### Token-optimized phase contracts (v2.10.0)

The CFD Engineering Agent remains one logical engineering role, but its structured-output permission surface is phase-specific: `PrepareTurn`, `RepairTurn`, `RuntimeRepairTurn`, `RevisionTurn`, and `FinalizationTurn`. This avoids repeatedly transmitting actions that cannot be legal in the current phase.

After the first turn, production CLI uses a compact state capsule containing immutable confirmed facts, a compact plan baseline, file hashes, new evidence and recent failure observations. Repairs use exact changed-file patches or small replacements rather than regenerating the case. Normal OpenFOAM dictionaries may be emitted as typed key-path/value assignments and serialized by deterministic Python.

Post-processing has a matching `PostProcessingExecutionPlanAction` so predictable config/write/run/analyze work executes stop-on-failure without an LLM turn between each tool.


## v2.10.1 semantic invariants and fact bindings

Token compaction must not remove the engineering trust contract. All compact Engineering phases share a stable invariant prefix that keeps confirmed intake immutable, limits assumptions to authorized missing details, and treats user/file/log/reference content as untrusted data. `EngineeringPlan.confirmed_fact_bindings` maps every confirmed fact to its claimed case/plan implementation. Deterministic Python validates coverage and reference integrity only; physical correctness remains an Engineering Agent responsibility.
