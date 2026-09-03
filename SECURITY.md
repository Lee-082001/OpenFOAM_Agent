# v2.4 Security and Trust Boundary

## Local Ollama transport boundary (v2.7)

The Ollama backend accepts only loopback base URLs (`localhost`, `127.0.0.1`, `::1`). The Agent never opens, binds, or exposes TCP port 11434 and never connects directly to `mlfm4.knu.ac.kr:11434`. The user establishes SSH local forwarding separately. `--base-url` exists for local tunnel/port customization, not for bypassing the loopback boundary.

Ollama uses the OpenAI-compatible API with a dummy `ollama` key. Failure to reach `/v1/models`, absence of a requested model, or a later connection loss fails closed. No request is silently rerouted to the OpenAI cloud backend.

For structured Agent turns, Ollama uses generic JSON mode rather than compiling the full Agent Pydantic schema into a local decoding grammar. The returned text is still untrusted model output: Python/Pydantic must validate it successfully before the existing action schema, workspace safety, command allowlist, evidence, sealing, approval, and native-execution gates can act on it. Failed validation is returned only as a bounded repair prompt to the same Ollama backend; after two repair turns the call fails closed.

Ollama multi-turn Intake additionally receives up to two semantic provenance repairs. These repairs do not relax the trust boundary: a `source=user` fact still requires verbatim evidence from exactly one supplied user turn/file name, explicit user numbers must remain represented, and a model that continues to fabricate or paraphrase evidence fails closed. The extra turns only provide the local model with clearer deterministic error/evidence context.

## Cloud disclosure boundary

With `--backend openai --confirm-api-calls`, confirmed CFD task data, bounded tool observations, post-processing evidence, and human feedback/revision-review payloads are intentionally sent to the configured OpenAI model. API keys are never inserted into prompts; known local paths are redacted and `store=False` is used by default. Do not use the cloud backend for data that policy forbids leaving the host; a local model backend would be required for a zero-egress deployment.

## Native process boundary

The model never receives a shell. `SafeRunner` resolves allowlisted executables inside trusted `WM_PROJECT_DIR`, sanitizes environment variables (including API secrets and user/site OpenFOAM override paths), restricts cwd to the private workspace, and applies timeouts. Allowed tools are limited to OpenFOAM preparation/validation, `foamRun`, `foamPostProcess`, and `foamDictionary` actions exposed by typed interfaces.

## Role-routed cloud models (v2.6)

Different workflow roles may use different OpenAI model IDs, but every routed model remains behind the same cloud-call authorization, context compaction, local-path redaction, typed Structured Outputs, tool allowlists, case-integrity checks, and approval gates. Routing is not an authority boundary: selecting a stronger engineering model cannot bypass deterministic validation or execute arbitrary commands. Reports record the resolved role map so mixed-model runs remain auditable.

## Native diagnostic disclosure boundary (v2.5)

Failed allowlisted native commands preserve their complete stdout/stderr only in private workspace logs. The cloud/user-facing `NativeFailureDiagnostic` is bounded and local absolute paths are redacted before transmission/display. The extractor classifies only observable failure markers and never maps them to a repair policy; diagnostic text is untrusted tool output and cannot bypass case, provenance, approval, or resource gates.

## Case-input integrity

Agent writes are sandboxed. Traversal, absolute paths, symlinked execution inputs, NUL content, executable directives (`#codeStream`, `#calc`, runtime coded constructs, `system(...)`, etc.) and non-allowlisted libraries are rejected. Before solve, every input under `0/`, `constant/`, and `system/` is SHA-256 sealed. `MESH_READY` additionally requires current trusted `checkMesh` evidence bound to a mesh-only manifest covering mesh-generation dictionaries, geometry inputs, and generated `constant/polyMesh`; non-mesh solver inputs remain protected by the full case seal and their own validators.

## Confirmed-fact semantic-drift boundary (v2.10.1)

Compact prompts retain a shared invariant that confirmed intake values must be implemented faithfully in the actual case, not merely preserved as fact IDs or metadata. `EngineeringPlan.confirmed_fact_bindings` gives every non-context confirmed fact an auditable implementation mapping. Python verifies binding coverage and referenced-file existence but deliberately does not decide CFD semantics (for example, it does not recompute Reynolds number or infer whether a BC is physically appropriate). If faithful implementation would require changing confirmed intake, the Agent is instructed to block and return to human review rather than silently changing the problem.

## Evidence-retrieval anti-thrash boundary (v2.14)

Prepare/runtime retrieval uses explicit evidence-gap IDs rather than arbitrary repeated search actions. Each gap ID is single-use. After retrieval it becomes `evidence_available` or `stagnant`; proceeding to execution marks available gaps `satisfied`. Additional retrieval must use a new more-specific gap with `refines_gap_id`, which supersedes the parent. Blocked repeats do not consume a retrieval cycle, and a small phase-specific hard fuse eventually removes retrieval from the Structured Output contract entirely. Query/ID normalization is permitted only for non-authoritative retrieval protocol metadata; Python still does not decide CFD engineering choices or infer whether external evidence is scientifically sufficient.

A bounded OpenAI protocol-repair attempt may correct a Pydantic/JSON shape failure before the workflow is failed. The correction prompt explicitly preserves confirmed CFD facts. This is not a safety bypass: deterministic workspace, provenance, solver/provider, semantic-binding, pre-solve, case-seal, native-evidence and human-approval gates run exactly as before after a valid object is obtained.

Runtime exact edits are grouped per file and applied in model-specified order. Each edit still requires its `old` fragment to occur exactly once in the current buffer at the moment it is applied; grouping does not permit fuzzy matching, shell execution, arbitrary paths, or solver changes.

## Runtime self-deception resistance

Runtime success is not accepted from model prose. It requires a successful native return status, at least one parsed `Time = ...` record (including OF13 unit-suffixed forms), an `End` marker, and no fatal/non-finite/SIGFPE evidence. Failed logs are preserved and returned to the engineering agent as observations.

## Post-processing integrity

Post-processing cannot modify sealed solve inputs. It writes only `postprocessConfig/`, and every executed dictionary is hash-bound before `foamPostProcess`. Native result reads are restricted to numeric time roots and `postProcessing/`. Deterministic force metrics are bound to both result-file and dictionary hashes and are discarded if either changes. Agent-provided `scientific_confidence` is advisory only and never changes deterministic execution evidence or human-review state.

## Human-feedback and revision integrity

Human feedback is stored separately from the immutable intake. The Review Agent has no file-write or native-execution tools. Each revision proposal is bound to the exact EngineeringPlan digest and case-manifest digest it reviewed. Any out-of-band plan/input change before `/confirm` blocks revision.

A required case revision cannot pass finalization with an unchanged solver-input manifest. On successful revision, deterministic code records before/after hashes and per-file diff. The revised case must pass current validation; a fresh `checkMesh` is required only when mesh-affecting artifacts changed, and every revision still requires a fresh `/solve`.
If automated feedback assessment fails, the workflow returns to the originating review gate with the human observation retained as unresolved provenance. If an unexpected failure occurs after a confirmed revision has begun, the workflow is marked `ENGINEERING_BLOCKED` and reports the private baseline/output archive path rather than pretending the partial revision is usable.

Before a confirmed revision starts, exact pre-revision execution inputs are copied to private `revision-history/rev-XXXX/baseline_inputs/`; previous numeric time outputs, `postProcessing/`, post-process configuration and logs are moved into the same archive. This prevents stale evidence from being mistaken for the new run while preserving rollback/audit/comparison material.

Feedback that changes a confirmed user fact must return to intake review instead of silently rewriting the confirmed problem definition.

## Human acceptance boundary

Solver/post-processing success ends at `RESULT_REVIEW_REQUIRED`. Only explicit `/accept` transitions to `COMPLETE`. `/feedback` instead opens a reviewed revision route, and `/reject` can discard an unconfirmed proposal without changing the sealed case. A plain natural-language prompt at a pending mesh/result/revision gate is not silently reinterpreted as a new intake; the CLI asks the user to use `/feedback`, `/accept`, `/solve`, `/confirm`, or `/new` as appropriate.

`COMPLETE` means the human accepted the reviewed result. It does **not** prove scientific truth, validation against experiment, mesh/time-step independence, or that an LLM diagnosis was correct.

## Resource exhaustion

Engineering LLM turns, deterministic engineering actions, native commands, mesh repair, runtime repair, solver attempts, post-processing actions and post-processing commands have independent bounds. Human-confirmed revisions receive fresh per-round engineering/native/mesh-repair budgets while old events remain audit provenance. A v2.8 EngineeringSequence does not bypass any bound: each member is dispatched through the same safety checks and the sequence stops on the first rejected/failed member. This avoids penalizing legitimate iterative review without allowing one autonomous round to run indefinitely.

## Residual risks

- A cloud model necessarily receives the task/feedback content intentionally sent to it.
- OpenFOAM itself and the host installation are trusted dependencies; a compromised trusted installation is outside the model sandbox.
- Deterministic gates prove authority/integrity/execution evidence, not CFD scientific correctness.
- Human feedback and LLM diagnoses can be mistaken; hypotheses must be checked with new tool/result evidence.
- `COMPLETE` is an explicit human workflow decision, not a scientific-certification primitive.

## Progress output is observational, not an authority channel

The v2.3 live progress stream cannot authorize case edits, commands, solver runs, feedback revisions, or acceptance. It receives already-selected action metadata and deterministic tool/runtime observations only. Model `rationale` fields and hidden reasoning are intentionally not rendered.

Progress is emitted to stderr by the CLI so JSON reports on stdout are not corrupted. Raw `foamRun` stdout is enabled only by `--progress verbose`; `normal` mode emits bounded `Time`/Courant summaries while the complete native log remains in the private workspace log directory. Output callbacks are wrapped so callback exceptions cannot kill the OpenFOAM subprocess or change its return status.

## Model-context resource boundary (v2.4)

Cloud payload size is now independently bounded from local provenance size. Engineering, post-processing, and feedback-review prompts have deterministic pre-API character ceilings and high-volume histories are projected into bounded summaries before transmission. Full runtime residual arrays are never copied into post-processing/review prompts; the complete local `RuntimeReport` remains available for deterministic code and audit.

The generic fallback compactor explicitly marks omitted string/list content. It is not an authority mechanism: a compacted model view cannot satisfy deterministic provenance checks that require original observed events or current file/native evidence.

The CLI configures an explicit OpenAI `max_output_tokens` ceiling (default 16000). This is a resource/rate-limit control only; it does not change which CFD decisions the Agent owns. If a future unusually large single structured action legitimately needs more output, the user can raise the CLI cap deliberately.

## Canonical engineering evidence

Engineering evidence authority is not delegated to model prose. Successful capability/reference tool observations receive deterministic opaque IDs (`ev_cap_*`, `ev_ref_*`) and are stored on local `EngineeringEvent` records. The cloud model receives a bounded `available_evidence` projection and may only select those IDs; finalization rejects any ID absent from the local registry. User facts, confirmed-intake binding, `checkMesh`, and case-manifest integrity are Python-owned deterministic checks and are intentionally excluded from model-selectable evidence.

## v2.10 repair/serializer boundary

Delta repair does not introduce fuzzy edits. `CaseFilePatch` is accepted only when its `old` fragment matches exactly once, and the resulting file is re-written through the existing workspace content/path safety checks. Typed OpenFOAM dictionaries do not choose engineering values; they only move brace/semicolon rendering into deterministic Python and the serialized content is still passed through the same workspace safety validation.
