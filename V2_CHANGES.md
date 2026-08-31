# v2.7.2

## Ollama JSON-mode structured repair

- Changes only the Ollama/local structured-output transport; the OpenAI strict Structured Outputs path is unchanged.
- Replaces Ollama `chat.completions.parse(..., response_format=PydanticModel)` with OpenAI-compatible JSON mode via `chat.completions.create(..., response_format={"type":"json_object"})`, avoiding constrained-decoding grammar compilation for large Agent schemas.
- Supplies the target Pydantic JSON schema in the local-model prompt as guidance, then treats Python `model_validate_json()` as the authoritative structured-output validator.
- Adds two bounded structured-repair turns (three total local generation attempts) that return deterministic Pydantic validation errors to the same Ollama model; invalid output never reaches Agent tool dispatch.
- Aggregates Ollama token usage across internal repair attempts so existing usage reporting reflects the full local-model cost.
- Adds regression coverage proving `EngineeringTurn` does not use Ollama grammar parsing, invalid JSON/schema output is repaired, retries are bounded, connection failures still do not fall back to OpenAI, and the existing OpenAI/backend routing behavior remains intact.
- Synchronizes package/module/CLI/report metadata at 2.7.2.

# v2.7.1

## Mesh-scoped checkMesh freshness

- Separates the full execution-input manifest from a narrower mesh-only manifest used to decide whether passing `checkMesh` evidence is still current.
- Mesh freshness tracks the allowlisted mesh pipeline: `system/blockMeshDict`, `system/surfaceFeatureExtractDict`, `system/snappyHexMeshDict`, `system/createPatchDict`, geometry inputs under `constant/` (including `constant/triSurface`), and generated `constant/polyMesh`.
- Editing/deleting `fvSchemes`, `fvSolution`, `controlDict`, initial fields under `0/`, physical-property dictionaries, or other non-mesh solver inputs no longer clears passing mesh evidence or forces redundant `checkMesh`.
- Full case sealing remains unchanged: every execution input under `0/`, `constant/`, and `system/` is still SHA-256 bound before solve/retry.
- Runtime-repair rehydration restores mesh freshness only after verifying the persisted full case seal and persisted passing mesh evidence.
- Engineering/revision/runtime prompts now instruct the Agent to validate changed solver inputs appropriately and rerun `checkMesh` only after mesh-affecting changes.
- Adds regression coverage for solver-only edits preserving mesh evidence, mesh edits invalidating it, runtime retry without redundant `checkMesh`, and rehydrated runtime repair.
- Synchronizes package/module/CLI metadata at 2.7.1.

# v2.7.0

## Local Ollama backend over SSH tunnel

- Adds `--backend ollama` while preserving the existing OpenAI and offline rule-based intake backends.
- Adds `OllamaLLM`, which implements the existing `StructuredLLM` protocol using Ollama's OpenAI-compatible `/v1/chat/completions` structured-output path; upper Agent/workflow code remains backend-agnostic.
- Defaults to `http://localhost:11434/v1`, model `gemma4:31b`, and dummy API key `ollama`; supports `--model`/`OLLAMA_MODEL`, `--base-url`/`OLLAMA_BASE_URL`, and role-specific `OLLAMA_*_MODEL` overrides.
- Enforces loopback-only Ollama URLs so the remote GPU server is reached only through user-managed SSH local port forwarding; direct `mlfm4.knu.ac.kr:11434` and `0.0.0.0` endpoints are rejected.
- Adds a startup `/v1/models` health/model check with a clear SSH-tunnel diagnostic and no automatic fallback to OpenAI.
- Runtime Ollama connection failures retain the same no-fallback behavior and surface a tunnel/service error.
- Preserves v2.6 role routing: runtime repair and confirmed revision engineering continue to use the engineering model when the Ollama backend is selected.
- Adds mock-HTTP health-check regressions plus structured-adapter, routing, loopback-security, connection-failure, and no-fallback tests.
- Synchronizes package/module/report metadata at 2.7.0.

# v2.6.0

## Role-based model routing

- Adds `WorkflowLLMs` with independent `intake`, `engineering`, `postprocessing`, and `review` structured-LLM roles while preserving the legacy single-LLM API through automatic uniform routing.
- Adds CLI overrides `--intake-model`, `--engineering-model`, `--postprocess-model`, and `--review-model`; `--model` remains the global backward-compatible default.
- Supports role environment variables `OPENAI_INTAKE_MODEL`, `OPENAI_ENGINEERING_MODEL`, `OPENAI_POSTPROCESS_MODEL`, and `OPENAI_REVIEW_MODEL`, with deterministic precedence: role CLI > role env > `--model` > `OPENAI_MODEL`.
- Runtime repair and confirmed revision engineering deliberately reuse the engineering route so one engineering model owns the full design/repair thread.
- Reuses one `OpenAILLM` adapter per unique resolved model name instead of constructing duplicate adapters when multiple roles share a model.
- Exposes resolved role routing in interactive startup output, JSON reports (`model_routes`), and engineering/postprocess/review `LLM-CONTEXT` and `LLM-USAGE` metrics.
- Adds routing/backward-compatibility regression tests and synchronizes package/module metadata at 2.6.0.

# v2.5.1

## Pre-solve completeness gate

- Adds `MESH_READY -> PRE_SOLVE_VALIDATION -> SOLVE_READY` to the interactive CLI path. `/solve` is accepted only from `SOLVE_READY`.
- The Engineering Agent now declares solver-specific `required_case_files`; Python validates existence instead of hard-coding solver field choices.
- Deterministic pre-solve validation checks `system/controlDict`, `system/fvSchemes`, `system/fvSolution`, every Agent-declared required input, OpenFOAM dictionary parsing, and mesh-patch coverage for declared `0/*` fields.
- Missing files such as `fvSchemes`, `fvSolution`, `0/p`, or missing patchField entries are returned to the Engineering Agent before any `foamRun` attempt.
- Fixes native diagnostic classification so the normal `sigFpe : Enabling floating point exception trapping` startup banner cannot mask a later `FOAM FATAL ERROR`.
- Direct low-level API compatibility retains legacy `MESH_READY` approval when the new CLI policy is not enabled; the shipped CLI enables the solve-ready gate by default.
- Version metadata synchronized at 2.5.1.

# v2.5.0

- Generalizes v2.4.2 `blockMesh`-only failure extraction into a shared `NativeFailureDiagnostic` path for all allowlisted OpenFOAM execution stages.
- Engineering failures from `foamDictionary`, `surfaceCheck`, `blockMesh`, `surfaceFeatureExtract`, `snappyHexMesh`, `createPatch`, and `checkMesh` now surface bounded native diagnostics to both CLI progress and the next Engineering Agent turn.
- Runtime `foamRun` failures now render the same diagnostic to the user and return the bounded fatal observation to runtime repair while preserving the complete solver log locally.
- `foamPostProcess` failures now render the same diagnostic and become the next PostProcessing Agent observation.
- `foamDictionary` and `surfaceCheck` now persist complete stdout/stderr logs alongside mesh/runtime/post-processing logs.
- Diagnostic classification recognizes OpenFOAM fatal/IO errors, aborts, segmentation faults, floating-point exceptions, and a bounded output-tail fallback without prescribing a repair.
- Adds path-redaction and regression coverage for `snappyHexMesh`, dictionary/surface validation, `foamRun`, and `foamPostProcess` failure propagation.
- Synchronizes package/module metadata at 2.5.0.

# v2.4.2

- `blockMesh` failures now preserve the complete native stdout/stderr in the workspace log while extracting a bounded raw fatal diagnostic for the agent observation and CLI progress.
- Fatal extraction recognizes OpenFOAM fatal/IO errors plus common abort, segmentation-fault, and floating-point-exception markers; when no explicit marker exists, a bounded output tail is used.
- Live progress now prints the redacted fatal diagnostic under the failed `blockMesh` event instead of exposing only `returnCode`. Local absolute paths remain redacted from model/user-visible observations.
- Added regression coverage proving the next Engineering Agent turn receives the fatal block and the full raw log remains available locally.

# v2.4.1

- Replaces free-form EngineeringPlan evidence claims (`kind/reference` strings) with canonical opaque evidence IDs issued by Python only after successful capability/reference observations.
- Adds `available_evidence` to the engineering model state; the LLM may select only `ev_cap_*` / `ev_ref_*` IDs from that registry. Unknown IDs are rejected by exact registry membership rather than substring matching.
- Separates deterministic bindings from optional LLM-selected support: confirmed intake digest/fact IDs, current `checkMesh` evidence, and case-manifest SHA remain Python-owned and must not be emitted as model evidence claims.
- Records canonical evidence objects directly on successful EngineeringEvents, so context compaction cannot change provenance identity.
- Adds regression coverage for the former `tool_result:checkMesh at preparation step ...` and `user_fact:confirmed_intake` failure class.

# v2.4.0

- Adds deterministic model-context compaction without deleting local provenance/audit history.
- Replaces full runtime residual-history transmission with a compact summary: total sample count, latest residual per recent field, and a small recent-sample tail.
- Limits engineering model context to 12 recent observations with bounded excerpts; cumulative capability/reference provenance remains separately summarized and final deterministic validation still reads the original full event history.
- Limits post-processing model context to 8 recent observations and an 80-file bounded inventory projection while retaining the full native result tree locally.
- Compacts feedback-review runtime reports and bounds transmitted feedback history.
- Adds hard pre-API prompt character budgets: 60k engineering, 40k post-processing, and 40k feedback-review. An unusual oversized payload is deterministically compacted again instead of being sent unbounded.
- Adds `[LLM-CONTEXT]` progress telemetry with prompt/schema size and a conservative tokenizer-free `approxTokens` estimate, plus `[LLM-USAGE]` exact input/output/total token metrics when returned by OpenAI.
- Adds CLI `--llm-max-output-tokens` with a default of 16000 so structured action calls do not leave the model's much larger output window uncapped.
- Adds stress regressions with 12,000 residual samples, large engineering observations, and oversized result inventories.
- Synchronizes package/module metadata at 2.4.0.

# v2.3.2

- Surface bounded, path-redacted deterministic `finish_preview` validation failures in live CLI progress under a `reason:` block.
- Keep model rationale hidden; only deterministic gate failures are exposed.


# v2.3.1

- `/confirm` checks only the trusted `checkMesh` executable before any autonomous engineering LLM turn.
- Missing/untrusted `checkMesh` transitions directly to `ENGINEERING_BLOCKED` with zero engineering actions.
- The preflight is intentionally minimal: solver- and mesh-strategy-specific executables remain agent-observed capabilities during engineering.
- `--dry-run` skips the native `checkMesh` preflight.

# v2.3.0

- Adds a shared live `ProgressEvent`/`ProgressReporter` layer across intake, engineering, human revision, runtime repair, solver execution, post-processing, and feedback review.
- Adds CLI `--progress quiet|normal|verbose` with `normal` as the default.
- Shows `/confirm` engineering actions and deterministic outcomes immediately instead of waiting for the final state report.
- Includes live `checkMesh` cells/non-orthogonality/skewness metrics when current evidence exists.
- Adds bounded OF13-compatible live solver summaries for `Time = ...s` and Courant number, throttled in normal mode.
- Keeps raw `foamRun` stdout for `--progress verbose`; normal mode preserves the complete log on disk without flooding the terminal.
- Emits post-processing and human-feedback/revision progress through the same event bus.
- Prevents progress rendering from becoming an execution authority: callback exceptions are swallowed by `SafeRunner` and cannot terminate the OpenFOAM subprocess.
- Keeps model rationale/private reasoning out of progress output and sends CLI progress to stderr so `--json` stdout remains machine-readable.
- Adds dedicated progress regression tests for verbosity filtering, engineering/checkMesh output, OF13 runtime parsing, post-processing, feedback review, and callback isolation.
- Synchronizes package/module metadata at 2.3.0.

# v2.2.0

- Replaces automatic terminal `DONE` semantics with `EXECUTION_DONE -> RESULT_REVIEW_REQUIRED -> /accept -> COMPLETE`.
- Adds `/feedback <text>` at `MESH_READY` and `RESULT_REVIEW_REQUIRED`; feedback is stored separately from immutable intake with scope/run/state/evidence-snapshot provenance.
- Adds `CFDFeedbackReviewAgent`, a non-executing structured-output reviewer that separates observations from diagnosis hypotheses and produces an auditable `RevisionProposal`.
- Keeps a reviewed case immutable when feedback assessment finds no justified case revision; the assessment remains provenance and the workflow returns to the original human review gate instead of forcing a mutation.
- Returns safely to the original review gate if automated feedback assessment fails, preserving the human observation as unresolved provenance rather than stranding the workflow in an intermediate state.
- Binds every proposal to the exact baseline EngineeringPlan and case manifest; out-of-band input changes block revision.
- Routes feedback that explicitly changes confirmed user facts back to intake review rather than silently modifying the case.
- Adds a second `/confirm` gate for `REVISION_READY`; no solver-input file can change before this confirmation. Adds `/reject` to discard an unconfirmed proposal and return to the unchanged prior review state while preserving proposal audit history.
- Adds fresh bounded engineering/native/mesh-repair accounting for each confirmed human revision while preserving all older EngineeringEvents as provenance.
- Requires a real solver-input manifest change when a proposal says case revision is required, preventing the Agent from claiming a revision without changing the case.
- Adds deterministic `RevisionRecord` before/after plan hashes, manifest hashes and per-file added/removed/modified SHA-256 diff.
- Snapshots exact pre-revision `0/constant/system` inputs under `revision-history/rev-XXXX/baseline_inputs/` and archives prior numeric time outputs, `postProcessing/`, `postprocessConfig/`, and logs before a revised run, preventing stale-result contamination while preserving rollback/audit evidence.
- Adds advisory post-processing scientific confidence/review reasons/recommended human checks; these fields never substitute for deterministic metrics or human acceptance.
- Adds interactive guards so pending review states cannot be accidentally destroyed by typing a new plain prompt; the user must choose `/feedback`, `/accept`, `/solve`, `/confirm`, or `/new`.
- Adds strict-schema, hash-binding, no-mutation-before-confirmation, unchanged-manifest rejection, archive isolation, confirmed-fact reroute, revision-diff, and human-acceptance regression tests.
- Synchronizes package/module metadata at 2.2.0.

# v2.1.0

- Adds automatic `foamRun -> CFDPostProcessingAgent -> DONE` continuation after a successful solve.
- Keeps post-processing strategy agent-owned while isolating deterministic responsibilities to safe execution, provenance, bounded reads, and evidence-backed numerical analysis.
- Adds isolated `postprocessConfig/` authoring that is excluded from the immutable solve-input seal; normal engineering actions cannot write this namespace.
- Adds trusted Foundation `foamPostProcess` execution with dictionary hash binding, native command budgets, sanitized environment, and executable provenance checks.
- Restricts post-processing result reads to numeric time directories and `postProcessing/`; arbitrary case/host paths remain unavailable and large text reads are bounded.
- Adds deterministic OpenFOAM `forceCoeffs` parsing for mean Cd, mean/RMS Cl, Cl-based shedding frequency, and Strouhal number using `magUInf` and `lRef` from the executed dictionary.
- Binds force metrics to SHA-256 of both `coefficient.dat` and the analysis dictionary and invalidates metrics if either changes before final reporting.
- Collects hashable vorticity/force artifacts when they exist and reports a direct `paraFoam` visualization hint.
- Preserves successful runtime evidence if post-processing fails or lacks enough cycles; scientific validation remains explicitly separate.
- Adds CLI controls `--postprocess-steps`, `--postprocess-native-budget`, and `--skip-postprocess`.
- Adds strict Structured Outputs, integration, tamper, failure-isolation, result-sandbox, and bounded-read regression tests.
- Synchronizes package/module metadata at 2.1.0.

# v2.0.5

- Fixes OpenFOAM Foundation 13 runtime completion parsing for unit-suffixed time lines such as `Time = 20s` and `Time = 2.5e-03 s`.
- Prevents a clean OF13 run from being misclassified as `Runtime log contains no Time progress evidence`, which previously could trigger unnecessary autonomous solver retries until the runtime retry budget was exhausted.
- Adds parser and end-to-end runtime regressions proving a clean `Time = ...s` + `End` log transitions directly to `DONE` on the first solver attempt.
- Keeps strict runtime success evidence: return code 0, at least one parsed time-progress record, `End`, no FOAM fatal error, and no NaN/Inf/SIGFPE evidence are still required.
- Synchronizes package and module version metadata at 2.0.5.

# v2.0.4

- Replaces the fixed 40-action preparation cap with a production-sized progress-aware budget: 120-action soft boundary, +20 action extensions, and a 200-action absolute hard cap by default.
- Extensions require deterministic novelty in recent action/result evidence; repeated equivalent tool/search/read loops do not earn additional budget.
- Expands the finalization-only window to 8 actions while preserving its no-tools/no-edits restriction.
- Separates resource budgets: 40 executed native OpenFOAM commands, 10 mesh-repair cycles, and 60 agent actions per runtime-repair cycle by default.
- Raises the default runtime policy to 8 autonomous repair/retry cycles after the initial solver execution (9 total solver attempts).
- Adds explicit resource markers to engineering events so native-command and mesh-command accounting survives state rehydration.
- Treats `checkMesh` event success according to parsed mesh evidence, not return code alone.
- Adds CLI controls for all production budget limits.
- Adds regression coverage for progress extension, stagnation blocking, hard-cap enforcement, independent native-command limits, mesh-repair-cycle limits, and eight runtime repair cycles.
- Synchronizes package and module version metadata at 2.0.4.

# v2.0.3

- Fixes a real OpenAI-run orchestration bug where a successful `checkMesh` on engineering step 40 could still end as `ENGINEERING_BLOCKED` because the Agent had no remaining turn to submit `finish_preview`.
- Adds a bounded finalization-only window (default: 4 turns). No additional tool or file actions are permitted there; only `finish_preview` or `block`.
- Adds compact cumulative provenance to the model state so long runs do not forget capability-provider observations that fell outside the recent-observation window.
- Exposes step-budget/finalization readiness in the engineering prompt state and tells the Agent to finalize promptly after a current passing `checkMesh`.
- Synchronizes package and module version metadata at 2.0.3.

# v2.0.2

- Fixed OpenAI Structured Outputs compatibility for `EngineeringTurn.action`.
- Replaced the Pydantic discriminated union (`oneOf` + `discriminator`) with a plain literal-tagged union that emits supported nested `anyOf`.
- Added local preflight rejection for `oneOf` and root-level `anyOf`, so schema incompatibilities fail before an API request.
- Added regression tests for the generated engineering schema and union dispatch.

# v2 Architecture Reset

v2 is a breaking rewrite from the Phase/v0.x prototype.

## Production paths deleted

- `openfoam_agent.case_factory`
- `openfoam_agent.solver_factory`
- requirement/physics/equation/dimension/solver-capability agent chain
- deterministic capability requirements/gap/ownership planners
- deterministic runtime diagnosis and repair planner
- phase-specific case validation/refinement/verification suites
- old generated-solver registry/evaluation schemas and benchmark harness
- template directory and reactive-scalar generator backend

These were removed because they made Python own CFD design decisions or maintained a second deterministic implementation path beside the LLM agent.

## Production paths added/replaced

- `CFDEngineeringAgent`: one bounded engineering/tool loop for all confirmed CFD problems;
- `EngineeringPlan`: auditable agent decisions without Python implementation recipes;
- `EngineeringTurn`: one structured tool action per model turn;
- `CapabilityCatalog`: read-only evidence retrieval;
- `OpenFOAMReferenceIndex`: local official tutorial/source search;
- `CaseWorkspace`: strict file sandbox and full pre-solve input sealing;
- `DeterministicSafetyGate`: provenance/integrity/native evidence only;
- `RuntimeOrchestrator`: real failed solver logs returned to the same engineering agent.

## Approval semantics

- Intake is immutable after `/confirm`.
- `/confirm` authorizes bounded case/mesh preparation only.
- `/solve` authorizes `foamRun` only for the exact sealed `MESH_READY` case.
- Runtime automatic repair may edit/revalidate but may not switch the approved solver.

## Adversarial integration verification

The v2 regression suite includes a hard dynamic-mesh scenario that exercises the full authority/safety boundary: capability and installed-source lookup, rejection of executable dictionary directives, mesh-generation/checkMesh failure observation, agent-authored repair, native mesh sealing, `/solve`-equivalent state rehydration, SIGFPE runtime failure, same-agent repair, mandatory fresh checkMesh evidence, resealing, and bounded retry of the unchanged approved solver.

## Compatibility

This is intentionally not API-compatible with v0.x/Phase27. Old imports and tests are expected to fail rather than silently invoking legacy planning behavior.

## v2.0.1 security hardening

- Native executable provenance is enforced under the trusted OpenFOAM installation instead of trusting the first matching name on PATH.
- OpenFOAM subprocesses receive a reduced environment; `OPENAI_API_KEY`/unrelated secrets and user/site binary/library override paths are not inherited.
- PATH and `LD_LIBRARY_PATH` are filtered to trusted OpenFOAM/system roots and runtime HOME is workspace-local.
- Model-bound environment/reference metadata no longer contains absolute local OpenFOAM paths; known and generic absolute paths in tool/log observations are redacted.
- Environment-derived OpenFOAM source/tutorial/etc roots must remain inside `WM_PROJECT_DIR`.
- Run workspace/case/log directories are private and agent-authored artifacts/logs are not group/world readable.
- Final solver-provider/evidence claims must correspond to successful observations from the same run.
- Runtime success requires actual `Time = ...` progress evidence in addition to return code 0, `End`, and no fatal/non-finite evidence.
- Dedicated security regression tests cover secret-environment stripping, PATH shadowing, reference-root escape, path redaction, private permissions, fabricated capability provenance, and false `End`-only success.
