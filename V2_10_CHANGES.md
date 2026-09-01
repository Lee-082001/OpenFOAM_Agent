# v2.10.0 — Token-optimized phase contracts and delta repair

v2.10 keeps the v2.9 principle that CFD decisions belong to the LLM while deterministic Python owns execution authority, safety, provenance and evidence. The change is primarily about reducing repeated LLM context/output without weakening those gates.

## 1. Phase-specific Structured Output contracts

Production CLI now uses smaller contracts for `prepare`, `repair`, `runtime_repair`, `human_revision`, and `finalization` instead of sending the all-phase `EngineeringTurn` union every time. Library callers retain the legacy contract unless `EngineeringPolicy.compact_phase_schemas=True`.

## 2. Delta-only repair

`RepairCasePlanAction` preserves the baseline plan and unchanged files in Python state. A repair turn returns only exact `CaseFilePatch` operations, small replacement files, or typed replacement dictionaries plus the validations/commands that need to be rerun. Runtime repair still cannot change the approved solver.

Exact patches are applied only when the old fragment occurs exactly once; otherwise Python rejects the patch rather than guessing.

## 3. Compact state capsule after the first turn

With `state_delta_context=True`, subsequent calls send confirmed facts, a compact baseline-plan capsule, current file hashes, recent observations, evidence and failure text instead of resending the full intake/environment/plan history. This works statelessly. If an OpenAI client is configured with `store=True`, the adapter can additionally chain `previous_response_id`; `store=False` does not require server-side conversation storage.

## 4. Post-processing execution plan

`PostProcessingExecutionPlanAction` can write postprocess configuration, run one or more `foamPostProcess` commands, request deterministic analyses and finalize the report in one LLM plan. Execution stops on the first real failure and the next LLM turn sees the diagnostic.

## 5. Typed OpenFOAM dictionary serializer

`TypedFoamDictionaryFile` represents nested dictionary assignments as key paths plus Agent-selected OpenFOAM value expressions. Python renders braces and semicolons deterministically. This removes syntax boilerplate from LLM output and reduces syntax-only repair loops while keeping physical/numerical values Agent-owned.

## 6. Shorter rationale fields

Action-level rationale is optional and capped at 200 characters on legacy actions; high-level plans rely on concise goals/diagnoses. `EngineeringDecision.rationale` is capped at 500 characters and evidence notes at 300.

## 7. Prompt-cache optimization and telemetry

The OpenAI adapter supplies a stable `prompt_cache_key`, enables current GPT-5.6 prompt-cache options (`implicit`, `30m`), and records `cachedInputTokens` / `cacheWriteTokens` when the API returns them. Stable system/schema content stays before dynamic state.

## 8. Lower production CLI turn budgets

The optimized CLI defaults are now 12 initial engineering turns / 24 hard cap, 2 finalization turns, 4 runtime-repair turns per cycle, and 4 post-processing planning turns. They remain configurable from the existing CLI flags.

## Regression and token measurements

The v2.10 tree has 164 passing tests, including new strict-schema, typed-serializer, exact-patch, delta-context, post-processing-plan and prompt-cache tests.

Using the repository's conservative tokenizer-free `structured_request_metrics()` on the same fixtures:

| Scenario | v2.9 | v2.10 | Reduction |
| --- | ---: | ---: | ---: |
| solve-ready engineering, no failure | 16,707 | 10,169 | 39.1% |
| engineering with one blockMesh failure + repair | 35,188 | 18,244 | 48.2% |
| simple post-processing workflow | 18,015 | 5,122 | 71.6% |
| engineering + simple post-processing, no failure | 34,722 | 15,291 | 56.0% |
| engineering + simple post-processing, one repair | 53,203 | 23,366 | 56.1% |

These are conservative local estimates of prompt + schema bytes, not provider-billed token counts. Prompt caching can reduce actual input cost further when the provider reports cache hits.
