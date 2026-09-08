# OpenFOAM Agent v2.14.0 changes

## Why this release exists

Production v2.13 runs showed a remaining class of failures where CFD reasoning and deterministic evidence retrieval were valid, but a harmless Structured Output protocol defect (for example an empty/overlong evidence query) caused Pydantic validation to terminate the whole workflow. The same architectural risk existed in other LLM phases even when it had not yet appeared in a production log.

## Evidence-gap lifecycle

- Evidence-gap IDs are single-use.
- A retrieved gap becomes `evidence_available` when new canonical evidence is observed, otherwise `stagnant`.
- When the Engineering Agent proceeds to case execution/runtime repair, available gaps become `satisfied`.
- Further retrieval must use a new gap ID with `refines_gap_id`; the parent becomes `superseded`.
- Reusing an already retrieved gap or refining a satisfied gap is deterministically refused.
- Refused repeat/refinement requests do not consume the retrieval-cycle hard fuse.

## Retrieval protocol normalization

The following fields are non-authoritative retrieval metadata and are normalized before Pydantic field constraints can abort a run: whitespace/length of evidence queries, opaque gap-ID formatting, invalid retrieval scope, read-top count, duplicate query strings and duplicate gap requests. If all supplied queries are empty, the Agent's own `missing_evidence` statement becomes the bounded fallback reference query. No CFD value, solver, BC, mesh choice or confirmed fact is inferred by this normalization.

## OpenAI bounded Structured Output repair

`OpenAILLM` now performs one bounded repair attempt when `responses.parse` (or final model validation) raises a Pydantic `ValidationError`. The second request carries the validation error and instructs the model to correct protocol shape only while preserving confirmed user facts. Ollama already had a bounded local JSON repair loop; v2.14 gives the OpenAI path equivalent resilience. Deterministic safety/evidence gates are unchanged.

## Controlled no-op repair handling

`repair_case_plan`, `repair_candidate_case_plan` and `repair_runtime_case` no longer reject an empty delta at schema construction time. Their executors convert it into an explicit unsuccessful engineering/runtime action (and do not mutate the workspace or silently retry the solver). This prevents large `anyOf`/union validation traces from turning a correctable protocol mistake into `state: FAILED`.

## Audit results

The compact Engineering phase contracts (prepare, decision-only, candidate replan, repair, revision, finalization and runtime repair), Intake, Post-processing and Feedback schemas were re-checked against the strict OpenAI Structured Output schema constraints. Semantic/safety-critical validators remain hard: unsafe paths, conflicting file representations, ambiguous exact patches, confirmed-fact binding coverage, solver/provider consistency, case integrity and native evidence are not normalized away.

Regression suite: **190 passed**.
