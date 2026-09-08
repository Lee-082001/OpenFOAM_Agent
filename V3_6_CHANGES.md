# OpenFOAM Agent v3.6.0 changes

## Bounded Engineering model context

v3.6 stops successful evidence retrieval from making every later Engineering prompt monotonically larger.

- Full `EngineeringEvidenceRecord` payloads remain durable local state.
- Model-facing evidence is compiled into a bounded relevance/recency capsule.
- Normal prepare turns show at most the configured `max_prepare_model_evidence_items`; decision turns may use a slightly larger bounded set.
- Recent active gap evidence is prioritized, broad query result sets are capped per gap, and only a small fallback capability/reference tail is exposed.
- Evidence detail text is independently bounded; dropping model-context detail never removes deterministic local provenance.
- The evidence-gap ledger now preserves ordered and last-new evidence IDs so the context compiler can keep the best recent search results without replaying the entire registry.

The production CLI defaults are intentionally tighter: 18,000 prompt characters, 10 evidence items in normal Engineering turns, 12 in decision turns, and two prepare retrieval cycles before a forced decision.

## Stateless transport fix

Codex CLI calls are deliberately `--ephemeral` and `store=False`. v3.5 could still label later capsules as `delta_from_previous_response` merely because the adapter accepted a `conversation_key` argument. v3.6 only uses response-chained delta context when the backend is actually stateful and stores prior responses. Stateless backends receive a fresh bounded capsule on every call.

## Staged engineering output

The largest structured response has been split into two contracts:

1. `DesignCaseAction` / `PrepareDesignTurn`: choose and validate the `EngineeringPlan` without OpenFOAM file payloads.
2. `CaseAuthoringAction` / `CaseAuthoringTurn`: author only the case files and deterministic native validation pipeline against the Python-held frozen plan.

Python validates capability provenance/default policy before accepting the design, freezes the plan, then combines the separate authoring response with that exact plan into the existing `ExecuteCasePlanAction` executor. Existing transactional bundle preflight, native tools, checkMesh, PreSolve, repair, and CaseSeal behavior is unchanged.

This reduces transport-schema size substantially. In regression telemetry, the legacy decision-only contract was about 9.6k approximate schema/system tokens, while staged design-decision was about 6.3k and case-authoring about 4.8k before prompt payload. A representative staged single-region run measured about 10.7k approximate tokens for design and 6.6k for authoring. After two retrieval cycles with more than twenty observed evidence items, the forced design-decision call remained below 15k approximate tokens.

## Targeted deterministic capability shortlist

When preloading is enabled, Python derives a few search phrases from the frozen intake (`classification`, `physics`, `objective`, `material`, `motion`) and deterministically pre-observes matching capability providers. This is evidence retrieval only: Python still does not choose a solver, fvModel, mesh method, or numerical scheme. The bounded model capsule prioritizes these intake-relevant providers so common cases can avoid an unnecessary generic evidence-search round trip.

## CLI controls

New tuning flags:

```bash
--engineering-context-chars 18000
--engineering-evidence-items 10
--engineering-retrieval-cycles 2
```

Reports also record the active Engineering prompt cap, evidence projection size, retrieval-cycle limit, and whether staged case authoring is enabled.

## Regression coverage

New tests cover:

- staged design -> case-authoring -> existing deterministic execution -> `SOLVE_READY`;
- staged schema-size reduction relative to the legacy one-shot decision contract;
- bounded evidence projection with 80 durable evidence items;
- two full retrieval cycles followed by a forced staged decision under a 15k approximate-token regression ceiling;
- stateless model adapters never claiming previous-response delta reuse in staged mode.

Full regression suite: **298 passed**.
