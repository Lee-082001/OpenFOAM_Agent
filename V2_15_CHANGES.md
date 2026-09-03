# OpenFOAM Agent v2.15.0 changes

## Why this release exists

A real v2.14 run reached `SOLVE_READY`, repaired two genuine OpenFOAM runtime errors, and completed successfully, but exposed a more important trust problem: Intake had recorded `classification.problem_type=internal_flow` as `[user]` even though the user only described a cylinder inside a rectangular computational domain. Engineering then faithfully implemented that confirmed interpretation with no-slip upper/lower walls. This is exactly the distinction between execution integrity and semantic fidelity: every native gate can pass while the executable problem is not the user's intended physical problem.

## Review-critical Intake provenance

- New IntakeAgent-issued specs are upgraded to `semantic_contract_version=2`.
- `classification.problem_type` and `temporal.behavior` are treated as review-critical provenance.
- They remain `source=user` only when exact user evidence explicitly states the normalized class/time behavior (including bounded common Korean/English direct cues).
- Unsupported direct attribution is deterministically demoted to `derived`; the selected value is not changed by Python.
- The Intake prompt explicitly warns that a rectangular/square computational domain around an obstacle does not by itself establish internal/channel flow.
- The CLI renders derived classification/temporal/motion/boundary facts as `derived/review-critical` and shows the derivation reason before `/confirm`.

## Evidence-carrying confirmed-fact bindings

`ConfirmedFactBinding` now supports:

- `case_assertions[]`: Agent-selected current case paths plus exact semantic snippets. Verification is whitespace-insensitive so deterministic serializer formatting does not create false stale-evidence failures.
- `numeric_relation`: a generic numerator-product / denominator-product relation. Each scalar term supplies a case path, an exact artifact excerpt, the numeric token found in that excerpt, and an optional Agent-selected multiplier (for example, `2 × radius`).

Python does **not** know what formula represents Reynolds number, Mach number, or any other CFD concept. The Agent selects the terms and relation. Python checks that the terms are present in the current case and recomputes the relation against the numeric target already frozen in the confirmed intake. `//` and `/* ... */` comments are removed before semantic-evidence matching, so comments cannot act as self-issued implementation proof.

For semantic-contract v2:

- confirmed `classification` and `temporal` facts require at least one current-case semantic assertion;
- direct user `physics`, `scale`, and `property` facts containing exactly one numeric target require a numeric relation assertion.

These checks are part of the existing deterministic plan gate and therefore run again after case repair/retry.

## Protocol resilience of the semantic layer

Security-sensitive structure remains strict (sandbox path validation, binding coverage, conflicting representations). Harmless/incomplete assertion content is not allowed to recreate the old Pydantic-union failure mode: blank snippets/terms are accepted structurally and become explicit deterministic semantic-validation failures that the next Engineering turn can repair.

## Upgrade compatibility

Adding schema fields must not invalidate already sealed cases. v2.15 therefore preserves canonical hashes:

- legacy/default semantic-contract-v1 intakes omit the new version field from `CFDIntakeSpec.digest()`;
- empty `case_assertions` and `numeric_relation` fields are omitted from `EngineeringPlan.digest()`.

Non-empty v2.15 semantic evidence remains hash-bound to the plan and CaseSeal.

## Validation

- Added production-log regression coverage for false `[user]` classification/temporal attribution.
- Added semantic-contract-v2 acceptance tests.
- Added numeric-drift rejection (`Re` claim remains 1000 while case terms recompute to 100).
- Added stale case-snippet assertion rejection.
- Added legacy intake/plan digest compatibility tests.
- Added incomplete semantic assertion controlled-failure coverage.
- Added comment-only self-claim rejection coverage.
- Existing strict Structured Output and full workflow regression suites remain green.

Regression suite: **200 passed**.
