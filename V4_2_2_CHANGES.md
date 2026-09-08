# OpenFOAM Agent v4.2.2

## Phase-specific authoring context

A live v4.2.1 run successfully accepted the engineering design for a gravity-assisted, full-pipe water bifurcation, then failed before the authoring model call with:

`ContextBudgetError: Indivisible file task system/controlDict exceeds context budget`

The failure was architectural rather than CFD-specific: production CLI used one 18,000-character cap for both compact engineering design and file authoring. A single authoring task still carried a near-complete frozen EngineeringPlan, confirmed intake and implementation context, so an ordinary `controlDict` could become "indivisible" even after file partitioning.

v4.2.2 changes this boundary:

- `--engineering-context-chars` remains 18,000 by default for design/reasoning turns.
- New `--engineering-authoring-context-chars` defaults to 32,000 for `author_case` only.
- `EngineeringPolicy.max_authoring_prompt_chars` provides the same library-level control.
- Authoring partitioning first tries the normal file-scoped projection, then automatically falls back to a compact plan projection when needed.
- Compact tasks preserve confirmed fact closure, execution identity, region/interface topology, completion intent, concrete engineering defaults, required-file identity and relevant fact/evidence bindings.
- Advisory long rationales, evidence bodies and result-analysis metadata are summarized or deferred. The immutable full EngineeringPlan remains Python-owned and is bound into every task by SHA256.
- Final task assembly still validates the SHA256 of the original full plan before any candidate case is executed.

## Safety unchanged

The context relaxation does not weaken path/content/native safety. Unsafe directives, coded execution, path traversal, unapproved libraries/commands, execution approval, case seals, mesh freshness and completion evidence remain fail-closed.

## Verification

The full regression set passes in split execution. Packaging verification applies the generated v4.2.1 -> v4.2.2 patch to a clean v4.2.1 baseline and compares the resulting release tree byte-for-byte with the v4.2.2 ZIP.

A live OpenFOAM 13 + Codex CLI end-to-end rerun is still required; the assistant environment does not contain the user's sourced OpenFOAM/Codex runtime.
