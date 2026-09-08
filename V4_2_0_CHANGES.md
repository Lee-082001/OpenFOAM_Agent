# OpenFOAM Agent v4.2.0 - Progress-First Validation

v4.2.0 rebalances the framework away from universal fail-closed validation. The goal is to let the CFD Engineering Agent make ordinary engineering progress while retaining hard deterministic gates only where failure would compromise user intent, execution integrity, security, or result validity.

## What remains strict

- confirmed intake digest and confirmed-fact coverage
- unsafe/path-traversal protection and workspace confinement
- explicit execution authorization and process/resource guards
- trusted provider/executable requirements when execution depends on them
- raw untrusted OpenFOAM authoring evidence where deterministic typed serialization cannot replace it
- mesh/checkMesh freshness, case sealing, restart integrity, runtime completion and result-source integrity
- claimed opaque evidence IDs: a claimed ID must have been deterministically issued

## What is now progress-first

- ordinary missing CFD values may be recorded as `engineering_default` without a separate exploratory authorization flag
- design-stage region/file completeness is deferred to authoring and pre-solve validation
- duplicate/redundant planning metadata is normalized rather than rejected
- redundant solver/provider mirrors are normalized from the structured execution contract
- safe case-file references from confirmed-fact bindings are promoted into `required_case_files`
- planning schemas ignore unknown extra metadata instead of failing on harmless forward-compatible fields
- QoI, conservation and completion objects default to `resolution_state=intent`; concrete runtime locators are required only for `resolved` analysis contracts
- placeholder runtime selectors such as `latestTime` may exist in design intent without blocking
- unresolved result-analysis intents do not make post-processing report success fail by themselves
- design no longer requires a separate same-run capability evidence pointer when the deterministic installed catalog already has sufficient provider evidence
- optional malformed runtime-analysis sections in `design_case` are automatically dropped/deferred when all validation errors are confined to those sections; hard execution/confirmed-fact/path errors are never auto-dropped
- LLM prompts explicitly prefer the smallest safe design and discourage evidence retrieval for ordinary CFD judgement

## Validation philosophy

The framework now follows:

`design first -> defer runtime-resolvable detail -> deterministic/native validation -> hard gate only at the stage where the fact becomes safety/execution critical`

This does not weaken command/path/security boundaries. It reduces false `ENGINEERING_BLOCKED` and structured-output failures caused by premature completeness requirements.

## Verification

Automated regression tests cover the progress-first boundaries, including the exact failure class that previously rejected mixed QoI sources and nonliteral future time-directory placeholders during `PrepareDesignTurn`.

No claim is made that actual OpenFOAM 13, MPI, or live Codex end-to-end CFD execution was performed in this packaging environment. Those remain live qualification steps.
