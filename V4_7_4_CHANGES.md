# OpenFOAM Agent v4.7.4 — Failure-Local Validation Repair

v4.7.4 addresses a live OpenFOAM Foundation 13 run in which staged authoring and mesh validation succeeded, but the zero-step native consumer found a real dictionary omission:

```text
keyword laplacianSchemes is undefined in dictionary "<WORKSPACE><LOCAL_PATH:fvSchemes>"
```

That native failure was correctly classified as a case failure. The defect was in the recovery topology: the next ordinary repair turn reused a broad Engineering prompt whose protected plan/contracts exceeded the deterministic 18,000-character budget, so the Agent never received a chance to repair the one implicated dictionary.

## Changes

### Dedicated committed-case repair capsule

Normal post-authoring validation repair now uses `case_validation_repair_v1` instead of the full Engineering context. It carries only:

- confirmed intake constraints/identity,
- a compact projection of the accepted/pending EngineeringPlan,
- the primary failed validation event,
- exact current files implicated by the diagnostic,
- a bounded case-file contract scan and case inventory,
- current mesh evidence/freshness,
- bounded supporting evidence and successful repair-support observations,
- controller bindings and remaining budgets.

The full EngineeringPlan, case workspace and audit history remain in Python/controller state and are not duplicated into the model prompt.

### Diagnostic-to-file resolution

Native path redaction can turn a real path into a diagnostic such as `<LOCAL_PATH:fvSchemes>`. Repair context now matches surviving case-relative paths and authored-file basenames against the diagnostic, so the exact current `system/fvSchemes` content is supplied first. Core numerical dictionaries are included only as lower-priority neighbours when budget allows.

### Repair-support continuity

Successful `read_case_file`, `read_reference`, `search_references`, `search_capabilities` and `gather_evidence` observations from the repair round are returned in bounded form on the next failure-local turn. This prevents an Agent from repeatedly requesting the same companion evidence because the dedicated compact context forgot it.

### Controller authority and selective revalidation

The LLM still returns a `RepairCasePlanAction`; it never directly schedules trusted actions. `CaseDeltaGraph` constructs and validates the complete effective case before transactional mutation.

- If only `system/fvSchemes` changes, the graph writes that dictionary, re-runs pre-solve/zero-step consumer validation and attempts final sealing. Existing current checkMesh evidence remains valid because the mesh inputs did not change.
- If a mesh-affecting input changes, existing mesh dependency invalidation and controller-owned `checkMesh` ordering still apply.
- Required-file closure, path/content safety, native toolchain preflight, CaseSeal and explicit `/solve` approval are unchanged.

### No Python numerical hardcoding

v4.7.4 intentionally does not solve the motivating failure by inserting a canned `laplacianSchemes` block in Python. Required native-consumer validity is deterministic, but the actual numerical scheme is an Engineering Agent decision and remains subject to native revalidation.

## Regression coverage

`tests/test_v474_failure_local_repair.py` covers:

1. 18k repair prompt boundedness with an intentionally bloated baseline plan and unrelated case files.
2. Basename recovery from redacted `<LOCAL_PATH:fvSchemes>` diagnostics.
3. Bounded handling of an oversized failed dictionary.
4. Dictionary-only CaseDeltaGraph repair with no blockMesh/checkMesh rerun.
5. Plan-metadata validation failures using the same bounded repair topology.
6. A complete synthetic native cycle: zero-step `foamRun` fails on missing `laplacianSchemes`, the Agent repairs only `fvSchemes`, zero-step validation passes on retry, and the case reaches `SOLVE_READY` without a mesh rerun.
7. Successful repair-support read results being available on the following compact repair turn.

The existing token-optimization regression was updated so ordinary committed-case failure recovery explicitly expects `case_validation_repair_v1` rather than the old full-conversation delta context.
