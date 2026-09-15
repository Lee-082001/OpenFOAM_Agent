# OpenFOAM Agent v5.2.0

v5.2.0 is the reliability-qualification release. It adds controller-owned dependency-scoped revalidation after case deltas and a reproducible benchmark aggregation harness. It does **not** claim that the benchmark has already been executed; the shipped benchmark manifest is explicitly `NOT_RUN` until real reports are supplied.

## Dependency-scoped revalidation

- `CaseDeltaGraph` now records the exact validation domains invalidated by a repair/revision: changed dictionaries, changed surfaces, affected mesh scopes, and pre-solve validation.
- Multi-region mesh evidence is reused by scope. A numerical-only `fvSolution`/`controlDict` repair does not rerun `blockMesh` or `checkMesh`; a mesh delta in one named region rebuilds/rechecks only that region while unaffected regions retain current evidence.
- Repair/revision native pipelines use the same stable producer->consumer dependency ordering as initial case authoring.
- Every committed prepare/runtime/strategy delta writes a `RevalidationRecord` with changed artifacts, affected/reused mesh scopes, planned native commands, and whether solver retry is required. This record is audit/benchmark telemetry only and never authorizes execution.

## Reliability observability

- Engineering provider usage is persisted in `CFDState.engineering_llm_usage_records` and summarized in CLI JSON reports.
- JSON reports expose `revalidation_records` plus `reliability_observables`, including engineering token totals, scoped revalidation count, mesh-validation scopes avoided, and runtime repair attempts.
- Existing safety/native evidence remains authoritative; metrics do not change pass/fail behavior.

## Benchmark harness

`research/V5_2_RELIABILITY_BENCHMARK.json` defines 12 labeled trials across three comparison variants (`plain_llm`, `native_validation`, `ours`), for 36 expected reports. The set includes normal CFD cases, runtime/pre-solve repair injections, scoped multi-region mesh repair, Reynolds-number false-acceptance injection, and a protected numerical-control mutation.

Aggregate completed reports with:

```bash
PYTHONPATH=src python -m openfoam_agent.qualification.reliability \
  --manifest research/V5_2_RELIABILITY_BENCHMARK.json \
  --results-dir benchmark-results \
  --output benchmark-summary.json
```

The evaluator keeps missing reports explicit and computes, per variant:

- case acceptance rate;
- execution success rate for trials expected to run;
- false acceptance rate;
- unnecessary blocking rate;
- repair success rate;
- requirement-violation detection rate;
- average engineering token use;
- average native process/command count;
- mesh-validation scopes avoided by scoped revalidation.

## Verification

v5.2.0 adds regression coverage for numerical-only mesh-evidence reuse, single-region mesh invalidation in a multi-region case, dependency ordering in repair pipelines, false-acceptance accounting, repair-success accounting, requirement-violation detection, and explicit NOT_RUN/missing-report semantics.
