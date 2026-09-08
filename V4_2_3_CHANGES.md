# OpenFOAM Agent v4.2.3

## Why this release exists

A live v4.2.2 run successfully completed intake, accepted the Engineering design, and generated a large staged `author_case` response. The response was then rejected before any file mutation with:

```text
execute_case_plan required_case_files must exactly match plan.required_case_files
```

The failure was caused by duplicated ownership. The frozen `EngineeringPlan` already contained the authoritative required-file manifest, but the LLM was also required to echo a second `required_case_files` list in the authoring response. A harmless omission/order/stale echo therefore became a hard failure even though deterministic pre-solve checks were available later.

## v4.2.3 policy

`EngineeringPlan.required_case_files` is the single source of truth. Authoring-stage `required_case_files` is optional compatibility metadata. Python overwrites it from the frozen plan when creating `ExecuteCasePlanAction`. Partitioned tasks are accepted based on their actual authored path coverage against controller-issued `authoring_task.paths`; a mismatched mirror no longer blocks the task.

The relaxed boundary does **not** weaken safety or completeness:

- authored file paths must still exactly cover the controller-assigned partition task;
- unsafe paths/directives/content remain rejected;
- the full frozen plan digest remains controller-owned;
- pre-solve validation still checks the canonical required-file manifest;
- native command allowlists, checkMesh freshness, case seals, resource limits and user solve approval remain strict.

## Additional progress-first normalization

Authoring metadata that is redundant rather than safety-critical is normalized before Pydantic semantic validation:

- duplicate `validate_dictionaries`, `surface_checks`, `mesh_commands`, and compatibility required-file mirrors are deduplicated;
- native commands accidentally emitted by an intermediate deferred partition task are dropped rather than causing a retry;
- when both legacy `mesh_commands` and `native_pipeline` are supplied, `native_pipeline` wins;
- exact duplicate native invocations are collapsed;
- existing `checkMesh` invocations are ordered after mesh-mutating utilities.

## Validation

The release adds regressions covering stale/omitted required-file echoes, partitioned authoring, real file-coverage rejection, redundant intermediate native commands, and duplicate native mesh validation. Final release verification also applies the generated v4.2.2 -> v4.2.3 patch to a clean v4.2.2 tree and compares it against the ZIP release.
