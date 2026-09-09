# OpenFOAM Agent v4.5.1

## Live failure

A v4.5.0 live run completed Intake and Engineering design, then the `author_case` Codex response failed structured-output validation twice:

```text
typed_dictionaries.1
Value error, Typed dictionary contains duplicate entry paths.
```

The failure happened before case authoring, mesh generation, OpenFOAM validation, or runtime. A repeated model echo inside one typed dictionary was being treated as an invalid structured response rather than normalized by Python.

## Root cause

`TypedFoamDictionaryFile` required every entry path to be unique in an `after` Pydantic validator. This placed semantic consistency at the wrong boundary. Even an identical duplicate assignment caused the entire `CaseAuthoringTurn` to fail parsing, so the Codex client retried the complete large structured-output call. A real conflicting duplicate also had no way to reach the smaller retained-candidate repair path because the object never became a valid Python model.

## New boundary

v4.5.1 separates three responsibilities:

1. **Structured shape validation** -- Pydantic validates types, bounds and safe path syntax.
2. **Deterministic normalization** -- harmless repeated authoring representations are collapsed/merged before nested validation.
3. **Semantic conflict routing** -- genuinely conflicting values are carried to the controller and block transactionally before any file write, then use retained-candidate delta repair.

### Typed leaf duplicates

- same path + equivalent value -> keep the first assignment; no model retry;
- equivalence permits only outer whitespace and a serializer-owned trailing semicolon difference; internal whitespace is preserved when comparing values;
- same path + different value -> keep the first serializable assignment and record a hidden `entry_conflicts` path; do **not** silently choose the later value.

### Repeated case-file representations

- repeated typed dictionary files with the same case path -> merge their entries and let leaf normalization/conflict detection decide;
- repeated identical raw files -> deduplicate;
- repeated raw files with different content -> record an authoring conflict;
- raw and typed representations claiming the same case path -> record an authoring conflict;
- incompatible repeated `foam_class` metadata -> record an authoring conflict.

`entry_conflicts` and `authoring_conflicts` are Python/controller metadata hidden from the LLM JSON schema using `SkipJsonSchema`. They survive Python model dumps so staged/partitioned authoring does not lose the conflict record.

## Conflict execution policy

`_execute_case_plan()` checks controller conflict metadata before serialization or workspace mutation. If conflicts exist it emits `authoring_semantic_conflict`, retains the complete candidate in memory, records the implicated path(s), and enters the existing compact case-plan repair path. The full case is not regenerated and no partial case files are written.

## Safety retained

This normalization does not resolve substantive CFD ambiguity automatically. Conflicting values still prevent commit. Unsafe path/content checks, typed serializer invariants, blockMesh topology, native command policy, checkMesh freshness, pre-solve completeness, CaseSeal, solve approval and runtime contracts are unchanged.

## Validation

New regressions cover the exact live failure, conflicting duplicate routing, repeated typed-file merge, repeated raw-file normalization, hidden conflict schema fields and retained-candidate transactional repair. The complete test suite is rerun after the v4.5.1 version/document changes and the release patch is applied to a clean v4.5.0 baseline for independent verification.
