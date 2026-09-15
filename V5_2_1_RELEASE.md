# OpenFOAM Agent v5.2.1

Patch release based on the supplied v5.2.0 archive. Full source, tests, configuration,
examples and historical documentation are retained. See `REVIEW_5_2_1_KO.md` for the
Korean handoff and exact verification results.

## Changes

- Critical quantitative bindings are compared to the frozen target, not merely to
  a model-supplied expected value or a text anchor. Arbitrary multipliers are rejected.
  Literal field dimensions and confirmed patch/region are checked. Supported direct
  velocity proof uses the magnitude of a fixed boundary value. Reynolds proof is
  bounded to dimensioned U * reference length / nu.
- Delta scheduling propagates dirty generated outputs to a fixed point. A blockMesh
  rebuild therefore schedules downstream snappy; a feature dictionary edit schedules
  extraction and downstream snappy. Explicit dictionary overrides retain ownership.
  Mesh-dependent utilities are ordered after base/snapped mesh; checkMesh remains last.
- Runtime numerical repairs require positive finite reductions within the initial
  approved values. Protected dictionary leaves are checked individually. All numerical
  files require the exact authorized target hash at the post-mutation seal check.
  Multi-file authorization is atomic and both runtime action forms use the same gate.
- Runtime repair success requires actual runtime success. Pre-solve repair success
  must be declared separately. Free text mentioning repair is not repair evidence.
  Aggregation exposes coverage and rates over expected cases as well as observed rates.
- Docker COPY no longer references the absent V2_CHANGES.md. Regression execution
  writes complete collected/executed identities and current source hashes without
  referencing an absent audit file or a hardcoded historical audit count.

## Compatibility and limits

Old checkpoint fields remain loadable with empty defaults. Previously modified
numerical files without corresponding authorization hashes must be reviewed/resealed;
this patch does not invent authorization for them. A legacy quantitative locator may
need an explicit literal entry/dimensions. Unsupported dynamic dictionaries,
non-fixed/time-dependent boundary proof and compound formulas are rejected with a
specific reason instead of being labeled verified. No generic physics/solver recipe
was added.

Input arithmetic assurance is not an independent geometry or CFD output oracle:
the selected reference length still needs engineering justification against the
actual geometry. Arbitrary repeated/in-place native transformations and every third
party utility are not covered by the bounded ordering registry. Native OpenFOAM,
MPI, Docker build, and live model execution were not qualified in this environment.
The benchmark remains a report aggregator, not an automatic physical-oracle runner.

Run complete regressions with `sh scripts/run_regression.sh`. The command returns a
failure status if any test fails and preserves all outcomes; no historical failures
are hidden or automatically marked expected.
