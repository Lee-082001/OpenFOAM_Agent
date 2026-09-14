# OpenFOAM Agent v5.0.0 — Authority & Evidence Hardening

v5.0.0 is an architecture-hardening release. It does not move CFD engineering decisions into Python. The release tightens the boundary between Agent-owned engineering choices and controller-owned observation, provenance, safety, validation, and execution authorization.

## Core invariant

> The Agent chooses CFD engineering. Python validates observed evidence, checks integrity/safety, compiles deterministic execution graphs, and authorizes bounded native tools. Python must not manufacture CFD meaning from names, file paths, or controller defaults.

## Major changes

### 1. Installed identity is no longer CFD semantic evidence

Installation discovery now proves only local identity/readiness. Names such as `solid`, `incompressibleFluid`, `heatSource`, and `foamMultiRun` no longer cause Python to inject physical capabilities such as conjugate heat transfer, incompressibility, or volumetric heating. Documented capability-graph semantics remain separate and may be joined with observed local readiness without manufacturing new meaning.

### 2. Execution owns modern multi-region topology

For modern plans with `execution.regions`, that list is the single authority for named regions and solver assignments. `region_layouts` and required-file paths are validation projections. A manifest cannot create an undeclared region or override a solver assignment.

### 3. Initial build and repair use shared region/native scope

A shared native-scope compiler is used for region-scoped dictionary consumers and final `checkMesh` routing. Repair deltas no longer fall back to root-only `blockMesh`/`checkMesh` semantics for named-region cases.

### 4. No hidden solver-file requirements in pre-solve validation

`EngineeringPlan.required_case_files` is the solve-input manifest. Pre-solve validation no longer silently adds solver-specific `fvSchemes`, `fvSolution`, or initial-field requirements that were not declared by the Agent/provider contract. Python validates the manifest rather than redesigning it.

### 5. User authorization and controller policy are separated

Engineering defaults may only be sealed when the request/session actually authorized exploratory completion. A controller progress policy cannot promote a false user authorization into a durable engineering default.

### 6. Native failure evidence is stricter

Only explicit OpenFOAM fatal diagnostics are generically promoted to case failures eligible for CFD repair. Unknown non-zero exits, crashes, segmentation faults, floating-point exceptions, and other ambiguous process failures remain `INCONCLUSIVE` tool/infra observations unless stronger case-level evidence exists.

### 7. Implementation evidence is explicit and content-bound

Documentary implementation evidence no longer binds to files by basename/content guessing. File coverage must be explicitly scoped. Each observed source excerpt receives a content-bound observation ID derived from its source, durable record, and excerpt hash. Source-level compatibility evidence IDs remain for backward compatibility.

### 8. Documentary syntax evidence is advisory, not a mutation token

The legacy `require_authoring_evidence()` path is retained only as an explicit audit helper. Production authoring authorization comes from workspace/security policy and deterministic serializers/parsers/native consumers. Missing documentary syntax evidence neither authorizes nor blocks a mutation by itself.

## Compatibility notes

- OpenFOAM Foundation 13/14 support is retained.
- Existing source-level evidence IDs remain readable.
- Legacy region-layout inference remains only as a compatibility fallback for plans that predate an explicit execution topology.
- Safety invariants such as sandboxed paths, command effects, execution approval, process/resource budgets, file hashes, seals, and native tool prerequisites remain deterministic controller responsibilities.

## Regression coverage

`tests/test_v500_authority_evidence.py` covers the v5.0 authority boundary, including:

- installed names cannot manufacture CFD semantics;
- execution regions are authoritative;
- manifests cannot create rogue regions;
- unknown non-zero native exits do not become case evidence;
- explicit OpenFOAM fatal diagnostics still become case failures;
- basename matching cannot manufacture implementation evidence;
- explicitly scoped source observations are content-bound.

## Validation status

The branch changes are structured for the repository test suite, but no claim of a full native OpenFOAM runtime qualification is made by this changelog. Run the complete pytest suite and representative Foundation 13 cases before merging/releasing.
