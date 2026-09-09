# OpenFOAM Agent v4.4.0 - Claim-Based Semantic Assurance

## Live failure that motivated the refactor

A v4.3.1 run for a fully flooded two-outlet pipe splitter completed the difficult parts successfully:

- staged Engineering design accepted;
- eight OpenFOAM case files authored;
- deterministic header checks passed;
- `blockMesh` succeeded;
- `checkMesh` succeeded with 36,000 cells, max non-orthogonality 0 and essentially zero skewness;
- pre-solve static checks passed.

The case was then rejected at `finish_preview` only because the derived routing fact `classification.problem_type=internal_flow` had no `case_assertion`. A secondary repair attempt then exceeded the model context budget. The CFD artifacts had not been shown invalid; the controller had confused *absence of an independent proof pointer* with *proof of an invalid case*.

## New assurance model

### 1. Provenance closure is mandatory

Every non-context confirmed fact must still be preserved by:

- exact `confirmed_intake_sha256`;
- exact `confirmed_fact_ids`;
- one `ConfirmedFactBinding` per fact;
- valid referenced case paths/plan fields.

Dropping a confirmed fact, changing the frozen intake digest, or referencing missing/unsafe implementation artifacts remains a hard failure.

### 2. Machine assertions are claim-based

`case_assertions` and `numeric_relation` are optional higher-assurance claims. If present, Python verifies them strictly against the current case. A stale snippet, wrong dictionary value or incorrect numeric recomputation remains a hard case failure.

### 3. Missing optional proof is advisory

Not every confirmed fact has a universal OpenFOAM token. In particular, classification/objective/routing metadata is not forced to invent a file snippet. User temporal/boundary/geometry/physics-style facts may generate an advisory semantic-assurance warning when no independent artifact assertion is available. This warning is visible but does not invalidate a case that passes real deterministic/native checks.

### 4. Finalization no longer repairs an assurance gap as CFD

`finish_preview` persists `semantic_assurance_warnings` and seals the case when hard gates pass. A missing optional assertion therefore cannot trigger a pointless CFD repair turn and secondary `ContextBudgetError`.

## Hard gates unchanged

- confirmed intake/fact preservation;
- contradiction in any explicitly claimed case assertion or numeric relation;
- workspace/path/content security;
- trusted execution/provider constraints;
- required case-file completeness;
- native mesh generation and `checkMesh` freshness;
- case seals and solve approval;
- runtime completion/result freshness.

## Compatibility

The semantic-contract wire format remains version 2 for rehydration compatibility. v4.4.0 changes the controller assurance policy, not the persisted intake schema. Existing v2 states therefore benefit from the new non-fabrication rule without invalidating their confirmed-intake digest.
