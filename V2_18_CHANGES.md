# OpenFOAM Agent v2.18.0 changes

## Runtime-repair state invariant

- Replaced the overloaded runtime repair boolean with `RuntimeRepairDecision`: `RETRY_SOLVER`, `NEEDS_USER_REVIEW`, `BLOCKED`, or `STRATEGY_REVISION`.
- All `CFDEngineeringAgent.repair_runtime()` exits close the transient `RUNTIME_REPAIR` state.
- `RuntimeOrchestrator` defensively rejects leaked repair state and retries that do not restore `SIMULATION`.
- The top-level workflow contains an explicit invariant guard, so an escaped repair state can no longer degrade into `No v2 handler for RUNTIME_REPAIR`.

## Cross-file boundary compatibility

- PreSolve parses current mesh patch types from `constant/polyMesh/boundary` and field patch types from required `0/*` dictionaries.
- Constraint patches `empty`, `wedge`, `symmetry`, `symmetryPlane`, `cyclic`, and `cyclicAMI` must match exactly between mesh and field.
- Ordinary mesh `wall`/`patch` types are not incorrectly equated with field BC classes such as `fixedValue` or `zeroGradient`.
- This catches the observed `mesh frontAndBack=empty` / `0/p frontAndBack=patch` failure before `/solve`.

## Revision-aware topology invalidation

- `blockMesh`, `snappyHexMesh`, and `createPatch` invalidate cached PreSolve and checkMesh freshness after execution attempts.
- Runtime-repair file edits invalidate the current CaseSeal until a validated `retry_solver` reseals the exact current case.
- Topology mutation clears persisted `mesh_evidence`, forcing the appropriate current `checkMesh` evidence before automatic solver retry.

## Codex CLI backend

- Added `--backend codex` and role-based `CODEX_*_MODEL` routing.
- Requires explicit `--confirm-api-calls`, an installed Codex CLI, and a successful `codex login status`.
- Uses `codex exec --ephemeral --sandbox read-only --output-schema ... --output-last-message ...` from an isolated temporary working directory.
- API-key/base-URL/org/project routing environment variables are removed for Codex subprocesses so the backend uses the Codex CLI login path.
- Pydantic remains authoritative after Codex structured output; one bounded protocol-repair call is allowed for schema-shape errors.
- Codex never writes the CFD workspace or runs OpenFOAM directly.

## Verification

- Added regression coverage for runtime state leakage, explicit decisions, mesh/field constraint mismatch, createPatch invalidation, Codex CLI capability/login checks, isolated Codex execution, and CLI backend routing.
- Full suite: **219 passed**.
