# OpenFOAM Agent v4.9.0

v4.9.0 fixes the staged-design boundary rather than adding another retry layer.

## Controller-sealed EngineeringPlan

The LLM-facing `design_case` contract now contains only Agent-owned CFD engineering choices: physics interpretation, execution-provider/solver choice, regions/interfaces, temporal/motion semantics, mesh strategy, engineering defaults, required case files, and optional result intent.

The following fields are removed from staged LLM output and sealed deterministically by Python from state that the controller already owns:

- confirmed intake SHA256
- confirmed fact ID closure
- confirmed fact audit bindings
- canonical evidence IDs
- implementation-evidence bindings
- Foundation target version (derived from the Agent-selected capability providers)
- redundant solver/provider mirrors when `execution` already identifies them

Python still does **not** choose the CFD solver or mesh strategy. It validates the provider IDs selected by the Agent and attaches immutable/audit metadata.

## Why

Earlier staged design asked the model to reproduce controller-owned hashes, opaque IDs, exact binding closure, provider/version metadata, and CFD design in one structured object. A harmless copy/audit mismatch caused the entire design to be rejected and regenerated, which could consume the full LLM budget without any native OpenFOAM progress.

v4.9.0 makes that failure mode structurally impossible: model output cannot override those fields because they are not in the staged design schema.

## Observability

Deterministic non-native validation failures now expose their bounded, path-redacted reason lines through the existing progress reporter. Native and non-native gate failures therefore use the same visible diagnostic path.

## Preserved authority

The Agent still chooses solver/execution topology, mesh strategy, BC/numerical intent, geometry defaults, material/default values, required files and repair deltas. Python continues to own only integrity, sandboxing, evidence validation, native execution authorization and case sealing.
