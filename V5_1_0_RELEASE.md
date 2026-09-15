# OpenFOAM Agent v5.1.0

## Reliability contract: requirement-to-case assurance and bounded runtime repair

### Critical quantitative requirement assurance

- Staged EngineeringDesign can now carry `requirement_bindings`, which the controller seals into `confirmed_fact_bindings` without allowing the model to override frozen fact identity or intake digest.
- Explicit user quantitative requirements are machine-verification mandatory when they are represented as typed physical quantities, dimensionless scale/property facts such as Reynolds number, or unit-bearing numeric geometry/material/physics/temporal/motion/boundary facts.
- Mandatory requirements must have a truthful `case_assertion` or `numeric_relation`; provenance alone is no longer enough for these facts.
- Numeric relations use SI-normalized typed quantities when available and are recomputed from actual authored case artifacts. A case that executes but implements the wrong Reynolds number/value is rejected before sealing/retry.

### Entry-level runtime repair authorization

- Runtime repair remains limited to numerical-control surfaces. `fvSchemes` and `fvSolution` remain numerical dictionaries.
- `system/controlDict` is no longer treated as an all-or-nothing file: only `deltaT`, `maxDeltaT`, `adjustTimeStep`, `maxCo`, and `maxAlphaCo` may change automatically.
- Application, end time, functions, libraries, physics/material/boundary files and other control entries remain blocked.
- Requirement bindings are projected into protected repair entries/files at solve approval. A user-bound numerical control (for example an explicitly requested `deltaT`) cannot be changed by automatic repair.
- Fine-grained controlDict authorization happens before mutation and the authorized target hash is carried into the post-mutation seal check.

### Regression coverage

- Missing machine evidence for a critical user quantity is a hard failure.
- Correct numeric relation passes; mismatched artifact value fails.
- `Re=1000` recomputed as 500 is rejected.
- Untyped user Reynolds-scale facts are still classified as mandatory quantitative requirements.
- Authorized `deltaT` repair passes; `endTime`/solver/physics-file changes are rejected.
- A frozen user-bound `deltaT` is rejected before workspace mutation.
