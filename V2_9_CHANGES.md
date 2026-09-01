# OpenFOAM Agent v2.9.0 changes

## Goal

Reduce token use by calling the Engineering LLM only at engineering decision points.

## Main change

`execute_case_plan` lets one LLM turn provide:

- a bounded OpenFOAM case-file bundle,
- dictionary and optional surface validations,
- an ordered mesh pipeline ending in `checkMesh`,
- `required_case_files`, and
- the final `EngineeringPlan`.

Python executes the expanded primitive actions with existing safety and stop-on-failure
semantics. A successful plan can reach `SOLVE_READY` with one Engineering LLM call. On a real
native failure, only the failed prefix is executed and the next LLM turn receives the native
diagnostic for repair.

## Token/call reductions

- CLI capability-provider evidence is preloaded for the small bundled graph.
- Redundant capability search is no longer required for the common path.
- A successful explicit pre-solve check is reused during finalization when its manifest binding
  is still current.
- Default engineering budget: 20 soft / 40 hard (previously 120 / 200).
- Runtime repair defaults: 3 repair cycles, 10 LLM turns per repair cycle.
- Post-processing default: 12 turns.

## Safety preserved

The execution plan is expanded into the existing `write_case_file`, validation, mesh,
`validate_pre_solve`, and `finish_preview` paths. It does not bypass sandboxing, OpenFOAM
command allowlists, tool/native budgets, `checkMesh` evidence, solver-provider validation,
pre-solve completeness, or CaseSeal generation.

## Regression tests

- successful execution plan reaches `SOLVE_READY` in one LLM turn;
- first native failure stops the plan and is included in the next LLM prompt;
- corrected second plan reaches `SOLVE_READY`;
- strict OpenAI structured-output schema remains valid.
