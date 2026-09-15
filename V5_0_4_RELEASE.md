# OpenFOAM Agent v5.0.4

## Scope

v5.0.4 is a maintenance release over v5.0.3. It preserves the v5.0.2 scope/interface fixes and the v5.0.3 strategy-revision recovery fix, while hardening pre-solve consumer coverage and runtime-repair context handling for multi-region transient cases.

## Runtime-repair bounded/partitioned context

Runtime repair no longer sends the complete approved EngineeringPlan into the model prompt. The complete plan remains Python-held authority and is bound by its digest. The model receives a failure-local projection containing the selected execution driver/scopes, implicated region layouts/interfaces, solve-critical or diagnostic-linked required files, focused case-file contents, the native failure, and bounded supporting evidence.

The runtime-repair prompt is deterministically partitioned under the configured engineering context budget. Successive projections reduce focused file count/content, supporting evidence, and diagnostic detail while retaining solver/execution identity and the observed native failure. If the smallest safe projection still cannot fit, no repair mutation is authorized and the original runtime failure remains intact.

## One-step shadow consumer validation

Transient pre-solve validation now complements the existing endTime=0 initialization probe with an isolated one-step shadow execution. The production case is never modified. The controller copies only 0/, constant/, and system/ into a bounded temporary validation case, forces serial startTime=0 execution with adjustTimeStep=false, and advances exactly one literal positive deltaT.

This catches failures that occur only when the first equation is assembled or solved, such as a missing fvSolution solver entry for a solid energy variable. Parallel execution, custom runtime arguments, non-empty function objects, ambiguous/non-literal deltaT, symlinked input, or oversized shadow input are treated as inconclusive rather than as fabricated case failures.

## Regression coverage

New v5.0.4 tests verify:

1. a two-solid foamMultiRun shadow case may pass zero-step initialization yet fail one-step on `keyword e is undefined`, and the production controlDict remains unchanged;
2. pre-solve validation classifies that one-step native failure as a case failure before user `/solve` approval;
3. an intentionally bloated multi-region runtime-repair payload is projected below the 18,000-character model budget without resending the full required-file/binding archive;
4. the real context-controller runtime-repair route uses the bounded projection before the model call;
5. package and project metadata report v5.0.4 consistently.

## Compatibility

No Y-pipe, solid, solver, or meshing strategy is hard-coded. The Agent still chooses CFD strategy and numerical settings. Python owns only safety, immutable authority, bounded context projection, native shadow validation, and transition/retry gates.
