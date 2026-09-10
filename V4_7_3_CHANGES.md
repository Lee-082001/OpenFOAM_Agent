# OpenFOAM Agent v4.7.3 — Pre-Commit Authoring Feasibility Recovery

v4.7.3 addresses a live v4.7.2 case in which the user requested a fully flooded two-branch pipe with junction turbulence. Intake and Engineering design succeeded, but the staged `author_case` turn returned a terminal block: `The required surface artifact is incomplete; I cannot truthfully return a complete, validation-ready case bundle.` The requested geometry was conceptual and no immutable CAD/STL had been supplied, so treating an incomplete Agent-generated surface as a terminal user-input failure violated the progress-first geometry ownership contract.

## Behavioral changes

1. `BlockAction.block_kind` adds `authoring_strategy_infeasible`. During staged authoring this means the frozen implementation representation cannot be completely authored within the bounded contract; it does not mean the user must provide geometry.
2. For compatibility with existing model output, Python also recognizes legacy `other` / `engineering_choice_missing` blocks whose diagnostic clearly refers to an incomplete surface/STL/CAD/geometry artifact, but only when the corresponding required geometry path is Agent-owned rather than an imported user asset.
3. Recoverable pre-commit authoring failures become a failed Engineering event with `failure_scope=strategy` and remain in `ENGINEERING`; they do not transition to terminal `ENGINEERING_BLOCKED`.
4. The next compact model turn uses the existing `StrategyRevisionAction` as a pre-commit plan-only replan. Python supplies a bounded strategy context containing confirmed facts, a compact baseline design projection, the authoring-feasibility trigger, Agent-owned generated geometry paths, environment/tool contracts, and bounded evidence.
5. In pre-commit mode `revise_mesh_strategy` may update `plan_patch` / `updated_plan` only. File patches/replacements/native hints are deliberately ignored because no case candidate has been committed. The next `author_case` turn authors the complete revised manifest from scratch.
6. A successful strategy revision resolves the previous strategy trigger, so the controller returns to staged authoring instead of repeatedly requesting strategy revision. A later new authoring/native failure creates a fresh trigger normally.
7. `EngineeringPlanPatch` may update `confirmed_fact_bindings` implementation mappings when a strategy changes the case files that implement a frozen fact. EngineeringPlan validation still requires exact confirmed fact-ID closure and the confirmed intake digest remains immutable.
8. Strategy context uses a smaller `project_strategy_plan` projection so an authoring-feasibility recovery does not reintroduce the broad 18k-context failure class fixed for human revision in v4.7.1.
9. Design/geometry prompts now discourage large Agent-generated triangulated artifacts when a compact procedural/blockMesh representation is sufficient, while leaving the engineering choice with the Agent.
10. Interactive slash commands strip ANSI cursor/style sequences and zero-width/BOM artifacts before command dispatch only. Natural-language prompts are not normalized. This makes a pasted `/confirm` resilient to an observed terminal-control artifact.

## Invariants retained

- Exact user-owned geometry assets remain immutable and are never replaced by this fallback.
- Python does not choose blockMesh, snappyHexMesh, dimensions, turbulence model, solver, or flow rate.
- `CaseBuildGraph` remains the only initial authoring action authority; `CaseDeltaGraph` remains the repair/revision authority after a case exists.
- Required-file closure, safe paths/content, native toolchain preflight, `surfaceCheck`, real mesh consumers, `checkMesh`, pre-solve validation, CaseSeal, `/solve` approval and runtime/resource limits remain hard gates.
- A genuinely unavailable user-required asset, unsafe content, or unsupported required native capability may still block.

## Regression coverage

`tests/test_v473_authoring_feasibility_recovery.py` covers explicit and legacy infeasible authoring blocks, imported-user-asset exclusion, compact pre-commit strategy context, plan-only strategy revision without workspace mutation, trigger resolution, implementation-binding updates, no-op replan rejection, and slash-command normalization.
