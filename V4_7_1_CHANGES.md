# OpenFOAM Agent v4.7.1

## Revision delta context and retry-safe human revision

v4.7.1 addresses a live v4.7.0 result-review revision failure. The initial case used the controller-owned CaseBuildGraph successfully, `blockMesh`/`checkMesh` and pre-solve validation passed, the first native solver error was repaired through runtime repair, and the second solver attempt completed. The subsequent confirmed human revision failed before its first LLM turn because the revision path reused a full Engineering context whose mandatory content exceeded the deterministic 18,000-character prompt budget.

### Changes

- Added `engineering/revision_context.py` with a dedicated bounded revision capsule.
- The first human-revision turn now uses `human_revision_delta_v1` rather than the full design/capability/history context.
- Added `EngineeringPlanPatch`; repair and strategy-revision actions may return `plan_patch` instead of duplicating the entire EngineeringPlan.
- Patch application preserves controller-owned confirmed/audit/evidence metadata by merging onto the complete sealed baseline held in Python.
- Fixed the feedback-context field access to use `HumanFeedback.statement` in the revision delta path.
- Existing runtime outputs/logs are archived only when a validated revision delta is about to mutate the case, not at revision confirmation time.
- Pre-mutation revision failures restore `REVISION_READY` / `revision_proposed`, retain the prior outputs/logs and create no revision archive, so the same proposal is retryable.
- Revision file/native changes continue through the v4.7 CaseDeltaGraph authority boundary.

### Safety invariants retained

- Confirmed intake and user-owned immutable inputs remain frozen.
- A plan patch cannot replace controller-owned confirmed/audit identity fields.
- Case mutations still require deterministic CaseDeltaGraph preflight and rollback-capable commit.
- Native mesh/solver failures, path/content safety, CaseSeal, execution approval and result provenance remain strict.
