# v2.10.4 — Internal Rationale Overflow Fix

- Fixes a deterministic execution-plan expansion bug where a long `execute_case_plan.goal` was copied into child action `rationale` fields capped at 200 characters.
- Internal child actions now use empty rationales; the parent execution-plan goal remains available for progress/audit output.
- Keeps the 200-character LLM-facing rationale cap, so the token optimization is preserved instead of weakening the schema.
- Adds a regression test proving a 1000-character execution-plan goal can reach `SOLVE_READY` without validation failure.
