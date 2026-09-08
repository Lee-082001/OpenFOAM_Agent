# OpenFOAM Agent v3.2.0

## Engineering Evidence / Assumption Contract

- Added durable `EngineeringEvidenceRecord` storage on `CFDState` for capability/reference retrieval payloads.
- `EngineeringEvent.output_excerpt` is now a bounded audit/display projection; evidence events carry `payload_ref` instead of embedding full JSON payloads.
- Fixed the previous truncation-marker overflow where a 12,000-character slice became longer than the Pydantic 12,000-character event limit after the marker was prepended.
- Model context is rebuilt from canonical structured evidence records rather than reparsing event strings.
- Added typed `EngineeringDefaultAssumption` records with explicit `source=engineering_default`, value/unit, basis, rationale, and optional canonical evidence IDs.
- Added `engineering_assumption_policy` to engineering context. Authorized exploratory/easy runs are expected to choose ordinary missing engineering details instead of treating absence of user evidence as a blocker.
- Added structured `BlockAction.block_kind` and `missing_items`; terminal blocks classified as `engineering_choice_missing` are rejected when exploratory completion is authorized.
- Added deterministic retrieval-failure fuse. One evidence infrastructure failure disables further retrieval for that phase; prepare switches to `PrepareDecisionOnlyTurn`, and runtime repair receives an explicit retrieval-unavailable policy.
- Added regression coverage for >50k evidence payloads, event truncation bounds, retrieval-failure escalation, engineering-default authorization/provenance, delegated block rejection, and Claude/Codex schema compilation.

## Verification

- Full regression suite: 267 tests passing before release-version packaging checks.
