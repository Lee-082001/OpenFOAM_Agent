# v2.12.0 changes

- Retain rejected `execute_case_plan` candidates in Python memory after pre-commit serialization/workspace-safety failure.
- Replace complete-plan re-generation with compact `repair_candidate_case_plan | block` retry contract.
- Candidate repair supports exact patches, raw replacements, typed-dictionary replacements, and dropping optional candidate paths.
- Re-run serialization and whole-bundle preflight after every candidate delta; workspace remains untouched until the complete candidate passes.
- Keep EngineeringPlan metadata repair out of the authoring-retry schema; it remains a normal post-commit RepairTurn responsibility.
- Add bounded retained-candidate context containing manifest plus only implicated artifact content.
- Reduce the static CasePlanRetry structured schema from about 8.4k characters in v2.11 to about 3.3k characters.
- Preserve the existing three-attempt authoring retry bound and all v2.11 transactional safety semantics.
