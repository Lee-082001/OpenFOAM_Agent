# OpenFOAM Agent v4.2.1

## Progress-first authoring correction

- Removed the stale runtime mutation gate that required an observed syntax excerpt before every `write_case_file` / `patch_case_file`.
- Removed bundle-level raw-file syntax-evidence blocking. Syntax/reference evidence remains available as advisory provenance only.
- Kept hard safety gates for sandbox paths, executable directives (`#codeStream`, `systemCall`, coded entries), includes, non-allowlisted libraries, native command allowlists, approval, case seals, mesh freshness and completion.
- Kept deterministic authoring validation: candidate bundle safety, `FoamFile` header checks, dictionary/native validation, pre-solve completeness and `checkMesh`.
- Repair/revision prompts now carry a compact evidence summary rather than a protected full evidence body.
- Optional QoI/conservation/evidence payloads may be compacted under context pressure; confirmed requirements and execution contracts remain protected.

## Verification scope

Regression and packaging checks are local/mock unless explicitly stated. No claim is made here that a live Codex + OpenFOAM 13 bypass case has completed end-to-end.
