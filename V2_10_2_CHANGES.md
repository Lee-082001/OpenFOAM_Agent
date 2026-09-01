# v2.10.2

Fixes confirmed-fact binding Structured Output failures observed in v2.10.1.

- Replaces the implicit `implementation_refs` string-prefix protocol with explicit `case_files` and `plan_fields` fields.
- Keeps a legacy loader for persisted v2.10.1 `case:` / `plan:` references.
- Makes the compact engineering invariant explicitly describe the binding wire format.
- Safety validation still checks referenced case files deterministically; Python does not infer CFD semantics.
