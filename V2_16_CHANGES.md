# OpenFOAM Agent v2.16.0

## Compact semantic evidence

- Replaces v2.15's preferred excerpt/value-token duplication with artifact pointers.
- Structural semantic assertions prefer `path + entry_path + expected_value`; short anchors are fallback for raw/list-heavy files.
- Numeric relation terms prefer `path + entry_path` or `path + anchor + number_index`; Python extracts the actual scalar from the current case before recomputation.
- Semantic evidence pointers count as implementation references, so duplicate `case_files` declarations are no longer required.
- Binding explanations are optional and shorter.
- Legacy v2.15 `contains`, `excerpt`, and `value_token` assertions remain accepted and their empty new defaults are stripped from canonical plan hashing for compatibility.
- Default LLM output ceiling increases from 16000 to 24000 tokens only as a truncation safety margin; the semantic representation itself is smaller.

## Verification

- Dictionary extraction is deterministic and comment-stripped.
- Raw numeric anchors are resolved against current artifact text and a bounded numeric index.
- Python still does not choose Reynolds-number formulas, reference scales, boundary conditions, solvers, or other CFD engineering decisions.

## Regression

- 207 tests pass in the v2.16 release tree.
- All production Structured Output schemas pass the strict-schema compatibility audit.
- Same-file multiple semantic pointers are allowed; exact duplicate assertions remain rejected.
