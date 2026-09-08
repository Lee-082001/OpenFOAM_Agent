# OpenFOAM Agent v3.0.1

## Structured-output transport compatibility

v3.0.1 fixes backend schema-dialect failures that could stop Claude or Codex before the model received the prompt.

### New transport schema compiler

Canonical Pydantic models remain the authoritative domain contracts. CLI backends now receive backend-specific compiled schemas:

- **Claude Code**: JSON Schema 2020-12 `prefixItems` emitted by Pydantic fixed tuples is normalized to portable `items` schemas while preserving fixed tuple cardinality (`minItems`/`maxItems`).
- **Codex CLI / OpenAI strict output schema**: every declared object property is marked `required`, `additionalProperties` is forced to `false`, Python `default` annotations are removed from the transport schema, and tuple `prefixItems` is normalized.
- Responses from both backends are still revalidated against the original Pydantic model. Backend normalization therefore does not replace or weaken deterministic domain validation.

### Failure modes fixed

Claude engineering calls no longer fail at CLI schema validation with diagnostics such as:

`strict mode: unknown keyword: "prefixItems"`

Codex intake calls no longer fail with OpenAI strict-schema diagnostics such as:

`'required' is required to be supplied and to be an array including every key in properties. Missing 'suggested_default'.`

### Regression coverage

New tests cover:

- nullable optional fields becoming explicit required fields for Codex strict output,
- removal of transport-inapplicable `default` annotations,
- fixed 3- and 8-element tuple normalization,
- no remaining `prefixItems` in Claude/Codex transport schemas,
- actual Claude CLI argument and Codex `--output-schema` file generation paths,
- final Pydantic revalidation after backend transport.

Full regression suite: **244 passed**.
