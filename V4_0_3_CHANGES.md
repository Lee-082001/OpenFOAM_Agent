# OpenFOAM Agent v4.0.3

This release addresses a live Codex engineering failure in which a normal structured
model call emitted a `command_execution`, `web_search`, MCP/action, file-change, or
unknown item event and v4.0.2 rejected the turn only after the event had already been
observed.

## Codex model-only transport hardening
- `codex exec` now receives an explicit model-only feature-disable profile in addition to
  `--ephemeral`, `--ignore-user-config`, and `--sandbox read-only`.
- The profile disables shell/unified exec, code mode, web/search, apps/plugins, browser,
  computer-use, image-generation, skill-search, and multi-agent surfaces used by the
  supported Codex CLI generation.
- `approval_policy="never"` is forced for the transport call.
- `check_codex_cli()` now requires the repeatable `--disable` feature surface.
- The post-call JSON event inspector remains fail-closed because a CLI/model regression
  could ignore a requested feature disable.
- Transport failures now identify the exact event and item type. For command execution,
  web search, MCP/tool calls, and file changes, a bounded command/query/tool/path hint is
  included instead of the previous generic "tool/action or unknown item" error.
- A successful event stream records that the model-only profile was requested and that
  the post-execution event guard ran. It deliberately does **not** claim independent
  verification of the server-side tool surface.

## Evidence retrieval projection
- Search/retrieval tools may still collect all useful candidates into the durable evidence
  record and gap ledger.
- A deterministic relevance/fairness selector promotes at most six newly retrieved items
  per evidence batch into the immediate model/event context by default.
- Full reference excerpts and stronger runtime/native capability evidence receive higher
  priority than shallow snippets or weak discovery-only evidence.
- Unpromoted retrieval candidates cannot leak back into the same bounded turn through
  targeted-capability or generic evidence fallback paths.
- The CLI exposes `--engineering-new-evidence-items` (default 6, allowed 1..12).
- Progress now distinguishes `retrieved` from `promoted to model context` counts.

## Verification
- Full regression: 455 passed, 0 failed, 0 skipped in the development container.
- Added model-only command construction and exact event-diagnostic tests.
- Added/updated evidence projection regression covering a 100-item retrieval with a
  six-item promotion limit while preserving all 100 items in durable state.
- Actual subscription-authenticated Codex CLI 0.153.0 execution on the user's workstation
  was not run in this build environment, so live tool-surface suppression is not claimed
  as verified until the user reruns the reproduced case.
