# OpenFOAM Agent v4.0.4

This release fixes a live Codex CLI 0.153.0 intake failure introduced by the stricter v4.0.3 JSONL transport guard.

## Codex diagnostic-event handling
- `item.type=error` is no longer classified as tool/action use. Codex can emit this item as a non-terminal diagnostic while a turn later completes successfully.
- Non-terminal error items are bounded and recorded in `last_transport.diagnostics`; they do not bypass the existing command/search/MCP/file-change fail-closed guard.
- Top-level `error` and `turn.failed` events remain hard failures and now expose their bounded message in `CodexTurnError` diagnostics.
- A stream containing diagnostics but no `turn.completed` still fails closed and reports the last diagnostic.
- Tool/action item types remain rejected exactly as in v4.0.3.

## Verification
- Added regression cases for non-terminal error items, top-level terminal errors, nested `turn.failed` messages, and end-to-end CodexLLM acceptance of a warning item followed by `turn.completed`.
- Actual subscription-authenticated Codex CLI 0.153.0 execution on the user workstation is not performed in the build environment.
