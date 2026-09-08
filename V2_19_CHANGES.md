# OpenFOAM Agent v2.19.0 changes

## Claude Code subscription backend

- Added `--backend claude` as a fourth autonomous `StructuredLLM` transport.
- Startup verifies the installed Claude Code CLI, requires reliable strict structured-output support, requires `--safe-mode`, and requires `claude auth status` to report `loggedIn=true` with `authMethod=claude.ai`.
- API/provider routing variables (`ANTHROPIC_API_KEY`, `ANTHROPIC_AUTH_TOKEN`, `ANTHROPIC_BASE_URL`, and Bedrock/Vertex/Foundry selectors) are removed from auth checks and model subprocesses so this backend cannot silently switch to a different billing/provider path. `CLAUDE_CODE_OAUTH_TOKEN` remains available as documented subscription OAuth.
- Each model call uses `claude -p --output-format json --json-schema ... --no-session-persistence --safe-mode --tools "" --strict-mcp-config` from an empty temporary working directory.
- Claude Code's `structured_output` is revalidated with the same Pydantic schema before any deterministic action. One bounded protocol-repair attempt may correct schema shape only.
- Added role routing through `CLAUDE_MODEL`, `CLAUDE_INTAKE_MODEL`, `CLAUDE_ENGINEERING_MODEL`, `CLAUDE_POSTPROCESS_MODEL`, and `CLAUDE_REVIEW_MODEL`; omitted model IDs use the Claude Code CLI default.
- Claude usage metadata is projected into the existing input/output/cache token telemetry when the CLI returns it.
- No OpenAI/Codex/Ollama fallback is constructed for `--backend claude`.

## Security boundary

Claude Code is not given the CFD workspace as its working directory and model-side tools/MCP are disabled. Unlike Codex, Claude Code has no equivalent OS-level read-only sandbox flag in this integration; managed enterprise hooks remain an external trusted-installation/policy boundary and are not described as sandboxed by OpenFOAM Agent. All CFD writes, OpenFOAM commands, safety gates, CaseSeal handling, solve approval, and result acceptance remain deterministic/human-owned exactly as before.

## Validation

- Added six Claude backend regression tests covering subscription-auth preflight, API/provider environment stripping, tool/MCP/session isolation flags, structured-output/Pydantic validation, role routing, and explicit cloud-call authorization.
- Full suite: **225 passed**.
