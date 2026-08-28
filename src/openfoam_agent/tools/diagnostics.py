from __future__ import annotations

import re

_FATAL_PATTERNS = (
    re.compile(r"FOAM\s+FATAL\s+(?:IO\s+)?ERROR", re.IGNORECASE),
    re.compile(r"\bfatal\s+error\b", re.IGNORECASE),
    re.compile(r"terminate called after throwing", re.IGNORECASE),
    re.compile(r"\baborted\b(?:\s*\(core dumped\))?", re.IGNORECASE),
    re.compile(r"\bsegmentation fault\b", re.IGNORECASE),
    re.compile(r"\bfloating point exception\b", re.IGNORECASE),
)


def extract_openfoam_failure_diagnostic(
    stdout: str,
    stderr: str,
    *,
    max_lines: int = 80,
    max_chars: int = 8_000,
    context_before: int = 2,
) -> str:
    """Return a bounded raw diagnostic block from a failed OpenFOAM command.

    The complete native output remains in the workspace log. This projection is
    intended for the next agent observation and live CLI progress, so it keeps
    the original diagnostic text/line breaks while bounding size. If no explicit
    fatal marker is present, the tail of the combined output is returned because
    OpenFOAM and libc often place the useful abort reason at the end.
    """

    if max_lines < 1 or max_chars < 1:
        raise ValueError("diagnostic bounds must be positive")

    parts = [part for part in (stdout, stderr) if part]
    combined = "\n".join(parts).replace("\r\n", "\n").replace("\r", "\n")
    lines = combined.splitlines()
    if not lines:
        return ""

    marker_index: int | None = None
    for index, line in enumerate(lines):
        if any(pattern.search(line) for pattern in _FATAL_PATTERNS):
            marker_index = index
            break

    if marker_index is None:
        selected = lines[-max_lines:]
    else:
        start = max(0, marker_index - max(0, context_before))
        selected = lines[start : start + max_lines]

    # Drop only blank padding around the raw block. Interior whitespace and line
    # structure are preserved because those details can matter for OpenFOAM IO
    # diagnostics (file/line/context blocks).
    while selected and not selected[0].strip():
        selected.pop(0)
    while selected and not selected[-1].strip():
        selected.pop()

    diagnostic = "\n".join(selected).strip()
    if len(diagnostic) <= max_chars:
        return diagnostic

    suffix = "\n... [diagnostic truncated]"
    keep = max(1, max_chars - len(suffix))
    return diagnostic[:keep].rstrip() + suffix
