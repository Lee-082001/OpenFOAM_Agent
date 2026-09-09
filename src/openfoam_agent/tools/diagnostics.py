from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from openfoam_agent.schemas.common import ToolResult

_FATAL_PATTERNS = (
    ("foam_fatal_io_error", re.compile(r"FOAM\s+FATAL\s+IO\s+ERROR", re.IGNORECASE)),
    ("foam_fatal_error", re.compile(r"FOAM\s+FATAL\s+ERROR", re.IGNORECASE)),
    ("fatal_error", re.compile(r"\bfatal\s+error\b", re.IGNORECASE)),
    ("terminate", re.compile(r"terminate called after throwing", re.IGNORECASE)),
    ("abort", re.compile(r"\baborted\b(?:\s*\(core dumped\))?", re.IGNORECASE)),
    ("segmentation_fault", re.compile(r"\bsegmentation fault\b", re.IGNORECASE)),
    ("floating_point_exception", re.compile(r"\bfloating point exception\b", re.IGNORECASE)),
)


@dataclass(frozen=True)
class NativeFailureDiagnostic:
    """Bounded observation of a failed allowlisted native OpenFOAM command.

    This object is descriptive only. It never selects a repair strategy. Complete
    stdout/stderr remains in the workspace log; ``excerpt`` is the bounded native
    projection that may be shown to the user and returned to an Agent.
    """

    command: str
    return_code: int
    kind: str
    excerpt: str

    def render(self) -> str:
        lines = [
            f"nativeCommand: {self.command}",
            f"returnCode: {self.return_code}",
            f"diagnosticKind: {self.kind}",
        ]
        if self.excerpt:
            lines.append(self.excerpt)
        else:
            lines.append("(no stdout/stderr diagnostic text was captured)")
        return "\n".join(lines)



ValidationStatus = Literal["pass", "fail", "inconclusive"]
FailureCategory = Literal["case", "tool", "infra", "security", "user_contract"]


@dataclass(frozen=True)
class NativeValidationAssessment:
    """Tri-state interpretation of a native validation observation.

    PASS proves the requested probe/consumer accepted the input. FAIL is reserved
    for explicit case-level evidence. INCONCLUSIVE means the tool/runner did not
    establish validity either way and must not trigger blind CFD repair.
    """

    status: ValidationStatus
    category: FailureCategory | None
    diagnostic: NativeFailureDiagnostic | None
    reason: str

    @property
    def workflow_success(self) -> bool:
        return self.status != "fail"


def classify_native_validation(
    result: ToolResult,
    *,
    command_name: str | None = None,
    probe: bool = False,
) -> NativeValidationAssessment:
    """Separate case invalidity from validator/runner uncertainty."""

    command = command_name or _logical_command_name(result)
    if result.success:
        return NativeValidationAssessment("pass", None, None, f"{command} completed successfully.")

    diagnostic = diagnose_openfoam_failure(result, command_name=command)
    if diagnostic.kind in {"foam_fatal_io_error", "foam_fatal_error", "fatal_error"}:
        return NativeValidationAssessment(
            "fail", "case", diagnostic, f"{command} reported an explicit OpenFOAM fatal diagnostic."
        )

    termination = (result.termination_reason or "").strip().lower()
    if not termination and result.return_code == 124:
        termination = "timeout"
    elif not termination and result.return_code in {125, 126}:
        termination = "runner_limit"
    if termination in {
        "timeout", "output_limit", "log_io_error", "spawn_or_io_error",
        "cancelled", "workspace_quota", "aggregate_cpu_limit",
        "aggregate_memory_limit", "runner_limit",
    }:
        category: FailureCategory = "tool" if termination == "timeout" else "infra"
        return NativeValidationAssessment(
            "inconclusive", category, diagnostic,
            f"{command} validation was inconclusive because the runner terminated with {termination}."
        )

    if diagnostic.kind in {"terminate", "abort", "segmentation_fault", "floating_point_exception"}:
        if probe:
            return NativeValidationAssessment(
                "inconclusive", "tool", diagnostic,
                f"{command} advisory validator terminated abnormally without proving the case invalid.",
            )
        return NativeValidationAssessment(
            "fail", "case", diagnostic,
            f"{command} consumer terminated abnormally while using the authored case.",
        )

    if probe:
        return NativeValidationAssessment(
            "inconclusive", "tool", diagnostic,
            f"{command} returned non-zero without proving the case invalid."
        )

    return NativeValidationAssessment(
        "fail", "case", diagnostic, f"{command} consumer returned non-zero without an infrastructure termination."
    )

def diagnose_openfoam_failure(
    result: ToolResult,
    *,
    command_name: str | None = None,
    max_lines: int = 80,
    max_chars: int = 8_000,
    context_before: int = 2,
) -> NativeFailureDiagnostic:
    """Build a command-agnostic failure observation from native stdout/stderr."""

    command = command_name or _logical_command_name(result)
    excerpt, kind = _extract_failure_block(
        result.stdout,
        result.stderr,
        max_lines=max_lines,
        max_chars=max_chars,
        context_before=context_before,
    )
    return NativeFailureDiagnostic(
        command=command,
        return_code=result.return_code,
        kind=kind,
        excerpt=excerpt,
    )


def extract_openfoam_failure_diagnostic(
    stdout: str,
    stderr: str,
    *,
    max_lines: int = 80,
    max_chars: int = 8_000,
    context_before: int = 2,
) -> str:
    """Backward-compatible raw diagnostic projection used by older callers/tests."""

    excerpt, _ = _extract_failure_block(
        stdout,
        stderr,
        max_lines=max_lines,
        max_chars=max_chars,
        context_before=context_before,
    )
    return excerpt


def _extract_failure_block(
    stdout: str,
    stderr: str,
    *,
    max_lines: int,
    max_chars: int,
    context_before: int,
) -> tuple[str, str]:
    if max_lines < 1 or max_chars < 1:
        raise ValueError("diagnostic bounds must be positive")

    parts = [part for part in (stdout, stderr) if part]
    combined = "\n".join(parts).replace("\r\n", "\n").replace("\r", "\n")
    lines = combined.splitlines()
    if not lines:
        return "", "empty_output"

    marker_index: int | None = None
    marker_kind = "output_tail"
    # Pattern priority matters more than first textual occurrence. OpenFOAM startup
    # commonly prints "sigFpe : Enabling floating point exception trapping", which
    # is not a failure. Prefer a later FOAM FATAL block over that harmless banner.
    for kind, pattern in _FATAL_PATTERNS:
        for index, line in enumerate(lines):
            if kind == "floating_point_exception" and "enabling floating point exception trapping" in line.lower():
                continue
            if pattern.search(line):
                marker_index = index
                marker_kind = kind
                break
        if marker_index is not None:
            break

    if marker_index is None:
        selected = lines[-max_lines:]
    else:
        start = max(0, marker_index - max(0, context_before))
        selected = lines[start : start + max_lines]

    while selected and not selected[0].strip():
        selected.pop(0)
    while selected and not selected[-1].strip():
        selected.pop()

    diagnostic = "\n".join(selected).strip()
    if len(diagnostic) <= max_chars:
        return diagnostic, marker_kind

    suffix = "\n... [diagnostic truncated]"
    keep = max(1, max_chars - len(suffix))
    return diagnostic[:keep].rstrip() + suffix, marker_kind


def _logical_command_name(result: ToolResult) -> str:
    if not result.command:
        return "unknown"
    return Path(result.command[0]).name or "unknown"
