from __future__ import annotations

import re
import sys
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Mapping, Protocol, TextIO


class ProgressLevel(str, Enum):
    QUIET = "quiet"
    NORMAL = "normal"
    VERBOSE = "verbose"


class ProgressImportance(str, Enum):
    NORMAL = "normal"
    VERBOSE = "verbose"


@dataclass(frozen=True)
class ProgressEvent:
    phase: str
    message: str
    status: str = "info"
    step: int | None = None
    limit: int | None = None
    importance: ProgressImportance = ProgressImportance.NORMAL
    metrics: Mapping[str, object] = field(default_factory=dict)
    details: tuple[str, ...] = field(default_factory=tuple)


class ProgressReporter(Protocol):
    def enabled(self, importance: ProgressImportance = ProgressImportance.NORMAL) -> bool: ...

    def emit(self, event: ProgressEvent) -> None: ...


class NullProgressReporter:
    def enabled(self, importance: ProgressImportance = ProgressImportance.NORMAL) -> bool:
        del importance
        return False

    def emit(self, event: ProgressEvent) -> None:
        del event


class CLIProgressReporter:
    """Render safe, deterministic workflow observations without model rationale.

    Progress goes to stderr by default so --json stdout remains machine-readable.
    """

    def __init__(
        self,
        level: str | ProgressLevel = ProgressLevel.NORMAL,
        *,
        stream: TextIO | None = None,
    ) -> None:
        self.level = ProgressLevel(level)
        self.stream = stream or sys.stderr

    def enabled(self, importance: ProgressImportance = ProgressImportance.NORMAL) -> bool:
        if self.level == ProgressLevel.QUIET:
            return False
        if importance == ProgressImportance.VERBOSE and self.level != ProgressLevel.VERBOSE:
            return False
        return True

    def emit(self, event: ProgressEvent) -> None:
        if not self.enabled(event.importance):
            return
        phase = event.phase.upper().replace("_", "-")
        if event.step is not None:
            if event.limit is not None:
                prefix = f"[{phase} {event.step:02d}/{event.limit}]"
            else:
                prefix = f"[{phase} {event.step:02d}]"
        else:
            prefix = f"[{phase}]"
        status = {
            "start": "",
            "info": "",
            "success": " OK",
            "failure": " FAIL",
            "warning": " WARN",
        }.get(event.status, f" {event.status.upper()}")
        print(f"{prefix}{status} {event.message}", file=self.stream, flush=True)
        if event.metrics:
            payload = ", ".join(
                f"{key}={_format_metric(value)}"
                for key, value in event.metrics.items()
                if value is not None
            )
            if payload:
                print(f"  {payload}", file=self.stream, flush=True)
        if event.details:
            label = "reason:" if event.status == "failure" else "details:"
            print(f"  {label}", file=self.stream, flush=True)
            for detail in event.details:
                text = _compact(str(detail), 800)
                if text:
                    print(f"    - {text}", file=self.stream, flush=True)


_ENGINEERING_VERBOSE = {
    "read_reference",
    "list_case_files",
    "read_case_file",
}
_POSTPROCESS_VERBOSE = {
    "read_postprocess_reference",
    "list_result_files",
    "read_result_file",
}


def action_importance(action_type: str) -> ProgressImportance:
    if action_type in _ENGINEERING_VERBOSE | _POSTPROCESS_VERBOSE:
        return ProgressImportance.VERBOSE
    return ProgressImportance.NORMAL


def describe_action(action: object) -> str:
    action_type = str(getattr(action, "type", "action"))
    mapping = {
        "inspect_environment": "OpenFOAM environment 확인",
        "search_capabilities": "capability graph 조회",
        "search_references": "OpenFOAM reference 탐색",
        "gather_evidence": "evidence gap batch 조회",
        "repair_runtime_case": "runtime case delta repair",
        "read_reference": "OpenFOAM reference 읽기",
        "list_case_files": "case 파일 목록 확인",
        "read_case_file": "case 파일 읽기",
        "write_case_file": "case 파일 작성",
        "delete_case_file": "case 파일 삭제",
        "validate_dictionary": "foamDictionary 검사",
        "surface_check": "surfaceCheck 실행",
        "run_mesh_command": "mesh command 실행",
        "validate_pre_solve": "pre-solve completeness 검사",
        "sequence": "engineering sequence 계획",
        "finish_preview": "engineering plan 최종 검증 및 case seal",
        "retry_solver": "runtime repair 검증 및 solver 재시도 요청",
        "block": "engineering 중단/검토 요청",
        "search_postprocess_references": "post-processing reference 탐색",
        "read_postprocess_reference": "post-processing reference 읽기",
        "write_postprocess_config": "post-processing config 작성",
        "run_foam_postprocess": "foamPostProcess 실행",
        "list_result_files": "결과 파일 목록 확인",
        "read_result_file": "결과 파일 읽기",
        "analyze_force_coefficients": "Cd/Cl 시계열 deterministic 분석",
        "finish_postprocessing": "post-processing 결과 최종화",
        "block_postprocessing": "post-processing 중단/검토 요청",
    }
    base = mapping.get(action_type, action_type)
    target = _safe_action_target(action)
    return f"{base}: {target}" if target else base


def _safe_action_target(action: object) -> str:
    action_type = str(getattr(action, "type", ""))
    if action_type in {"write_case_file", "delete_case_file", "read_case_file", "validate_dictionary", "surface_check"}:
        return str(getattr(action, "path", ""))
    if action_type == "run_mesh_command":
        return str(getattr(action, "command", ""))
    if action_type == "sequence":
        return _compact(str(getattr(action, "goal", "")), 120)
    if action_type in {"search_capabilities", "search_references", "search_postprocess_references"}:
        return _compact(str(getattr(action, "query", "")), 100)
    if action_type == "gather_evidence":
        gaps = getattr(action, "gaps", [])
        return ", ".join(str(getattr(item, "gap_id", "")) for item in gaps[:4])
    if action_type == "repair_runtime_case":
        return _compact(str(getattr(action, "diagnosis", "")), 100)
    if action_type in {"read_reference", "read_postprocess_reference"}:
        reference = str(getattr(action, "reference", ""))
        return reference.rsplit("/", 1)[-1] if reference else ""
    if action_type == "write_postprocess_config":
        return str(getattr(action, "path", ""))
    if action_type == "run_foam_postprocess":
        return str(getattr(action, "dictionary_path", ""))
    if action_type == "read_result_file":
        return str(getattr(action, "path", ""))
    if action_type == "analyze_force_coefficients":
        return str(getattr(action, "coefficient_path", ""))
    return ""


def _compact(value: str, limit: int) -> str:
    value = re.sub(r"\s+", " ", value).strip()
    return value if len(value) <= limit else value[: limit - 3] + "..."


def _format_metric(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


class SolverProgressTracker:
    """Parse a live foamRun stream into bounded progress events.

    Normal mode is wall-clock throttled to avoid printing thousands of CFD time
    steps. Verbose mode emits every Time marker plus residual lines.
    """

    _TIME = re.compile(
        r"^\s*Time\s*=\s*(?P<time>[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*(?:s)?\s*$",
        re.IGNORECASE,
    )
    _CO = re.compile(
        r"Courant Number\s+mean:\s*(?P<mean>[+\-0-9.eE]+)\s+max:\s*(?P<max>[+\-0-9.eE]+)",
        re.IGNORECASE,
    )
    _RESIDUAL = re.compile(
        r"Solving for\s+(?P<field>[^,]+),\s+Initial residual =\s*(?P<initial>[+\-0-9.eE]+),\s+Final residual =\s*(?P<final>[+\-0-9.eE]+)",
        re.IGNORECASE,
    )

    def __init__(
        self,
        reporter: ProgressReporter,
        *,
        attempt: int,
        attempt_limit: int,
        normal_interval_seconds: float = 1.0,
    ) -> None:
        self.reporter = reporter
        self.attempt = attempt
        self.attempt_limit = attempt_limit
        self.normal_interval_seconds = normal_interval_seconds
        self.latest_co_mean: float | None = None
        self.latest_co_max: float | None = None
        self._last_normal_emit = 0.0
        self._seen_time = False

    def feed(self, line: str) -> None:
        co = self._CO.search(line)
        if co:
            try:
                self.latest_co_mean = float(co.group("mean"))
                self.latest_co_max = float(co.group("max"))
            except ValueError:
                pass
            return

        time_match = self._TIME.match(line)
        if time_match:
            try:
                current_time = float(time_match.group("time"))
            except ValueError:
                return
            now = time.monotonic()
            verbose = self.reporter.enabled(ProgressImportance.VERBOSE)
            should_normal = (
                not self._seen_time
                or now - self._last_normal_emit >= self.normal_interval_seconds
            )
            if verbose or should_normal:
                self.reporter.emit(
                    ProgressEvent(
                        phase="runtime",
                        message="foamRun 진행",
                        status="info",
                        importance=(
                            ProgressImportance.VERBOSE if verbose else ProgressImportance.NORMAL
                        ),
                        metrics={
                            "attempt": f"{self.attempt}/{self.attempt_limit}",
                            "Time": current_time,
                            "CoMean": self.latest_co_mean,
                            "CoMax": self.latest_co_max,
                        },
                    )
                )
                if should_normal:
                    self._last_normal_emit = now
            self._seen_time = True
            return

        residual = self._RESIDUAL.search(line)
        if residual and self.reporter.enabled(ProgressImportance.VERBOSE):
            self.reporter.emit(
                ProgressEvent(
                    phase="runtime",
                    message=f"linear solve: {residual.group('field').strip()}",
                    importance=ProgressImportance.VERBOSE,
                    metrics={
                        "initial": residual.group("initial"),
                        "final": residual.group("final"),
                    },
                )
            )
