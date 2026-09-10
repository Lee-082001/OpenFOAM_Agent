from __future__ import annotations

import os
import hashlib
import json
import signal
import uuid
from contextlib import contextmanager
import queue
import re
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Callable, Mapping, Sequence

from openfoam_agent.schemas.common import ToolResult
from openfoam_agent.contracts.models import ResourceLimits
from .execution_policy import (ExecutionContext, ValidationExecutionContext, ExecutionPolicyError, ProcessBudget, command_effect)
from openfoam_agent.tools.installation import OpenFOAMInstallationDiscovery


class UnsafeCommandError(ExecutionPolicyError):
    pass


class SafeRunner:
    """Execute only trusted OpenFOAM utilities inside an optional workspace root.

    Security properties:
    - executable names are discovered from the sourced, trusted OpenFOAM installation;
    - the resolved executable must live under a trusted OpenFOAM installation root;
    - subprocesses receive a reduced OpenFOAM/runtime environment, not the parent
      process environment (so API keys/tokens are not inherited);
    - cwd is confined to the configured workspace.
    """

    # Backward-compatible alias for offline/unit-test environments only. It is never
    # the authority when a sourced Foundation 13/14 installation is discovered.
    OFFLINE_FALLBACK_ALLOWED = {
        "blockMesh",
        "surfaceFeatureExtract",
        "surfaceCheck",
        "snappyHexMesh",
        "createPatch",
        "checkMesh",
        "foamRun",
        "foamPostProcess",
        "foamDictionary",
    }
    DEFAULT_ALLOWED = OFFLINE_FALLBACK_ALLOWED
    _SAFE_ENV_EXACT = {
        "HOME",
        "USER",
        "LOGNAME",
        "LANG",
        "LC_ALL",
        "TERM",
        "TMPDIR",
        "TEMP",
        "TMP",
        "PATH",
        "LD_LIBRARY_PATH",
        "LIBRARY_PATH",
        "CPATH",
        "CPLUS_INCLUDE_PATH",
        "SCOTCH_ARCH_PATH",
        "BOOST_ARCH_PATH",
        "CGAL_ARCH_PATH",
        "FFTW_ARCH_PATH",
        "ParaView_DIR",
        "ParaView_VERSION",
    }
    _SAFE_ENV_PREFIXES = ("WM_", "FOAM_", "MPI_", "OMPI_", "OPAL_", "PMI_", "OMP_")
    _UNTRUSTED_OPENFOAM_ENV = {
        "FOAM_USER_APPBIN",
        "FOAM_USER_LIBBIN",
        "FOAM_SITE_APPBIN",
        "FOAM_SITE_LIBBIN",
        "FOAM_RUN",
    }
    _TRUSTED_PATH_ENV = {
        "WM_PROJECT_DIR",
        "WM_PROJECT_INST_DIR",
        "WM_THIRD_PARTY_DIR",
        "FOAM_APPBIN",
        "FOAM_LIBBIN",
        "FOAM_EXT_LIBBIN",
        "SCOTCH_ARCH_PATH",
        "BOOST_ARCH_PATH",
        "CGAL_ARCH_PATH",
        "FFTW_ARCH_PATH",
        "FOAM_ETC",
        "FOAM_SRC",
        "FOAM_TUTORIALS",
        "FOAM_APPBIN",
        "FOAM_MODULES",
        "FOAM_SOLVERS",
        "FOAM_UTILITIES",
        "FOAM_APP",
    }
    _SYSTEM_PATH_ROOTS = tuple(Path(item) for item in ("/usr/bin", "/bin"))
    _SYSTEM_LIBRARY_ROOTS = tuple(
        Path(item) for item in ("/usr/lib", "/usr/lib64", "/lib", "/lib64")
    )

    def __init__(
        self,
        allowed_commands: set[str] | None = None,
        *,
        workspace_root: str | Path | None = None,
        max_timeout: int = 86400,
        trusted_executable_roots: Sequence[str | Path] | None = None,
        trusted_library_roots: Sequence[str | Path] | None = None,
        base_env: Mapping[str, str] | None = None,
        resource_limits: ResourceLimits | None = None,
    ) -> None:
        self.workspace_root = (
            Path(workspace_root).expanduser().resolve() if workspace_root else None
        )
        self.max_timeout = max_timeout
        self.resource_limits = resource_limits or ResourceLimits(wall_seconds=max_timeout)
        self.budget = ProcessBudget(limit=self.resource_limits.max_native_processes,
                                    ledger_path=(self.workspace_root / "process-ledger.json" if self.workspace_root else None))
        self._execution_context: ExecutionContext | None = None
        self._validation_context: ValidationExecutionContext | None = None
        self._mpi_ranks = 1
        self._mpi_launcher: str | None = None
        self.process_observer = None
        self.isolation = None
        self._base_env = dict(os.environ if base_env is None else base_env)
        self.trusted_executable_roots = self._resolve_trusted_roots(
            trusted_executable_roots
        )
        self.trusted_library_roots = self._resolve_trusted_library_roots(
            trusted_library_roots
        )
        self.installation = OpenFOAMInstallationDiscovery(
            base_env=self._base_env,
            trusted_roots=self.trusted_executable_roots,
        ).discover()
        if allowed_commands is None:
            discovered = self.installation.executable_names
            # Offline/unit-test environments may not have a sourced installation. Keep the
            # historical minimum only as a non-native compatibility fallback; once a trusted
            # installation is present, the installation itself is the command authority.
            self.allowed_commands = set(discovered or self.OFFLINE_FALLBACK_ALLOWED)
        else:
            self.allowed_commands = set(allowed_commands)

    def run(
        self,
        command: list[str],
        cwd: str | Path | None = None,
        timeout: int = 3600,
        stream_output: bool = False,
        output_callback: Callable[[str], None] | None = None,
    ) -> ToolResult:
        if not command:
            raise ValueError("command must not be empty")
        if timeout <= 0 or timeout > self.max_timeout:
            raise UnsafeCommandError(
                f"Requested timeout {timeout}s exceeds bounded runner policy."
            )

        raw_exe = command[0]
        exe = Path(raw_exe).name
        if raw_exe != exe:
            raise UnsafeCommandError(
                f"Command must use a bare allowlisted executable name: {raw_exe}"
            )
        if exe not in self.allowed_commands:
            raise UnsafeCommandError(f"Command is not allowlisted: {exe}")

        resolved_cwd = self._validate_cwd(cwd)
        self._validate_arguments(command[1:], resolved_cwd)
        env = self.sanitized_environment()
        executable = self.resolve_trusted_executable(exe, env=env)
        actual_command = [str(executable), *command[1:]]

        effect = command_effect(exe, command[1:], self.installation)
        if effect in {"unknown", "destructive"}:
            raise UnsafeCommandError(f"Command effect {effect!r} is not authorized: {exe}")
        if effect == "solve":
            if resolved_cwd is None:
                raise UnsafeCommandError("Solver execution requires a bounded case directory.")
            if self._execution_context is not None:
                self._execution_context.validate(command, resolved_cwd, timeout, ranks=self._mpi_ranks)
            elif self._validation_context is not None:
                self._validation_context.validate(command, resolved_cwd, timeout, ranks=self._mpi_ranks)
            else:
                raise UnsafeCommandError("Main calculation requires explicit runtime approval; only Python-owned zero-step shadow validation is permitted before approval.")
        if effect == "reconstruction" and self._execution_context is None:
            raise UnsafeCommandError("Reconstruction requires the approved runtime execution context.")
        if effect == "decomposition":
            if resolved_cwd is None:
                raise UnsafeCommandError("Decomposition requires a case directory.")
            import re
            text = (resolved_cwd / "system/decomposeParDict").read_text(encoding="utf-8")
            match = re.search(r"\bnumberOfSubdomains\s+(\d+)\s*;", text)
            if not match or not 1 <= int(match.group(1)) <= self.resource_limits.max_ranks:
                raise UnsafeCommandError("Decomposition rank count is missing or exceeds the configured rank cap.")
        if self._mpi_launcher is not None:
            launcher = self._resolve_mpi_launcher(self._mpi_launcher, env)
            actual_command = [str(launcher), "-np", str(self._mpi_ranks), *actual_command]
        if os.name != "posix" and (self.resource_limits.cpu_seconds or self.resource_limits.memory_bytes):
            raise UnsafeCommandError("Requested OS resource limits are unsupported on this platform.")
        from openfoam_agent.tools.linux_isolation import workspace_execution_lock, bounded_tree_bytes, IsolationUnavailable
        root = self.workspace_root or resolved_cwd
        if root is None:
            raise UnsafeCommandError("A bounded workspace is required for native execution.")
        with workspace_execution_lock(root):
            # Multiple runner instances must not use stale in-memory process budgets.
            if self.budget.ledger_path and self.budget.ledger_path.exists():
                fresh = json.loads(self.budget.ledger_path.read_text())["records"]
                if fresh != self.budget.records:
                    raise UnsafeCommandError("Process ledger changed; reload/reconcile before executing.")
            if any(row.get("status") in {"running", "spawn_intent"} for row in self.budget.records):
                raise UnsafeCommandError("An earlier subprocess has uncertain completion; no automatic replay.")
            if bounded_tree_bytes(root) > self.resource_limits.max_case_bytes:
                raise UnsafeCommandError("Aggregate workspace byte quota exceeded before spawn.")
            spent_wall = sum(row.get("wall_seconds", 0) for row in self.budget.records)
            spent_cpu = sum((row.get("aggregate_cpu_seconds") or 0) for row in self.budget.records)
            if spent_wall >= self.resource_limits.total_wall_seconds:
                raise UnsafeCommandError("Cumulative native wall budget exhausted.")
            if self.resource_limits.total_cpu_seconds is not None and spent_cpu >= self.resource_limits.total_cpu_seconds:
                raise UnsafeCommandError("Cumulative native CPU budget exhausted.")
            if self.resource_limits.total_cpu_seconds is not None and any(row.get("pid") and row.get("aggregate_cpu_seconds") is None for row in self.budget.records):
                raise UnsafeCommandError("Prior native CPU accounting is unknown; a cumulative guarantee cannot be invented on resume.")
            if self.resource_limits.total_cpu_seconds is not None and self.isolation is None:
                raise UnsafeCommandError("Aggregate CPU accounting requires the strict cgroup isolation backend.")
            if sum(row.get("output_bytes", 0) for row in self.budget.records) >= self.resource_limits.max_total_output_bytes:
                raise UnsafeCommandError("Cumulative native output budget exhausted.")
            if self.isolation is not None:
                self.isolation.preflight()
            row = self.budget.reserve(command, effect)
            return self._run_streaming(
                actual_command, logical_command=command, cwd=resolved_cwd,
                timeout=min(timeout, self.resource_limits.total_wall_seconds-spent_wall),
                env=env, echo_output=stream_output, output_callback=output_callback, record=row,
            )

    @contextmanager
    def validation_execution(self, context: ValidationExecutionContext):
        if self._execution_context is not None:
            raise UnsafeCommandError("Validation execution cannot overlap an approved production runtime.")
        previous = self._validation_context
        self._validation_context = context
        try:
            yield
        finally:
            self._validation_context = previous

    @contextmanager
    def approved_execution(self, context: ExecutionContext):
        previous = self._execution_context
        old_limits, old_limit = self.resource_limits, self.budget.limit
        approved = context.approval.resource_limits
        values = {}
        for key, value in old_limits.model_dump().items():
            other = getattr(approved, key)
            values[key] = other if value is None else value if other is None else min(value, other)
        self.resource_limits = ResourceLimits(**values)
        self.budget.limit = min(old_limit, self.resource_limits.max_native_processes)
        self._execution_context = context
        try:
            yield
        finally:
            self._execution_context = previous
            self.resource_limits, self.budget.limit = old_limits, old_limit

    def _resolve_mpi_launcher(self, name: str, env: Mapping[str, str]) -> Path:
        if name not in {"mpirun", "mpiexec"}:
            raise UnsafeCommandError("Only operator-selected local MPI launchers are supported.")
        found = shutil.which(name, path=env.get("PATH", ""))
        if not found:
            raise UnsafeCommandError("MPI launcher is not installed; no solver was started.")
        path = Path(found).resolve()
        roots = (*self.trusted_executable_roots, *self._SYSTEM_PATH_ROOTS)
        if not any(_is_within(path, root.resolve()) for root in roots):
            raise UnsafeCommandError("MPI launcher is outside trusted system/installation roots.")
        return path

    def run_mpi(self, command: list[str], *, ranks: int, launcher: str, **kwargs) -> ToolResult:
        if self._execution_context is None or ranks < 2:
            raise UnsafeCommandError("Local MPI requires explicit runtime approval and at least two ranks.")
        approved = self._execution_context.approval.execution.get("parallel", {})
        if approved.get("mode") != "local_mpi" or approved.get("launcher") != launcher:
            raise UnsafeCommandError("MPI launcher/mode differs from approval.")
        old = self._mpi_ranks, self._mpi_launcher
        self._mpi_ranks, self._mpi_launcher = ranks, launcher
        try:
            return self.run([*command, "-parallel"], **kwargs)
        finally:
            self._mpi_ranks, self._mpi_launcher = old

    def executable_status(self, exe: str) -> dict[str, object]:
        try:
            path = self.resolve_trusted_executable(exe, env=self.sanitized_environment())
        except UnsafeCommandError as exc:
            return {"name": exe, "available": False, "trusted": False, "reason": str(exc)}
        return {"name": exe, "available": True, "trusted": True, "path_redacted": path.name}

    def resolve_trusted_executable(
        self,
        exe: str,
        *,
        env: Mapping[str, str] | None = None,
    ) -> Path:
        if exe not in self.allowed_commands:
            raise UnsafeCommandError(f"Command is not allowlisted: {exe}")
        if not self.trusted_executable_roots:
            raise UnsafeCommandError(
                "No trusted OpenFOAM installation root is configured. Source the "
                "OpenFOAM environment (WM_PROJECT_DIR) before native execution."
            )
        search_env = dict(env or self.sanitized_environment())
        found = shutil.which(exe, path=search_env.get("PATH", ""))
        if not found:
            raise UnsafeCommandError(f"Allowlisted OpenFOAM executable is unavailable: {exe}")
        resolved = Path(found).expanduser().resolve()
        if not resolved.is_file():
            raise UnsafeCommandError(f"Resolved executable is not a file: {exe}")
        if not any(_is_within(resolved, root) for root in self.trusted_executable_roots):
            raise UnsafeCommandError(
                f"Executable '{exe}' resolved outside the trusted OpenFOAM installation."
            )
        return resolved

    def sanitized_environment(self) -> dict[str, str]:
        env: dict[str, str] = {}
        for key, value in self._base_env.items():
            if key in self._UNTRUSTED_OPENFOAM_ENV:
                continue
            if key in self._SAFE_ENV_EXACT or key.startswith(self._SAFE_ENV_PREFIXES):
                env[key] = value

        for key in self._TRUSTED_PATH_ENV:
            value = env.get(key, "").strip()
            if not value:
                continue
            try:
                resolved = Path(value).expanduser().resolve()
            except OSError:
                env.pop(key, None)
                continue
            roots = (
                self.trusted_library_roots
                if key in {
                    "WM_PROJECT_INST_DIR", "WM_THIRD_PARTY_DIR", "FOAM_LIBBIN", "FOAM_EXT_LIBBIN",
                    "SCOTCH_ARCH_PATH", "BOOST_ARCH_PATH", "CGAL_ARCH_PATH",
                    "FFTW_ARCH_PATH",
                }
                else self.trusted_executable_roots
            )
            if not any(_is_within(resolved, root) for root in roots):
                env.pop(key, None)

        if self.workspace_root is not None:
            runtime_home = self.workspace_root / ".runtime-home"
            runtime_home.mkdir(parents=True, exist_ok=True)
            env["HOME"] = str(runtime_home)

        # Never let PATH/loader search user-controlled directories. OpenFOAM paths
        # under the trusted installation are preserved; standard root-owned system
        # paths remain available for normal utilities and libc dependencies.
        env["PATH"] = self._filtered_search_path(
            self._base_env.get("PATH", ""),
            allowed_system_roots=self._SYSTEM_PATH_ROOTS,
        )
        if "LD_LIBRARY_PATH" in self._base_env:
            env["LD_LIBRARY_PATH"] = self._filtered_search_path(
                self._base_env.get("LD_LIBRARY_PATH", ""),
                allowed_system_roots=self._SYSTEM_LIBRARY_ROOTS,
                trusted_roots=self.trusted_library_roots,
            )
        return env

    def _filtered_search_path(
        self,
        value: str,
        *,
        allowed_system_roots: Sequence[Path],
        trusted_roots: Sequence[Path] | None = None,
    ) -> str:
        accepted: list[str] = []
        seen: set[str] = set()
        trust = tuple(trusted_roots or self.trusted_executable_roots)
        for raw in value.split(os.pathsep):
            if not raw:
                continue
            try:
                path = Path(raw).expanduser().resolve()
            except OSError:
                continue
            trusted = any(_is_within(path, root) for root in trust)
            system = any(path == root or _is_within(path, root) for root in allowed_system_roots)
            if (trusted or system) and str(path) not in seen:
                accepted.append(str(path))
                seen.add(str(path))
        # Ensure basic system tools are available even when the inherited PATH was sparse.
        for root in allowed_system_roots:
            if root.is_dir() and str(root.resolve()) not in seen:
                accepted.append(str(root.resolve()))
                seen.add(str(root.resolve()))
        return os.pathsep.join(accepted)

    def _resolve_trusted_roots(
        self,
        supplied: Sequence[str | Path] | None,
    ) -> tuple[Path, ...]:
        raw_roots: list[str | Path] = list(supplied or [])
        if supplied is None:
            project = self._base_env.get("WM_PROJECT_DIR", "").strip()
            if project:
                raw_roots.append(project)
        roots: list[Path] = []
        for raw in raw_roots:
            path = Path(raw).expanduser().resolve()
            if path.is_dir() and path not in roots:
                roots.append(path)
        return tuple(roots)

    def _resolve_trusted_library_roots(
        self,
        supplied: Sequence[str | Path] | None,
    ) -> tuple[Path, ...]:
        """Resolve library-only trust roots without broadening executable trust.

        Foundation source installations commonly place ThirdParty-* beside the
        OpenFOAM project tree. v4.7.1 filtered LD_LIBRARY_PATH only against
        WM_PROJECT_DIR, which could remove FOAM_EXT_LIBBIN/Scotch paths and make a
        healthy snappyHexMesh fail with ``libscotch.so`` missing. Library roots are
        therefore tracked separately from executable roots.
        """
        if supplied is not None:
            raw_roots: list[str | Path] = list(supplied)
        else:
            raw_roots = list(self.trusted_executable_roots)
            project_text = self._base_env.get("WM_PROJECT_DIR", "").strip()
            project = Path(project_text).expanduser().resolve() if project_text else None
            inst_text = self._base_env.get("WM_PROJECT_INST_DIR", "").strip()
            inst = Path(inst_text).expanduser().resolve() if inst_text else None
            third_text = self._base_env.get("WM_THIRD_PARTY_DIR", "").strip()
            third = Path(third_text).expanduser().resolve() if third_text else None

            # WM_PROJECT_INST_DIR is the conventional common parent of OpenFOAM-*
            # and ThirdParty-*. Trust it for libraries only when it actually contains
            # the already-trusted project root.
            if inst is not None and inst.is_dir() and project is not None and inst == project.parent:
                raw_roots.append(inst)
            if third is not None and third.is_dir():
                sibling = project is not None and third.parent == project.parent
                under_inst = inst is not None and inst.is_dir() and inst == project.parent and _is_within(third, inst)
                if sibling or under_inst:
                    raw_roots.append(third)

            # Accept explicit library locations only when they are inside a library
            # trust anchor established above. Never promote them to executable roots.
            provisional: list[Path] = []
            for raw in raw_roots:
                try:
                    candidate = Path(raw).expanduser().resolve()
                except OSError:
                    continue
                if candidate.is_dir() and candidate not in provisional:
                    provisional.append(candidate)
            for key in (
                "FOAM_LIBBIN", "FOAM_EXT_LIBBIN", "SCOTCH_ARCH_PATH",
                "BOOST_ARCH_PATH", "CGAL_ARCH_PATH", "FFTW_ARCH_PATH",
            ):
                value = self._base_env.get(key, "").strip()
                if not value:
                    continue
                try:
                    candidate = Path(value).expanduser().resolve()
                except OSError:
                    continue
                if candidate.is_dir() and any(_is_within(candidate, root) for root in provisional):
                    raw_roots.append(candidate)

        roots: list[Path] = []
        for raw in raw_roots:
            try:
                path = Path(raw).expanduser().resolve()
            except OSError:
                continue
            if path.is_dir() and path not in roots:
                roots.append(path)
        return tuple(roots)

    def executable_dependency_status(self, exe: str) -> dict[str, object]:
        """Inspect trusted ELF loader dependencies under the sanitized runtime env.

        This is a bounded static/toolchain preflight, not a CFD consumer run. ``ldd``
        is used only for an already trusted ELF executable, with shell disabled and
        the sanitized environment. Unknown/non-ELF cases remain non-blocking; only
        explicit missing shared libraries are reported as not ready.
        """
        status = self.executable_status(exe)
        if not status.get("available"):
            return {**status, "runtime_ready": False, "dependency_checked": False}
        try:
            path = self.resolve_trusted_executable(exe, env=self.sanitized_environment())
            with path.open("rb") as handle:
                magic = handle.read(4)
        except (OSError, UnsafeCommandError) as exc:
            return {**status, "runtime_ready": False, "dependency_checked": False, "reason": str(exc)}
        if magic != b"\x7fELF" or os.name != "posix":
            return {**status, "runtime_ready": True, "dependency_checked": False}
        ldd = shutil.which("ldd", path="/usr/bin:/bin")
        if not ldd:
            return {**status, "runtime_ready": True, "dependency_checked": False}
        env = self.sanitized_environment()
        try:
            completed = subprocess.run(
                [ldd, str(path)],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                errors="replace",
                env=env,
                cwd=str(self.workspace_root) if self.workspace_root is not None else None,
                timeout=8,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            return {**status, "runtime_ready": True, "dependency_checked": False, "probe_note": str(exc)}
        output = (completed.stdout or "")[:65536]
        missing = []
        for line in output.splitlines():
            match = re.match(r"\s*([^\s]+)\s*=>\s*not found\s*$", line)
            if match:
                missing.append(match.group(1))
        if missing:
            return {
                **status,
                "runtime_ready": False,
                "dependency_checked": True,
                "missing_libraries": sorted(dict.fromkeys(missing)),
                "reason": "Missing shared libraries under the sanitized OpenFOAM runtime environment: "
                + ", ".join(sorted(dict.fromkeys(missing))),
            }
        return {**status, "runtime_ready": True, "dependency_checked": True}

    def _validate_cwd(self, cwd: str | Path | None) -> Path | None:
        if cwd is None:
            return None
        resolved = Path(cwd).expanduser().resolve()
        if self.workspace_root is not None and not (
            resolved == self.workspace_root or self.workspace_root in resolved.parents
        ):
            raise UnsafeCommandError(f"Command cwd escapes workspace: {resolved}")
        return resolved

    def _validate_arguments(self, args: Sequence[str], cwd: Path | None) -> None:
        """Reject generic arguments that can redirect a tool outside the workspace.

        subprocess is always shell=False, so shell metacharacters have no special meaning.
        The remaining risk is an OpenFOAM application's own path flags.  Python owns the case
        working directory; callers may not override root/case paths, use parent traversal, or
        pass absolute paths outside the workspace.
        """
        forbidden = {"-root", "-roots", "-hostRoots", "-lib", "-libs", "--host", "-host", "-hostfile"}
        for index, token in enumerate(args):
            if token in forbidden or any(token.startswith(flag + "=") for flag in forbidden | {"-case"}):
                raise UnsafeCommandError(f"Native path/library/host override is forbidden: {token}")
            if token == "-case":
                if index + 1 >= len(args) or cwd is None or Path(args[index + 1]).resolve() != cwd:
                    raise UnsafeCommandError("Native -case must equal the Python-owned cwd.")
            if token == "-parallel" and self._mpi_launcher is None:
                raise UnsafeCommandError("Parallel execution requires the structured MPI path.")
        for raw in args:
            if not isinstance(raw, str) or not raw or len(raw) > 1000:
                raise UnsafeCommandError("Native OpenFOAM arguments must be bounded non-empty strings.")
            if "\x00" in raw or "\n" in raw or "\r" in raw:
                raise UnsafeCommandError("Native OpenFOAM arguments cannot contain control characters.")
            candidate = Path(raw)
            if ".." in candidate.parts:
                raise UnsafeCommandError(f"Native OpenFOAM argument contains parent traversal: {raw}")
            if candidate.is_absolute():
                resolved = candidate.expanduser().resolve()
                root = self.workspace_root or cwd
                if root is None or not _is_within(resolved, root):
                    raise UnsafeCommandError(
                        f"Native OpenFOAM argument escapes the workspace: {raw}"
                    )

    def _run_streaming(
        self, command: list[str], *, logical_command: list[str], cwd: Path | None,
        timeout: int, env: Mapping[str, str], echo_output: bool,
        output_callback: Callable[[str], None] | None, record: dict,
    ) -> ToolResult:
        # One bounded chunk queue, bounded memory tail, complete byte-for-byte disk log.
        log_dir = (self.workspace_root or cwd or Path.cwd()) / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"native-{record['id']:05d}-{uuid.uuid4().hex[:8]}.log"
        output_queue: queue.Queue[bytes | None] = queue.Queue(maxsize=8)
        stop_reader = threading.Event()
        tail = bytearray()
        digest = hashlib.sha256()
        total = 0
        started = time.monotonic()
        termination = "exited"
        partial_line = ""
        limits = self.resource_limits
        guard = Path(__file__).with_name("exec_guard.py")
        aggregate_metrics = {}
        cgroup = None
        if self.isolation is not None:
            try:
                cgroup = self.isolation.start(limits)
                command = self.isolation.command(command, cwd, env)
            except BaseException as exc:
                if self.isolation.active is not None:
                    self.isolation.finish()
                self.budget.update(record, status="isolation_unavailable", error=str(exc))
                raise
        guarded = [sys.executable, str(guard), json.dumps({"cpu_seconds": limits.cpu_seconds,
                   "memory_bytes": limits.memory_bytes if self.isolation is None else None,
                   "file_bytes": limits.max_case_bytes, "cgroup": cgroup}), *command]
        spent_cpu = sum((row.get("aggregate_cpu_seconds") or 0) for row in self.budget.records)
        spent_output = sum(row.get("output_bytes", 0) for row in self.budget.records)
        next_quota_check = started
        proc = None
        reader = None
        handle = None
        try:
            handle = log_path.open("xb")
            os.chmod(log_path, 0o600)
            if self.process_observer is not None:
                self.process_observer("spawn_intent", record)
            proc = subprocess.Popen(guarded, cwd=str(cwd) if cwd else None,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=dict(env),
                start_new_session=(os.name == "posix"), bufsize=0)
            self.budget.update(record, pid=proc.pid, status="running", cgroup_path=cgroup)
            if self.process_observer is not None:
                self.process_observer("running", record)
            assert proc.stdout is not None

            def read_output() -> None:
                try:
                    while not stop_reader.is_set():
                        try:
                            chunk = os.read(proc.stdout.fileno(), 16384)
                        except OSError:
                            break
                        if not chunk:
                            break
                        while not stop_reader.is_set():
                            try:
                                output_queue.put(chunk, timeout=0.1); break
                            except queue.Full:
                                continue
                finally:
                    while not stop_reader.is_set():
                        try:
                            output_queue.put(None, timeout=0.1); break
                        except queue.Full:
                            continue

            reader = threading.Thread(target=read_output, name="native-log-reader", daemon=True)
            reader.start()
            eof = False
            killed_at = None
            exited_at = None
            group_cleanup_after_exit = False
            drain_grace_seconds = 0.5
            while True:
                now = time.monotonic()
                return_code = proc.poll()
                if return_code is not None and exited_at is None:
                    exited_at = now

                # Quotas and timeout govern the live primary process only. Waiting
                # for pipe EOF after the primary has exited can otherwise misclassify
                # a successful utility as timeout when a descendant inherited stdout.
                if return_code is None and now >= next_quota_check and killed_at is None:
                    from openfoam_agent.tools.linux_isolation import bounded_tree_bytes
                    next_quota_check = now + 0.1
                    if bounded_tree_bytes(self.workspace_root or cwd) > limits.max_case_bytes:
                        termination = "workspace_quota"; self._terminate_group(proc); killed_at = now
                    if self.isolation is not None:
                        metrics = self.isolation.metrics()
                        if metrics.get("oom_kill", 0):
                            termination = "aggregate_memory_limit"; self._terminate_group(proc); killed_at = now
                        elif limits.total_cpu_seconds is not None and spent_cpu + metrics["cpu_seconds"] >= limits.total_cpu_seconds:
                            termination = "aggregate_cpu_limit"; self._terminate_group(proc); killed_at = now
                if return_code is None and now - started >= timeout and killed_at is None:
                    termination = "timeout"; self._terminate_group(proc); killed_at = now
                if killed_at is not None and now - killed_at > 2.0:
                    break
                if return_code is not None and eof:
                    break
                if return_code is not None and exited_at is not None and now - exited_at >= drain_grace_seconds and output_queue.empty():
                    if not eof and os.name == "posix":
                        # The primary process is done, but a descendant still owns
                        # the inherited pipe. Clean the process group without turning
                        # the already-successful primary result into a timeout.
                        self._terminate_group(proc)
                        group_cleanup_after_exit = True
                    break

                try:
                    chunk = output_queue.get(timeout=0.05)
                except queue.Empty:
                    continue
                if chunk is None:
                    eof = True
                    continue
                remaining = min(limits.max_output_bytes - total, limits.max_total_output_bytes - spent_output - total)
                if len(chunk) > remaining:
                    chunk = chunk[:max(0, remaining)]
                    termination = "output_limit"
                    self._terminate_group(proc); killed_at = now
                try:
                    handle.write(chunk)
                except OSError:
                    termination = "log_io_error"; self._terminate_group(proc); killed_at = now
                    break
                total += len(chunk); digest.update(chunk)
                tail.extend(chunk)
                if len(tail) > 65536:
                    del tail[:-65536]
                text = chunk.decode("utf-8", errors="replace")
                if echo_output:
                    sys.stdout.write(text); sys.stdout.flush()
                if output_callback is not None:
                    partial_line += text
                    while "\n" in partial_line:
                        line, partial_line = partial_line.split("\n", 1)
                        try:
                            output_callback(line + "\n")
                        except Exception:
                            pass
                    if len(partial_line) > 16384:
                        partial_line = partial_line[-16384:]
            if proc.poll() is None:
                self._terminate_group(proc)
            code = proc.wait(timeout=5)
            if termination == "exited" and code < 0:
                termination = "signal"
            if termination == "timeout":
                code = 124
            elif termination == "output_limit":
                code = 125
            elif termination == "log_io_error":
                code = 126
            if partial_line and output_callback is not None:
                try:
                    output_callback(partial_line)
                except Exception:
                    pass
            handle.flush(); os.fsync(handle.fileno())
        except BaseException as exc:
            if proc is not None:
                self._terminate_group(proc)
                proc.wait(timeout=5)
            termination = "cancelled" if isinstance(exc, (KeyboardInterrupt, SystemExit)) else "spawn_or_io_error"
            self.budget.update(record, status=termination, error=type(exc).__name__)
            raise
        finally:
            stop_reader.set()
            if proc is not None and proc.stdout is not None:
                try:
                    proc.stdout.close()
                except OSError:
                    pass
            if reader is not None:
                reader.join(timeout=1)
            if handle is not None:
                handle.close()
            if self.isolation is not None and self.isolation.active is not None:
                aggregate_metrics = self.isolation.finish()
                if aggregate_metrics.get("oom_kill", 0):
                    termination = "aggregate_memory_limit"
                if limits.total_cpu_seconds is not None and spent_cpu + aggregate_metrics.get("cpu_seconds", 0) >= limits.total_cpu_seconds:
                    termination = "aggregate_cpu_limit"
        elapsed = time.monotonic() - started
        from openfoam_agent.tools.linux_isolation import bounded_tree_bytes
        if bounded_tree_bytes(self.workspace_root or cwd) > limits.max_case_bytes:
            termination = "workspace_quota"
        if termination in {"workspace_quota","aggregate_cpu_limit","aggregate_memory_limit"}:
            code = 125
        self.budget.update(record, status=termination, return_code=code, wall_seconds=elapsed,
                           log_path=str(log_path), log_sha256=digest.hexdigest(), output_bytes=total,
                           aggregate_cpu_seconds=aggregate_metrics.get("cpu_seconds", 0) if self.isolation else None,
                           isolation_metrics=aggregate_metrics,
                           isolation_mode="strict_linux" if self.isolation else "local_no_os_isolation")
        if self.process_observer is not None:
            self.process_observer("finished", record)
        return ToolResult(success=code == 0 and termination == "exited", command=logical_command,
            return_code=code, stdout=tail.decode("utf-8", errors="replace"), stderr="",
            termination_reason=termination, log_path=str(log_path), log_sha256=digest.hexdigest(),
            output_bytes=total, output_truncated=total > len(tail), wall_seconds=elapsed,
            process_id=proc.pid, process_group_terminated=(termination != "exited" or group_cleanup_after_exit))

    @staticmethod
    def _terminate_group(proc) -> None:
        try:
            if os.name == "posix":
                os.killpg(proc.pid, signal.SIGKILL)
            elif proc.poll() is None:
                proc.kill()
        except ProcessLookupError:
            pass


def _is_within(path: Path, root: Path) -> bool:
    return path == root or root in path.parents
