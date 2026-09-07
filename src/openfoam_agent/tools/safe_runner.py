from __future__ import annotations

import os
import hashlib
import json
import signal
import uuid
from contextlib import contextmanager
import queue
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Callable, Mapping, Sequence

from openfoam_agent.schemas.common import ToolResult
from openfoam_agent.contracts.models import ResourceLimits
from .execution_policy import (ExecutionContext, ExecutionPolicyError, ProcessBudget, command_effect)
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
        "FOAM_APPBIN",
        "FOAM_LIBBIN",
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
        self._mpi_ranks = 1
        self._mpi_launcher: str | None = None
        self.process_observer = None
        self._base_env = dict(os.environ if base_env is None else base_env)
        self.trusted_executable_roots = self._resolve_trusted_roots(
            trusted_executable_roots
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
            if self._execution_context is None or resolved_cwd is None:
                raise UnsafeCommandError("Main calculation requires an explicit execution approval and sealed runtime context.")
            self._execution_context.validate(command, resolved_cwd, timeout, ranks=self._mpi_ranks)
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
        row = self.budget.reserve(command, effect)
        return self._run_streaming(
            actual_command, logical_command=command, cwd=resolved_cwd, timeout=timeout,
            env=env, echo_output=stream_output, output_callback=output_callback, record=row,
        )

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
            if not any(_is_within(resolved, root) for root in self.trusted_executable_roots):
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
            )
        return env

    def _filtered_search_path(
        self,
        value: str,
        *,
        allowed_system_roots: Sequence[Path],
    ) -> str:
        accepted: list[str] = []
        seen: set[str] = set()
        for raw in value.split(os.pathsep):
            if not raw:
                continue
            try:
                path = Path(raw).expanduser().resolve()
            except OSError:
                continue
            trusted = any(_is_within(path, root) for root in self.trusted_executable_roots)
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
        guarded = [sys.executable, str(guard), json.dumps({"cpu_seconds": limits.cpu_seconds,
                   "memory_bytes": limits.memory_bytes, "file_bytes": limits.max_case_bytes}), *command]
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
            self.budget.update(record, pid=proc.pid, status="running")
            if self.process_observer is not None:
                self.process_observer("running", record)
            assert proc.stdout is not None

            def read_output() -> None:
                try:
                    while not stop_reader.is_set():
                        chunk = os.read(proc.stdout.fileno(), 16384)
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
            while not eof or proc.poll() is None:
                now = time.monotonic()
                if now - started >= timeout and killed_at is None:
                    termination = "timeout"; self._terminate_group(proc); killed_at = now
                if killed_at is not None and now - killed_at > 2.0:
                    break
                try:
                    chunk = output_queue.get(timeout=0.05)
                except queue.Empty:
                    continue
                if chunk is None:
                    eof = True
                    continue
                remaining = limits.max_output_bytes - total
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
                    # A malicious/no-newline log cannot grow this observation buffer.
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
            if reader is not None:
                reader.join(timeout=1)
            if proc is not None and proc.stdout is not None:
                proc.stdout.close()
            if handle is not None:
                handle.close()
        elapsed = time.monotonic() - started
        self.budget.update(record, status=termination, return_code=code, wall_seconds=elapsed,
                           log_path=str(log_path), log_sha256=digest.hexdigest(), output_bytes=total)
        if self.process_observer is not None:
            self.process_observer("finished", record)
        return ToolResult(success=code == 0 and termination == "exited", command=logical_command,
            return_code=code, stdout=tail.decode("utf-8", errors="replace"), stderr="",
            termination_reason=termination, log_path=str(log_path), log_sha256=digest.hexdigest(),
            output_bytes=total, output_truncated=total > len(tail), wall_seconds=elapsed,
            process_id=proc.pid, process_group_terminated=termination != "exited")

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
