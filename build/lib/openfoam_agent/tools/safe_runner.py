from __future__ import annotations

import os
import queue
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Callable, Mapping, Sequence

from openfoam_agent.schemas.common import ToolResult


class UnsafeCommandError(RuntimeError):
    pass


class SafeRunner:
    """Execute only trusted OpenFOAM utilities inside an optional workspace root.

    Security properties:
    - only fixed executable names are allowlisted;
    - the resolved executable must live under a trusted OpenFOAM installation root;
    - subprocesses receive a reduced OpenFOAM/runtime environment, not the parent
      process environment (so API keys/tokens are not inherited);
    - cwd is confined to the configured workspace.
    """

    DEFAULT_ALLOWED = {
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
        max_timeout: int = 3600,
        trusted_executable_roots: Sequence[str | Path] | None = None,
        base_env: Mapping[str, str] | None = None,
    ) -> None:
        self.allowed_commands = set(
            self.DEFAULT_ALLOWED if allowed_commands is None else allowed_commands
        )
        self.workspace_root = (
            Path(workspace_root).expanduser().resolve() if workspace_root else None
        )
        self.max_timeout = max_timeout
        self._base_env = dict(os.environ if base_env is None else base_env)
        self.trusted_executable_roots = self._resolve_trusted_roots(
            trusted_executable_roots
        )

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
        env = self.sanitized_environment()
        executable = self.resolve_trusted_executable(exe, env=env)
        actual_command = [str(executable), *command[1:]]

        if stream_output or output_callback is not None:
            return self._run_streaming(
                actual_command,
                logical_command=command,
                cwd=resolved_cwd,
                timeout=timeout,
                env=env,
                echo_output=stream_output,
                output_callback=output_callback,
            )

        proc = subprocess.run(
            actual_command,
            cwd=str(resolved_cwd) if resolved_cwd else None,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
            env=env,
        )
        return ToolResult(
            success=proc.returncode == 0,
            command=command,
            return_code=proc.returncode,
            stdout=proc.stdout,
            stderr=proc.stderr,
        )

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

    @staticmethod
    def _run_streaming(
        command: list[str],
        *,
        logical_command: list[str],
        cwd: Path | None,
        timeout: int,
        env: Mapping[str, str],
        echo_output: bool,
        output_callback: Callable[[str], None] | None,
    ) -> ToolResult:
        proc = subprocess.Popen(
            command,
            cwd=str(cwd) if cwd else None,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=dict(env),
        )
        assert proc.stdout is not None
        output_queue: queue.Queue[str | None] = queue.Queue()

        def read_output() -> None:
            for line in proc.stdout:
                output_queue.put(line)
            output_queue.put(None)

        reader = threading.Thread(target=read_output, daemon=True)
        reader.start()
        started = time.monotonic()
        chunks: list[str] = []
        finished = False
        while not finished:
            if time.monotonic() - started > timeout:
                proc.kill()
                reader.join(timeout=1)
                raise subprocess.TimeoutExpired(logical_command, timeout)
            try:
                item = output_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            if item is None:
                finished = True
                continue
            chunks.append(item)
            if output_callback is not None:
                try:
                    output_callback(item)
                except Exception:
                    # Progress/reporting callbacks are observational only and must
                    # never terminate or alter an OpenFOAM subprocess.
                    pass
            if echo_output:
                sys.stdout.write(item)
                sys.stdout.flush()
        return_code = proc.wait(timeout=5)
        return ToolResult(
            success=return_code == 0,
            command=logical_command,
            return_code=return_code,
            stdout="".join(chunks),
            stderr="",
        )


def _is_within(path: Path, root: Path) -> bool:
    return path == root or root in path.parents
