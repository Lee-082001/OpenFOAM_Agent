from __future__ import annotations

import os
import hashlib
import json
import re
import shutil
import uuid
from pathlib import Path
from typing import Callable

from openfoam_agent.schemas.engineering import OpenFOAMExecutionSpec
from openfoam_agent.schemas.common import ToolResult
from openfoam_agent.tools.execution_policy import ValidationExecutionContext

from .safe_runner import SafeRunner


class OpenFOAMTools:
    """Trusted wrappers over a dynamically discovered Foundation v13/v14 installation."""

    def __init__(self, runner: SafeRunner | None = None):
        self.runner = runner or SafeRunner()
        self._dictionary_cache = {}

    @property
    def installed_openfoam(self):
        return self.runner.installation

    @classmethod
    def for_workspace(cls, workspace_root: str | Path) -> "OpenFOAMTools":
        return cls(SafeRunner(workspace_root=workspace_root))

    def detected_foundation_version(self) -> str | None:
        if self.runner.installation.installation_configured and self.runner.installation.version:
            return self.runner.installation.version
        value = os.environ.get("WM_PROJECT_VERSION", "").strip()
        matched = re.fullmatch(r"(?:v)?(13|14)", value, re.IGNORECASE)
        return matched.group(1) if matched else None

    def check_mesh_preflight(self) -> dict[str, object]:
        """Return trusted-executable availability for the minimal native preflight."""
        return self.runner.executable_status("checkMesh")

    def native_command_preflight(self, command: str) -> dict[str, object]:
        """Return cached executable + loader dependency health for one native tool.

        This does not run the CFD utility. It asks SafeRunner to inspect the trusted
        ELF executable under the same sanitized runtime environment that will be
        used later, so missing ThirdParty/shared-library dependencies are caught
        before candidate case mutation or an LLM repair loop.
        """
        cache = getattr(self, "_native_dependency_cache", None)
        if cache is None:
            cache = {}
            self._native_dependency_cache = cache
        if command not in cache:
            cache[command] = self.runner.executable_dependency_status(command)
        return dict(cache[command])

    def environment_snapshot(self) -> dict[str, object]:
        """Return a compact, path-free installation capsule for model context.

        The full executable inventory lives in CapabilityCatalog and is retrieved by
        evidence query when needed. Repeating one status dictionary per installed utility
        in every model turn made large Foundation installations dominate the prompt.
        """
        installation = self.installed_openfoam
        execution_drivers = sorted(
            item.name for item in installation.executables if item.category == "execution_driver"
        )
        return {
            "wm_project": os.environ.get("WM_PROJECT", ""),
            "wm_project_version": os.environ.get("WM_PROJECT_VERSION", ""),
            "foundation_version": self.detected_foundation_version(),
            "trusted_installation_configured": bool(self.runner.trusted_executable_roots),
            "installed_ir_fingerprint": installation.fingerprint,
            "installed_executable_count": len(installation.executables),
            "installed_execution_drivers": execution_drivers,
            "installed_solver_modules": sorted(installation.solver_modules),
            "installed_fv_models": sorted(installation.fv_models),
            "capability_inventory_queryable": True,
            "reference_scopes_configured": {
                "tutorials": bool(os.environ.get("FOAM_TUTORIALS")),
                "source": bool(os.environ.get("FOAM_SRC")),
                "etc": bool(os.environ.get("FOAM_ETC")),
                "modules": bool(os.environ.get("FOAM_MODULES")),
            },
        }


    @staticmethod
    def mesh_tool_contracts() -> list[dict[str, object]]:
        """Deterministic execution contracts, not CFD strategy choices."""
        return [
            {
                "command": "snappyHexMesh",
                "precondition": "base mesh must be fully 3D during snapping/mesh relaxation",
                "deterministic_observation": "an existing polyMesh boundary with type empty proves this precondition is not met",
            }
        ]

    @staticmethod
    def mesh_command_precondition(command: str, case_dir: str | Path) -> tuple[bool, str]:
        """Check narrow executable prerequisites before consuming a native command.

        This does not select a meshing strategy. It only enforces a tool contract that
        the executable itself requires.
        """
        if command != "snappyHexMesh":
            return True, ""
        case = Path(case_dir).resolve()
        boundary = case / "constant" / "polyMesh" / "boundary"
        if not boundary.exists():
            return False, "snappyHexMesh requires an existing base polyMesh before snapping."
        try:
            text = boundary.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            return False, f"Could not inspect snappyHexMesh base-mesh boundary file: {exc}"
        if re.search(r"\btype\s+empty\s*;", text):
            return False, (
                "snappyHexMesh requires a fully 3D base mesh during snapping/mesh relaxation, "
                "but constant/polyMesh/boundary contains an empty patch."
            )
        return True, ""

    def run_native_command(
        self,
        command: str,
        case_dir: str | Path,
        *,
        arguments: list[str] | None = None,
        timeout: int = 900,
        stream_output: bool = False,
        output_callback: Callable[[str], None] | None = None,
    ):
        """Execute any application discovered in the trusted OpenFOAM installation.

        The LLM never supplies an executable path and cannot override the case/root.
        Every invocation is shell=False, case-workspace confined, and re-resolved under
        WM_PROJECT_DIR immediately before execution.
        """
        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_.+-]*", command):
            raise ValueError(f"Unsafe OpenFOAM command identifier: {command!r}")
        args = list(arguments or [])
        reserved = {"-case", "-root", "-hostRoots", "-roots"}
        if any(arg in reserved for arg in args):
            raise ValueError("Agent native commands cannot override Python-owned case/root paths.")
        return self.runner.run(
            [command, *args],
            cwd=case_dir,
            timeout=timeout,
            stream_output=stream_output,
            output_callback=output_callback,
        )

    def run_execution(
        self,
        case_dir: str | Path,
        execution: OpenFOAMExecutionSpec,
        *,
        stream_output: bool = False,
        timeout: int = 3600,
        output_callback: Callable[[str], None] | None = None,
    ):
        args = list(execution.arguments)
        if execution.driver == "foamRun":
            assert execution.solver_module is not None
            args = ["-solver", execution.solver_module, *args]
        # MPI is a Python-owned launch contract, never an arbitrary LLM command.
        if execution.parallel.mode == "local_mpi":
            return self.runner.run_mpi(
                [execution.driver, *args], ranks=execution.parallel.ranks,
                launcher=execution.parallel.launcher, cwd=case_dir,
                stream_output=stream_output, timeout=timeout, output_callback=output_callback,
            )
        # foamMultiRun obtains region->solver semantics from controlDict.regionSolvers.
        return self.run_native_command(
            execution.driver,
            case_dir,
            arguments=args,
            stream_output=stream_output,
            timeout=timeout,
            output_callback=output_callback,
        )

    def block_mesh(self, case_dir: str | Path):
        return self.runner.run(["blockMesh", "-case", str(Path(case_dir).resolve())], cwd=case_dir)

    def surface_feature_extract(self, case_dir: str | Path):
        return self.runner.run(
            ["surfaceFeatureExtract", "-case", str(Path(case_dir).resolve())],
            cwd=case_dir,
        )

    def surface_check(self, geometry_path: str | Path, cwd: str | Path | None = None):
        return self.runner.run(
            ["surfaceCheck", str(Path(geometry_path).resolve())],
            cwd=cwd,
            timeout=120,
        )

    def snappy_hex_mesh(self, case_dir: str | Path):
        return self.runner.run(
            ["snappyHexMesh", "-overwrite", "-case", str(Path(case_dir).resolve())],
            cwd=case_dir,
        )

    def create_patch(self, case_dir: str | Path):
        return self.runner.run(
            ["createPatch", "-overwrite", "-case", str(Path(case_dir).resolve())],
            cwd=case_dir,
        )

    def check_mesh(self, case_dir: str | Path):
        return self.runner.run(
            ["checkMesh", "-case", str(Path(case_dir).resolve())],
            cwd=case_dir,
        )

    def foam_run(
        self,
        case_dir: str | Path,
        solver: str,
        *,
        stream_output: bool = False,
        timeout: int = 3600,
        output_callback: Callable[[str], None] | None = None,
    ):
        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", solver):
            raise ValueError(f"Unsafe solver identifier: {solver!r}")
        return self.runner.run(
            ["foamRun", "-case", str(Path(case_dir).resolve()), "-solver", solver],
            cwd=case_dir,
            stream_output=stream_output,
            timeout=timeout,
            output_callback=output_callback,
        )

    def foam_post_process(
        self,
        case_dir: str | Path,
        dictionary_path: str | Path,
        *,
        solver: str | None = None,
        region: str = "",
        latest_time: bool = False,
        timeout: int = 900,
    ):
        command = [
            "foamPostProcess",
            "-case",
            str(Path(case_dir).resolve()),
            "-dict",
            str(Path(dictionary_path).resolve()),
        ]
        if solver is not None:
            if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", solver):
                raise ValueError(f"Unsafe solver identifier: {solver!r}")
            command.extend(["-solver", solver])
        if region:
            if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_.-]*", region):
                raise ValueError("Unsafe post-processing region.")
            command.extend(["-region", region])
        if latest_time:
            command.append("-latestTime")
        return self.runner.run(command, cwd=case_dir, timeout=timeout)

    def foam_dictionary_validate(
        self,
        file_path: str | Path,
        cwd: str | Path | None = None,
    ):
        path = Path(file_path).resolve()
        # Cache only successful, literal-file validation, keyed by the exact file,
        # installation and executable bytes, options and case context.
        executable = self.runner.resolve_trusted_executable("foamDictionary", env=self.runner.sanitized_environment())
        binary_hash = None
        if executable and Path(str(executable)).is_file():
            with Path(str(executable)).open("rb") as stream:
                binary_hash = hashlib.file_digest(stream, "sha256").hexdigest()
        key = hashlib.sha256(json.dumps({
            "path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "cwd": str(Path(cwd).resolve()) if cwd else None,
            "environment": self.runner.installation.fingerprint,
            "binary": binary_hash, "options": ["-keywords"],
        }, sort_keys=True).encode()).hexdigest()
        if key in self._dictionary_cache:
            return self._dictionary_cache[key].model_copy(update={"cache_hit": True})
        result = self.runner.run(["foamDictionary", "-keywords", str(path)], cwd=cwd, timeout=30)
        if result.success:
            self._dictionary_cache[key] = result.model_copy(deep=True)
        return result


    def zero_step_consumer_validate(
        self,
        case_dir: str | Path,
        execution: OpenFOAMExecutionSpec,
        *,
        timeout: int = 60,
        max_shadow_bytes: int = 512_000_000,
    ) -> tuple[ToolResult | None, str]:
        """Initialize the selected OpenFOAM consumer in a temporary endTime=0 case.

        The production case is never modified. If a bounded, side-effect-minimized
        shadow cannot be proven safe, return ``None`` and let the caller report
        INCONCLUSIVE rather than inventing case invalidity.
        """

        case = Path(case_dir).resolve()
        if execution.parallel.mode != "serial" or execution.parallel.ranks != 1:
            return None, "Zero-step consumer validation is deferred for parallel execution topologies."
        if execution.arguments:
            return None, "Custom production solver arguments are not replayed before approval; zero-step consumer validation deferred."
        control = case / "system" / "controlDict"
        if not control.is_file():
            return None, "system/controlDict is missing; zero-step consumer validation was not attempted."

        total = 0
        for top in ("0", "constant", "system"):
            root = case / top
            if not root.exists():
                continue
            for item in root.rglob("*"):
                if item.is_symlink():
                    return None, "Zero-step shadow validation refuses symlinked case inputs."
                if item.is_file():
                    total += item.stat().st_size
                    if total > max_shadow_bytes:
                        return None, "Case inputs exceed the bounded zero-step shadow-copy budget."

        text = control.read_text(encoding="utf-8", errors="replace")
        function_marker = re.search(r"(?m)^\s*functions\s*\{", text)
        if function_marker and not re.search(r"(?ms)^\s*functions\s*\{\s*\}\s*", text):
            return None, "Non-empty controlDict functions are deferred to approved runtime; zero-step probe skipped."

        def set_entry(source: str, key: str, value: str) -> str:
            pattern = re.compile(rf"(?m)^(?P<indent>[ \t]*){re.escape(key)}\s+[^;\n]+;")
            matches = list(pattern.finditer(source))
            if len(matches) > 1:
                raise ValueError(f"controlDict contains duplicate literal {key} entries.")
            replacement = f"{key}    {value};"
            if matches:
                match = matches[0]
                if match.group("indent"):
                    raise ValueError(f"controlDict {key} is not a literal top-level entry.")
                return source[:match.start()] + replacement + source[match.end():]
            return source.rstrip() + "\n" + replacement + "\n"

        try:
            for key, value in (
                ("startFrom", "startTime"),
                ("startTime", "0"),
                ("stopAt", "endTime"),
                ("endTime", "0"),
                ("writeControl", "timeStep"),
                ("writeInterval", "1"),
                ("purgeWrite", "0"),
                ("runTimeModifiable", "false"),
            ):
                text = set_entry(text, key, value)
        except ValueError as exc:
            return None, f"Zero-step controlDict normalization was inconclusive: {exc}"

        shadow_root = (self.runner.workspace_root or case.parent) / ".validation-shadow"
        shadow = shadow_root / uuid.uuid4().hex
        shadow_root.mkdir(parents=True, exist_ok=True)
        shadow.mkdir(parents=True, exist_ok=False)
        try:
            for top in ("0", "constant", "system"):
                source = case / top
                if source.exists():
                    shutil.copytree(source, shadow / top)
            (shadow / "system" / "controlDict").write_text(text, encoding="utf-8")

            args = list(execution.arguments)
            if execution.driver == "foamRun":
                if not execution.solver_module:
                    return None, "foamRun zero-step validation requires a selected solver module."
                args = ["-solver", execution.solver_module, *args]
            command = [execution.driver, *args]
            context = ValidationExecutionContext(
                expected_command=command,
                case_dir=shadow,
                workspace_root=(self.runner.workspace_root or case.parent),
                max_wall_seconds=min(timeout, 60),
            )
            with self.runner.validation_execution(context):
                result = self.runner.run(command, cwd=shadow, timeout=min(timeout, 60))
            return result, "Zero-step consumer validation executed in a temporary shadow case."
        finally:
            shutil.rmtree(shadow, ignore_errors=True)
            try:
                shadow_root.rmdir()
            except OSError:
                pass

    def run_mesh_command(self, command: str, case_dir: str | Path):
        dispatch = {
            "blockMesh": self.block_mesh,
            "surfaceFeatureExtract": self.surface_feature_extract,
            "snappyHexMesh": self.snappy_hex_mesh,
            "createPatch": self.create_patch,
            "checkMesh": self.check_mesh,
        }
        tool = dispatch.get(command)
        if tool is not None:
            return tool(case_dir)
        # Any other installed Foundation utility is allowed through the same trusted
        # native runner. Strategy choice remains Agent-owned; Python only verifies
        # installation provenance and workspace confinement.
        return self.run_native_command(command, case_dir)
