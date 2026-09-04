from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Callable

from openfoam_agent.schemas.engineering import OpenFOAMExecutionSpec

from .safe_runner import SafeRunner


class OpenFOAMTools:
    """Trusted wrappers over a dynamically discovered Foundation v13/v14 installation."""

    def __init__(self, runner: SafeRunner | None = None):
        self.runner = runner or SafeRunner()

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
        if latest_time:
            command.append("-latestTime")
        return self.runner.run(command, cwd=case_dir, timeout=timeout)

    def foam_dictionary_validate(
        self,
        file_path: str | Path,
        cwd: str | Path | None = None,
    ):
        return self.runner.run(
            ["foamDictionary", "-keywords", str(Path(file_path).resolve())],
            cwd=cwd,
            timeout=30,
        )

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
