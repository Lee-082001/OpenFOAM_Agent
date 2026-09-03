from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Callable

from .safe_runner import SafeRunner


class OpenFOAMTools:
    """Narrow wrappers around allowlisted, provenance-checked OpenFOAM commands."""

    MESH_COMMANDS = (
        "blockMesh",
        "surfaceFeatureExtract",
        "snappyHexMesh",
        "createPatch",
        "checkMesh",
    )

    def __init__(self, runner: SafeRunner | None = None):
        self.runner = runner or SafeRunner()

    @classmethod
    def for_workspace(cls, workspace_root: str | Path) -> "OpenFOAMTools":
        return cls(SafeRunner(workspace_root=workspace_root))

    @staticmethod
    def detected_foundation_version() -> str | None:
        value = os.environ.get("WM_PROJECT_VERSION", "").strip()
        matched = re.fullmatch(r"(?:v)?(13|14)", value, re.IGNORECASE)
        return matched.group(1) if matched else None

    def check_mesh_preflight(self) -> dict[str, object]:
        """Return trusted-executable availability for the minimal native preflight."""
        return self.runner.executable_status("checkMesh")

    def environment_snapshot(self) -> dict[str, object]:
        # Deliberately expose no absolute local paths to the remote model.
        commands = sorted(self.runner.allowed_commands)
        return {
            "wm_project": os.environ.get("WM_PROJECT", ""),
            "wm_project_version": os.environ.get("WM_PROJECT_VERSION", ""),
            "foundation_version": self.detected_foundation_version(),
            "trusted_installation_configured": bool(self.runner.trusted_executable_roots),
            "commands": [self.runner.executable_status(command) for command in commands],
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
        try:
            tool = dispatch[command]
        except KeyError as exc:
            raise ValueError(f"Unsupported mesh command: {command}") from exc
        return tool(case_dir)
