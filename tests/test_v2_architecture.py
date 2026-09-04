from __future__ import annotations

from pathlib import Path

from openfoam_agent import __version__
from openfoam_agent.tools.safe_runner import SafeRunner


ROOT = Path(__file__).resolve().parents[1]


def test_release_version():
    assert __version__ == "3.2.0"


def test_legacy_engineering_modules_are_deleted():
    deleted = [
        "src/openfoam_agent/case_factory",
        "src/openfoam_agent/solver_factory",
        "src/openfoam_agent/runtime/repair.py",
        "src/openfoam_agent/capabilities/planner.py",
        "src/openfoam_agent/agents/physics.py",
        "src/openfoam_agent/agents/equation.py",
    ]
    assert all(not (ROOT / item).exists() for item in deleted)


def test_production_source_has_no_known_phase27_case_contracts():
    text = "\n".join(
        path.read_text(encoding="utf-8", errors="ignore")
        for path in (ROOT / "src" / "openfoam_agent").rglob("*.py")
    )
    forbidden = [
        "_REQUIRED_INCOMPRESSIBLE_FILES",
        "_foundation_case_contract",
        "CapabilityGapPlanner",
        "RuntimeRepairPlanner",
        "use_square_obstacle_template",
        "always_use_snappy_hex_mesh",
    ]
    assert all(token not in text for token in forbidden)


def test_command_allowlist_is_narrow_openfoam_only():
    assert SafeRunner.DEFAULT_ALLOWED == {
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
