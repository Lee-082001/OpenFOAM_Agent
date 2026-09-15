from __future__ import annotations

import inspect
from pathlib import Path
import re

import openfoam_agent
from openfoam_agent.schemas.engineering import OpenFOAMExecutionSpec
import openfoam_agent.engineering.design_seal as design_seal


def test_v501_package_and_project_versions_are_consistent():
    assert openfoam_agent.__version__ == "5.0.1"
    root = Path(__file__).resolve().parents[1]
    text = (root / "pyproject.toml").read_text(encoding="utf-8")
    match = re.search(r'(?m)^version\s*=\s*"([^"]+)"', text)
    assert match is not None
    assert match.group(1) == "5.0.1"


def test_v501_design_seal_has_no_legacy_region_layout_authority():
    source = inspect.getsource(design_seal.materialize_engineering_plan)
    assert "_canonical_region_layouts" not in source
    assert 'data["region_layouts"] = []' in source


def test_v501_execution_schema_keeps_scopes_as_agent_topology_contract():
    properties = OpenFOAMExecutionSpec.model_json_schema()["properties"]
    assert "scopes" in properties
    assert "regions" not in properties
    assert "solver_module" not in properties
    assert "solver_provider_id" not in properties
