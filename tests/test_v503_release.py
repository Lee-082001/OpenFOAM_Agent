from __future__ import annotations

from pathlib import Path
import re

import openfoam_agent


def test_v503_package_and_project_versions_remain_consistent_after_maintenance_releases():
    root = Path(__file__).resolve().parents[1]
    text = (root / "pyproject.toml").read_text(encoding="utf-8")
    match = re.search(r'(?m)^version\s*=\s*"([^"]+)"', text)
    assert match is not None
    assert match.group(1) == openfoam_agent.__version__
