from __future__ import annotations

from pathlib import Path
import re

import openfoam_agent


def test_v505_package_and_project_versions_are_consistent():
    assert openfoam_agent.__version__ == "5.0.5"
    root = Path(__file__).resolve().parents[1]
    text = (root / "pyproject.toml").read_text(encoding="utf-8")
    match = re.search(r'(?m)^version\s*=\s*"([^"]+)"', text)
    assert match is not None
    assert match.group(1) == "5.0.5"
