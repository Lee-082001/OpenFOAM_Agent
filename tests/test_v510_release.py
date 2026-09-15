from pathlib import Path
import re
import openfoam_agent


def test_v510_package_and_project_versions_remain_consistent_after_minor_releases():
    root = Path(__file__).resolve().parents[1]
    text = (root / "pyproject.toml").read_text(encoding="utf-8")
    match = re.search(r'(?m)^version\s*=\s*"([^"]+)"', text)
    assert match is not None
    assert match.group(1) == openfoam_agent.__version__


def test_v510_release_note_exists():
    root = Path(__file__).resolve().parents[1]
    assert (root / "V5_1_0_RELEASE.md").is_file()
