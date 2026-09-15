from __future__ import annotations

import json
from pathlib import Path

import openfoam_agent
from openfoam_agent.qualification.reliability import BenchmarkManifest


def test_v520_version_and_release_notes():
    root = Path(__file__).resolve().parents[1]
    import tomllib
    assert openfoam_agent.__version__ == tomllib.loads((root / "pyproject.toml").read_text())["project"]["version"]
    assert (root / "V5_2_0_RELEASE.md").is_file()


def test_v520_benchmark_manifest_is_explicitly_not_run_and_has_36_expected_reports():
    root = Path(__file__).resolve().parents[1]
    raw = json.loads((root / "research/V5_2_RELIABILITY_BENCHMARK.json").read_text(encoding="utf-8"))
    manifest = BenchmarkManifest.model_validate(raw)
    assert manifest.release == "5.2.0"
    assert manifest.status == "NOT_RUN"
    assert len(manifest.trials) == 12
    assert len(manifest.variants) == 3
    assert len(manifest.trials) * len(manifest.variants) == 36
