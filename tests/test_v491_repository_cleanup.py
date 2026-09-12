from __future__ import annotations

import ast
from pathlib import Path

from openfoam_agent import __version__

ROOT = Path(__file__).resolve().parents[1]
CONTROLLERS = [
    ROOT / "src/openfoam_agent/engineering/agent.py",
    *sorted((ROOT / "src/openfoam_agent/engineering/phases").glob("*_controller.py")),
]


def _unused_module_imports(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    tree = ast.parse(text)
    used = {
        node.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            import re
            used.update(re.findall(r"\b[A-Za-z_][A-Za-z0-9_]*\b", node.value))
    unused = []
    for node in tree.body:
        if not isinstance(node, (ast.Import, ast.ImportFrom)):
            continue
        if isinstance(node, ast.ImportFrom) and node.module == "__future__":
            continue
        if any(alias.name == "*" for alias in node.names):
            continue
        for alias in node.names:
            bound = alias.asname or (alias.name.split(".", 1)[0] if isinstance(node, ast.Import) else alias.name)
            if bound not in used:
                unused.append(bound)
    return unused


def test_release_version():
    assert __version__ == "4.9.1"


def test_split_engineering_controllers_do_not_keep_monolith_import_bloat():
    failures = {str(path.relative_to(ROOT)): _unused_module_imports(path) for path in CONTROLLERS}
    failures = {path: names for path, names in failures.items() if names}
    assert not failures, failures


def test_generated_historical_verification_bundle_is_not_source_controlled():
    assert not (ROOT / "verification").exists()
    assert not (ROOT / "examples/v4rc2").exists()
    assert not (ROOT / "research/V2_EVAL_PLAN.md").exists()


def test_runtime_source_has_no_stale_v2_branding():
    cli = (ROOT / "src/openfoam_agent/cli.py").read_text(encoding="utf-8")
    workflow = (ROOT / "src/openfoam_agent/workflow/engine.py").read_text(encoding="utf-8")
    assert "OpenFOAM Agent v2:" not in cli
    assert "No v2 handler" not in workflow
    assert '"""v2 workflow:' not in workflow
