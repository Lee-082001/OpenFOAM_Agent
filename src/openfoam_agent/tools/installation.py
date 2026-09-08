from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Mapping, Sequence

from openfoam_agent.schemas.installation import (
    InstalledComponent,
    InstalledExecutable,
    InstalledOpenFOAMIR,
)


# Documented solver applications are used only to classify discovered executables.
# They do not create installed capability evidence.
DOCUMENTED_SOLVER_APPLICATIONS: dict[str, tuple[str, ...]] = {
    "13": (
        "foamRun", "foamMultiRun", "boundaryFoam", "chemFoam", "potentialFoam",
        "electrostaticFoam", "magneticFoam", "mhdFoam", "laplacianFoam",
        "financialFoam", "dsmcFoam", "mdEquilibrationFoam", "mdFoam",
        "adjointShapeOptimizationFoam", "icoFoam", "shallowWaterFoam",
        "porousSimpleFoam", "rhoPorousSimpleFoam", "PDRFoam",
    ),
    "14": (
        "foamRun", "foamMultiRun", "boundaryFoam", "chemFoam", "potentialFoam",
        "electrostaticFoam", "magneticFoam", "mhdFoam", "laplacianFoam",
        "financialFoam", "dsmcFoam", "mdEquilibrationFoam", "mdFoam",
        "adjointShapeOptimizationFoam", "icoFoam", "shallowWaterFoam",
        "porousSimpleFoam", "rhoPorousSimpleFoam", "PDRFoam",
    ),
}


class OpenFOAMInstallationDiscovery:
    """Discover all trusted Foundation applications plus runtime-selectable components.

    Executables are discovered from trusted OpenFOAM PATH entries and FOAM_APPBIN, not
    from a hand-maintained command allowlist.  Source-tree discovery is additive and bounded. Documented v13/v14 fallback evidence
    lives in the capability graphs and is never promoted into InstalledOpenFOAMIR.
    """

    def __init__(
        self,
        *,
        base_env: Mapping[str, str] | None = None,
        trusted_roots: Sequence[str | Path] = (),
    ) -> None:
        self.base_env = dict(os.environ if base_env is None else base_env)
        self.trusted_roots = tuple(Path(root).expanduser().resolve() for root in trusted_roots)

    def discover(self) -> InstalledOpenFOAMIR:
        version = self._version()
        configured = self._foundation_configured(version)
        executables = self._executables(version) if configured else []
        components = self._components(version) if configured else []
        scopes = [
            key
            for key in ("FOAM_APPBIN", "FOAM_MODULES", "FOAM_SOLVERS", "FOAM_UTILITIES", "FOAM_SRC")
            if self._trusted_directory(self.base_env.get(key, "")) is not None
        ]
        return InstalledOpenFOAMIR(
            version=version,
            installation_configured=configured,
            executables=executables,
            components=components,
            source_scopes=scopes,
        )

    def _version(self) -> str | None:
        raw = self.base_env.get("WM_PROJECT_VERSION", "").strip().lower().lstrip("v")
        return raw if raw in {"13", "14"} else None

    def _foundation_configured(self, version: str | None) -> bool:
        project = self.base_env.get("WM_PROJECT", "").strip().casefold()
        return bool(self.trusted_roots and version in {"13", "14"} and project == "openfoam")

    def _executables(self, version: str | None) -> list[InstalledExecutable]:
        names: set[str] = set()
        candidate_dirs: list[Path] = []
        appbin = self._trusted_directory(self.base_env.get("FOAM_APPBIN", ""))
        if appbin is not None:
            candidate_dirs.append(appbin)
        for raw in self.base_env.get("PATH", "").split(os.pathsep):
            path = self._trusted_directory(raw)
            if path is not None and path not in candidate_dirs:
                candidate_dirs.append(path)
        for directory in candidate_dirs:
            try:
                entries = list(directory.iterdir())
            except OSError:
                continue
            for entry in entries:
                try:
                    resolved = entry.resolve()
                    if not resolved.is_file() or not os.access(resolved, os.X_OK):
                        continue
                except OSError:
                    continue
                if not self._within_trusted_root(resolved):
                    continue
                if _safe_executable_name(entry.name):
                    names.add(entry.name)

        documented_solvers = set(DOCUMENTED_SOLVER_APPLICATIONS.get(version or "", ()))
        result: list[InstalledExecutable] = []
        for name in sorted(names):
            if name in {"foamRun", "foamMultiRun"}:
                category = "execution_driver"
            elif name in documented_solvers:
                category = "solver_application"
            elif name.startswith("foam") and name in {"foamInfo", "foamGet", "foamVersion"}:
                category = "script"
            else:
                category = "utility"
            result.append(InstalledExecutable(name=name, category=category, trusted=True))
        return result

    def _components(self, version: str | None) -> list[InstalledComponent]:
        items: dict[tuple[str, str], InstalledComponent] = {}
        modules_root = self._trusted_directory(self.base_env.get("FOAM_MODULES", ""))
        if modules_root is not None:
            for name in _source_component_names(modules_root, max_files=2500):
                items[("solver_module", name)] = InstalledComponent(
                    name=name, category="solver_module", source="installed_source"
                )

        src_root = self._trusted_directory(self.base_env.get("FOAM_SRC", ""))
        if src_root is not None:
            for relative, category, base_types in (
                ("fvModels", "fv_model", ("fvModel",)),
                ("functionObjects", "function_object", ("functionObject",)),
            ):
                root = src_root / relative
                if not root.is_dir():
                    continue
                # Library/component directories are useful coarse evidence, but OpenFOAM's
                # actual runtime-selectable type names are registered in C++ macros.  Parse
                # those bounded registration sites so InstalledOpenFOAMIR reflects what the
                # sourced installation can instantiate, not merely which libraries exist.
                names = _source_component_names(root, max_files=5000)
                names.update(
                    _runtime_selection_names(
                        root,
                        base_types=base_types,
                        max_files=8000,
                        max_file_bytes=1_000_000,
                    )
                )
                for name in names:
                    items[(category, name)] = InstalledComponent(
                        name=name, category=category, source="installed_source"
                    )
        return [items[key] for key in sorted(items)]

    def _trusted_directory(self, raw: str | Path) -> Path | None:
        if not raw:
            return None
        try:
            path = Path(raw).expanduser().resolve()
        except OSError:
            return None
        if not path.is_dir() or not self._within_trusted_root(path):
            return None
        return path

    def _within_trusted_root(self, path: Path) -> bool:
        return any(path == root or root in path.parents for root in self.trusted_roots)


def _safe_executable_name(name: str) -> bool:
    if not name or len(name) > 160 or not name[0].isalpha():
        return False
    return all(ch.isalnum() or ch in "_.+-" for ch in name)


def _source_component_names(root: Path, *, max_files: int) -> set[str]:
    """Return bounded source-component directory names without interpreting code text."""
    names: set[str] = set()
    inspected = 0
    try:
        iterator = root.rglob("Make/files")
        for make_file in iterator:
            if inspected >= max_files:
                break
            inspected += 1
            try:
                resolved = make_file.resolve()
            except OSError:
                continue
            if not resolved.is_file() or root not in resolved.parents:
                continue
            parent = resolved.parent.parent
            if parent == root or not _safe_executable_name(parent.name):
                continue
            names.add(parent.name)
    except OSError:
        return names
    return names


_RUNTIME_SELECTION = re.compile(
    r"\baddToRunTimeSelectionTable\s*\(\s*"
    r"(?P<base>[A-Za-z_][A-Za-z0-9_:]*)\s*,\s*"
    r"(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*,",
    re.MULTILINE,
)
_RUNTIME_SELECTION_NAMED = re.compile(
    r"\baddNamedToRunTimeSelectionTable\s*\(\s*"
    r"(?P<base>[A-Za-z_][A-Za-z0-9_:]*)\s*,\s*"
    r"(?P<type>[A-Za-z_][A-Za-z0-9_]*)\s*,\s*"
    r"[A-Za-z_][A-Za-z0-9_]*\s*,\s*"
    r"(?P<lookup>[A-Za-z_][A-Za-z0-9_]*)\s*\)",
    re.MULTILINE,
)


def _runtime_selection_names(
    root: Path,
    *,
    base_types: Sequence[str],
    max_files: int,
    max_file_bytes: int,
) -> set[str]:
    """Discover bounded OpenFOAM run-time selection registrations from source.

    This is installation evidence, not a hand-maintained capability list.  Only simple
    identifier registrations are retained; templated/generated registrations that cannot be
    resolved without compiling OpenFOAM are deliberately left to documented fallback
    profiles or native runtime evidence.
    """

    wanted = {item.split("::")[-1] for item in base_types}
    names: set[str] = set()
    inspected = 0
    try:
        iterator = root.rglob("*.C")
        for source in iterator:
            if inspected >= max_files:
                break
            inspected += 1
            try:
                resolved = source.resolve()
                if not resolved.is_file() or root not in resolved.parents:
                    continue
                if resolved.stat().st_size > max_file_bytes:
                    continue
                text = resolved.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            for match in _RUNTIME_SELECTION.finditer(text):
                base = match.group("base").split("::")[-1]
                name = match.group("name")
                if base in wanted and _safe_executable_name(name):
                    names.add(name)
            for match in _RUNTIME_SELECTION_NAMED.finditer(text):
                base = match.group("base").split("::")[-1]
                lookup = match.group("lookup")
                if base in wanted and _safe_executable_name(lookup):
                    names.add(lookup)
    except OSError:
        return names
    return names
