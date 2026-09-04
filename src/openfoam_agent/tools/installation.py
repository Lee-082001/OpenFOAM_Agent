from __future__ import annotations

import os
from pathlib import Path
from typing import Mapping, Sequence

from openfoam_agent.schemas.installation import (
    InstalledComponent,
    InstalledExecutable,
    InstalledOpenFOAMIR,
)


# Operational solver modules documented in the Foundation v13/v14 User Guides.
# Base classes (fluidSolver, twoPhaseSolver, ...) are intentionally excluded because
# they are not direct engineering execution targets.
DOCUMENTED_SOLVER_MODULES: dict[str, tuple[str, ...]] = {
    "13": (
        "fluid", "incompressibleDenseParticleFluid", "incompressibleFluid",
        "multicomponentFluid", "shockFluid", "XiFluid", "compressibleMultiphaseVoF",
        "compressibleVoF", "incompressibleDriftFlux", "incompressibleMultiphaseVoF",
        "incompressibleVoF", "isothermalFluid", "multiphaseEuler", "solid",
        "solidDisplacement", "isothermalFilm", "film", "functions", "movingMesh",
    ),
    "14": (
        "fluid", "incompressibleDenseParticleFluid", "incompressibleFluid",
        "multicomponentFluid", "shockFluid", "XiFluid", "compressibleMultiphaseVoF",
        "compressibleVoF", "incompressibleDriftFlux", "incompressibleMultiphaseVoF",
        "incompressibleVoF", "isothermalFluid", "multiphaseEuler", "solid",
        "solidDisplacement", "isothermalFilm", "film", "functions", "movingMesh",
    ),
}

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

# These are documented/runtime-selectable models that matter for capability routing even
# when source trees are not installed.  Installed source discovery supplements this list.
DOCUMENTED_FV_MODELS: dict[str, tuple[str, ...]] = {
    "13": ("heatSource",),
    "14": ("heatSource",),
}


class OpenFOAMInstallationDiscovery:
    """Discover all trusted Foundation applications plus runtime-selectable components.

    Executables are discovered from trusted OpenFOAM PATH entries and FOAM_APPBIN, not
    from a hand-maintained command allowlist.  Source-tree discovery is additive and
    bounded; documented v13/v14 profiles provide solver-module/model fallback names.
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
        if version in DOCUMENTED_SOLVER_MODULES:
            for name in DOCUMENTED_SOLVER_MODULES[version]:
                items[("solver_module", name)] = InstalledComponent(
                    name=name, category="solver_module", source="documented_profile"
                )
        if version in DOCUMENTED_FV_MODELS:
            for name in DOCUMENTED_FV_MODELS[version]:
                items[("fv_model", name)] = InstalledComponent(
                    name=name, category="fv_model", source="documented_profile"
                )

        modules_root = self._trusted_directory(self.base_env.get("FOAM_MODULES", ""))
        if modules_root is not None:
            for name in _source_component_names(modules_root, max_files=2500):
                items[("solver_module", name)] = InstalledComponent(
                    name=name, category="solver_module", source="installed_source"
                )

        src_root = self._trusted_directory(self.base_env.get("FOAM_SRC", ""))
        if src_root is not None:
            for relative, category in (("fvModels", "fv_model"), ("functionObjects", "function_object")):
                root = src_root / relative
                if not root.is_dir():
                    continue
                for name in _source_component_names(root, max_files=5000):
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
