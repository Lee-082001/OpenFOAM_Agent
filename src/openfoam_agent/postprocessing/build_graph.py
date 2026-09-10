from __future__ import annotations

from dataclasses import dataclass

from openfoam_agent.tools.foam_serializer import FoamSerializationError, serialize_foam_dictionary


@dataclass(frozen=True)
class PostProcessGraph:
    config_files: dict[str, str]
    run_specs: tuple[object, ...]
    failures: tuple[str, ...]
    warnings: tuple[str, ...]

    @property
    def valid(self) -> bool:
        return not self.failures


def compile_postprocess_graph(workspace, plan) -> PostProcessGraph:
    """Compile config ownership and run dependencies before any postprocess write."""
    configs: dict[str, str] = {}
    failures: list[str] = []
    warnings: list[str] = []

    def add(path: str, content: str) -> None:
        if not path.startswith("postprocessConfig/") or ".." in path:
            failures.append(f"Unsafe post-processing config path: {path}")
            return
        if path in configs and configs[path] != content:
            failures.append(f"Conflicting post-processing config content for {path}.")
            return
        configs[path] = content

    for item in plan.configs:
        add(item.path, item.content)
    for item in plan.typed_configs:
        try:
            add(item.path, serialize_foam_dictionary(item))
        except FoamSerializationError as exc:
            failures.append(f"Typed post-processing config serialization failed for {item.path}: {exc}")

    failures.extend(workspace.validate_candidate_bundle(configs))
    available = set(configs)
    for run in plan.runs:
        path = run.dictionary_path
        if not path.startswith("postprocessConfig/") or ".." in path:
            failures.append(f"Post-processing run dictionary must live under postprocessConfig/: {path}")
            continue
        if path in available:
            continue
        try:
            workspace.resolve_case_path(path, must_exist=True)
        except (FileNotFoundError, OSError):
            failures.append(f"Post-processing run references missing config: {path}")


    for analysis in plan.force_analyses:
        path = analysis.dictionary_path
        if not path.startswith("postprocessConfig/") or ".." in path:
            failures.append(f"Force-analysis dictionary must live under postprocessConfig/: {path}")
            continue
        if path in available:
            continue
        try:
            workspace.resolve_case_path(path, must_exist=True)
        except (FileNotFoundError, OSError):
            failures.append(f"Force analysis references missing config: {path}")

    return PostProcessGraph(
        config_files=configs,
        run_specs=tuple(plan.runs),
        failures=tuple(dict.fromkeys(failures)),
        warnings=tuple(dict.fromkeys(warnings)),
    )
