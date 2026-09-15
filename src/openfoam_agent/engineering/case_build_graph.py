from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from openfoam_agent.schemas.engineering import ExecuteCasePlanAction, NativeOpenFOAMCommand
from openfoam_agent.contracts.execution_scopes import (
    auto_mesh_consumers,
    final_mesh_validation_commands,
    render_scope_template,
    scope_for_native_arguments,
)
from openfoam_agent.tools.native_contracts import (
    command_permitted,
    native_command_region,
    native_tool_contract,
    required_dictionary,
)


# Files that are authored inputs but are not OpenFOAM dictionaries/fields. They are
# validated by their actual consumer (surfaceCheck, mesh utility, solver) instead of
# the FoamFile/header parser.
_NON_DICTIONARY_EXTENSIONS = {
    ".stl", ".obj", ".off", ".vtk", ".vtp", ".csv", ".dat", ".emesh", ".gz",
}
_SURFACE_EXTENSIONS = {".stl", ".obj", ".off", ".vtk", ".vtp"}

# Native utility prerequisites/effects are centralized in native_tool_contracts.py.


@dataclass(frozen=True)
class CaseBuildGraph:
    """Controller-compiled build graph for one frozen case candidate.

    The LLM owns engineering content. Python owns manifest coverage, validation target
    discovery and executable action ordering. This prevents a stale model-authored
    validation list from referring to a file that was never authored.
    """

    required_paths: tuple[str, ...]
    authored_paths: tuple[str, ...]
    authored_required_paths: tuple[str, ...]
    deferred_native_required_paths: tuple[str, ...]
    missing_required_paths: tuple[str, ...]
    dictionary_paths: tuple[str, ...]
    surface_paths: tuple[str, ...]
    native_pipeline: tuple[NativeOpenFOAMCommand, ...]
    failures: tuple[str, ...]
    warnings: tuple[str, ...]

    @property
    def valid(self) -> bool:
        return not self.failures


def _dict_override(arguments: list[str]) -> str | None:
    for flag in ("-dict", "-dictFile"):
        if flag in arguments:
            index = arguments.index(flag)
            if index + 1 < len(arguments):
                return arguments[index + 1]
    return None


def _dedupe_invocations(items: list[NativeOpenFOAMCommand]) -> list[NativeOpenFOAMCommand]:
    unique: list[NativeOpenFOAMCommand] = []
    seen: set[tuple[str, tuple[str, ...], str]] = set()
    for item in items:
        token = (item.command, tuple(item.arguments), item.role)
        if token in seen:
            continue
        seen.add(token)
        unique.append(item)
    return unique


def _path_is_within(path: str, root: str) -> bool:
    normalized = str(PurePosixPath(path)).rstrip("/")
    normalized_root = str(PurePosixPath(root)).rstrip("/")
    return normalized == normalized_root or normalized.startswith(normalized_root + "/")


def _deferred_native_required_paths(
    plan,
    required_paths: tuple[str, ...],
    native_pipeline: list[NativeOpenFOAMCommand],
) -> tuple[str, ...]:
    """Return manifest paths whose producer is an executable native graph node.

    This is producer-contract based, not a filename exception.  A required polyMesh
    artifact is deferred only when the compiled case graph contains a native utility
    whose declared ``mesh_dependency_outputs`` cover that exact scope/path.  External
    or pre-existing mesh inputs therefore remain ordinary required inputs when no
    producer exists in the current graph.
    """
    output_roots: list[str] = []
    for invocation in native_pipeline:
        contract = native_tool_contract(invocation.command)
        if not contract.mesh_dependency_outputs:
            continue
        try:
            scope = scope_for_native_arguments(plan, invocation.arguments)
        except ValueError:
            # An invocation that cannot be bound to one declared scope cannot prove
            # ownership of any required output. Other validation will surface the
            # malformed/ambiguous native strategy.
            continue
        for template in contract.mesh_dependency_outputs:
            root = render_scope_template(template, scope)
            if root not in output_roots:
                output_roots.append(root)
    return tuple(
        path
        for path in required_paths
        if any(_path_is_within(path, root) for root in output_roots)
    )


def compile_case_build_graph(
    execution: ExecuteCasePlanAction,
    candidate_bundle: dict[str, str],
) -> CaseBuildGraph:
    """Compile one authoritative authoring/validation/native graph.

    ``EngineeringPlan.required_case_files`` is the only solve-input manifest. LLM
    fields such as ``validate_dictionaries`` and ``surface_checks`` are compatibility
    hints and never create actions independently. Native mesh commands are strategy
    hints; Python removes documentary probes, verifies strong prerequisites, infers
    obvious mesh consumers from authored dictionaries, and owns final checkMesh order.
    """

    required = tuple(dict.fromkeys(execution.plan.required_case_files))
    authored = tuple(candidate_bundle.keys())
    authored_set = set(authored)
    failures: list[str] = []
    warnings: list[str] = []

    if not required:
        failures.append(
            "EngineeringPlan.required_case_files is empty; the Agent must declare at least one solve-input path before authoring."
        )

    dictionary_paths = tuple(
        path
        for path in authored
        if Path(path).suffix.lower() not in _NON_DICTIONARY_EXTENSIONS
    )
    surface_paths = tuple(
        path
        for path in authored
        if Path(path).suffix.lower() in _SURFACE_EXTENSIONS
    )

    if execution.native_pipeline:
        hints = list(execution.native_pipeline)
    else:
        hints = [
            NativeOpenFOAMCommand(command=command, role="mesh")
            for command in execution.mesh_commands
        ]

    # Documentary probes/finalizers are never model execution authority. Native
    # consumers inferred from authored dictionaries come from package-data contracts.
    strategy = [
        item for item in hints
        if item.command not in {"foamDictionary", "surfaceCheck"}
        and not native_tool_contract(item.command).controller_finalizer
    ]
    strategy = _dedupe_invocations(strategy)

    try:
        inferred = [item for _, item in auto_mesh_consumers(execution.plan, authored)]
    except ValueError as exc:
        failures.append(f"Could not compile scope-bound mesh consumers: {exc}")
        inferred = []
    strategy = _dedupe_invocations([*inferred, *strategy])

    executable_strategy: list[NativeOpenFOAMCommand] = []
    for invocation in strategy:
        try:
            required_dict = required_dictionary(invocation.command, invocation.arguments)
        except ValueError as exc:
            failures.append(f"Invalid native invocation {invocation.command}: {exc}")
            continue
        if not command_permitted(invocation.command, "authoring"):
            warnings.append(f"Dropped native hint {invocation.command}: command is not permitted in authoring phase.")
            continue
        if required_dict and required_dict not in authored_set:
            warnings.append(
                f"Dropped stale native hint {invocation.command}: required authored input {required_dict} is absent."
            )
            continue
        executable_strategy.append(invocation)

    authored_required = tuple(path for path in required if path in authored_set)
    deferred_native = _deferred_native_required_paths(
        execution.plan,
        tuple(path for path in required if path not in authored_set),
        executable_strategy,
    )
    deferred_set = set(deferred_native)
    missing = tuple(
        path for path in required
        if path not in authored_set and path not in deferred_set
    )
    if missing:
        failures.append(
            "Required case manifest is not fully authored or backed by a compiled native producer: "
            + ", ".join(missing)
        )

    try:
        checks = final_mesh_validation_commands(execution.plan)
    except ValueError as exc:
        failures.append(f"Could not compile scope-bound mesh validation: {exc}")
        checks = []

    # Surface checking is controller-owned and derived from actual artifacts. It is
    # represented separately by SurfaceCheckAction, not duplicated in native_pipeline.
    pipeline = tuple(executable_strategy + checks)

    # Validation mirrors can still be useful diagnostics in reports, but stale paths
    # are explicitly ignored rather than executed.
    for hinted in list(execution.validate_dictionaries) + list(execution.surface_checks):
        if hinted not in authored_set:
            warnings.append(
                f"Ignored stale validation hint for unauthored path: {hinted}"
            )

    return CaseBuildGraph(
        required_paths=required,
        authored_paths=authored,
        authored_required_paths=authored_required,
        deferred_native_required_paths=deferred_native,
        missing_required_paths=missing,
        dictionary_paths=dictionary_paths,
        surface_paths=surface_paths,
        native_pipeline=pipeline,
        failures=tuple(dict.fromkeys(failures)),
        warnings=tuple(dict.fromkeys(warnings)),
    )
