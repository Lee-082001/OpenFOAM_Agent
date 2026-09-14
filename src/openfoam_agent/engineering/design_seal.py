from __future__ import annotations

from dataclasses import dataclass

from openfoam_agent.schemas.engineering import (
    EngineeringDefaultAssumption,
    EngineeringDesign,
    EngineeringPlan,
)
from openfoam_agent.tools.execution_contracts import (
    provider_requirements as execution_provider_requirements,
    sealed_solver_mirror as execution_solver_mirror,
    validate_execution_contract,
)


class DesignSealError(ValueError):
    """The Agent-owned design cannot be sealed against deterministic controller state."""


@dataclass(frozen=True)
class _ProviderRequirement:
    provider_id: str
    expected_name: str
    allowed_types: frozenset[str]
    label: str


def _provider_requirements(design: EngineeringDesign) -> list[_ProviderRequirement]:
    execution = design.execution
    if execution is None:
        assert design.solver is not None and design.solver_provider_id is not None
        return [
            _ProviderRequirement(
                design.solver_provider_id,
                design.solver,
                frozenset({"solver", "solver_module", "generated_solver"}),
                "solver",
            )
        ]

    result: list[_ProviderRequirement] = []
    for provider_id, expected_name, label in execution_provider_requirements(execution):
        allowed = (
            frozenset({"execution_driver", "solver_application"})
            if label == "execution driver"
            else frozenset({"solver", "solver_module", "generated_solver"})
        )
        result.append(_ProviderRequirement(provider_id, expected_name, allowed, label))
    return result


def _sealed_solver_mirror(design: EngineeringDesign) -> tuple[str, str]:
    execution = design.execution
    if execution is None:
        assert design.solver is not None and design.solver_provider_id is not None
        return design.solver, design.solver_provider_id
    return execution_solver_mirror(execution)


def materialize_engineering_plan(design: EngineeringDesign, state, catalog) -> EngineeringPlan:
    """Seal Agent decisions with controller-owned identity/provenance metadata.

    This function never chooses a solver, mesh, BC, material value, or numerical
    setting. It only validates provider identities the Agent already chose and attaches
    deterministic state that Python already owns.
    """
    if state.intake is None or not state.intake_digest:
        raise DesignSealError("Confirmed intake is unavailable; staged design cannot be sealed.")
    if not design.required_case_files:
        raise DesignSealError(
            "Agent-owned required case manifest is empty; staged design cannot be sealed."
        )

    if design.execution is not None:
        execution_failures = validate_execution_contract(design.execution)
        if execution_failures:
            raise DesignSealError("Execution topology violates the selected provider contract: " + " | ".join(execution_failures))

    versions: set[str] = set()
    seen: set[str] = set()
    for requirement in _provider_requirements(design):
        if requirement.provider_id in seen:
            continue
        seen.add(requirement.provider_id)
        provider = catalog.provider(requirement.provider_id)
        if provider is None:
            raise DesignSealError(
                f"{requirement.label.title()} provider '{requirement.provider_id}' does not exist in the capability catalog."
            )
        if provider.provider_type not in requirement.allowed_types:
            raise DesignSealError(
                f"Capability provider '{requirement.provider_id}' has type {provider.provider_type!r}, "
                f"which is not valid for {requirement.label}."
            )
        if provider.name != requirement.expected_name:
            raise DesignSealError(
                f"{requirement.label.title()} '{requirement.expected_name}' disagrees with capability provider "
                f"'{requirement.provider_id}' ({provider.name})."
            )
        versions.add(str(provider.openfoam_version))

    if len(versions) != 1:
        raise DesignSealError(
            "Selected execution providers do not resolve to one OpenFOAM Foundation version: "
            + ", ".join(sorted(versions))
        )
    openfoam_version = next(iter(versions))
    if openfoam_version not in {"13", "14"}:
        raise DesignSealError(
            f"Selected capability providers target unsupported Foundation version {openfoam_version}."
        )

    solver, solver_provider_id = _sealed_solver_mirror(design)
    fact_ids = [fact.id for fact in state.intake.facts if fact.category != "context"]
    # Frozen intake digest + confirmed_fact_ids are the identity closure.  Do not
    # manufacture implementation evidence merely to make every fact appear artifact-backed.
    bindings = []
    defaults = [
        EngineeringDefaultAssumption(
            parameter=item.parameter,
            value=item.value,
            unit=item.unit,
            basis=item.basis,
            rationale=item.rationale,
            source="engineering_default",
            evidence_ids=[],
        )
        for item in design.engineering_defaults
    ]

    data = design.model_dump(mode="python")
    data["region_layouts"] = [
        item.model_dump(mode="python") for item in _canonical_region_layouts(design)
    ]
    data.update(
        schema_version="2.0",
        solver=solver,
        solver_provider_id=solver_provider_id,
        engineering_defaults=[item.model_dump(mode="python") for item in defaults],
        implementation_evidence_bindings=[],
        openfoam_distribution="foundation",
        openfoam_version=openfoam_version,
        plan_conflicts=[],
        confirmed_fact_ids=fact_ids,
        confirmed_fact_bindings=[item.model_dump(mode="python") for item in bindings],
        evidence=[],
        confirmed_intake_sha256=state.intake_digest,
    )
    return EngineeringPlan.model_validate(data)
