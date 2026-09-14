from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

from conftest import make_state
from openfoam_agent.agents.intake import validate_intake_provenance
from openfoam_agent.contracts.evidence_policy import provider_is_sufficient
from openfoam_agent.contracts.regions import region_layouts
from openfoam_agent.engineering.design_seal import materialize_engineering_plan
from openfoam_agent.engineering.native_scope import final_check_mesh_commands, mesh_affecting_path
from openfoam_agent.schemas.common import ToolResult
from openfoam_agent.schemas.engineering import (
    EngineeringDesign,
    ObservedEngineeringEvidence,
    OpenFOAMExecutionSpec,
    RegionSolverAssignment,
)
from openfoam_agent.schemas.installation import InstalledComponent, InstalledExecutable, InstalledOpenFOAMIR
from openfoam_agent.schemas.intake import CFDIntakeSpec, IntakeFact
from openfoam_agent.schemas.request import UserRequest
from openfoam_agent.tools.capability_catalog import CapabilityCatalog
from openfoam_agent.tools.diagnostics import classify_native_validation
from openfoam_agent.tools.native_contracts import native_tool_contract, required_dictionary
from openfoam_agent.tools.references import normalize_query


class _Catalog:
    def __init__(self):
        self.items = {
            "driver.foamMultiRun": SimpleNamespace(
                id="driver.foamMultiRun", name="foamMultiRun",
                provider_type="execution_driver", openfoam_version="14",
            ),
            "solver.solid": SimpleNamespace(
                id="solver.solid", name="solid",
                provider_type="solver_module", openfoam_version="14",
            ),
        }

    def provider(self, provider_id):
        return self.items.get(provider_id)


def test_v500_installation_identity_does_not_invent_physics_capabilities(tmp_path):
    graph = tmp_path / "graph.json"
    graph.write_text(json.dumps({
        "schema_version": "1.0",
        "graph_id": "v500-test",
        "openfoam_distribution": "foundation",
        "openfoam_version": "14",
        "providers": [],
    }), encoding="utf-8")
    installation = InstalledOpenFOAMIR(
        version="14",
        installation_configured=True,
        components=[InstalledComponent(name="solid", category="solver_module", source="installed_source")],
    )
    catalog = CapabilityCatalog(graph, installation=installation)
    provider = catalog.provider("installed.solver_module.solid")
    assert provider is not None
    assert provider.capabilities == ["solver.module.solid"]
    assert "heat_transfer.solid" not in provider.capabilities
    assert "heat_transfer.conjugate" not in provider.capabilities
    assert provider.metadata["semantic_authority_documented_graph"] is False
    assert not provider_is_sufficient(provider, executable=False)


def test_v500_documented_semantics_are_strengthened_by_installed_presence(tmp_path):
    graph = tmp_path / "graph.json"
    graph.write_text(json.dumps({
        "schema_version": "1.0",
        "graph_id": "v500-test",
        "openfoam_distribution": "foundation",
        "openfoam_version": "14",
        "providers": [{
            "id": "driver.foamMultiRun",
            "name": "foamMultiRun",
            "provider_type": "execution_driver",
            "capabilities": ["execution.multiregion"],
            "openfoam_version": "14",
            "verified": True,
            "verification_level": "documented",
            "evidence": [{"kind": "user_guide", "reference": "guide:foamMultiRun", "note": "documented"}],
        }],
    }), encoding="utf-8")
    installation = InstalledOpenFOAMIR(
        version="14", installation_configured=True,
        executables=[InstalledExecutable(name="foamMultiRun", category="execution_driver", trusted=True)],
    )
    provider = CapabilityCatalog(graph, installation=installation).provider("driver.foamMultiRun")
    assert provider is not None
    assert provider.capabilities == ["execution.multiregion"]
    assert provider.verification_level == "binary_present"
    assert provider.metadata["semantic_authority_documented_graph"] is True
    assert provider_is_sufficient(provider, executable=True)


def test_v500_design_seal_keeps_identity_closure_without_fake_fact_bindings():
    state = make_state()
    design = EngineeringDesign(
        case_name="twoSolid",
        execution=OpenFOAMExecutionSpec(
            driver="foamMultiRun",
            driver_provider_id="driver.foamMultiRun",
            regions=[
                RegionSolverAssignment(region="battery", solver_module="solid", provider_id="solver.solid"),
                RegionSolverAssignment(region="heater", solver_module="solid", provider_id="solver.solid"),
            ],
        ),
        problem_interpretation="Transient two-solid heat-transfer preparation.",
        temporal_behavior="transient",
        motion_kind="static",
        mesh_motion_requirement="static",
        mesh_strategy="Agent-selected two-region blockMesh strategy.",
        required_case_files=[
            "system/controlDict",
            "system/battery/blockMeshDict", "system/heater/blockMeshDict",
            "0/battery/T", "0/heater/T",
        ],
    )
    plan = materialize_engineering_plan(design, state, _Catalog())
    expected = [fact.id for fact in state.intake.facts if fact.category != "context"]
    assert plan.confirmed_fact_ids == expected
    assert plan.confirmed_fact_bindings == []
    layouts = region_layouts(plan)
    assert [(x.region, x.solver_module) for x in layouts] == [
        ("battery", "solid"), ("heater", "solid")
    ]


def test_v500_execution_regions_are_authority_even_if_legacy_layout_disagrees():
    plan = SimpleNamespace(
        execution=OpenFOAMExecutionSpec(
            driver="foamMultiRun", driver_provider_id="driver.foamMultiRun",
            regions=[
                RegionSolverAssignment(region="battery", solver_module="solid", provider_id="solver.solid"),
                RegionSolverAssignment(region="heater", solver_module="solid", provider_id="solver.solid"),
            ],
        ),
        region_layouts=[SimpleNamespace(region="wrong", solver_module="wrong")],
        required_case_files=["0/battery/T", "0/heater/T"],
    )
    layouts = region_layouts(plan)
    assert [(x.region, x.solver_module) for x in layouts] == [
        ("battery", "solid"), ("heater", "solid")
    ]


def test_v500_region_native_scope_is_shared_by_mesh_change_and_final_validation():
    plan = SimpleNamespace(
        execution=OpenFOAMExecutionSpec(
            driver="foamMultiRun", driver_provider_id="driver.foamMultiRun",
            regions=[
                RegionSolverAssignment(region="battery", solver_module="solid", provider_id="solver.solid"),
                RegionSolverAssignment(region="heater", solver_module="solid", provider_id="solver.solid"),
            ],
        ),
        region_layouts=[],
        required_case_files=["0/battery/T", "0/heater/T"],
    )
    assert required_dictionary("blockMesh", ["-region", "battery"]) == "system/battery/blockMeshDict"
    assert mesh_affecting_path("system/battery/blockMeshDict")
    assert mesh_affecting_path("constant/heater/polyMesh/boundary")
    checks = final_check_mesh_commands(plan)
    assert [(x.command, x.arguments) for x in checks] == [
        ("checkMesh", ["-region", "battery"]),
        ("checkMesh", ["-region", "heater"]),
    ]


def test_v500_native_tool_contracts_are_loaded_from_package_data():
    contract = native_tool_contract("blockMesh")
    assert contract.effect == "mesh"
    assert contract.required_dictionary == "system/blockMeshDict"
    assert native_tool_contract("foamRun").effect == "solve"


def test_v500_unknown_nonzero_native_result_is_inconclusive_not_case_evidence():
    result = ToolResult(success=False, command=["customUtility"], return_code=1, stderr="unclassified native error")
    assessment = classify_native_validation(result, command_name="customUtility", probe=False)
    assert assessment.status == "inconclusive"
    assert assessment.category == "tool"


def test_v500_user_fact_gets_controller_issued_immutable_span_locator():
    request = UserRequest(prompt="Re=1000 square obstacle", exploratory_completion_authorized=False)
    intake = CFDIntakeSpec(
        semantic_contract_version="2",
        title="test",
        facts=[
            IntakeFact(id="request.summary", category="context", label="request",
                       value=request.prompt, source="derived", reason="normalized"),
            IntakeFact(id="classification.problem_type", category="classification", label="class",
                       value="custom", source="derived", reason="not explicitly classified",
                       depends_on=["request.summary"]),
            IntakeFact(id="objective.primary", category="objective", label="objective",
                       value="square obstacle", source="user", evidence="square obstacle"),
            IntakeFact(id="operating.reynolds_number", category="scale", label="Re",
                       value="1000", source="user", evidence="Re=1000"),
        ],
        status="ready_for_review",
    )
    validate_intake_provenance(intake, request)
    fact = intake.fact("operating.reynolds_number")
    assert fact is not None and fact.evidence_locator is not None
    locator = fact.evidence_locator
    assert locator.source_kind == "conversation_turn"
    assert locator.source_index == 0
    assert request.prompt[locator.start_char:locator.end_char] == "Re=1000"
    assert locator.source_sha256 == hashlib.sha256(request.prompt.encode()).hexdigest()


def test_v500_observed_evidence_descriptor_is_content_bound():
    item = ObservedEngineeringEvidence(
        evidence_id="ev_cap_0123456789abcdef0123",
        kind="capability",
        reference="driver.foamMultiRun",
        summary="binary present + documented semantics",
    )
    assert item.observation_sha256 is not None
    copied = item.model_copy(update={"summary": "tampered"})
    try:
        ObservedEngineeringEvidence.model_validate(copied.model_dump(mode="python"))
    except ValueError:
        pass
    else:
        raise AssertionError("tampered observation descriptor must not retain the old hash")


def test_v500_reference_search_does_not_expand_domain_aliases():
    assert normalize_query("열전달") == ["열전달"]
    assert "heat" not in normalize_query("열전달")
