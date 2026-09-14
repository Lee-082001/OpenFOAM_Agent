from types import SimpleNamespace

from openfoam_agent.contracts.evidence import implementation_evidence_pack, observed_syntax
from openfoam_agent.contracts.regions import region_layouts
from openfoam_agent.schemas.common import ToolResult
from openfoam_agent.schemas.installation import InstalledComponent, InstalledExecutable, InstalledOpenFOAMIR
from openfoam_agent.tools.capability_catalog import CapabilityCatalog
from openfoam_agent.tools.diagnostics import classify_native_validation


def test_installed_identity_does_not_manufacture_cfd_semantics():
    installation = InstalledOpenFOAMIR(
        version="13",
        installation_configured=True,
        executables=[
            InstalledExecutable(name="foamMultiRun", category="execution_driver"),
        ],
        components=[
            InstalledComponent(name="solid", category="solver_module", source="installed_source"),
            InstalledComponent(name="heatSource", category="fv_model", source="installed_source"),
        ],
    )
    providers = {item.id: item for item in CapabilityCatalog._installed_providers(installation)}

    driver = providers["installed.application.foamMultiRun"]
    solid = providers["installed.solver_module.solid"]
    heat_source = providers["installed.fv_model.heatSource"]

    assert driver.capabilities == ["execution.driver.foamMultiRun"]
    assert solid.capabilities == ["solver.module.solid"]
    assert heat_source.capabilities == ["fvModel.heatSource"]
    forbidden = {
        "heat_transfer.conjugate",
        "heat_transfer.solid",
        "equation.energy.solid",
        "source.heat.volumetric",
        "heat_generation.volumetric",
    }
    assert forbidden.isdisjoint(driver.capabilities + solid.capabilities + heat_source.capabilities)
    assert all(item.metadata["semantic_capabilities_inferred"] is False for item in providers.values())


def test_execution_regions_are_authoritative_over_manifest_projection():
    plan = SimpleNamespace(
        required_case_files=[
            "system/controlDict",
            "0/battery/T",
            "0/heater/T",
            "system/battery/fvSchemes",
            "system/heater/fvSchemes",
        ],
        execution=SimpleNamespace(
            regions=[
                SimpleNamespace(region="battery", solver_module="solid"),
                SimpleNamespace(region="heater", solver_module="solid"),
            ]
        ),
        region_layouts=[],
    )
    layouts = region_layouts(plan)
    assert [(item.region, item.solver_module) for item in layouts] == [
        ("battery", "solid"),
        ("heater", "solid"),
    ]
    assert layouts[0].required_fields == ["T"]
    assert layouts[1].required_fields == ["T"]


def test_manifest_cannot_create_region_outside_execution_topology():
    plan = SimpleNamespace(
        required_case_files=["0/battery/T", "0/rogue/T"],
        execution=SimpleNamespace(
            regions=[SimpleNamespace(region="battery", solver_module="solid")]
        ),
        region_layouts=[],
    )
    try:
        region_layouts(plan)
    except ValueError as exc:
        assert "outside the authoritative execution topology" in str(exc)
    else:
        raise AssertionError("manifest-created rogue region was accepted")


def test_unknown_native_nonzero_is_inconclusive_not_case_evidence():
    result = ToolResult(
        success=False,
        command=["foamRun"],
        return_code=17,
        stdout="consumer stopped without an OpenFOAM fatal marker",
        stderr="",
    )
    assessment = classify_native_validation(result, command_name="foamRun", probe=False)
    assert assessment.status == "inconclusive"
    assert assessment.category == "tool"


def test_explicit_openfoam_fatal_remains_case_failure():
    result = ToolResult(
        success=False,
        command=["foamRun"],
        return_code=1,
        stdout="--> FOAM FATAL IO ERROR:\nUnknown transport type constIso",
        stderr="",
    )
    assessment = classify_native_validation(result, command_name="foamRun", probe=False)
    assert assessment.status == "fail"
    assert assessment.category == "case"


def _record(record_id: str, *, content: str, target_case_files=None):
    return SimpleNamespace(
        record_id=record_id,
        payload={
            "reference": "source:solvers/fvSolution.C",
            "content": content,
            "target_case_files": list(target_case_files or []),
        },
    )


def _plan(bindings=()):
    return SimpleNamespace(
        required_case_files=["system/fvSolution"],
        implementation_evidence_bindings=list(bindings),
        openfoam_version="13",
    )


def test_documentary_evidence_never_binds_by_basename_guess():
    state = SimpleNamespace(
        engineering_evidence_records=[
            _record("evrec_aaaaaaaaaaaaaaaaaaaa", content="fvSolution syntax example")
        ]
    )
    pack = implementation_evidence_pack(state, _plan(), max_chars=None)
    assert pack["records"] == []
    assert pack["file_coverage"] == [
        {"path": "system/fvSolution", "evidence_ids": [], "status": "missing"}
    ]
    assert pack["complete"] is False


def test_explicit_scoped_evidence_is_content_bound():
    state = SimpleNamespace(
        engineering_evidence_records=[
            _record(
                "evrec_bbbbbbbbbbbbbbbbbbbb",
                content="solver { T { solver smoothSolver; } }",
                target_case_files=["system/fvSolution"],
            )
        ]
    )
    available, targets = observed_syntax(state, "13")
    assert targets["system/fvSolution"]
    item = next(iter(available.values()))
    assert len(item["observation_ids"]) == 1
    assert item["observation_ids"][0].startswith("obs_ref_")

    pack = implementation_evidence_pack(state, _plan(), max_chars=None)
    assert pack["complete"] is True
    assert pack["file_coverage"][0]["status"] == "explicit"


def test_observation_identity_changes_when_excerpt_changes():
    state_a = SimpleNamespace(
        engineering_evidence_records=[
            _record("evrec_cccccccccccccccccccc", content="alpha")
        ]
    )
    state_b = SimpleNamespace(
        engineering_evidence_records=[
            _record("evrec_cccccccccccccccccccc", content="beta")
        ]
    )
    obs_a = next(iter(observed_syntax(state_a, "13")[0].values()))["observation_ids"][0]
    obs_b = next(iter(observed_syntax(state_b, "13")[0].values()))["observation_ids"][0]
    assert obs_a != obs_b
