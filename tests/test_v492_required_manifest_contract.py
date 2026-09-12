from __future__ import annotations

from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from conftest import make_state
from openfoam_agent.engineering.case_build_graph import compile_case_build_graph
from openfoam_agent.engineering.design_seal import DesignSealError, materialize_engineering_plan
from openfoam_agent.schemas.engineering import EngineeringDesign, OpenFOAMExecutionSpec


class _Catalog:
    def __init__(self):
        self.items = {
            "driver.foamRun": SimpleNamespace(
                id="driver.foamRun",
                name="foamRun",
                provider_type="execution_driver",
                openfoam_version="14",
            ),
            "solver.incompressibleFluid": SimpleNamespace(
                id="solver.incompressibleFluid",
                name="incompressibleFluid",
                provider_type="solver_module",
                openfoam_version="14",
            ),
        }

    def provider(self, provider_id):
        return self.items.get(provider_id)


def _design(required_case_files):
    return EngineeringDesign(
        case_name="manifestContract",
        execution=OpenFOAMExecutionSpec(
            driver="foamRun",
            driver_provider_id="driver.foamRun",
            solver_module="incompressibleFluid",
            solver_provider_id="solver.incompressibleFluid",
        ),
        problem_interpretation="Controller-sealed CFD design manifest contract regression.",
        temporal_behavior="steady",
        motion_kind="static",
        mesh_motion_requirement="static",
        mesh_strategy="Agent-selected static mesh.",
        required_case_files=required_case_files,
    )


def test_staged_design_schema_requires_nonempty_required_manifest():
    schema = EngineeringDesign.model_json_schema()
    field = schema["properties"]["required_case_files"]
    assert field["minItems"] == 1
    with pytest.raises(ValidationError):
        _design([])


def test_sealer_defensively_rejects_model_copy_with_empty_manifest():
    valid = _design(["system/controlDict"])
    bypassed = valid.model_copy(update={"required_case_files": []})
    with pytest.raises(DesignSealError, match="required case manifest is empty"):
        materialize_engineering_plan(bypassed, make_state(), _Catalog())


def test_case_build_graph_fails_closed_for_empty_durable_manifest():
    execution = SimpleNamespace(
        plan=SimpleNamespace(required_case_files=[]),
        native_pipeline=[],
        mesh_commands=[],
        validate_dictionaries=[],
        surface_checks=[],
    )
    graph = compile_case_build_graph(execution, {"system/controlDict": "dummy"})
    # Real CaseBuildGraph exposes .valid/.failures. Keep the assertion explicit so
    # malformed legacy/controller state cannot be accepted merely because missing=[] .
    assert not graph.valid
    assert any("required_case_files is empty" in item for item in graph.failures)
