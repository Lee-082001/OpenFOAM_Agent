from __future__ import annotations

import pytest

from conftest import FakeOpenFOAMTools, ScriptedLLM, foam_header, make_plan, make_state
from openfoam_agent.engineering import CFDEngineeringAgent
from openfoam_agent.schemas.engineering import FoamDictionaryEntry, TypedFoamDictionaryFile
from openfoam_agent.tools.foam_file import validate_foam_file_header
from openfoam_agent.tools.foam_serializer import FoamSerializationError, serialize_foam_dictionary
from openfoam_agent.tools.workspace import CaseWorkspace
from openfoam_agent.verification.presolve import PreSolveCompletenessGate


def _typed(path: str, *pairs: tuple[str, str], foam_class: str | None = None) -> TypedFoamDictionaryFile:
    return TypedFoamDictionaryFile(
        path=path,
        foam_class=foam_class,
        entries=[FoamDictionaryEntry(path=key, value=value) for key, value in pairs],
    )


def _mesh_boundary() -> str:
    return """FoamFile
{
    version 2.0;
    format ascii;
    class polyBoundaryMesh;
    location \"constant/polyMesh\";
    object boundary;
}
1
(
    wall { type wall; }
)
"""


def _scalar_field(path: str) -> str:
    return foam_header(path, "volScalarField") + """dimensions [0 0 0 0 0 0 0];
internalField uniform 0;
boundaryField
{
    wall { type zeroGradient; }
}
"""


def test_typed_serializer_emits_canonical_dictionary_header_and_owns_object_location():
    text = serialize_foam_dictionary(
        _typed(
            "system/controlDict",
            ("solver", "incompressibleFluid"),
            ("startTime", "0"),
        )
    )
    assert text.startswith("FoamFile\n{")
    assert "class dictionary;" in text
    assert 'location "system";' in text
    assert "object controlDict;" in text
    assert "solver incompressibleFluid;" in text


def test_typed_serializer_infers_initial_field_class_from_internal_field():
    vector = serialize_foam_dictionary(
        _typed(
            "0/U",
            ("dimensions", "[0 1 -1 0 0 0 0]"),
            ("internalField", "uniform (1 0 0)"),
            ("boundaryField.wall.type", "noSlip"),
        )
    )
    scalar = serialize_foam_dictionary(
        _typed(
            "0/p",
            ("dimensions", "[0 2 -2 0 0 0 0]"),
            ("internalField", "uniform 0"),
            ("boundaryField.wall.type", "zeroGradient"),
        )
    )
    assert "class volVectorField;" in vector
    assert "object U;" in vector
    assert "class volScalarField;" in scalar
    assert "object p;" in scalar


def test_typed_serializer_accepts_matching_legacy_header_metadata_but_does_not_duplicate_it():
    text = serialize_foam_dictionary(
        _typed(
            "0/U",
            ("FoamFile.object", "U"),
            ("FoamFile.class", "volVectorField"),
            ("dimensions", "[0 1 -1 0 0 0 0]"),
            ("internalField", "uniform (0 0 0)"),
        )
    )
    assert text.count("FoamFile") == 1
    assert text.count("object U;") == 1
    assert "FoamFile.object" not in text


def test_typed_serializer_rejects_header_metadata_that_conflicts_with_path():
    with pytest.raises(FoamSerializationError, match="conflicts with path-derived object"):
        serialize_foam_dictionary(
            _typed(
                "0/U",
                ("FoamFile.object", "p"),
                ("internalField", "uniform (0 0 0)"),
            )
        )


def test_typed_serializer_requires_explicit_class_when_field_shape_is_indeterminate():
    with pytest.raises(FoamSerializationError, match="Cannot prove the FoamFile class"):
        serialize_foam_dictionary(
            _typed(
                "0/customField",
                ("dimensions", "[0 0 0 0 0 0 0]"),
                ("internalField", "$initialValue"),
            )
        )
    text = serialize_foam_dictionary(
        _typed(
            "0/customField",
            ("dimensions", "[0 0 0 0 0 0 0]"),
            ("internalField", "$initialValue"),
            foam_class="volScalarField",
        )
    )
    assert "class volScalarField;" in text


def test_header_validator_rejects_headerless_and_wrong_object_files():
    missing = validate_foam_file_header("system/fvSchemes", "ddtSchemes {}\n", expected_class="dictionary")
    assert not missing.valid
    assert any("header missing" in failure for failure in missing.failures)

    wrong = validate_foam_file_header(
        "system/fvSchemes",
        foam_header("system/fvSolution") + "ddtSchemes {}\n",
        expected_class="dictionary",
    )
    assert not wrong.valid
    assert any("object mismatch" in failure for failure in wrong.failures)


def test_presolve_blocks_headerless_file_before_native_dictionary_check(tmp_path):
    workspace = CaseWorkspace(tmp_path)
    boundary = workspace.case_dir / "constant/polyMesh/boundary"
    boundary.parent.mkdir(parents=True, exist_ok=True)
    boundary.write_text(_mesh_boundary(), encoding="utf-8")
    workspace.write_text("system/controlDict", foam_header("system/controlDict") + "solver incompressibleFluid;\n")
    workspace.write_text("system/fvSchemes", "ddtSchemes {}\n")
    workspace.write_text("system/fvSolution", foam_header("system/fvSolution") + "solvers {}\n")
    workspace.write_text("0/p", _scalar_field("0/p"))

    tools = FakeOpenFOAMTools()
    state = make_state()
    plan = make_plan(state.intake).model_copy(update={"required_case_files": ["0/p"]})
    result = PreSolveCompletenessGate(tools, workspace).validate(plan)

    assert not result.valid
    assert any("header missing in system/fvSchemes" in failure for failure in result.failures)
    assert not any(call.endswith("system/fvSchemes") for call in tools.dictionary_calls)


def test_runtime_contract_scan_reports_systematic_header_failures_together(tmp_path, graph_path):
    state = make_state()
    plan = make_plan(state.intake).model_copy(update={"required_case_files": ["0/U", "0/p"]})
    state.engineering_plan = plan
    agent = CFDEngineeringAgent(
        ScriptedLLM([]),
        workspace=tmp_path,
        capability_db=graph_path,
        tools=FakeOpenFOAMTools(),
    )
    agent.workspace.write_text("system/controlDict", "solver incompressibleFluid;\n")
    agent.workspace.write_text("system/fvSchemes", "ddtSchemes {}\n")
    agent.workspace.write_text("system/fvSolution", "solvers {}\n")
    agent.workspace.write_text("0/U", "dimensions [0 1 -1 0 0 0 0];\ninternalField uniform (0 0 0);\n")
    agent.workspace.write_text("0/p", "dimensions [0 2 -2 0 0 0 0];\ninternalField uniform 0;\n")

    scan = agent._runtime_case_file_contract_scan(state)
    assert scan["invalid_count"] == 5
    paths = {item["path"] for item in scan["invalid"]}
    assert paths == {"system/controlDict", "system/fvSchemes", "system/fvSolution", "0/U", "0/p"}
