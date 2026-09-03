from __future__ import annotations

import hashlib
import json

from conftest import FakeOpenFOAMTools, make_plan
from openfoam_agent.schemas.engineering import (
    CaseContentAssertion,
    ConfirmedFactBinding,
    NumericEvidenceTerm,
    NumericRelationAssertion,
)
from openfoam_agent.schemas.intake import CFDIntakeSpec, IntakeFact
from openfoam_agent.tools.workspace import CaseWorkspace
from openfoam_agent.verification.safety import DeterministicSafetyGate


def _intake() -> CFDIntakeSpec:
    return CFDIntakeSpec(
        semantic_contract_version="2",
        title="Compact pointer case",
        facts=[
            IntakeFact(
                id="request.summary",
                category="context",
                label="Request",
                value="2D cylinder wake at Re=1000",
                source="derived",
                reason="summary",
            ),
            IntakeFact(
                id="classification.problem_type",
                category="classification",
                label="Problem type",
                value="external_flow",
                source="derived",
                reason="agent interpretation",
            ),
            IntakeFact(
                id="temporal.behavior",
                category="temporal",
                label="Temporal",
                value="transient",
                source="derived",
                reason="agent interpretation",
            ),
            IntakeFact(
                id="property.reynolds_number",
                category="property",
                label="Reynolds number",
                value="1000",
                source="user",
                evidence="1000",
            ),
            IntakeFact(
                id="objective.vortex_shedding",
                category="objective",
                label="Objective",
                value="observe vortex shedding",
                source="user",
                evidence="vortex shedding",
            ),
        ],
        status="ready_for_review",
    )


def _bindings(intake: CFDIntakeSpec) -> list[ConfirmedFactBinding]:
    result: list[ConfirmedFactBinding] = []
    for fact in intake.facts:
        if fact.category == "context":
            continue
        if fact.id == "classification.problem_type":
            result.append(
                ConfirmedFactBinding(
                    fact_id=fact.id,
                    plan_fields=["problem_interpretation"],
                    case_assertions=[
                        CaseContentAssertion(
                            path="0/U",
                            entry_path="boundaryField.top.type",
                            expected_value="slip",
                        )
                    ],
                )
            )
        elif fact.id == "temporal.behavior":
            result.append(
                ConfirmedFactBinding(
                    fact_id=fact.id,
                    plan_fields=["temporal_behavior"],
                    case_assertions=[
                        CaseContentAssertion(
                            path="system/controlDict",
                            entry_path="adjustTimeStep",
                            expected_value="yes",
                        )
                    ],
                )
            )
        elif fact.id == "property.reynolds_number":
            result.append(
                ConfirmedFactBinding(
                    fact_id=fact.id,
                    numeric_relation=NumericRelationAssertion(
                        numerator=[
                            NumericEvidenceTerm(
                                path="0/U",
                                entry_path="boundaryField.inlet.value",
                                number_index=0,
                            ),
                            NumericEvidenceTerm(
                                path="system/blockMeshDict",
                                anchor="(0.5 0 0)",
                                number_index=0,
                                multiplier=2.0,
                            ),
                        ],
                        denominator=[
                            NumericEvidenceTerm(
                                path="constant/physicalProperties",
                                entry_path="nu",
                            )
                        ],
                    ),
                )
            )
        else:
            result.append(
                ConfirmedFactBinding(
                    fact_id=fact.id,
                    plan_fields=["problem_interpretation"],
                )
            )
    return result


def _write_case(ws: CaseWorkspace, *, nu: float = 0.001) -> None:
    ws.write_text(
        "0/U",
        """boundaryField
{
    inlet
    {
        type fixedValue;
        value uniform (1 0 0);
    }
    top
    {
        type slip;
    }
}
""",
    )
    ws.write_text(
        "system/controlDict",
        "solver incompressibleFluid;\nadjustTimeStep yes;\nendTime 50;\n",
    )
    ws.write_text("system/blockMeshDict", "vertices\n(\n    (0.5 0 0)\n);\n")
    ws.write_text("constant/physicalProperties", f"nu {nu};\n")


def test_compact_artifact_pointers_verify_current_case_without_repeated_value_tokens(tmp_path):
    intake = _intake()
    plan = make_plan(intake).model_copy(update={"confirmed_fact_bindings": _bindings(intake)})
    ws = CaseWorkspace(tmp_path)
    _write_case(ws)

    result = DeterministicSafetyGate(FakeOpenFOAMTools(), ws).validate_plan(plan, intake)

    assert result.valid, result.failures
    payload = plan.model_dump_json()
    assert '"entry_path":"boundaryField.inlet.value"' in payload
    assert '"value_token":"1"' not in payload
    assert '"excerpt":"value uniform (1 0 0);"' not in payload


def test_compact_numeric_pointer_rejects_drift_from_actual_artifact(tmp_path):
    intake = _intake()
    plan = make_plan(intake).model_copy(update={"confirmed_fact_bindings": _bindings(intake)})
    ws = CaseWorkspace(tmp_path)
    _write_case(ws, nu=0.01)

    result = DeterministicSafetyGate(FakeOpenFOAMTools(), ws).validate_plan(plan, intake)

    assert not result.valid
    assert any("recomputes to 100.0" in failure for failure in result.failures)


def test_semantic_evidence_pointer_path_counts_as_case_implementation_ref():
    binding = ConfirmedFactBinding(
        fact_id="temporal.behavior",
        case_assertions=[
            CaseContentAssertion(
                path="system/controlDict",
                entry_path="adjustTimeStep",
                expected_value="yes",
            )
        ],
    )
    assert binding.case_files == []
    assert binding.case_assertions[0].path == "system/controlDict"


def test_compact_numeric_encoding_is_smaller_than_v215_excerpt_encoding():
    legacy = NumericRelationAssertion(
        numerator=[
            NumericEvidenceTerm(
                path="0/U",
                excerpt="boundaryField inlet value uniform (1 0 0);",
                value_token="1",
            ),
            NumericEvidenceTerm(
                path="system/blockMeshDict",
                excerpt="arc point defining cylinder radius (0.5 0 0)",
                value_token="0.5",
                multiplier=2.0,
            ),
        ],
        denominator=[
            NumericEvidenceTerm(
                path="constant/physicalProperties",
                excerpt="transport model kinematic viscosity nu 0.001;",
                value_token="0.001",
            )
        ],
    )
    compact = NumericRelationAssertion(
        numerator=[
            NumericEvidenceTerm(path="0/U", entry_path="boundaryField.inlet.value"),
            NumericEvidenceTerm(path="system/blockMeshDict", anchor="(0.5 0 0)", multiplier=2.0),
        ],
        denominator=[
            NumericEvidenceTerm(path="constant/physicalProperties", entry_path="nu")
        ],
    )
    legacy_payload = legacy.model_dump_json(exclude_defaults=True)
    compact_payload = compact.model_dump_json(exclude_defaults=True)
    assert len(compact_payload) < len(legacy_payload) * 0.75


def test_v215_semantic_plan_digest_ignores_empty_v216_pointer_defaults():
    intake = _intake()
    legacy_binding = ConfirmedFactBinding(
        fact_id="classification.problem_type",
        case_files=["0/U"],
        plan_fields=["problem_interpretation"],
        case_assertions=[CaseContentAssertion(path="0/U", contains=["farField", "cylinder"])],
        explanation="legacy explanation",
    )
    bindings = []
    for fact in intake.facts:
        if fact.category == "context":
            continue
        if fact.id == legacy_binding.fact_id:
            bindings.append(legacy_binding)
        else:
            bindings.append(
                ConfirmedFactBinding(
                    fact_id=fact.id,
                    plan_fields=["problem_interpretation"],
                    explanation="legacy explanation",
                )
            )
    plan = make_plan(intake).model_copy(update={"confirmed_fact_bindings": bindings})
    legacy_payload = plan.model_dump(mode="json")
    for binding in legacy_payload["confirmed_fact_bindings"]:
        for assertion in binding.get("case_assertions", []):
            assertion.pop("entry_path", None)
            assertion.pop("expected_value", None)
            assertion.pop("anchor", None)
        relation = binding.get("numeric_relation")
        if relation:
            for term in [*relation.get("numerator", []), *relation.get("denominator", [])]:
                term.pop("entry_path", None)
                term.pop("anchor", None)
                term.pop("number_index", None)
                term.pop("occurrence", None)
        if not binding.get("case_assertions"):
            binding.pop("case_assertions", None)
        if binding.get("numeric_relation") is None:
            binding.pop("numeric_relation", None)
    expected = hashlib.sha256(
        json.dumps(legacy_payload, ensure_ascii=True, separators=(",", ":"), sort_keys=True).encode()
    ).hexdigest()
    assert plan.digest() == expected


def test_llm_schema_hides_legacy_excerpt_fields_and_redundant_binding_prose():
    binding_schema = ConfirmedFactBinding.model_json_schema()
    assertion_schema = CaseContentAssertion.model_json_schema()
    term_schema = NumericEvidenceTerm.model_json_schema()
    assert "case_files" not in binding_schema["properties"]
    assert "explanation" not in binding_schema["properties"]
    assert "contains" not in assertion_schema["properties"]
    assert "excerpt" not in term_schema["properties"]
    assert "value_token" not in term_schema["properties"]


def test_multiple_semantic_pointers_may_reference_the_same_case_file():
    binding = ConfirmedFactBinding(
        fact_id="classification.problem_type",
        case_assertions=[
            CaseContentAssertion(path="0/U", entry_path="boundaryField.top.type", expected_value="slip"),
            CaseContentAssertion(path="0/U", entry_path="boundaryField.bottom.type", expected_value="slip"),
        ],
    )
    assert len(binding.case_assertions) == 2
