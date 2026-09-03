from __future__ import annotations

from dataclasses import dataclass

from conftest import FakeOpenFOAMTools, make_plan
from openfoam_agent.agents.intake import IntakeAgent, validate_intake_provenance
from openfoam_agent.schemas.engineering import (
    CaseContentAssertion,
    ConfirmedFactBinding,
    NumericEvidenceTerm,
    NumericRelationAssertion,
)
from openfoam_agent.schemas.intake import CFDIntakeSpec, IntakeFact
from openfoam_agent.schemas.request import UserRequest
from openfoam_agent.tools.workspace import CaseWorkspace
from openfoam_agent.verification.safety import DeterministicSafetyGate
from openfoam_agent.workflow.state import CFDState
from openfoam_agent.workflow.states import State


def _misattributed_intake() -> CFDIntakeSpec:
    return CFDIntakeSpec(
        title="Cylinder vortex shedding",
        facts=[
            IntakeFact(
                id="request.summary",
                category="context",
                label="Request",
                value="Cylinder vortex shedding in a rectangular computational domain",
                source="derived",
                reason="Normalized request summary.",
            ),
            IntakeFact(
                id="classification.problem_type",
                category="classification",
                label="Problem type",
                value="internal_flow",
                source="user",
                evidence="직사각형 geometry안에 있는 cylinder",
            ),
            IntakeFact(
                id="geometry.type",
                category="geometry",
                label="Geometry",
                value="rectangular domain with cylinder",
                source="user",
                evidence="직사각형 geometry안에 있는 cylinder",
            ),
            IntakeFact(
                id="physics.reynolds_number",
                category="physics",
                label="Reynolds number",
                value="1000",
                unit="dimensionless",
                source="user",
                evidence="1000",
            ),
            IntakeFact(
                id="temporal.behavior",
                category="temporal",
                label="Temporal behavior",
                value="transient/unsteady",
                source="user",
                evidence="Vortex shedding",
            ),
            IntakeFact(
                id="objective.vortex_shedding",
                category="objective",
                label="Objective",
                value="Observe vortex shedding",
                source="user",
                evidence="Vortex shedding",
            ),
        ],
        status="ready_for_review",
    )


def _request() -> UserRequest:
    return UserRequest(
        prompt="reynolds number가 1000정도인 직사각형 geometry안에 있는 cylinder 장애물 Vortex shedding 2d simulation보고싶어",
        exploratory_completion_authorized=True,
    )


def test_review_critical_user_misattribution_is_demoted_before_confirmation():
    spec = _misattributed_intake()
    validate_intake_provenance(spec, _request())

    classification = spec.fact("classification.problem_type")
    temporal = spec.fact("temporal.behavior")
    assert classification is not None and classification.source == "derived"
    assert temporal is not None and temporal.source == "derived"
    assert classification.evidence is None
    assert temporal.evidence is None
    assert "interpretation" in (classification.reason or "").casefold()
    assert "geometry.type" in classification.depends_on
    # Direct user facts stay direct.
    assert spec.fact("physics.reynolds_number").source == "user"


@dataclass
class _OneShotIntakeLLM:
    output: CFDIntakeSpec

    def generate(self, schema, prompt, *, system_prompt=None):
        del prompt, system_prompt
        assert schema is CFDIntakeSpec
        return self.output.model_copy(deep=True)


def test_new_intake_agent_runs_upgrade_to_semantic_contract_v2():
    llm = _OneShotIntakeLLM(_misattributed_intake())
    state = CFDState(run_id="semantic-contract", user_request=_request())

    result = IntakeAgent(llm).run(state)

    assert result.current_state == State.INTAKE_REVIEW_REQUIRED
    assert result.intake is not None
    assert result.intake.semantic_contract_version == "2"
    assert result.intake.fact("classification.problem_type").source == "derived"


def _semantic_intake() -> CFDIntakeSpec:
    return CFDIntakeSpec(
        semantic_contract_version="2",
        title="External cylinder wake",
        facts=[
            IntakeFact(
                id="request.summary",
                category="context",
                label="Request",
                value="2D cylinder wake at Re=1000",
                source="derived",
                reason="Normalized request.",
            ),
            IntakeFact(
                id="classification.problem_type",
                category="classification",
                label="Problem type",
                value="external_flow",
                source="derived",
                reason="Agent interpretation of flow around an isolated obstacle.",
                depends_on=["geometry.type"],
            ),
            IntakeFact(
                id="geometry.type",
                category="geometry",
                label="Geometry",
                value="cylinder in rectangular domain",
                source="user",
                evidence="cylinder",
            ),
            IntakeFact(
                id="physics.reynolds_number",
                category="physics",
                label="Reynolds number",
                value="1000",
                unit="dimensionless",
                source="user",
                evidence="1000",
            ),
            IntakeFact(
                id="temporal.behavior",
                category="temporal",
                label="Temporal behavior",
                value="transient",
                source="derived",
                reason="Vortex shedding requires time-resolved evolution.",
                depends_on=["objective.vortex_shedding"],
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


def _semantic_bindings(intake: CFDIntakeSpec, *, nu: float = 0.001) -> list[ConfirmedFactBinding]:
    bindings: list[ConfirmedFactBinding] = []
    for fact in intake.facts:
        if fact.category == "context":
            continue
        if fact.id == "classification.problem_type":
            bindings.append(
                ConfirmedFactBinding(
                    fact_id=fact.id,
                    case_files=["0/U"],
                    plan_fields=["problem_interpretation"],
                    case_assertions=[
                        CaseContentAssertion(path="0/U", contains=["farField", "cylinder"])
                    ],
                    explanation="The Agent claims the far-field/cylinder patch setup implements the external-flow interpretation.",
                )
            )
        elif fact.id == "temporal.behavior":
            bindings.append(
                ConfirmedFactBinding(
                    fact_id=fact.id,
                    case_files=["system/controlDict"],
                    plan_fields=["temporal_behavior"],
                    case_assertions=[
                        CaseContentAssertion(
                            path="system/controlDict",
                            contains=["endTime 50;", "deltaT 0.01;"],
                        )
                    ],
                    explanation="The time-control dictionary carries the Agent's transient implementation claim.",
                )
            )
        elif fact.id == "physics.reynolds_number":
            bindings.append(
                ConfirmedFactBinding(
                    fact_id=fact.id,
                    case_files=["0/U", "system/blockMeshDict", "constant/physicalProperties"],
                    plan_fields=["problem_interpretation", "assumptions"],
                    numeric_relation=NumericRelationAssertion(
                        numerator=[
                            NumericEvidenceTerm(
                                path="0/U",
                                excerpt="value uniform (1 0 0);",
                                value_token="1",
                            ),
                            NumericEvidenceTerm(
                                path="system/blockMeshDict",
                                excerpt="(0.5 0 0)",
                                value_token="0.5",
                                multiplier=2.0,
                            ),
                        ],
                        denominator=[
                            NumericEvidenceTerm(
                                path="constant/physicalProperties",
                                excerpt=f"nu {nu};",
                                value_token=str(nu),
                            )
                        ],
                    ),
                    explanation="The Agent selected U, D and nu and carries the arithmetic relation that implements Re=1000.",
                )
            )
        else:
            bindings.append(
                ConfirmedFactBinding(
                    fact_id=fact.id,
                    plan_fields=["problem_interpretation"],
                    explanation="The confirmed fact is represented in the engineering interpretation.",
                )
            )
    return bindings


def _write_semantic_case(ws: CaseWorkspace, *, nu: float = 0.001) -> None:
    ws.write_text(
        "system/controlDict",
        "solver incompressibleFluid;\nstartTime 0;\nendTime 50;\ndeltaT 0.01;\n",
    )
    ws.write_text("0/U", "value uniform (1 0 0);\nfarField\ncylinder\n")
    ws.write_text("system/blockMeshDict", "vertices ((0.5 0 0));\n")
    ws.write_text("constant/physicalProperties", f"nu {nu};\n")


def test_semantic_contract_v2_accepts_case_assertions_and_recomputes_numeric_relation(tmp_path):
    intake = _semantic_intake()
    plan = make_plan(intake).model_copy(
        update={"confirmed_fact_bindings": _semantic_bindings(intake)}
    )
    ws = CaseWorkspace(tmp_path)
    _write_semantic_case(ws)

    result = DeterministicSafetyGate(FakeOpenFOAMTools(), ws).validate_plan(plan, intake)

    assert result.valid, result.failures


def test_semantic_contract_v2_rejects_numeric_drift_even_when_fact_id_and_files_exist(tmp_path):
    intake = _semantic_intake()
    plan = make_plan(intake).model_copy(
        update={"confirmed_fact_bindings": _semantic_bindings(intake, nu=0.01)}
    )
    ws = CaseWorkspace(tmp_path)
    _write_semantic_case(ws, nu=0.01)

    result = DeterministicSafetyGate(FakeOpenFOAMTools(), ws).validate_plan(plan, intake)

    assert not result.valid
    assert any("recomputes to 100.0" in failure for failure in result.failures)


def test_semantic_contract_v2_rejects_stale_case_snippet_assertion(tmp_path):
    intake = _semantic_intake()
    bindings = _semantic_bindings(intake)
    classification_index = next(
        index for index, item in enumerate(bindings) if item.fact_id == "classification.problem_type"
    )
    bindings[classification_index] = bindings[classification_index].model_copy(
        update={
            "case_assertions": [
                CaseContentAssertion(path="0/U", contains=["topAndBottom slip"])
            ]
        }
    )
    plan = make_plan(intake).model_copy(update={"confirmed_fact_bindings": bindings})
    ws = CaseWorkspace(tmp_path)
    _write_semantic_case(ws)

    result = DeterministicSafetyGate(FakeOpenFOAMTools(), ws).validate_plan(plan, intake)

    assert not result.valid
    assert any("Semantic assertion for classification.problem_type is not present" in failure for failure in result.failures)


def test_legacy_v1_intake_digest_ignores_new_contract_default_for_rehydration():
    import hashlib
    import json

    spec = _semantic_intake().model_copy(update={"semantic_contract_version": "1"})
    legacy_payload = spec.model_dump(mode="json")
    legacy_payload.pop("semantic_contract_version", None)
    expected = hashlib.sha256(
        json.dumps(
            legacy_payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()

    assert spec.digest() == expected


def test_legacy_plan_digest_ignores_empty_v215_assertion_defaults():
    import hashlib
    import json

    intake = _semantic_intake().model_copy(update={"semantic_contract_version": "1"})
    plan = make_plan(intake)
    legacy_payload = plan.model_dump(mode="json")
    for binding in legacy_payload["confirmed_fact_bindings"]:
        binding.pop("case_assertions", None)
        binding.pop("numeric_relation", None)
    expected = hashlib.sha256(
        json.dumps(
            legacy_payload,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()

    assert plan.digest() == expected


def test_explicit_external_and_transient_language_remains_direct_user_provenance():
    request = UserRequest(
        prompt="이건 외부유동 external flow이고 transient로 계산해. Re=1000 vortex shedding을 본다."
    )
    spec = CFDIntakeSpec(
        title="Explicit external transient case",
        facts=[
            IntakeFact(
                id="request.summary",
                category="context",
                label="Request",
                value=request.prompt,
                source="derived",
                reason="Normalized request.",
            ),
            IntakeFact(
                id="classification.problem_type",
                category="classification",
                label="Problem type",
                value="external_flow",
                source="user",
                evidence="external flow",
            ),
            IntakeFact(
                id="temporal.behavior",
                category="temporal",
                label="Temporal behavior",
                value="transient",
                source="user",
                evidence="transient",
            ),
            IntakeFact(
                id="physics.reynolds_number",
                category="physics",
                label="Reynolds number",
                value="1000",
                source="user",
                evidence="Re=1000",
            ),
            IntakeFact(
                id="objective.primary",
                category="objective",
                label="Objective",
                value="vortex shedding",
                source="user",
                evidence="vortex shedding",
            ),
        ],
        status="ready_for_review",
    )

    validate_intake_provenance(spec, request)

    assert spec.fact("classification.problem_type").source == "user"
    assert spec.fact("temporal.behavior").source == "user"


def test_incomplete_semantic_assertion_shape_becomes_safety_failure_not_schema_crash(tmp_path):
    intake = _semantic_intake()
    bindings = _semantic_bindings(intake)
    re_index = next(
        index for index, item in enumerate(bindings) if item.fact_id == "physics.reynolds_number"
    )
    bindings[re_index] = bindings[re_index].model_copy(
        update={"numeric_relation": NumericRelationAssertion(numerator=[], denominator=[])}
    )
    plan = make_plan(intake).model_copy(update={"confirmed_fact_bindings": bindings})
    ws = CaseWorkspace(tmp_path)
    _write_semantic_case(ws)

    result = DeterministicSafetyGate(FakeOpenFOAMTools(), ws).validate_plan(plan, intake)

    assert not result.valid
    assert any("requires at least one numerator term" in failure for failure in result.failures)


def test_semantic_assertions_do_not_accept_comment_only_self_claims(tmp_path):
    intake = _semantic_intake()
    bindings = _semantic_bindings(intake)
    classification_index = next(
        index for index, item in enumerate(bindings) if item.fact_id == "classification.problem_type"
    )
    bindings[classification_index] = bindings[classification_index].model_copy(
        update={
            "case_assertions": [
                CaseContentAssertion(path="0/U", contains=["claimedExternalFlow"])
            ]
        }
    )
    plan = make_plan(intake).model_copy(update={"confirmed_fact_bindings": bindings})
    ws = CaseWorkspace(tmp_path)
    _write_semantic_case(ws)
    ws.write_text("0/U", "value uniform (1 0 0);\nfarField\ncylinder\n// claimedExternalFlow\n")

    result = DeterministicSafetyGate(FakeOpenFOAMTools(), ws).validate_plan(plan, intake)

    assert not result.valid
    assert any("Semantic assertion for classification.problem_type is not present" in failure for failure in result.failures)
