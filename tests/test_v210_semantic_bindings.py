from openfoam_agent.schemas.engineering import ConfirmedFactBinding


def test_confirmed_fact_binding_uses_structured_case_and_plan_fields():
    binding = ConfirmedFactBinding(
        fact_id="physics.reynolds_number",
        case_files=["constant/physicalProperties", "0/U"],
        plan_fields=["problem_interpretation"],
        explanation="Reynolds number is implemented by velocity, transport properties and the plan interpretation.",
    )
    assert binding.case_files == ["constant/physicalProperties", "0/U"]
    assert binding.plan_fields == ["problem_interpretation"]
    assert binding.implementation_refs == [
        "case:constant/physicalProperties",
        "case:0/U",
        "plan:problem_interpretation",
    ]


def test_confirmed_fact_binding_accepts_legacy_prefixed_refs_when_loading_state():
    binding = ConfirmedFactBinding.model_validate({
        "fact_id": "physics.reynolds_number",
        "implementation_refs": ["case:0/U", "plan:problem_interpretation"],
        "explanation": "legacy state",
    })
    assert binding.case_files == ["0/U"]
    assert binding.plan_fields == ["problem_interpretation"]
