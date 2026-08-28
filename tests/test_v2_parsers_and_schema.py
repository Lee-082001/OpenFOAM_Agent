from __future__ import annotations

from openfoam_agent.llm.openai_client import validate_structured_output_schema
from openfoam_agent.schemas.engineering import (
    BlockAction,
    EngineeringTurn,
    InspectEnvironmentAction,
    SearchCapabilitiesAction,
)
from openfoam_agent.tools.parsers import parse_runtime_log


def test_engineering_turn_is_strict_structured_output_compatible():
    validate_structured_output_schema(EngineeringTurn)
    schema = EngineeringTurn.model_json_schema()
    action_schema = schema["properties"]["action"]
    assert "anyOf" in action_schema
    assert "oneOf" not in action_schema
    assert "discriminator" not in action_schema


def test_engineering_turn_plain_union_dispatches_by_literal_type():
    inspect = EngineeringTurn(
        action=InspectEnvironmentAction(
            type="inspect_environment", rationale="Inspect the installed OpenFOAM environment."
        )
    )
    search = EngineeringTurn.model_validate(
        {
            "action": {
                "type": "search_capabilities",
                "query": "transient incompressible flow",
                "rationale": "Find solver capability evidence.",
            }
        }
    )
    blocked = EngineeringTurn.model_validate(
        {
            "action": {
                "type": "block",
                "reason": "Need geometry clarification.",
                "needs_user_input": True,
                "rationale": "Cannot safely proceed without this fact.",
            }
        }
    )
    assert isinstance(inspect.action, InspectEnvironmentAction)
    assert isinstance(search.action, SearchCapabilitiesAction)
    assert isinstance(blocked.action, BlockAction)


def test_runtime_parser_accepts_clean_completed_log():
    result = parse_runtime_log(
        "Time = 1\nCourant Number mean: 0.1 max: 0.3\nEnd\n",
        return_code=0,
    )
    assert result.success
    assert result.last_time == 1
    assert result.courant_max == 0.3


def test_runtime_parser_accepts_openfoam13_seconds_suffix():
    result = parse_runtime_log(
        (
            "Courant Number mean: 0.02095074 max: 0.034800772\n"
            "Time = 20s\n"
            "smoothSolver:  Solving for Ux, Initial residual = 1.9344907e-05, "
            "Final residual = 2.2917407e-10, No Iterations 2\n"
            "GAMG:  Solving for p, Initial residual = 5.7748509e-05, "
            "Final residual = 5.337679e-06, No Iterations 2\n"
            "End\n"
        ),
        return_code=0,
    )
    assert result.success
    assert result.last_time == 20.0
    assert result.courant_max == 0.034800772
    assert not result.evidence_failures


def test_runtime_parser_accepts_seconds_suffix_with_whitespace_and_exponent():
    result = parse_runtime_log(
        "Time = 2.5e-03 s\nCourant Number mean: 0.01 max: 0.02\nEnd\n",
        return_code=0,
    )
    assert result.success
    assert result.last_time == 2.5e-03


def test_runtime_parser_rejects_nonfinite_or_fatal_log():
    result = parse_runtime_log(
        "Time = 1\n--> FOAM FATAL ERROR: bad thing\nnan\n",
        return_code=1,
    )
    assert not result.success
    assert result.fatal_error is not None
    assert result.non_finite_detected


def test_preflight_rejects_discriminated_union_one_of():
    from typing import Annotated, Literal

    import pytest
    from pydantic import BaseModel, ConfigDict, Field

    from openfoam_agent.llm.openai_client import StructuredOutputSchemaError

    class A(BaseModel):
        model_config = ConfigDict(extra="forbid")
        type: Literal["a"]
        value: str

    class B(BaseModel):
        model_config = ConfigDict(extra="forbid")
        type: Literal["b"]
        value: str

    class BadTurn(BaseModel):
        model_config = ConfigDict(extra="forbid")
        action: Annotated[A | B, Field(discriminator="type")]

    with pytest.raises(StructuredOutputSchemaError, match="oneOf"):
        validate_structured_output_schema(BadTurn)
