from __future__ import annotations

from openfoam_agent.engineering import CFDEngineeringAgent, EngineeringPolicy
from openfoam_agent.llm.context import (
    build_bounded_json_prompt,
    compact_runtime_result,
    structured_request_metrics,
)
from openfoam_agent.postprocessing.agent import CFDPostProcessingAgent
from openfoam_agent.postprocessing import PostProcessingPolicy
from openfoam_agent.review.agent import CFDFeedbackReviewAgent
from openfoam_agent.schemas.engineering import EngineeringEvent, InspectEnvironmentAction
from openfoam_agent.schemas.feedback import FeedbackAssessment
from openfoam_agent.schemas.postprocessing import FinishPostProcessingAction, PostProcessingTurn
from openfoam_agent.schemas.simulation import ResidualSample, RuntimeReport, SimulationAttempt, SimulationResult
from openfoam_agent.tools.workspace import CaseWorkspace
from openfoam_agent.workflow.states import State

from conftest import ScriptedLLM, control_dict, make_plan, make_state


def _large_runtime_result(sample_count: int = 12_000) -> SimulationResult:
    fields = ("Ux", "Uy", "p")
    residuals = [
        ResidualSample(
            time=index * 0.001,
            field=fields[index % len(fields)],
            initial_residual=1.0 / (index + 1),
            final_residual=1.0 / (index + 2),
        )
        for index in range(sample_count)
    ]
    return SimulationResult(
        success=True,
        completed=True,
        return_code=0,
        last_time=12.0,
        residuals=residuals,
        courant_max=0.04,
        continuity_error=1e-10,
        end_marker_found=True,
        log_sha256="a" * 64,
    )


def _runtime_report(sample_count: int = 12_000) -> RuntimeReport:
    result = _large_runtime_result(sample_count)
    return RuntimeReport(
        success=True,
        attempts=[SimulationAttempt(attempt=1, result=result)],
        final_result=result,
    )


def test_runtime_compaction_replaces_full_residual_history():
    compacted = compact_runtime_result(_large_runtime_result())
    summary = compacted["residual_summary"]
    assert summary["total_samples"] == 12_000
    assert summary["field_count"] == 3
    assert len(summary["latest_by_field"]) == 3
    assert len(summary["recent_samples"]) == 8
    assert "residuals" not in compacted


def test_engineering_prompt_stays_bounded_with_large_recent_observations(tmp_path, graph_path):
    state = make_state()
    for index in range(20):
        state.engineering_events.append(
            EngineeringEvent(
                step=index + 1,
                action_type="read_reference",
                success=True,
                summary=f"large reference observation {index}",
                output_excerpt=(f"reference-{index}-" + "x" * 11_500),
            )
        )
    llm = ScriptedLLM([
        InspectEnvironmentAction(type="inspect_environment", rationale="Inspect environment."),
    ])
    agent = CFDEngineeringAgent(
        llm,
        workspace=tmp_path,
        capability_db=graph_path,
        policy=EngineeringPolicy(max_model_prompt_chars=60_000),
    )

    agent._generate_turn(state, step=21, phase="prepare", native_execution=False)

    assert len(llm.prompts[-1]) < 60_000
    assert "[model-context compacted]" in llm.prompts[-1]


def test_postprocessing_prompt_does_not_transmit_full_runtime_residuals(tmp_path):
    state = make_state()
    state.engineering_plan = make_plan(state.intake)
    state.runtime_report = _runtime_report()
    state.simulation = state.runtime_report.final_result

    # Make the result inventory intentionally larger than the model-facing cap.
    case_dir = tmp_path / "case"
    for index in range(180):
        path = case_dir / "postProcessing" / "probe" / f"{index:04d}.dat"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("0 0\n", encoding="utf-8")

    llm = ScriptedLLM([
        FinishPostProcessingAction(
            type="finish_postprocessing",
            summary="bounded context test",
            scientific_confidence="unknown",
            rationale="Finish test turn.",
        )
    ])
    agent = CFDPostProcessingAgent(
        llm,
        workspace=tmp_path,
        policy=PostProcessingPolicy(
            max_model_prompt_chars=40_000,
            max_model_result_inventory=80,
        ),
    )

    agent._generate_turn(state, step=1)
    prompt = llm.prompts[-1]

    assert len(prompt) < 40_000
    assert '"total_samples": 12000' in prompt
    assert '"total_files": 180' in prompt
    assert '"shown_files": 80' in prompt
    assert '"residuals": [' not in prompt


def test_feedback_review_compacts_large_runtime_report(tmp_path):
    state = make_state()
    plan = make_plan(state.intake)
    workspace = CaseWorkspace(tmp_path)
    workspace.write_text("system/controlDict", control_dict())
    state.engineering_plan = plan
    state.case_seal = workspace.seal(plan)
    state.case_dir = str(workspace.case_dir)
    state.runtime_report = _runtime_report()
    state.simulation = state.runtime_report.final_result
    state.current_state = State.RESULT_REVIEW_REQUIRED

    assessment = FeedbackAssessment(
        diagnosis_summary="Need additional evidence.",
        hypotheses=[],
        proposed_changes=[],
        expected_cost="unknown",
        requires_case_revision=False,
        review_limitations=["No case change proposed."],
    )
    llm = ScriptedLLM([assessment])
    reviewer = CFDFeedbackReviewAgent(llm)

    reviewer.review(state, "와류가 이상해 보인다")

    prompt = llm.prompts[-1]
    assert len(prompt) < 40_000
    assert '"attempt_count": 1' in prompt
    assert '"total_samples": 12000' in prompt
    assert '"residuals": [' not in prompt


def test_bounded_prompt_has_deterministic_hard_cap_and_request_estimate():
    result = build_bounded_json_prompt(
        "instruction\n",
        {"large": "x" * 20_000, "items": list(range(200))},
        max_chars=8_000,
    )
    metrics = structured_request_metrics(PostProcessingTurn, result.prompt)

    assert len(result.prompt) <= 8_000
    assert result.compacted is True
    assert metrics["promptChars"] == len(result.prompt)
    assert metrics["approxTokens"] > 0


def test_openai_adapter_sends_explicit_output_token_cap():
    from pydantic import BaseModel, ConfigDict

    from openfoam_agent.llm.openai_client import OpenAILLM

    class TinyResponse(BaseModel):
        model_config = ConfigDict(extra="forbid")
        value: str

    class FakeResponses:
        def __init__(self):
            self.request = None

        def parse(self, **request):
            self.request = request
            usage = type(
                "Usage",
                (),
                {"input_tokens": 321, "output_tokens": 45, "total_tokens": 366},
            )()
            return type(
                "Response",
                (),
                {"output_parsed": TinyResponse(value="ok"), "usage": usage},
            )()

    class FakeClient:
        def __init__(self):
            self.responses = FakeResponses()

    client = FakeClient()
    llm = OpenAILLM(model="test-model", client=client, max_output_tokens=16_000)
    result = llm.generate(TinyResponse, "return ok")

    assert result.value == "ok"
    assert client.responses.request["max_output_tokens"] == 16_000
    assert llm.last_usage == {
        "inputTokens": 321,
        "outputTokens": 45,
        "totalTokens": 366,
    }


def test_cli_defaults_to_bounded_llm_output_tokens():
    from openfoam_agent.cli import build_parser

    args = build_parser().parse_args(["--interactive", "--backend", "openai", "--confirm-api-calls"])
    assert args.llm_max_output_tokens == 16_000
