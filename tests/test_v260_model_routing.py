from __future__ import annotations

from argparse import Namespace

import pytest

from openfoam_agent.cli import _resolve_openai_model_names, build_parser
from openfoam_agent.llm import LLMConfigurationError, WorkflowLLMs
from openfoam_agent.workflow.engine import CFDWorkflow

from conftest import FakeOpenFOAMTools, ScriptedLLM


class NamedLLM(ScriptedLLM):
    def __init__(self, name: str):
        super().__init__([])
        self.model = name


def _args(**overrides):
    values = {
        "model": None,
        "intake_model": None,
        "engineering_model": None,
        "postprocess_model": None,
        "review_model": None,
    }
    values.update(overrides)
    return Namespace(**values)


def test_workflow_llms_uniform_preserves_backward_compatibility():
    llm = NamedLLM("one-model")
    routes = WorkflowLLMs.coerce(llm)
    assert routes.intake is llm
    assert routes.engineering is llm
    assert routes.postprocessing is llm
    assert routes.review is llm


def test_workflow_routes_each_agent_and_runtime_repair_to_engineering(tmp_path, graph_path):
    intake = NamedLLM("intake-model")
    engineering = NamedLLM("engineering-model")
    postprocess = NamedLLM("postprocess-model")
    review = NamedLLM("review-model")
    routes = WorkflowLLMs(
        intake=intake,
        engineering=engineering,
        postprocessing=postprocess,
        review=review,
    )
    workflow = CFDWorkflow(
        llm=routes,
        capability_db=graph_path,
        workspace=tmp_path,
        openfoam_tools=FakeOpenFOAMTools(),
    )
    assert workflow.intake.llm is intake
    assert workflow.engineering.llm is engineering
    assert workflow.postprocess.llm is postprocess
    assert workflow.review.llm is review
    assert workflow.runtime.engineering is workflow.engineering
    assert workflow.runtime.engineering.llm is engineering


def test_role_cli_override_beats_role_env_and_global_default():
    default, resolved = _resolve_openai_model_names(
        _args(model="global", engineering_model="eng-cli"),
        environ={
            "OPENAI_MODEL": "env-global",
            "OPENAI_ENGINEERING_MODEL": "eng-env",
            "OPENAI_REVIEW_MODEL": "review-env",
        },
    )
    assert default == "global"
    assert resolved == {
        "intake": "global",
        "engineering": "eng-cli",
        "postprocessing": "global",
        "review": "review-env",
    }


def test_role_models_can_be_fully_configured_without_global_default():
    default, resolved = _resolve_openai_model_names(
        _args(
            intake_model="luna",
            engineering_model="sol",
            postprocess_model="luna",
            review_model="sol",
        ),
        environ={},
    )
    assert default is None
    assert resolved == {
        "intake": "luna",
        "engineering": "sol",
        "postprocessing": "luna",
        "review": "sol",
    }


def test_missing_role_without_global_default_is_rejected():
    with pytest.raises(LLMConfigurationError, match="postprocessing"):
        _resolve_openai_model_names(
            _args(intake_model="luna", engineering_model="sol", review_model="sol"),
            environ={},
        )


def test_cli_exposes_role_specific_model_flags():
    parser = build_parser()
    args = parser.parse_args(
        [
            "demo",
            "--backend",
            "openai",
            "--confirm-api-calls",
            "--model",
            "luna",
            "--engineering-model",
            "sol",
            "--review-model",
            "sol",
        ]
    )
    assert args.model == "luna"
    assert args.engineering_model == "sol"
    assert args.review_model == "sol"


def test_build_llm_reuses_adapter_per_unique_model(monkeypatch):
    import openfoam_agent.cli as cli

    class DummyOpenAILLM:
        created: list[str] = []

        def __init__(self, *, model: str, max_output_tokens: int | None = None):
            self.model = model
            self.max_output_tokens = max_output_tokens
            self.last_usage = None
            self.__class__.created.append(model)

    monkeypatch.setattr(cli, "OpenAILLM", DummyOpenAILLM)
    args = Namespace(
        backend="openai",
        model="luna",
        intake_model=None,
        engineering_model="sol",
        postprocess_model=None,
        review_model="sol",
        llm_max_output_tokens=16000,
    )
    llms, backend, default_model = cli._build_llm(args)
    assert backend == "openai"
    assert default_model == "luna"
    assert llms.intake.model == "luna"
    assert llms.postprocessing is llms.intake
    assert llms.engineering.model == "sol"
    assert llms.review is llms.engineering
    assert DummyOpenAILLM.created == ["luna", "sol"]
