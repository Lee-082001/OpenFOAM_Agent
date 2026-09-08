from __future__ import annotations

import json
import socket
import threading
from argparse import Namespace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import pytest
from pydantic import BaseModel

from openfoam_agent.schemas.engineering import EngineeringTurn

from openfoam_agent.cli import (
    _build_llm,
    _resolve_ollama_model_names,
    _validate_args,
    build_parser,
)
from openfoam_agent.llm import (
    DEFAULT_OLLAMA_BASE_URL,
    DEFAULT_OLLAMA_MODEL,
    LLMConfigurationError,
    OllamaConnectionError,
    OllamaLLM,
    check_ollama_health,
    normalize_ollama_base_url,
)


class DemoOutput(BaseModel):
    action: str
    reason: str


class _ModelsHandler(BaseHTTPRequestHandler):
    models = [DEFAULT_OLLAMA_MODEL]

    def do_GET(self):  # noqa: N802 - stdlib handler name
        if self.path == "/v1/models":
            body = json.dumps({"object": "list", "data": [{"id": name} for name in self.models]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        self.send_response(404)
        self.end_headers()

    def log_message(self, format, *args):  # noqa: A002 - stdlib signature
        return


@pytest.fixture
def mock_ollama_http():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _ModelsHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address
        yield f"http://{host}:{port}/v1"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def _args(**overrides):
    values = {
        "backend": "ollama",
        "model": None,
        "intake_model": None,
        "engineering_model": None,
        "postprocess_model": None,
        "review_model": None,
        "base_url": None,
        "ollama_health_timeout": 1.0,
        "llm_max_output_tokens": 16000,
    }
    values.update(overrides)
    return Namespace(**values)


def test_ollama_defaults_and_role_env_routing():
    default, resolved = _resolve_ollama_model_names(
        _args(engineering_model="eng-cli"),
        environ={"OLLAMA_REVIEW_MODEL": "review-env"},
    )
    assert default == DEFAULT_OLLAMA_MODEL
    assert resolved == {
        "intake": DEFAULT_OLLAMA_MODEL,
        "engineering": "eng-cli",
        "postprocessing": DEFAULT_OLLAMA_MODEL,
        "review": "review-env",
    }


def test_ollama_health_check_uses_openai_compatible_models_endpoint(mock_ollama_http):
    available = check_ollama_health(
        base_url=mock_ollama_http,
        models=[DEFAULT_OLLAMA_MODEL],
        timeout=1.0,
    )
    assert DEFAULT_OLLAMA_MODEL in available


def test_ollama_health_check_reports_missing_model_clearly(mock_ollama_http):
    with pytest.raises(LLMConfigurationError, match="were not found"):
        check_ollama_health(
            base_url=mock_ollama_http,
            models=["missing:31b"],
            timeout=1.0,
        )


def test_ollama_health_check_reports_missing_tunnel_clearly():
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    with pytest.raises(LLMConfigurationError) as exc:
        check_ollama_health(
            base_url=f"http://127.0.0.1:{port}/v1",
            models=[DEFAULT_OLLAMA_MODEL],
            timeout=0.2,
        )
    text = str(exc.value)
    assert "Cannot connect to Ollama" in text
    assert "SSH tunnel to mlfm4.knu.ac.kr" in text


def test_ollama_rejects_direct_remote_base_url():
    with pytest.raises(LLMConfigurationError, match="loopback"):
        normalize_ollama_base_url("http://mlfm4.knu.ac.kr:11434/v1")
    with pytest.raises(LLMConfigurationError, match="loopback"):
        normalize_ollama_base_url("http://0.0.0.0:11434/v1")


def test_ollama_structured_adapter_uses_json_mode_then_python_validation():
    captured = {}

    class FakeCompletions:
        def create(self, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(
                            content=json.dumps({"action": "run", "reason": "ok"}),
                            refusal=None,
                        )
                    )
                ],
                usage=SimpleNamespace(prompt_tokens=123, completion_tokens=17, total_tokens=140),
            )

        def parse(self, **kwargs):
            raise AssertionError("Ollama must not compile Pydantic schemas as grammars")

    fake_client = SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions()))
    llm = OllamaLLM(
        model=DEFAULT_OLLAMA_MODEL,
        base_url=DEFAULT_OLLAMA_BASE_URL,
        client=fake_client,
        max_output_tokens=2048,
    )
    result = llm.generate(DemoOutput, "Return one action.", system_prompt="Be concise.")
    assert result.action == "run"
    assert captured["model"] == DEFAULT_OLLAMA_MODEL
    assert captured["response_format"] == {"type": "json_object"}
    assert captured["temperature"] == 0
    assert captured["max_tokens"] == 2048
    assert captured["messages"][0] == {"role": "system", "content": "Be concise."}
    assert "Return one action." in captured["messages"][1]["content"]
    assert "OUTPUT CONTRACT FOR LOCAL JSON MODE" in captured["messages"][1]["content"]
    assert '"action"' in captured["messages"][1]["content"]
    assert llm.last_usage == {
        "inputTokens": 123,
        "outputTokens": 17,
        "totalTokens": 140,
    }


def test_ollama_repairs_invalid_json_with_pydantic_errors_and_sums_usage():
    calls = []
    payloads = [
        {"action": "run"},  # missing reason
        {"action": "run", "reason": "repaired"},
    ]

    class FakeCompletions:
        def create(self, **kwargs):
            calls.append(kwargs)
            payload = payloads[len(calls) - 1]
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(content=json.dumps(payload), refusal=None)
                    )
                ],
                usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=15),
            )

    fake_client = SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions()))
    llm = OllamaLLM(model=DEFAULT_OLLAMA_MODEL, client=fake_client)

    result = llm.generate(DemoOutput, "Return one action.")

    assert result == DemoOutput(action="run", reason="repaired")
    assert len(calls) == 2
    repair_messages = calls[1]["messages"]
    assert repair_messages[-2] == {"role": "assistant", "content": '{"action": "run"}'}
    assert "VALIDATION ERROR" in repair_messages[-1]["content"]
    assert "reason" in repair_messages[-1]["content"]
    assert llm.last_usage == {
        "inputTokens": 20,
        "outputTokens": 10,
        "totalTokens": 30,
    }


def test_ollama_structured_repair_stops_after_three_total_attempts():
    calls = []

    class FakeCompletions:
        def create(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(content='{"action":"run"}', refusal=None)
                    )
                ],
                usage=None,
            )

    fake_client = SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions()))
    llm = OllamaLLM(
        model=DEFAULT_OLLAMA_MODEL,
        client=fake_client,
        structured_repair_attempts=2,
    )

    from openfoam_agent.llm import StructuredOutputError

    with pytest.raises(StructuredOutputError, match=r"after 3 attempt\(s\)"):
        llm.generate(DemoOutput, "Return one action.")
    assert len(calls) == 3


def test_ollama_engineering_turn_avoids_complex_schema_grammar():
    captured = {}

    class FakeCompletions:
        def create(self, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(
                            content=json.dumps(
                                {
                                    "action": {
                                        "type": "inspect_environment",
                                        "rationale": "Inspect the OpenFOAM environment first.",
                                    }
                                }
                            ),
                            refusal=None,
                        )
                    )
                ],
                usage=None,
            )

        def parse(self, **kwargs):
            raise AssertionError("EngineeringTurn must not be sent to Ollama grammar parsing")

    llm = OllamaLLM(
        model=DEFAULT_OLLAMA_MODEL,
        client=SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions())),
    )
    turn = llm.generate(EngineeringTurn, "Choose the next engineering action.")

    assert turn.action.type == "inspect_environment"
    assert captured["response_format"] == {"type": "json_object"}
    assert "EngineeringTurn" not in str(captured["response_format"])
    assert '"anyOf"' in captured["messages"][1]["content"]


def test_ollama_generate_connection_failure_never_falls_back_to_openai():
    class APIConnectionError(Exception):
        pass

    class FakeCompletions:
        def create(self, **kwargs):
            raise APIConnectionError("connection refused")

    fake_client = SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions()))
    llm = OllamaLLM(model=DEFAULT_OLLAMA_MODEL, client=fake_client)
    with pytest.raises(OllamaConnectionError, match="SSH tunnel to mlfm4.knu.ac.kr"):
        llm.generate(DemoOutput, "test")


def test_build_llm_ollama_reuses_adapter_and_does_not_create_openai(monkeypatch):
    import openfoam_agent.cli as cli

    health_calls = []

    def fake_health(**kwargs):
        health_calls.append(kwargs)
        return ["gemma4:31b", "other:7b"]

    class DummyOllamaLLM:
        created = []

        def __init__(self, *, model, base_url, api_key, max_output_tokens):
            self.model = model
            self.base_url = base_url
            self.last_usage = None
            self.__class__.created.append((model, base_url, api_key, max_output_tokens))

    class ForbiddenOpenAILLM:
        def __init__(self, **kwargs):
            raise AssertionError("OpenAI fallback must not be constructed for --backend ollama")

    monkeypatch.setattr(cli, "check_ollama_health", fake_health)
    monkeypatch.setattr(cli, "OllamaLLM", DummyOllamaLLM)
    monkeypatch.setattr(cli, "OpenAILLM", ForbiddenOpenAILLM)

    args = _args(
        model="gemma4:31b",
        engineering_model="other:7b",
        review_model="other:7b",
        base_url="http://localhost:11434/v1",
    )
    llms, backend, default_model = _build_llm(args)
    assert backend == "ollama"
    assert default_model == "gemma4:31b"
    assert llms.intake.model == "gemma4:31b"
    assert llms.postprocessing is llms.intake
    assert llms.engineering.model == "other:7b"
    assert llms.review is llms.engineering
    assert len(DummyOllamaLLM.created) == 2
    assert len(health_calls) == 1


def test_cli_accepts_ollama_without_cloud_confirmation_and_exposes_base_url():
    parser = build_parser()
    args = parser.parse_args(
        [
            "demo",
            "--backend",
            "ollama",
            "--model",
            "gemma4:31b",
            "--base-url",
            "http://localhost:11434/v1",
        ]
    )
    assert args.backend == "ollama"
    assert args.model == "gemma4:31b"
    assert args.base_url == "http://localhost:11434/v1"
    assert args.confirm_api_calls is False
    assert _validate_args(args, parser) == "demo"
