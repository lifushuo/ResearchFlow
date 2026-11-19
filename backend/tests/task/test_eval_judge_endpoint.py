"""裁判与研究模型即使都走 custom，也必须使用各自的端点和凭据。"""

from dataclasses import replace

import pytest

from evals import benchmark, harness
from evals.judge_config import judge_client_config
from research_engine import llm_factory
from research_engine.runconfig import get_run_config, reset_run_config, set_run_config


@pytest.fixture
def dedicated_judge(monkeypatch):
    monkeypatch.setenv("EVAL_JUDGE_MODEL", "custom:kimi-k3")
    monkeypatch.setenv("EVAL_JUDGE_BASE_URL", "https://judge.invalid/v1")
    monkeypatch.setenv("EVAL_JUDGE_API_KEY", "judge-test-key")
    base = get_run_config()
    token = set_run_config(
        replace(
            base,
            llm_mode="real",
            enforce_ssrf_guards=False,
            models={role: "custom:deepseek-chat" for role in base.models},
            provider_keys={
                "custom": "pipeline-test-key",
                "custom_base_url": "https://pipeline.invalid/v1",
            },
        )
    )
    user_token = llm_factory.set_user_keys({"custom": "user-test-key"})
    yield
    llm_factory.reset_user_keys(user_token)
    reset_run_config(token)


@pytest.mark.parametrize("host", ["harness", "benchmark"])
def test_real_clients_keep_separate_endpoints_and_keys(monkeypatch, dedicated_judge, host):
    before = get_run_config()
    if host == "harness":
        monkeypatch.setattr(harness, "JUDGE_ROUTE", "custom:kimi-k3")
        judge = harness._judge_llm()
    else:
        judge, route = benchmark._build_judge(benchmark.DEFAULT_JUDGE_MODEL)
        assert route == "custom:kimi-k3"
    assert judge.model_name == "kimi-k3"
    assert str(judge.openai_api_base) == "https://judge.invalid/v1"
    assert judge.openai_api_key.get_secret_value() == "judge-test-key"
    assert get_run_config() is before
    pipeline = llm_factory.get_llm("critic")
    assert pipeline.model_name == "deepseek-chat"
    assert str(pipeline.openai_api_base) == "https://pipeline.invalid/v1"
    assert pipeline.openai_api_key.get_secret_value() == "user-test-key"


def test_factory_failure_restores_routing_and_byok(dedicated_judge):
    before = get_run_config()
    with pytest.raises(RuntimeError, match="construction failed"):
        with judge_client_config("custom:kimi-k3"):
            assert llm_factory.api_key_for("custom") == "judge-test-key"
            raise RuntimeError("construction failed")
    assert get_run_config() is before
    assert llm_factory.api_key_for("custom") == "user-test-key"


async def test_real_sdk_sends_judge_and_pipeline_to_their_own_endpoints(
    monkeypatch, dedicated_judge
):
    import json
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer

    seen = []

    class Provider(BaseHTTPRequestHandler):
        def do_POST(self):  # noqa: N802
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            seen.append(
                (
                    self.path,
                    self.headers["Authorization"],
                    body["model"],
                    body.get("max_completion_tokens", body.get("max_tokens")),
                )
            )
            response = json.dumps(
                {
                    "id": "chatcmpl-test",
                    "object": "chat.completion",
                    "created": 0,
                    "model": body["model"],
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": "OK"},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                }
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(response)))
            self.end_headers()
            self.wfile.write(response)

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Provider)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    endpoint = f"http://127.0.0.1:{server.server_port}"
    monkeypatch.setenv("EVAL_JUDGE_BASE_URL", endpoint + "/judge/v1")
    monkeypatch.setattr(harness, "JUDGE_ROUTE", "custom:kimi-k3")
    base = get_run_config()
    token = set_run_config(
        replace(
            base, provider_keys={**base.provider_keys, "custom_base_url": endpoint + "/pipeline/v1"}
        )
    )
    try:
        assert (await harness._judge_llm().ainvoke("test")).content == "OK"
        assert (await llm_factory.get_llm("critic").ainvoke("test")).content == "OK"
    finally:
        reset_run_config(token)
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
    assert seen == [
        ("/judge/v1/chat/completions", "Bearer judge-test-key", "kimi-k3", 4096),
        ("/pipeline/v1/chat/completions", "Bearer user-test-key", "deepseek-chat", 1000),
    ]


@pytest.mark.parametrize("missing", ["EVAL_JUDGE_API_KEY", "EVAL_JUDGE_BASE_URL"])
def test_partial_endpoint_pair_refuses_before_build(monkeypatch, dedicated_judge, missing):
    monkeypatch.setenv(missing, "")
    with pytest.raises(ValueError, match="必须同时配置"):
        with judge_client_config("custom:kimi-k3"):
            pytest.fail("不完整的配置进入了客户端构建步骤")


def test_explicit_native_route_does_not_use_custom_judge_credentials(dedicated_judge):
    with judge_client_config("anthropic:claude-sonnet-4-6"):
        assert llm_factory.api_key_for("custom") == "user-test-key"


def test_benchmark_specific_route_takes_precedence(monkeypatch, dedicated_judge):
    monkeypatch.setenv("BENCHMARK_JUDGE_MODEL", "custom:another-judge")
    judge, route = benchmark._build_judge(benchmark.DEFAULT_JUDGE_MODEL)
    assert route == "custom:another-judge"
    assert judge.model_name == "another-judge"


async def test_harness_reads_named_judge_from_environment(monkeypatch, dedicated_judge, tmp_path):
    import sys

    monkeypatch.setattr(harness, "RUN_CONFIG", get_run_config())
    monkeypatch.setattr(harness, "JUDGE_ROUTE", None)
    monkeypatch.setattr(
        sys, "argv", ["harness", "--limit", "1", "--out", str(tmp_path / "run.json")]
    )
    called = []

    async def stop_before_spending(*args, **kwargs):
        called.append(harness.JUDGE_ROUTE)
        raise RuntimeError("stop before spending")

    monkeypatch.setattr(harness, "run_one", stop_before_spending)
    with pytest.raises(RuntimeError, match="stop before spending"):
        await harness.main()
    assert called == ["custom:kimi-k3"]
