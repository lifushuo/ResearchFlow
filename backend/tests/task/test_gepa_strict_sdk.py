"""Actual OpenAI SDK over MockTransport: no hidden retries or unbudgeted requests."""

import httpx
import pytest
from langchain_openai import ChatOpenAI
from openai import InternalServerError

from evals.paid_call_budget import BudgetConfig, BudgetExhausted, PaidCallBudget, UnknownCallBlocked
from research_engine import llm_factory
from research_engine.runconfig import RunConfig, reset_run_config, set_run_config
from research_engine.token_usage import PaidRequestUncertain, gate_scope, invoke


@pytest.mark.parametrize("status", [200, 500])
async def test_sdk_attempts_never_escape_request_gate(tmp_path, monkeypatch, status):
    requests = []

    async def handle(request):
        requests.append(request)
        if status == 500:
            return httpx.Response(
                500, json={"error": {"message": "offline simulated error", "type": "test"}}
            )
        return httpx.Response(
            200,
            json={
                "id": "offline",
                "object": "chat.completion",
                "created": 1,
                "model": "offline-model",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "ok"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            },
        )

    budget = PaidCallBudget(
        tmp_path / "ledger",
        BudgetConfig("sdk-test", {"planner": 1, "judge": 1, "reflection": 0}, 2, 1),
    )
    budget.preflight()
    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        model = ChatOpenAI(
            model="offline-model", api_key="not-a-real-key", http_async_client=client
        )
        assert model.root_async_client.max_retries == 2
        monkeypatch.setattr(llm_factory, "_build", lambda *_: model)
        cfg = RunConfig(
            llm_mode="real", models={**RunConfig().models, "planner": "custom:offline-model"}
        )
        token = set_run_config(cfg)
        try:
            with llm_factory.no_sdk_retries(), gate_scope(budget):
                guarded = llm_factory.get_llm("planner")
                assert guarded.root_client.max_retries == guarded.root_async_client.max_retries == 0
                if status == 500:
                    with pytest.raises(PaidRequestUncertain) as failed:
                        await invoke(guarded, [], role="planner")
                    assert isinstance(failed.value.__cause__, InternalServerError)
                    with pytest.raises(UnknownCallBlocked):
                        await invoke(guarded, [], role="planner")
                else:
                    await invoke(guarded, [], role="planner")
                    with pytest.raises(BudgetExhausted):
                        await invoke(guarded, [], role="planner")
        finally:
            reset_run_config(token)
    assert len(requests) == 1
    assert budget.state()["used_total"] == 1


def test_live_judge_independence_is_checked_before_enqueue(tmp_path, monkeypatch):
    import uuid

    from app.services.planner_evolution_jobs import create_job

    for role in ("PLANNER", "EXECUTOR", "CRITIC", "SYNTHESIZER", "CHAT"):
        monkeypatch.setenv("MODEL_" + role, "custom:offline-same-model")
    monkeypatch.setenv("CUSTOM_BASE_URL", "https://invalid.example/v1")
    monkeypatch.setenv("CUSTOM_API_KEY", "offline-placeholder")
    monkeypatch.setenv("EVAL_JUDGE_MODEL", "custom:offline-same-model")
    monkeypatch.setenv("RF_GEPA_LIVE_ENABLED", "1")
    monkeypatch.setenv("RF_GEPA_ALLOWED_OWNERS", "alice")
    with pytest.raises(ValueError):
        create_job(
            tmp_path,
            "alice",
            request_id=str(uuid.uuid4()),
            mode="live",
            max_evals=8,
            proposals=1,
            seed=42,
            consent="I_CONFIRM_GEPA_MODEL_API_COST",
            dataset_preset="smoke",
        )
    assert not list(tmp_path.rglob("request.json"))
