"""Reproduce the actual classified Planner connection failure without any network."""

import json

import httpx
import pytest
from langchain_openai import ChatOpenAI
from test_gepa_web_v3_e2e import web as web

from evals.agent_eval_ext.planner_evolution.experiment import Case
from evals.agent_eval_ext.planner_evolution.runtime import LiveRuntime
from evals.paid_call_budget import (
    BudgetConfig,
    PaidCallBudget,
    RequestOutcomeUnknown,
    UnknownCallBlocked,
)
from research_engine import llm_factory
from research_engine.runconfig import RunConfig


async def test_planner_connection_error_stops_before_graph_retry(tmp_path, monkeypatch):
    sent = []

    async def fail(request):
        sent.append(request)
        raise httpx.ConnectError("offline simulated disconnect", request=request)

    budget = PaidCallBudget(
        tmp_path / "ledger",
        BudgetConfig("connection", {"planner": 256, "judge": 64, "reflection": 4}, 324, 64),
    )
    budget.preflight()
    async with httpx.AsyncClient(transport=httpx.MockTransport(fail)) as client:
        model = ChatOpenAI(model="offline-model", api_key="offline", http_async_client=client)
        monkeypatch.setattr(llm_factory, "_build", lambda *_: model)
        cfg = RunConfig(
            llm_mode="real", models={**RunConfig().models, "planner": "custom:offline-model"}
        )
        runtime = LiveRuntime(
            cfg,
            "custom:offline-judge",
            "custom:offline-model",
            30,
            (),
            metric_version="v3",
            budget=budget,
        )
        case = Case(
            case_id="reranking",
            group="test",
            split="validation",
            query="Compare reranking benchmark evidence and latency",
            required_aspects=["benchmark evidence", "measured latency"],
        )
        with pytest.raises(RequestOutcomeUnknown) as error:
            await runtime.observe(case, None)
        assert error.value.slot_id == "planner-000001"
        assert error.value.role == "planner"
        assert "ConnectionError" in error.value.cause_type
    assert len(sent) == 1
    state = budget.state()
    assert state["used_total"] == 1 and state["unknown"] == ["planner-000001"]
    with pytest.raises(UnknownCallBlocked):
        budget.reserve("judge", "forbidden successor")
    events = [
        json.loads(line)
        for line in (tmp_path / "ledger/slots/planner-000001.events.jsonl").read_text().splitlines()
    ]
    assert events[-1]["status"] == "UNKNOWN" and events[-1]["cause_types"]
    assert events[-1]["call_stack"]


async def test_judge_and_reflection_keep_classification_on_failure(tmp_path):
    for role in ("judge", "reflection"):
        b = PaidCallBudget(
            tmp_path / role, BudgetConfig(role, {"planner": 1, "judge": 1, "reflection": 1}, 3, 1)
        )
        b.preflight()
        rt = LiveRuntime(RunConfig(), "custom:judge", "custom:planner", 30, (), budget=b)

        async def fail():
            raise ConnectionError("offline")

        with pytest.raises(RequestOutcomeUnknown) as error:
            await rt._paid(role, role, fail)
        assert error.value.role == role and error.value.slot_id == role + "-000001"


def test_complete_64_budget_fake_search(web):
    from test_gepa_web_v3_e2e import start

    from evals.agent_eval_ext.planner_evolution_job_worker import run_claimed

    client, root = web
    job, folder, request = start(web, max_evals=64, proposals=4, dataset_preset="full")
    run_claimed(root, folder, request)
    audit = client.get("/api/v1/evaluation/evolution/jobs/" + job["job_id"] + "/audit").json()
    assert audit["state"] == "pending_final"
    assert audit["journal"]["reflection_count"] >= 1
    assert audit["journal"]["observation_count"] <= 64
    assert audit["budget"]["configured"]["total_cap"] == 324
    assert audit["budget"]["used_total"] <= 324 and audit["budget"]["unknown"] == []
    assert not audit["journal"]["holdout_observed"]
