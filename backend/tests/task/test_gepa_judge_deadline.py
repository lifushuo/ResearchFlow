import asyncio

import pytest

from app.services.planner_gepa_contract import request_timeouts
from evals.agent_eval_ext.planner_evolution.runtime import JudgeDeadlineExceeded, LiveRuntime
from evals.paid_call_budget import (
    BudgetConfig,
    PaidCallBudget,
    RequestOutcomeUnknown,
    UnknownCallBlocked,
)
from research_engine.runconfig import RunConfig


async def test_local_judge_deadline_keeps_unknown_and_blocks_successor(tmp_path):
    b = PaidCallBudget(
        tmp_path / "ledger",
        BudgetConfig("short", {"planner": 1, "judge": 1, "reflection": 1}, 3, 1),
    )
    b.preflight()
    rt = LiveRuntime(
        RunConfig(), "fake:judge", "fake:reflection", 90, (), budget=b, judge_timeout=0.005
    )
    calls = []

    async def slow():
        calls.append(1)
        await asyncio.sleep(0.04)
        return {"ok": True}

    with pytest.raises(RequestOutcomeUnknown) as failed:
        await rt._paid("judge", "deadline", lambda: rt._judge_wait(slow))
    assert failed.value.cause_type == "JudgeDeadlineExceeded"
    assert b.state()["unknown"] == ["judge-000001"] and len(calls) == 1
    with pytest.raises(UnknownCallBlocked):
        b.reserve("planner", "no retry")


async def test_separate_judge_allowance_waits_for_same_single_call(tmp_path):
    b = PaidCallBudget(
        tmp_path / "ledger", BudgetConfig("long", {"planner": 1, "judge": 1, "reflection": 1}, 3, 1)
    )
    b.preflight()
    rt = LiveRuntime(
        RunConfig(), "fake:judge", "fake:reflection", 0.005, (), budget=b, judge_timeout=0.2
    )
    calls = []

    async def slow():
        calls.append(1)
        await asyncio.sleep(0.04)
        return {"ok": True}

    assert await rt._paid("judge", "deadline", lambda: rt._judge_wait(slow)) == {"ok": True}
    assert b.state()["unknown"] == [] and b.state()["used_total"] == 1 and len(calls) == 1
    assert rt.timeout == 0.005


async def test_provider_timeout_is_not_labelled_local_deadline():
    rt = LiveRuntime(RunConfig(), "fake:judge", "fake:reflection", 90, (), judge_timeout=240)

    async def fail():
        raise TimeoutError("provider timeout")

    with pytest.raises(TimeoutError) as error:
        await rt._judge_wait(fail)
    assert not isinstance(error.value, JudgeDeadlineExceeded)


def test_new_deadlines_are_bounded_and_old_cli_fallback_preserved(monkeypatch):
    monkeypatch.delenv("RF_GEPA_JUDGE_TIMEOUT_SECONDS", raising=False)
    assert request_timeouts() == {"planner": 90, "judge": 240, "reflection": 90}
    rt = LiveRuntime(RunConfig(), "fake:judge", "fake:reflection", 90, ())
    assert rt.judge_timeout == 90
    for invalid in ["0", "601", "abc"]:
        monkeypatch.setenv("RF_GEPA_JUDGE_TIMEOUT_SECONDS", invalid)
        with pytest.raises(ValueError):
            request_timeouts()
