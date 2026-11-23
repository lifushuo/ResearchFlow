"""Free tests for the hard paid-call budget gate. All LLMs here are Fakes.

Zero real model calls, zero GEPA, zero holdout scoring.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from evals.paid_call_budget import (
    ROLE_JUDGE,
    ROLE_PLANNER,
    ROLE_REFLECTION,
    BudgetConfig,
    BudgetError,
    BudgetExhausted,
    BudgetPreflightError,
    ExecutionLock,
    PaidCallBudget,
    RunAlreadyActive,
    UnknownCallBlocked,
)


def cfg(planner=2, judge=2, reflection=1, total=5, run_id="test-run"):
    return BudgetConfig(
        run_id=run_id,
        quotas={ROLE_PLANNER: planner, ROLE_JUDGE: judge, ROLE_REFLECTION: reflection},
        total_cap=total,
        planned_evaluations=64,
    )


def budget(tmp_path: Path, **kwargs) -> PaidCallBudget:
    b = PaidCallBudget(tmp_path / "ledger", cfg(**kwargs))
    b.preflight()
    return b


def test_01_reservation_is_atomic_under_concurrency(tmp_path: Path):
    """20 concurrent reservations with quota 5 must yield exactly 5 winners."""
    b = budget(tmp_path, planner=5, judge=0, reflection=0, total=5)

    async def attempt(i: int):
        try:
            return b.reserve(ROLE_PLANNER, f"call-{i}")
        except BudgetExhausted:
            return None

    # asyncio.gather needs a coroutine wrapper
    async def main():
        return await asyncio.gather(*[attempt(i) for i in range(20)])

    results = asyncio.run(main())
    assert sum(1 for r in results if r is not None) == 5
    assert b.state()["used_by_role"][ROLE_PLANNER] == 5
    assert b.state()["remaining_by_role"][ROLE_PLANNER] == 0
    slots = [r.slot_id for r in results if r is not None]
    assert len(set(slots)) == 5


def test_02_second_call_after_exhaustion_is_denied_and_traced(tmp_path: Path):
    b = budget(tmp_path, planner=1, judge=0, reflection=0, total=1)
    r = b.reserve(ROLE_PLANNER, "first")
    r.mark_sent()
    r.mark_completed(result_fingerprint="abc")
    with pytest.raises(BudgetExhausted):
        b.reserve(ROLE_PLANNER, "second")
    trace = [json.loads(line) for line in b.trace_path.read_text(encoding="utf-8").splitlines()]
    assert any(t.get("event") == "denied" and t.get("reason") == "role_quota" for t in trace)


def test_03_unknown_status_blocks_all_future_calls(tmp_path: Path):
    """Response-before-commit crash leaves UNKNOWN; no auto retry permitted."""
    b = budget(tmp_path, planner=2, judge=2, reflection=1, total=5)
    r = b.reserve(ROLE_PLANNER, "crashed")
    r.mark_sent()
    r.mark_unknown(reason="timeout after request sent")

    with pytest.raises(UnknownCallBlocked):
        b.preflight()

    # explicit human resolution required, with evidence
    with pytest.raises(BudgetError, match="evidence is required"):
        b.acknowledge_unknown(r.slot_id, decision="resolved_completed", evidence="")
    b.acknowledge_unknown(r.slot_id, decision="abandoned", evidence="provider log shows no charge")
    assert r.slot_id not in b.state()["unknown"]
    b.preflight()  # now allowed


def test_04_sent_without_completion_counts_as_unknown(tmp_path: Path):
    b = budget(tmp_path, planner=1, judge=1, reflection=1, total=3)
    r = b.reserve(ROLE_JUDGE, "sent-only")
    r.mark_sent()
    assert r.slot_id in b.state()["unknown"]
    with pytest.raises(UnknownCallBlocked):
        b.preflight()


def test_05_timeout_marks_unknown_and_never_auto_retries(tmp_path: Path):
    calls = []

    def fake_llm(_payload):
        calls.append(1)
        raise TimeoutError("fake provider timeout")

    b = budget(tmp_path, planner=1, judge=0, reflection=0, total=1)
    with pytest.raises(TimeoutError):
        b.paid_call(ROLE_PLANNER, "timeout", lambda: fake_llm({}))
    assert len(calls) == 1
    assert b.state()["unknown"]
    with pytest.raises(UnknownCallBlocked):
        b.preflight()


def test_06_retries_consume_new_slots_and_cannot_loop_forever(tmp_path: Path):
    attempts = []

    def flaky():
        attempts.append(1)
        if len(attempts) < 2:
            raise RuntimeError("fake transient failure")
        return {"ok": True}

    b = budget(tmp_path, planner=2, judge=0, reflection=0, total=2)
    # First attempt becomes UNKNOWN, so further calls are blocked: no free retry.
    with pytest.raises(RuntimeError):
        b.paid_call(ROLE_PLANNER, "flaky", flaky)
    with pytest.raises(UnknownCallBlocked):
        b.paid_call(ROLE_PLANNER, "flaky-retry", flaky)
    assert len(attempts) == 1, "automatic retry must never happen"


def test_07_execution_lock_blocks_duplicate_start(tmp_path: Path):
    b = budget(tmp_path)
    with ExecutionLock(b):
        with pytest.raises(RunAlreadyActive):
            with ExecutionLock(b):
                pass
    # lock released when the owning block exits
    with ExecutionLock(b):
        pass


def test_08_paid_call_async_completes_and_records_fingerprint(tmp_path: Path):
    async def fake():
        await asyncio.sleep(0)
        return {"verdict": "ok"}

    b = budget(tmp_path, planner=0, judge=1, reflection=0, total=1)
    result = asyncio.run(b.paid_call_async(ROLE_JUDGE, "judge-1", fake, input_data={"q": "x"}))
    assert result == {"verdict": "ok"}
    state = b.state()
    assert state["used_by_role"][ROLE_JUDGE] == 1
    assert all(s == "COMPLETED" for s in state["statuses"].values())


def test_09_preflight_rejects_unwritable_output_before_any_call(tmp_path: Path, monkeypatch):
    import evals.paid_call_budget as mod

    b = PaidCallBudget(tmp_path / "ledger", cfg())
    called = []
    monkeypatch.setattr(
        mod, "_mkstemp_readback", lambda root: (_ for _ in ()).throw(OSError("read-only"))
    )
    with pytest.raises(BudgetPreflightError):
        b.paid_call(ROLE_PLANNER, "must-not-charge", lambda: called.append(1))
    assert called == []


def test_10_initial_budget_guards_are_enforced(tmp_path: Path):
    bad = BudgetConfig(
        run_id="bad",
        quotas={ROLE_PLANNER: 10, ROLE_JUDGE: 10, ROLE_REFLECTION: 10},
        total_cap=5,
        planned_evaluations=64,
    )
    b = PaidCallBudget(tmp_path / "ledger", bad)
    with pytest.raises(BudgetPreflightError):
        b.preflight()


def test_11_state_distinguishes_evaluations_from_paid_calls(tmp_path: Path):
    b = budget(tmp_path, planner=1, judge=2, reflection=1, total=4)
    s = b.state()
    assert s["planned_evaluations"] == 64
    assert s["used_total"] == 0
    assert sum(s["used_by_role"].values()) == s["used_total"]
    cfg_disk = json.loads(b.config_path.read_text(encoding="utf-8"))
    assert cfg_disk["planned_evaluations"] == 64
    assert cfg_disk["total_cap"] == 4


def test_12_human_acknowledgement_records_no_fabricated_score(tmp_path: Path):
    b = budget(tmp_path, planner=1, judge=1, reflection=1, total=3)
    r = b.reserve(ROLE_PLANNER, "unknown")
    r.mark_sent()
    r.mark_unknown(reason="no response")
    b.acknowledge_unknown(
        r.slot_id, decision="resolved_completed", evidence="billing log confirms one completed call"
    )
    event = json.loads(b._events_path(r.role, r.index).read_text(encoding="utf-8").splitlines()[-1])
    assert event["status"] == "COMPLETED"
    assert event["human_decision"] is True
    assert "task_specificity" not in event and "quality" not in event


def test_13_planner_retries_and_fallback_each_charge_a_slot(tmp_path: Path):
    """A single Planner observation can issue up to 4 requests; all are charged."""
    from research_engine.runconfig import RunConfig, reset_run_config, set_run_config
    from research_engine.token_usage import gate_scope
    from research_engine.token_usage import invoke as usage_invoke

    cfg = BudgetConfig(
        run_id="per-request",
        quotas={ROLE_PLANNER: 6, ROLE_JUDGE: 0, ROLE_REFLECTION: 0},
        total_cap=6,
        planned_evaluations=1,
    )
    b = PaidCallBudget(tmp_path / "ledger", cfg)
    b.preflight()

    class FakeModel:
        async def ainvoke(self, messages):
            return {"ok": True}

    token = set_run_config(RunConfig())
    try:
        with gate_scope(b):
            for _ in range(4):  # 2 attempts x (function_calling + JSON fallback)
                asyncio.run(usage_invoke(FakeModel(), [], role="planner"))
    finally:
        reset_run_config(token)

    state = b.state()
    assert state["used_by_role"][ROLE_PLANNER] == 4
    assert all(v == "COMPLETED" for v in state["statuses"].values())
