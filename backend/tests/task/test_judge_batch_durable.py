"""Free fault-injection tests for durable Judge batch execution.

Every Judge here is a local fake; NO paid model call is made.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from evals.judge_batch_durable import (
    EVENT_COMPLETED,
    EVENT_STARTED,
    EVENT_UNMEASURED,
    DurableJudgeBatch,
    JudgeCallSpec,
    PersistenceFailure,
    PreflightError,
    UnknownCallError,
)


def specs(n: int = 4) -> list[JudgeCallSpec]:
    return [
        JudgeCallSpec(
            case_id=f"case-{i}",
            variant="candidate" if i % 2 else "baseline",
            repeat=1,
            payload={"question": f"question {i}", "tasks": {1: f"task {i}"}},
        )
        for i in range(1, n + 1)
    ]


def good_judge(counter: dict[str, int]):
    async def judge(spec: JudgeCallSpec):
        counter[spec.logical_id] = counter.get(spec.logical_id, 0) + 1
        return {
            "raw_response": f'{{"verdict": "ok for {spec.logical_id}"}}',
            "task_specificity": [2],
            "plan_quality": [2],
            "tokens": {"input": 10, "output": 5},
            "latency_seconds": 0.01,
        }

    return judge


def test_01_forced_termination_keeps_completed_records(tmp_path: Path):
    """Kill between calls: completed results survive, in-flight call is UNKNOWN."""
    log = tmp_path / "batch.jsonl"
    counter: dict[str, int] = {}

    async def judge_with_kill(spec: JudgeCallSpec):
        counter[spec.logical_id] = counter.get(spec.logical_id, 0) + 1
        if len(counter) == 3:
            raise SystemExit("simulated process kill after 2 completed calls")
        return {"raw_response": "ok", "task_specificity": [2], "plan_quality": [2]}

    batch = DurableJudgeBatch(log, run_source="fault-injection", route="fake-judge")
    with pytest.raises(SystemExit):
        asyncio.run(batch.run(specs(), judge_with_kill))

    fresh = DurableJudgeBatch(log, run_source="resume", route="fake-judge")
    state = fresh.statuses()
    done = [k for k, v in state.items() if v == "DONE"]
    unknown = [k for k, v in state.items() if v == "UNKNOWN"]
    assert len(done) == 2, state
    assert len(unknown) == 1, state
    assert len(fresh.completed_results()) == 2
    # the two results that survived are fully readable
    for rec in fresh.completed_results().values():
        assert rec["result"]["raw_response"] == "ok"
        assert rec["input_fingerprint"]
        assert rec["case_id"] and rec["variant"] and rec["repeat"] == 1


def test_02_resume_skips_completed_calls(tmp_path: Path):
    """Re-running a finished batch must not issue any additional paid call."""
    log = tmp_path / "batch.jsonl"
    counter: dict[str, int] = {}
    batch = DurableJudgeBatch(log, run_source="first", route="fake-judge")
    asyncio.run(batch.run(specs(3), good_judge(counter)))
    before = log.read_bytes()

    never_call_counter: dict[str, int] = {}

    def must_not_call(spec: JudgeCallSpec):  # pragma: no cover - should never run
        never_call_counter[spec.logical_id] = 1
        return {"raw_response": "SHOULD NOT HAPPEN"}

    resumed = DurableJudgeBatch(log, run_source="resume", route="fake-judge")
    report = asyncio.run(resumed.run(specs(3), must_not_call))
    assert report["attempted_this_run"] == 0
    assert report["skipped_already_done"] == 3
    assert never_call_counter == {}
    assert log.read_bytes() == before  # resume did not modify old evidence


def test_03_unwritable_output_fails_before_any_paid_call(tmp_path: Path, monkeypatch):
    """Preflight failure raises before judge_call is invoked even once."""
    import evals.judge_batch_durable as mod

    log = tmp_path / "batch.jsonl"
    batch = DurableJudgeBatch(log, run_source="preflight-test", route="fake-judge")
    called: list[str] = []

    def raise_on_probe(*args, **kwargs):
        raise OSError("simulated read-only filesystem")

    monkeypatch.setattr(mod.tempfile, "mkstemp", raise_on_probe)

    with pytest.raises(PreflightError):
        asyncio.run(batch.run(specs(2), lambda spec: called.append(spec.logical_id)))
    assert called == []
    assert not log.exists()


def test_04_judge_success_but_save_failure_stops_and_marks_unknown(tmp_path: Path, monkeypatch):
    """Judge returned (billed) but result could not be saved: halt, record UNKNOWN."""
    log = tmp_path / "batch.jsonl"
    batch = DurableJudgeBatch(log, run_source="save-failure-test", route="fake-judge")
    counter: dict[str, int] = {}

    original_append = batch._append
    failed_once: list[str] = []

    def failing_append(record):
        if record["event"] == EVENT_COMPLETED and not failed_once:
            failed_once.append(record["logical_id"])
            raise PersistenceFailure("simulated ENOSPC on result write")
        return original_append(record)

    monkeypatch.setattr(batch, "_append", failing_append)

    with pytest.raises(PersistenceFailure):
        asyncio.run(batch.run(specs(4), good_judge(counter)))

    assert len(failed_once) == 1
    assert len(counter) == 1, "batch must stop after a persistence failure"
    fresh = DurableJudgeBatch(log, run_source="resume", route="fake-judge")
    assert fresh.statuses()[failed_once[0]] == "UNKNOWN"
    assert fresh.incident_path.exists()
    incident = json.loads(fresh.incident_path.read_text(encoding="utf-8").strip().splitlines()[-1])
    assert incident["phase"] == "result_persist_failed"
    assert incident["status"] == "UNKNOWN"
    with pytest.raises(UnknownCallError):
        fresh.pending(specs(4))


def test_05_single_timeout_does_not_damage_other_records(tmp_path: Path):
    """One Judge timeout is recorded as UNMEASURED; other results stay intact."""
    log = tmp_path / "batch.jsonl"
    counter: dict[str, int] = {}

    async def flaky_judge(spec: JudgeCallSpec):
        counter[spec.logical_id] = counter.get(spec.logical_id, 0) + 1
        if spec.case_id == "case-2":
            raise TimeoutError("simulated provider timeout")
        return {"raw_response": "ok", "task_specificity": [1], "plan_quality": [1]}

    batch = DurableJudgeBatch(log, run_source="timeout-test", route="fake-judge")
    report = asyncio.run(batch.run(specs(4), flaky_judge))
    assert report["completed_this_run"] == 3
    assert report["unmeasured_this_run"] == 1
    state = batch.statuses()
    assert state["case-2|baseline|1"] == "UNMEASURED"
    assert sum(1 for v in state.values() if v == "DONE") == 3
    events = [json.loads(line)["event"] for line in log.read_text(encoding="utf-8").splitlines()]
    assert events.count(EVENT_STARTED) == 4
    assert events.count(EVENT_COMPLETED) == 3
    assert events.count(EVENT_UNMEASURED) == 1


def test_06_explicit_override_needed_for_unmeasured_resume(tmp_path: Path):
    """UNMEASURED calls are never silently re-billed; explicit opt-in works."""
    log = tmp_path / "batch.jsonl"
    counter: dict[str, int] = {}

    def fail_second(spec: JudgeCallSpec):
        counter[spec.logical_id] = counter.get(spec.logical_id, 0) + 1
        if spec.case_id == "case-2":
            raise TimeoutError("timeout")
        return {"raw_response": "ok", "task_specificity": [2], "plan_quality": [2]}

    first = DurableJudgeBatch(log, run_source="first", route="fake-judge")
    asyncio.run(first.run(specs(3), fail_second))

    resumed = DurableJudgeBatch(log, run_source="resume", route="fake-judge")
    with pytest.raises(UnknownCallError):
        resumed.pending(specs(3))
    before = counter.copy()
    allowed = resumed.pending(specs(3), allow_unmeasured=True)
    assert [s.logical_id for s in allowed] == ["case-2|baseline|1"]
    assert counter == before  # pending() itself never calls the Judge


def test_07_input_fingerprint_is_logged_and_binding(tmp_path: Path):
    """Every record carries the input fingerprint and run source for audit."""
    log = tmp_path / "batch.jsonl"
    batch = DurableJudgeBatch(log, run_source="audit-test", route="fake-judge")
    items = specs(1)
    asyncio.run(batch.run(items, good_judge({})))
    records = batch.load()
    assert len(records) == 2
    assert records[0]["event"] == EVENT_STARTED
    assert records[1]["event"] == EVENT_COMPLETED
    for rec in records:
        assert rec["input_fingerprint"] == items[0].input_fingerprint
        assert rec["run_source"] == "audit-test"
        assert rec["route"] == "fake-judge"
        assert rec["case_id"] == items[0].case_id
        assert rec["variant"] == items[0].variant
        assert rec["repeat"] == items[0].repeat
    assert records[1]["result"]["tokens"] == {"input": 10, "output": 5}
