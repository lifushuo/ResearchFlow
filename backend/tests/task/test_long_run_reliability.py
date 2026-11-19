"""P0 long-run reliability: fake models, controllable clocks, fault injection.

The real incident is ``quantum-error-correction``: research reached the human gate
at ~182.84 s, the judge then ran ~235.29 s with eight unconditional 15 s gaps, and
the 420 s worker kill discarded everything because the only checkpoint was an
in-process ``MemorySaver``.

These tests never contact a provider, never use a real clock and never write to
``evals/results``. They pin the replacement behaviour: staged deadlines, a
provider throttle that waits only before a real next request, atomic per-batch
progress, a refusal to replay a completed batch, and a cooperative stop that
keeps the draft and every measured verdict.
"""

from __future__ import annotations

import asyncio
import json
import sys
from dataclasses import replace
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage

from evals import harness, metrics
from evals.agent_eval_ext import long_run
from evals.agent_eval_ext.judge_runtime import trace_judge_case
from evals.agent_eval_ext.judge_runtime_inspect import summarize

CASE_ID = "quantum-error-correction"
QUERY = {
    "id": CASE_ID,
    "domain": "physics",
    "query": "quantum error correction",
    "depth": "balanced",
}
REPORT = (
    "# Report\n\n## Findings\n"
    + "\n".join(f"- Fact {i} is present [1]." for i in range(1, 10))
    + "\n\n## Sources\n[1] https://example.org/1\n"
)
SOURCES = [{"index": 1, "snippet": "A fact is present."}]
CLAIMS = [c for c in metrics.claim_lines(REPORT) if metrics.CITE_RE.search(c)]
TOTAL_BATCHES = 3  # 9 claims / JUDGE_BATCH_SIZE=4

_DEADLINE_ENV = (
    "RF_EVAL_WORKER_HARD_CAP_S",
    "RF_EVAL_WORKER_CASES",
    "RF_EVAL_RESEARCH_DEADLINE_S",
    "RF_EVAL_JUDGE_DEADLINE_S",
    "RF_EVAL_TOTAL_DEADLINE_S",
    "RF_EVAL_JUDGE_REQUEST_TIMEOUT_S",
    "RF_EVAL_STOP_GRACE_S",
    "RF_EVAL_JUDGE_MIN_INTERVAL_S",
    "RF_EVAL_QUERY_MIN_INTERVAL_S",
    "RF_EVAL_JUDGE_RETRY_AFTER_MAX_S",
    "RF_EVAL_RELIABILITY_DIR",
    "RF_EVAL_RESUME_FROM",
    "RF_EVAL_BATCH_LEDGER_ROOT",
    "RF_EVAL_TRACE",
    "RF_EVAL_TRACE_DIR",
    "RF_JUDGE_BATCH_TRACE_DIR",
    "RF_EVAL_CONTENT_TRACE",
    "RF_EVAL_EXECUTION_ROOT",
)


class Clock:
    """A clock that only moves when the injected sleep says it did."""

    def __init__(self) -> None:
        self.t = 0.0
        self.sleeps: list[float] = []

    def now(self) -> float:
        return self.t

    async def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.t += seconds


class ScriptedJudge:
    """One fake model call per batch. `faults[call_no]` is an exception or a reply."""

    def __init__(self, faults: dict[int, object] | None = None, on_call=None) -> None:
        self.calls = 0
        self.faults = faults or {}
        self.on_call = on_call

    async def ainvoke(self, messages):
        self.calls += 1
        if self.on_call is not None:
            self.on_call(self.calls)
        fault = self.faults.get(self.calls)
        if isinstance(fault, BaseException):
            raise fault
        if isinstance(fault, str):
            text = fault
        else:
            # The batch prompt contains one "\nEvidence N:" line per claim.
            count = messages[-1].content.count("\nEvidence ")
            text = "\n".join(f"Claim {i}: YES" for i in range(1, count + 1))
        return AIMessage(content=text, response_metadata={"model_name": "scripted-judge"})


class FakeGraph:
    """Process-local research graph; `ainvoke` calls are counted explicitly."""

    def __init__(self, marker: list[str] | None = None) -> None:
        self.marker = marker if marker is not None else []

    async def ainvoke(self, initial, config):
        self.marker.append("ainvoke")
        interrupt = SimpleNamespace(value={"type": "HITL_READY", "word_count": 9})
        return {"__interrupt__": (interrupt,)}

    async def aget_state(self, config):
        return SimpleNamespace(
            values={
                "draft_report": REPORT,
                "sources": SOURCES,
                "evidence": [{"source": 1, "text": "A fact is present."}],
                "tokens_input": 1200,
                "tokens_output": 340,
                "cost_usd": 0.021,
            }
        )


@pytest.fixture(autouse=True)
def isolated_env(monkeypatch):
    for name in _DEADLINE_ENV:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(harness, "RUN_CONFIG", replace(harness.RUN_CONFIG, llm_mode="real"))
    monkeypatch.setattr(harness, "JUDGE_ROUTE", None)
    monkeypatch.setattr(harness, "build_graph", lambda *_: FakeGraph())


def config(case_id: str = CASE_ID) -> dict:
    return long_run.build_config(
        run_config=harness.RUN_CONFIG,
        judge_route=harness.JUDGE_ROUTE,
        judge_prompt=harness.JUDGE_PROMPT,
        case_id=case_id,
        query=QUERY["query"],
        depth=QUERY["depth"],
    )


def make_context(
    tmp_path,
    clock: Clock,
    *,
    name: str = "progress",
    min_interval_s: float = 0.0,
    resume_root=None,
):
    ctx = long_run.build_context(
        root=tmp_path / name,
        judge_provider="fake-provider",
        resume_root=resume_root,
        clock=clock.now,
        sleep=clock.sleep,
    )
    ctx.throttle.min_interval_s = min_interval_s
    long_run.preflight_root(ctx.root)
    store = ctx.store_for(config())
    store.preflight()
    ctx.store = store
    return ctx, store


def draft_doc() -> dict:
    return {
        "state": long_run.RESEARCH_DRAFT_READY,
        "draft_report": REPORT,
        "sources": SOURCES,
        "evidence": [],
        "citations": [],
        "claim_lines": CLAIMS,
        "report_sha256": long_run.sha256_text(REPORT),
        "sources_sha256": long_run.canonical_hash(SOURCES),
        "claims_sha256": long_run.canonical_hash(CLAIMS),
        "tokens_input": 1200,
        "tokens_output": 340,
        "cost_usd": 0.021,
        "latency_s": 182.84,
        "error": None,
        "interrupt": {
            "type": "HITL_READY",
            "gate": "hitl_gate",
            "approved": False,
            "resume_action": "not_taken",
        },
    }


def complete_batch_doc(batch_no: int) -> dict:
    start = (batch_no - 1) * long_run.JUDGE_BATCH_SIZE
    count = min(long_run.JUDGE_BATCH_SIZE, len(CLAIMS) - start)
    return {
        "report_sha256": long_run.sha256_text(REPORT),
        "sources_sha256": long_run.canonical_hash(SOURCES),
        "claim_sha256": long_run.canonical_hash(CLAIMS[start : start + count]),
        "expected_count": count,
        "judged_count": count,
        "unjudged_count": 0,
        "yes_count": count,
        "no_count": 0,
        "status": "complete",
        "reason_hint": None,
        "verdicts": {
            str(start + i): {"supported": True, "served_model": "scripted-judge"}
            for i in range(count)
        },
        "served_model": "scripted-judge",
        "application_level_calls": 1,
        "elapsed_ms": 1.0,
        "error_type": None,
        "http_status": None,
    }


# ── 1. Explicit stage deadlines; the 420 s hard cap is never raised ───────────────


def test_default_deadlines_leave_room_for_a_cooperative_stop():
    d = long_run.Deadlines()
    assert d.hard_cap_s == 420.0
    assert d.total_s + d.stop_grace_s <= d.hard_cap_s
    assert d.research_s <= d.total_s and d.judge_s <= d.total_s
    assert d.request_s <= d.judge_s
    assert d.as_dict()["worker_hard_cap_s"] == 420.0


@pytest.mark.parametrize(
    "env",
    [
        {"RF_EVAL_TOTAL_DEADLINE_S": "410"},  # 410 + 15 > 420
        {"RF_EVAL_TOTAL_DEADLINE_S": "1000"},
        {"RF_EVAL_JUDGE_REQUEST_TIMEOUT_S": "400"},  # > shared total budget
        {"RF_EVAL_RESEARCH_DEADLINE_S": "400"},  # > total budget
        {"RF_EVAL_TOTAL_DEADLINE_S": "not-a-number"},
    ],
)
def test_budgets_that_would_overrun_the_worker_are_refused(monkeypatch, env):
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    with pytest.raises(long_run.ReliabilityError):
        long_run.Deadlines.from_env()


def test_budget_checks_both_the_stage_and_the_overall_cap():
    clock = Clock()
    d = long_run.Deadlines(research_s=10, judge_s=5, total_s=12, request_s=2, stop_grace_s=1)
    budget = long_run.Budget(deadlines=d, clock=clock.now)
    assert budget.check("judge") == 5
    budget.spend("judge", 5)
    with pytest.raises(long_run.StageDeadlineExceeded) as exc:
        budget.check("judge")
    assert exc.value.stage == "judge"
    clock.t = 13
    with pytest.raises(long_run.StageDeadlineExceeded) as exc:
        budget.check("research")
    assert exc.value.stage == "research"


async def test_a_slow_request_raises_a_named_stage_deadline(monkeypatch):
    clock = Clock()
    d = long_run.Deadlines(judge_s=5, request_s=2, total_s=390)
    budget = long_run.Budget(deadlines=d, clock=clock.now)

    def fake_wait_for(awaitable, timeout):
        if hasattr(awaitable, "close"):
            awaitable.close()
        raise TimeoutError("controlled slow request")

    monkeypatch.setattr(long_run, "_wait_for", fake_wait_for)
    with pytest.raises(long_run.StageDeadlineExceeded) as exc:
        await long_run.run_request(asyncio.sleep(0), budget=budget)
    assert exc.value.stage == "judge_request"


# ── 2. Provider throttle, Retry-After, and recorded wait reasons ──────────────────


def test_retry_after_accepts_seconds_http_date_and_bounds_the_value(monkeypatch):
    monkeypatch.setenv("RF_EVAL_JUDGE_RETRY_AFTER_MAX_S", "9")

    class RateLimited(Exception):
        headers = {"Retry-After": "7"}

    assert long_run.retry_after_seconds(RateLimited()) == 7

    class InThePast(Exception):
        headers = {"retry-after": "Wed, 21 Oct 2015 07:28:00 GMT"}

    assert long_run.retry_after_seconds(InThePast()) == 0

    class TooLong(Exception):
        headers = {"Retry-After": "120"}

    assert long_run.retry_after_seconds(TooLong()) == 120
    assert long_run.retry_after_seconds(Exception()) is None
    assert long_run.retry_after_seconds(None) is None


async def test_no_wait_at_all_when_the_provider_needs_no_throttle(monkeypatch, tmp_path):
    clock = Clock()
    ctx, store = make_context(tmp_path, clock)
    store.save_draft(CASE_ID, draft_doc())
    judge = ScriptedJudge()
    monkeypatch.setattr(harness, "_judge_llm", lambda: judge)
    with long_run.active(ctx):
        rate, rows = await harness.judge_citation_support(REPORT, SOURCES)
    assert rate == 1.0 and len(rows) == len(CLAIMS)
    assert judge.calls == TOTAL_BATCHES
    assert clock.sleeps == []
    assert ctx.last_judge_status == long_run.JUDGE_COMPLETED
    assert (
        list((store.batches_dir).glob("batch-*.json"))
        and len(list((store.batches_dir).glob("batch-*.json"))) == TOTAL_BATCHES
    )
    section = long_run.reliability_section(ctx, status=ctx.last_judge_status, reason=None)
    assert section["total_wait_s"] == 0
    assert ctx.batches_completed == TOTAL_BATCHES


async def test_throttle_waits_only_before_a_real_next_request(monkeypatch, tmp_path):
    clock = Clock()
    ctx, store = make_context(tmp_path, clock, min_interval_s=4.0)
    store.save_draft(CASE_ID, draft_doc())
    judge = ScriptedJudge()
    monkeypatch.setattr(harness, "_judge_llm", lambda: judge)
    trace_dir = tmp_path / "judge-trace"
    with long_run.active(ctx):
        with trace_judge_case(case_id=CASE_ID, trace_dir=trace_dir):
            await harness.judge_citation_support(REPORT, SOURCES)
    # 3 requests -> 2 inter-request waits, never one after the final batch.
    assert clock.sleeps == [4.0, 4.0]
    events = [json.loads(line) for line in next(trace_dir.glob("*.jsonl")).read_text().splitlines()]
    waits = [e for e in events if e["event"] == "wait_started"]
    assert len(waits) == 2
    assert {w["wait_reason"] for w in waits} == {"provider_throttle"}
    assert {w["provider"] for w in waits} == {"fake-provider"}
    assert [w["wait_seconds"] for w in waits] == [4.0, 4.0]
    records = [w for w in ctx.throttle.records if w.reason != "not_required"]
    assert [r.batch_no for r in records] == [1, 2]
    assert all(r.actual_s == 4.0 and r.provider == "fake-provider" for r in records)
    # Every wait is visible in progress.json too, not only in the JSONL trace.
    progress = json.loads((store.dir / "progress.json").read_text())
    assert len([w for w in progress["waits"] if w["reason"] != "not_required"]) == 2


class RateLimited429(Exception):
    status_code = 429
    headers = {"Retry-After": "7"}


async def test_429_paces_the_following_batch_without_resending_the_failed_one(
    monkeypatch, tmp_path
):
    clock = Clock()
    ctx, store = make_context(tmp_path, clock)
    store.save_draft(CASE_ID, draft_doc())
    judge = ScriptedJudge(faults={1: RateLimited429("rate limited")})
    monkeypatch.setattr(harness, "_judge_llm", lambda: judge)
    trace_dir = tmp_path / "judge-429"
    with long_run.active(ctx):
        with trace_judge_case(case_id=CASE_ID, trace_dir=trace_dir):
            rate, rows = await harness.judge_citation_support(REPORT, SOURCES)
    assert judge.calls == TOTAL_BATCHES, "the 429 batch must never be re-requested"
    assert clock.sleeps == [7.0]
    events = [json.loads(line) for line in next(trace_dir.glob("*.jsonl")).read_text().splitlines()]
    wait = next(e for e in events if e["event"] == "wait_started")
    assert wait["wait_reason"] == "retry_after"
    assert wait["retry_after_seconds"] == 7.0
    assert wait["batch_no"] == 1
    # The failed batch's 4 claims stay UNJUDGED, excluded from the denominator;
    # the remaining 4 + 1 claims are the only ones that can be measured.
    assert sum(r["judged"] for r in rows) == 5
    assert rate == 1.0
    assert ctx.last_judge_status == long_run.JUDGE_PARTIAL
    assert ctx.exit_code == long_run.EXIT_JUDGE_PARTIAL
    # An incomplete batch is a diagnostic attempt, never a resumable score.
    assert not list(store.batches_dir.glob("batch-0001.json"))
    assert len(list(store.batches_dir.glob("batch-0001.attempt-*.json"))) == 1
    assert len(list(store.batches_dir.glob("batch-0002.json"))) == 1


async def test_empty_reply_stays_unmeasured_and_is_re_scorable(monkeypatch, tmp_path):
    clock = Clock()
    ctx, store = make_context(tmp_path, clock)
    store.save_draft(CASE_ID, draft_doc())
    judge = ScriptedJudge(faults={1: ""})
    monkeypatch.setattr(harness, "_judge_llm", lambda: judge)
    with long_run.active(ctx):
        rate, rows = await harness.judge_citation_support(REPORT, SOURCES)
    assert rate == 1.0 and sum(r["judged"] for r in rows) == 5
    assert ctx.last_judge_status == long_run.JUDGE_PARTIAL
    assert not list(store.batches_dir.glob("batch-0001.json"))
    assert len(list(store.batches_dir.glob("batch-0001.attempt-*.json"))) == 1


async def test_timeout_during_scoring_keeps_completed_batches_and_stops(monkeypatch, tmp_path):
    clock = Clock()
    ctx, store = make_context(tmp_path, clock)
    store.save_draft(CASE_ID, draft_doc())
    judge = ScriptedJudge()

    real_wait_for = long_run._wait_for
    calls = {"n": 0}

    def flaky_wait_for(awaitable, timeout):
        calls["n"] += 1
        if calls["n"] == 2:
            if hasattr(awaitable, "close"):
                awaitable.close()
            raise TimeoutError("controlled slow request")
        return real_wait_for(awaitable, timeout)

    monkeypatch.setattr(long_run, "_wait_for", flaky_wait_for)
    monkeypatch.setattr(harness, "_judge_llm", lambda: judge)
    with long_run.active(ctx):
        rate, rows = await harness.judge_citation_support(REPORT, SOURCES)
    assert judge.calls == 1, "no further provider call after the request deadline"
    assert rate == 1.0 and sum(r["judged"] for r in rows) == 4
    assert ctx.last_judge_status == long_run.JUDGE_PARTIAL
    assert len(list(store.batches_dir.glob("batch-0001.json"))) == 1
    assert not list(store.batches_dir.glob("batch-0003.json"))
    final = json.loads((store.dir / "final-status.json").read_text())
    assert final["status"] == long_run.JUDGE_PARTIAL
    assert "judge_deadline" in final["reason"]


async def test_cancellation_during_a_request_flushes_a_named_partial(monkeypatch, tmp_path):
    clock = Clock()
    ctx, store = make_context(tmp_path, clock)
    store.save_draft(CASE_ID, draft_doc())

    class Judge:
        calls = 0

        async def ainvoke(self, messages):
            self.calls += 1
            if self.calls == 2:
                raise asyncio.CancelledError()
            return await ScriptedJudge().ainvoke(messages)

    monkeypatch.setattr(harness, "_judge_llm", lambda: Judge())
    with long_run.active(ctx):
        rate, rows = await harness.judge_citation_support(REPORT, SOURCES)
    # Batch 1 was already atomically saved; the cancelled batch never becomes a zero.
    assert rate == 1.0 and sum(r["judged"] for r in rows) == 4
    assert ctx.last_judge_status == long_run.JUDGE_PARTIAL
    assert ctx.exit_code == long_run.EXIT_JUDGE_PARTIAL
    final = json.loads((store.dir / "final-status.json").read_text())
    assert final["status"] == long_run.JUDGE_PARTIAL
    assert final["reason"] == "cancelled"
    assert len(list(store.batches_dir.glob("batch-0001.json"))) == 1
    assert len(list(store.batches_dir.glob("batch-0002.attempt-*.json"))) == 1


# ── 3. Atomic, write-once artifacts and resume validation ─────────────────────────


def test_atomic_create_never_overwrites_a_saved_batch(tmp_path):
    path = tmp_path / "batch-0001.json"
    long_run.atomic_create(path, {"score": 1})
    with pytest.raises(long_run.ArtifactConflict):
        long_run.atomic_create(path, {"score": 0})
    assert json.loads(path.read_text())["score"] == 1


def test_only_completed_batches_are_restored(tmp_path):
    cfg = config()
    store = long_run.ProgressStore(tmp_path / "run1", config=cfg)
    store.preflight()
    store.save_draft(CASE_ID, draft_doc())
    store.save_batch(1, complete_batch_doc(1))
    store.save_batch(
        2,
        {
            **complete_batch_doc(2),
            "status": "unjudged",
            "judged_count": 0,
            "unjudged_count": 4,
            "verdicts": {},
        },
    )

    resumed = long_run.ProgressStore(tmp_path / "run2", config=cfg, source_dir=store.dir).load(
        case_id=CASE_ID
    )
    assert list(resumed.batches) == [1], "an unfinished batch must be re-scored, not replayed"
    assert len(list(resumed.batches[1]["verdicts"])) == 4


@pytest.mark.parametrize(
    "mutate",
    ["judge_config_version", "config_fingerprint", "report", "sources", "batch"],
)
def test_resume_refuses_a_changed_report_or_judge_configuration(tmp_path, mutate):
    cfg = config()
    store = long_run.ProgressStore(tmp_path / "run1", config=cfg)
    store.preflight()
    store.save_draft(CASE_ID, draft_doc())
    store.save_batch(1, complete_batch_doc(1))
    if mutate == "judge_config_version":
        path = store.dir / "progress.json"
        doc = json.loads(path.read_text())
        doc["judge_config_version"] = "rf-judge-reliability/0"
        path.write_text(json.dumps(doc))
    elif mutate == "config_fingerprint":
        path = store.dir / "progress.json"
        doc = json.loads(path.read_text())
        doc["config_fingerprint"] = "changed"
        path.write_text(json.dumps(doc))
    elif mutate == "report":
        path = store.draft_path(CASE_ID)
        doc = json.loads(path.read_text())
        doc["draft_report"] += "\nA new claim [1]."
        path.write_text(json.dumps(doc))
    elif mutate == "sources":
        path = store.draft_path(CASE_ID)
        doc = json.loads(path.read_text())
        doc["sources"] = [{"index": 1, "snippet": "different"}]
        path.write_text(json.dumps(doc))
    else:
        path = store.batches_dir / "batch-0001.json"
        doc = json.loads(path.read_text())
        doc["config_fingerprint"] = "changed"
        path.write_text(json.dumps(doc))
    with pytest.raises(long_run.ResumeRejected):
        long_run.ProgressStore(tmp_path / "run2", config=cfg, source_dir=store.dir).load(
            case_id=CASE_ID
        )


def test_build_config_is_allowlisted_and_has_no_secret(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "SECRET-must-not-be-fingerprinted")
    cfg = config()
    assert "SECRET" not in json.dumps(cfg)
    assert cfg["judge_config_version"] == long_run.JUDGE_CONFIG_VERSION
    assert cfg["judge_batch_size"] == long_run.JUDGE_BATCH_SIZE
    assert cfg["judge_generation"] == {"max_output_tokens": 4096}
    assert set(cfg) == {
        "judge_generation",
        "schema_version",
        "judge_config_version",
        "llm_mode",
        "models",
        "limits",
        "judge_route",
        "judge_prompt_sha256",
        "judge_batch_size",
        "case_id",
        "query",
        "depth",
        "phase",
        "prompt_overrides_sha256",
    }


# ── 4. Cooperative stop ───────────────────────────────────────────────────────────


def test_stop_child_prefers_cooperative_exit():
    sent, waited = [], []

    def wait(timeout):
        waited.append(timeout)
        return 3

    out = long_run.stop_child(send=sent.append, wait=wait, grace_s=15)
    assert sent == ["SIGTERM"] and waited == [15]
    assert out == {"mode": "cooperative", "forced": False, "exit_code": 3}


def test_stop_child_forces_only_after_the_grace_period():
    sent, waited = [], []

    def wait(timeout):
        waited.append(timeout)
        if timeout is not None:
            raise long_run.ChildStillRunning()
        return -9

    out = long_run.stop_child(send=sent.append, wait=wait, grace_s=15)
    assert sent == ["SIGTERM", "SIGKILL"] and waited == [15, None]
    assert out["forced"] is True and out["mode"] == "forced"


def test_worker_timeout_sends_sigterm_before_any_forced_kill(monkeypatch, tmp_path):
    """The 420 s cap is unchanged; only the way we stop inside it is cooperative."""
    import signal
    import subprocess

    from evals.agent_eval_ext import evaluation_pipeline_worker_v13 as worker

    sent: list[int] = []
    monkeypatch.setattr(worker.os, "killpg", lambda pid, sig: sent.append(sig), raising=False)

    class Proc:
        pid = 4242
        calls = 0

        def wait(self, timeout=None):
            self.calls += 1
            if self.calls == 1:
                raise subprocess.TimeoutExpired("cmd", timeout)
            return 3  # the child exited inside the cooperative grace period

    monkeypatch.setattr(worker.subprocess, "Popen", lambda *a, **k: Proc())
    result = tmp_path / "pipelines" / "p1" / "artifacts" / "raw_result.json"
    result.parent.mkdir(parents=True)
    assert worker._run_fixed("report-benchmark", result, 1) == 124
    assert sent == [signal.SIGTERM], "SIGKILL must not be used when the child cooperates"
    doc = json.loads((result.parent / "process.json").read_text())
    assert doc["stop_mode"] == "cooperative"
    assert doc["timed_out"] is True


async def test_cooperative_stop_does_not_start_another_paid_research_graph(monkeypatch, tmp_path):
    clock = Clock()
    ctx, _ = make_context(tmp_path, clock, name="stop-progress")
    ctx.stop.request("signal_SIGTERM")
    with long_run.active(ctx):
        row = await harness.run_one(QUERY)
    assert row["completed"] is False
    assert "cooperative_stop" in row["error"]


# ── 5. Draft persistence + breakpoint resume through run_one ──────────────────────


async def test_draft_is_saved_before_the_first_judge_call_with_an_unapproved_gate(
    monkeypatch, tmp_path
):
    clock = Clock()
    ctx, store = make_context(tmp_path, clock, name="draft-progress")
    seen: list[bool] = []
    judge = ScriptedJudge(on_call=lambda n: seen.append(store.draft_path(CASE_ID).is_file()))
    monkeypatch.setattr(harness, "_judge_llm", lambda: judge)
    with long_run.active(ctx):
        row = await harness.run_one(QUERY)
    assert seen and all(seen), "the draft must be on disk before the first judge call"
    assert row["completed"] is True
    assert row["research_state"] == long_run.RESEARCH_DRAFT_READY
    assert row["judge_status"] == long_run.JUDGE_COMPLETED
    draft = json.loads(store.draft_path(CASE_ID).read_text())
    assert draft["draft_report"] == REPORT
    assert draft["sources"] == SOURCES
    assert draft["evidence"] and draft["citations"]
    assert draft["tokens_input"] == 1200 and draft["tokens_output"] == 340
    assert draft["report_sha256"] == long_run.sha256_text(REPORT)
    assert draft["config_fingerprint"] == store.fingerprint
    assert draft["interrupt"]["approved"] is False
    assert draft["interrupt"]["resume_action"] == "not_taken"
    assert draft["interrupt"]["type"] == "HITL_READY"


async def test_breakpoint_resume_skips_research_and_completed_batches(monkeypatch, tmp_path):
    first_clock = Clock()
    first_ctx, first_store = make_context(tmp_path, first_clock, name="run1")
    first_graph_calls: list[str] = []
    monkeypatch.setattr(harness, "build_graph", lambda *_: FakeGraph(first_graph_calls))

    def stop_after_first_batch(call_no):
        if call_no == 1:
            first_ctx.stop.request("test_breakpoint")

    monkeypatch.setattr(
        harness, "_judge_llm", lambda: ScriptedJudge(on_call=stop_after_first_batch)
    )
    with long_run.active(first_ctx):
        first_row = await harness.run_one(QUERY)
    assert first_graph_calls == ["ainvoke"], "the first run really did run the research graph"
    assert first_row["judge_status"] == long_run.JUDGE_PARTIAL
    assert len(list(first_store.batches_dir.glob("batch-*.json"))) == 1

    # Second process: a fresh context, a fresh graph that must never be invoked.
    second_clock = Clock()
    second_ctx, second_store = make_context(
        tmp_path, second_clock, name="run2", resume_root=first_ctx.root
    )

    class MustNotRun:
        async def ainvoke(self, *args):
            raise AssertionError("research graph re-executed across processes")

        async def aget_state(self, *args):
            raise AssertionError("research graph re-executed across processes")

    monkeypatch.setattr(harness, "build_graph", lambda *_: MustNotRun())
    resume_judge = ScriptedJudge()
    monkeypatch.setattr(harness, "_judge_llm", lambda: resume_judge)
    trace_dir = tmp_path / "resume-trace"
    monkeypatch.setenv("RF_JUDGE_BATCH_TRACE_DIR", str(trace_dir))
    with long_run.active(second_ctx):
        second_row = await harness.run_one(QUERY)
    assert second_row["research_reused"] is True
    assert second_row["judge_status"] == long_run.JUDGE_COMPLETED
    assert second_ctx.research_reused == 1
    assert second_ctx.batches_restored == 1
    assert resume_judge.calls == TOTAL_BATCHES - 1, "the completed batch was replayed"
    assert len(second_row["claim_verdicts"]) == len(CLAIMS)
    assert all(r["judged"] for r in second_row["claim_verdicts"])
    # The restored batch is visible in the trace and costs no provider call.
    trace_path = next(trace_dir.glob("*.jsonl"))
    events = [json.loads(line) for line in trace_path.read_text().splitlines()]
    restored = [e for e in events if e["event"] == "batch_restored"]
    assert len(restored) == 1 and restored[0]["application_level_calls"] == 0
    assert summarize(trace_path)["restored_batches"] == 1
    assert list(second_store.batches_dir.glob("batch-0001.json")), "adopted evidence is present"


async def test_resume_refusal_makes_no_model_calls(monkeypatch, tmp_path):
    first_clock = Clock()
    first_ctx, first_store = make_context(tmp_path, first_clock, name="run1")
    monkeypatch.setattr(harness, "_judge_llm", lambda: ScriptedJudge())
    with long_run.active(first_ctx):
        await harness.run_one(QUERY)
    # Same progress, but a different judge prompt -> different config fingerprint.
    path = first_store.dir / "progress.json"
    doc = json.loads(path.read_text())
    doc["config"]["judge_prompt_sha256"] = "changed"
    path.write_text(json.dumps(doc))

    second_clock = Clock()
    second_ctx, _ = make_context(tmp_path, second_clock, name="run2", resume_root=first_ctx.root)
    monkeypatch.setattr(harness, "_judge_llm", lambda: ScriptedJudge())
    with long_run.active(second_ctx):
        with pytest.raises(long_run.ResumeRejected):
            await harness.run_one(QUERY)


# ── 6. Whole-run result: legacy fields, additive reliability, quality gate ─────────


async def test_partial_run_keeps_legacy_schema_and_does_not_clear_the_quality_gate(
    monkeypatch, tmp_path
):
    out = tmp_path / "raw_result.json"
    progress = tmp_path / "reliability"
    monkeypatch.setenv("RF_EVAL_RELIABILITY_DIR", str(progress))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "harness",
            "--mode",
            "report",
            "--case-id",
            CASE_ID,
            "--judge",
            "anthropic:claude-test-judge",
            "--out",
            str(out),
        ],
    )
    judge = ScriptedJudge(faults={1: RateLimited429("rate limited")})
    monkeypatch.setattr(harness, "_judge_llm", lambda: judge)

    with pytest.raises(SystemExit) as exc:
        await harness.main()
    assert exc.value.code == long_run.EXIT_JUDGE_PARTIAL

    payload = json.loads(out.read_text())
    for key in (
        "date",
        "generated_at",
        "llm_mode",
        "mode",
        "models",
        "method",
        "aggregate",
        "release_criteria",
        "results",
        "judge",
    ):
        assert key in payload, f"legacy field {key} was lost"
    assert payload["results"][0]["research_state"] == long_run.RESEARCH_DRAFT_READY
    assert payload["reliability"]["status"] == long_run.JUDGE_PARTIAL
    assert payload["reliability"]["batches_completed"] == TOTAL_BATCHES - 1
    assert payload["reliability"]["deadlines"]["worker_hard_cap_s"] == 420.0
    assert payload["reliability"]["throttle"]["provider"]
    assert payload["release_criteria"]["citation_support_ok"] is None
    root_status = long_run.read_root_status(progress)
    assert root_status["status"] == long_run.JUDGE_PARTIAL
    assert root_status["exit_code"] == long_run.EXIT_JUDGE_PARTIAL
