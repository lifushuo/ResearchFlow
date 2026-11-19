"""Run inside ResearchFlow worker environment; no external model calls."""

from __future__ import annotations

import asyncio
import json
from dataclasses import replace
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage

from evals import harness
from evals.agent_eval_ext.judge_runtime import trace_judge_case
from evals.agent_eval_ext.judge_runtime_inspect import summarize

REPORT = (
    "# Report\n\n## Findings\n"
    "- An alpha fact is present [1].\n"
    "- A beta fact is present [2].\n\n"
    "## Sources\n[1] https://example.org/1\n[2] https://example.org/2\n"
)
SOURCES = [
    {"index": 1, "snippet": "An alpha fact is present."},
    {"index": 2, "snippet": "There is no beta fact."},
]


class StubJudge:
    async def ainvoke(self, messages):
        return AIMessage(
            content="Claim 1: YES\nClaim 2: NO", response_metadata={"model_name": "mock-judge"}
        )


class PartialJudge:
    async def ainvoke(self, messages):
        return AIMessage(content="Claim 1: YES", response_metadata={"model_name": "mock-judge"})


async def test_real_judge_function_same_scores_with_observer_enabled(monkeypatch, tmp_path):
    monkeypatch.setattr(harness, "_judge_llm", lambda: StubJudge())

    async def no_sleep(_):
        return None

    monkeypatch.setattr(harness.asyncio, "sleep", no_sleep)
    without = await harness.judge_citation_support(REPORT, SOURCES)
    with trace_judge_case(case_id="example", trace_dir=tmp_path):
        observed = await harness.judge_citation_support(REPORT, SOURCES)
    assert observed == without
    traces = list(tmp_path.glob("*.jsonl"))
    assert len(traces) == 1
    rows = [json.loads(line) for line in traces[0].read_text().splitlines()]
    batch = next(e for e in rows if e["event"] == "batch_finished")
    assert batch["yes_count"] == 1
    assert batch["no_count"] == 1
    assert any(e["event"] == "batch_started" for e in rows)


async def test_partial_reply_stays_unjudged(monkeypatch, tmp_path):
    monkeypatch.setattr(harness, "_judge_llm", lambda: PartialJudge())

    async def no_sleep(_):
        return None

    monkeypatch.setattr(harness.asyncio, "sleep", no_sleep)
    with trace_judge_case(case_id="partial", trace_dir=tmp_path):
        rate, rows = await harness.judge_citation_support(REPORT, SOURCES)
    assert rate == 1.0
    assert [r["judged"] for r in rows] == [True, False]
    batch = read_batches(next(tmp_path.glob("*.jsonl")))[0]
    assert batch["unjudged_count"] == 1
    assert batch["status"] == "partial"


def read_events(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def read_batches(path):
    return [e for e in read_events(path) if e["event"] == "batch_finished"]


async def test_throttled_wait_is_observed_and_cancel_still_flushes_progress(monkeypatch, tmp_path):
    """The wait is no longer unconditional: it exists only before a *real* next
    request, is sized by the configured provider throttle, and records its reason.

    The old test pinned an unconditional 15 s sleep after the only batch. Here two
    batches are needed, the sleep is injected through the controllable clock, and
    cancelling it must still leave a named partial progress artifact.
    """
    from evals.agent_eval_ext import long_run

    class FullJudge:
        async def ainvoke(self, messages):
            return AIMessage(
                content="\n".join(f"Claim {i}: YES" for i in range(1, 5)),
                response_metadata={"model_name": "mock-judge"},
            )

    monkeypatch.setattr(harness, "RUN_CONFIG", replace(harness.RUN_CONFIG, llm_mode="real"))
    monkeypatch.setattr(harness, "_judge_llm", lambda: FullJudge())
    five_claims = "\n".join(f"- Fact {i} is present [1]." for i in range(1, 6))
    clock_value = [0.0]
    entered = asyncio.Event()

    async def block(seconds):
        assert seconds == 15
        clock_value[0] += seconds
        entered.set()
        await asyncio.Event().wait()

    ctx = long_run.build_context(
        root=tmp_path / "progress",
        judge_provider="fake-provider",
        clock=lambda: clock_value[0],
        sleep=block,
    )
    ctx.throttle.min_interval_s = 15.0
    long_run.preflight_root(ctx.root)
    config = long_run.build_config(
        run_config=harness.RUN_CONFIG,
        judge_route=harness.JUDGE_ROUTE,
        judge_prompt=harness.JUDGE_PROMPT,
        case_id="wait-cancel",
        query="synthetic report",
        depth="balanced",
    )
    store = ctx.store_for(config)
    store.preflight()
    from evals import metrics

    claim_lines = [c for c in metrics.claim_lines(five_claims) if metrics.CITE_RE.search(c)]
    store.save_draft(
        "wait-cancel",
        {
            "state": long_run.RESEARCH_DRAFT_READY,
            "draft_report": five_claims,
            "sources": SOURCES,
            "claim_lines": claim_lines,
            "report_sha256": long_run.sha256_text(five_claims),
            "sources_sha256": long_run.canonical_hash(SOURCES),
            "claims_sha256": long_run.canonical_hash(claim_lines),
        },
    )
    ctx.store = store

    async def run():
        with long_run.active(ctx):
            with trace_judge_case(case_id="wait-cancel", trace_dir=tmp_path):
                await harness.judge_citation_support(five_claims, SOURCES)

    task = asyncio.create_task(run())
    await asyncio.wait_for(entered.wait(), timeout=5)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    rows = read_events(next(tmp_path.glob("*.jsonl")))
    assert [e["event"] for e in rows] == [
        "case_start",
        "batch_started",
        "batch_finished",
        "wait_started",
        "wait_finished",
        "case_end",
    ]
    assert rows[3]["wait_seconds"] == 15
    assert rows[3]["wait_reason"] == "provider_throttle"
    assert rows[3]["provider"] == "fake-provider"
    assert rows[4]["status"] == "cancelled"
    # Only batch 1 ever reached the provider; batch 2 was never requested.
    assert len(read_batches(next(tmp_path.glob("*.jsonl")))) == 1
    summary = summarize(next(tmp_path.glob("*.jsonl")))
    assert summary["trace_complete"] is True and summary["observed_application_level_calls"] == 1
    assert (
        json.loads((ctx.root / "cases/wait-cancel/progress.json").read_text())["status"]
        == long_run.JUDGE_PARTIAL
    )
    assert (
        json.loads((ctx.root / "cases/wait-cancel/final-status.json").read_text())["reason"]
        == "cancelled_during_throttle"
    )
    assert list((ctx.root / "cases/wait-cancel/batches").glob("batch-0001.json"))


@pytest.fixture
def fast_judge(monkeypatch):
    async def no_sleep(_):
        pass

    monkeypatch.setattr(harness.asyncio, "sleep", no_sleep)
    monkeypatch.setattr(harness, "RUN_CONFIG", replace(harness.RUN_CONFIG, llm_mode="real"))


@pytest.mark.parametrize(
    "content,reason",
    [
        ("", "empty_or_nontext_response"),
        ([{"type": "text", "text": "Claim 1: YES"}], "empty_or_nontext_response"),
        ("Answer is yes", "unparseable_response"),
        ("Claim 1: YES", "partial_response"),
        ("Claim 1: YES\nClaim 2: NO", None),
    ],
)
async def test_actual_parser_outputs_unchanged_across_response_shapes(
    monkeypatch, tmp_path, fast_judge, content, reason
):
    class Judge:
        calls = 0

        async def ainvoke(self, messages):
            self.calls += 1
            return AIMessage(content=content, response_metadata={"model_name": "scripted-judge"})

    judge = Judge()
    monkeypatch.setattr(harness, "_judge_llm", lambda: judge)
    original = await harness.judge_citation_support(REPORT, SOURCES)
    with trace_judge_case(case_id="shapes", trace_dir=tmp_path):
        observed = await harness.judge_citation_support(REPORT, SOURCES)
    assert original == observed and judge.calls == 2
    event = read_batches(next(tmp_path.glob("*.jsonl")))[0]
    assert event["reason_hint"] == reason
    assert event["application_level_calls"] == 1 and event["sdk_retry_attempts"] is None


async def test_429_is_observed_without_extra_retry(monkeypatch, tmp_path, fast_judge):
    class RateLimited(Exception):
        status_code = 429

    class Judge:
        calls = 0

        async def ainvoke(self, messages):
            self.calls += 1
            raise RateLimited("SECRET-provider-message")

    judge = Judge()
    monkeypatch.setattr(harness, "_judge_llm", lambda: judge)
    with trace_judge_case(case_id="429", trace_dir=tmp_path):
        rate, rows = await harness.judge_citation_support(REPORT, SOURCES)
    assert rate is None and not any(r["judged"] for r in rows)
    assert judge.calls == 1
    path = next(tmp_path.glob("*.jsonl"))
    event = read_batches(path)[0]
    assert (event["http_status"], event["reason_hint"]) == (429, "invoke_exception")
    assert "SECRET" not in path.read_text()


async def test_returned_none_is_processing_exception(monkeypatch, tmp_path, fast_judge):
    class Judge:
        async def ainvoke(self, messages):
            return None

    monkeypatch.setattr(harness, "_judge_llm", Judge)
    with trace_judge_case(case_id="none", trace_dir=tmp_path):
        rate, rows = await harness.judge_citation_support(REPORT, SOURCES)
    event = read_batches(next(tmp_path.glob("*.jsonl")))[0]
    assert event["response_received"] is True
    assert event["reason_hint"] == "response_processing_exception"
    assert rate is None and len(rows) == 2


async def test_actual_cancellation_propagates_and_closes_trace(monkeypatch, tmp_path, fast_judge):
    started, block = asyncio.Event(), asyncio.Event()

    class Judge:
        async def ainvoke(self, messages):
            started.set()
            await block.wait()

    monkeypatch.setattr(harness, "_judge_llm", Judge)

    async def run():
        with trace_judge_case(case_id="cancel", trace_dir=tmp_path):
            await harness.judge_citation_support(REPORT, SOURCES)

    task = asyncio.create_task(run())
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    path = next(tmp_path.glob("*.jsonl"))
    rows = read_events(path)
    assert (
        next(e for e in rows if e["event"] == "batch_finished")["status"]
        == rows[-1]["status"]
        == "cancelled"
    )
    assert summarize(path)["application_level_calls"] == 1


async def test_out_of_range_labels_are_observed_without_polluting_next_batch(
    monkeypatch, tmp_path, fast_judge
):
    report = "\n".join(f"- Fact number {i} is present [1]." for i in range(1, 6))

    class Judge:
        calls = 0

        async def ainvoke(self, messages):
            self.calls += 1
            text = "\n".join(f"Claim {i}: YES" for i in range(1, 6)) if self.calls % 2 else ""
            return AIMessage(content=text)

    judge = Judge()
    monkeypatch.setattr(harness, "_judge_llm", lambda: judge)
    without = await harness.judge_citation_support(report, SOURCES)
    with trace_judge_case(case_id="out-of-range", trace_dir=tmp_path):
        observed = await harness.judge_citation_support(report, SOURCES)
    assert observed == without
    batches = read_batches(next(tmp_path.glob("*.jsonl")))
    assert batches[0]["out_of_range_labels"] == 1
    assert batches[1]["expected_count"] == 1
    assert batches[1]["preexisting_judged_count"] == 0
    assert batches[1]["recognizable_labels"] == 0
    assert observed[1][-1]["judged"] is False


async def test_run_one_phase_paths_and_failed_trace_preserve_results(
    monkeypatch, tmp_path, fast_judge
):
    from research_engine import eval_trace

    class Graph:
        async def ainvoke(self, *args):
            pass

        async def aget_state(self, *args):
            return SimpleNamespace(values={"draft_report": REPORT, "sources": SOURCES})

    class Judge:
        async def ainvoke(self, messages):
            assert eval_trace._ACTIVE.get() is None
            return AIMessage(content="Claim 1: YES\nClaim 2: NO")

    monkeypatch.setattr(harness, "build_graph", lambda *_: Graph())
    monkeypatch.setattr(harness, "_judge_llm", Judge)
    monkeypatch.delenv("RF_JUDGE_BATCH_TRACE_DIR", raising=False)
    query = {"id": "host-boundary", "query": "synthetic report"}
    disabled = await harness.run_one(query)
    assert not any(key.startswith("judge_batch_trace") for key in disabled)
    monkeypatch.setenv("RF_JUDGE_BATCH_TRACE_DIR", str(tmp_path / "judge"))
    baseline = await harness.run_one(query, trace_artifact_root=tmp_path)
    candidate = await harness.run_one(query, prompt_overrides={}, trace_artifact_root=tmp_path)
    assert baseline["claim_verdicts"] == candidate["claim_verdicts"] == disabled["claim_verdicts"]
    assert baseline["judge_batch_trace"] != candidate["judge_batch_trace"]
    for result, phase in ((baseline, "baseline"), (candidate, "candidate")):
        path = tmp_path / result["judge_batch_trace"]
        assert summarize(path)["phase"] == phase
        assert result["judge_batch_trace_status"] == "recorded"
    blocked = tmp_path / "blocked"
    blocked.write_text("not a directory")
    monkeypatch.setenv("RF_JUDGE_BATCH_TRACE_DIR", str(blocked / "child"))
    failed = await harness.run_one(query)
    assert failed["judge_batch_trace_status"] == "unavailable"
    assert failed["claim_verdicts"] == disabled["claim_verdicts"]


async def test_write_failure_inside_actual_judge_keeps_scores(monkeypatch, tmp_path, fast_judge):
    class Judge:
        async def ainvoke(self, messages):
            return AIMessage(content="Claim 1: YES\nClaim 2: NO")

    class BrokenFlush:
        def __init__(self, stream):
            self.stream = stream

        def write(self, text):
            self.stream.write(text)

        def flush(self):
            raise OSError("controlled write failure")

        def close(self):
            self.stream.close()

    monkeypatch.setattr(harness, "_judge_llm", Judge)
    original = await harness.judge_citation_support(REPORT, SOURCES)
    with trace_judge_case(case_id="write-fail", trace_dir=tmp_path) as rec:
        rec._handle = BrokenFlush(rec._handle)
        observed = await harness.judge_citation_support(REPORT, SOURCES)
    assert observed == original and rec.failed


async def test_diagnostic_property_failure_does_not_mask_score(
    monkeypatch, tmp_path, fast_judge, caplog
):
    class Response:
        reads = 0
        response_metadata = {"model_name": "scripted"}

        @property
        def content(self):
            self.reads += 1
            if self.reads > 2:
                raise RuntimeError("SECRET-property")
            return "Claim 1: YES\nClaim 2: NO"

    class Judge:
        async def ainvoke(self, messages):
            return Response()

    monkeypatch.setattr(harness, "_judge_llm", Judge)
    without = await harness.judge_citation_support(REPORT, SOURCES)
    with trace_judge_case(case_id="metadata-failure", trace_dir=tmp_path):
        observed = await harness.judge_citation_support(REPORT, SOURCES)
    assert observed == without
    assert "SECRET" not in caplog.text
    assert read_batches(next(tmp_path.glob("*.jsonl")))[0]["response_content_kind"] == "unavailable"


async def test_parallel_actual_judges_have_separate_identities(monkeypatch, tmp_path, fast_judge):
    ready = asyncio.Event()
    calls = []

    class Judge:
        async def ainvoke(self, messages):
            calls.append(1)
            if len(calls) == 2:
                ready.set()
            await ready.wait()
            return AIMessage(content="Claim 1: YES\nClaim 2: NO")

    monkeypatch.setattr(harness, "_judge_llm", Judge)

    async def run(case):
        with trace_judge_case(case_id=case, trace_dir=tmp_path):
            return await harness.judge_citation_support(REPORT, SOURCES)

    a, b = await asyncio.gather(run("a"), run("b"))
    assert a == b
    summaries = [summarize(p) for p in tmp_path.glob("*.jsonl")]
    assert {s["case_id"] for s in summaries} == {"a", "b"}
    assert all(s["application_level_calls"] == 1 and s["trace_complete"] for s in summaries)
