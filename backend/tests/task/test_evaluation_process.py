"""Full-input metrics, shared deadlines and durable cross-process batch ownership."""

import asyncio
import json
import subprocess
import sys
from dataclasses import replace

import pytest
import test_long_run_reliability as f
from langchain_core.messages import AIMessage

from evals import harness, metrics
from evals.agent_eval_ext import long_run
from evals.agent_eval_ext.batch_ledger import BatchLease
from evals.agent_eval_ext.judge_runtime import trace_judge_batch, trace_judge_case
from evals.agent_eval_ext.judge_runtime_inspect import summarize

isolated_env = f.isolated_env


def saved_context(*args, **kwargs):
    ctx, store = f.make_context(*args, **kwargs)
    if kwargs.get("resume_root") is None:
        store.save_draft(f.CASE_ID, f.draft_doc())
    return ctx, store


def test_full_input_and_attempted_coverage_are_distinct(tmp_path):
    with trace_judge_case(case_id="large", trace_dir=tmp_path, total_claims=74) as trace:
        for number in range(1, 16):
            start = (number - 1) * 4
            good = number < 15
            trace_judge_batch(
                batch_no=number,
                batch_start=start,
                expected_count=4,
                verdict_by_claim={i: True for i in range(start, start + 4)} if good else {},
                response=AIMessage(content="") if good else None,
                text="",
                error=None if good else TimeoutError(),
                elapsed_ms=1,
                response_received=good,
            )
    s = summarize(trace.path)
    assert (s["successful_batches"], s["attempted_batches"], s["total_batches"]) == (14, 15, 19)
    assert (s["judged_claims_total"], s["total_claims"], s["unjudged_claims_total"]) == (56, 74, 18)
    assert s["full_judge_coverage"] == pytest.approx(56 / 74)
    assert s["attempted_judge_coverage"] == pytest.approx(56 / 60)
    assert s["successful_requests"] == 14 and s["new_requested_batches"] == 15
    assert s["trace_complete"] and not s["scoring_complete"]


def test_unknown_historical_full_input_is_never_inferred(tmp_path):
    with trace_judge_case(case_id="old", trace_dir=tmp_path) as trace:
        trace_judge_batch(
            batch_no=1,
            batch_start=0,
            expected_count=1,
            verdict_by_claim={0: True},
            response=AIMessage(content="Claim 1: YES"),
            text="Claim 1: YES",
            error=None,
            elapsed_ms=1,
        )
    old = summarize(trace.path)
    assert old["judge_coverage"] is None and old["total_batches"] is None
    assert old["attempted_judge_coverage"] == 1
    assert summarize(trace.path, total_claims=5)["full_judge_coverage"] == 0.2


def test_returned_response_is_not_a_valid_score(tmp_path):
    with trace_judge_case(case_id="empty", trace_dir=tmp_path, total_claims=4) as trace:
        trace_judge_batch(
            batch_no=1,
            batch_start=0,
            expected_count=4,
            verdict_by_claim={},
            response=AIMessage(content=""),
            text="",
            error=None,
            elapsed_ms=1,
        )
    s = summarize(trace.path)
    assert s["successful_requests"] == 1 and s["valid_result_batches"] == 0
    assert s["judged_claims_total"] == 0 and not s["scoring_complete"]


def test_quality_gate_requires_full_input_even_when_support_is_perfect():
    agg = harness.aggregate(
        [
            {
                "completed": True,
                "claim_verdicts": [{"judged": True}] * 56 + [{"judged": False}] * 18,
                "citation_support_rate": 1.0,
            }
        ]
    )
    result = harness.check_release_criteria(agg)
    assert result["citation_support_ok"] is True
    assert result["full_judge_coverage_ok"] is False and result["gate_passed"] is False


@pytest.mark.parametrize("research", [10, 171, 239])
def test_judge_can_use_remaining_shared_deadline(monkeypatch, research):
    d = long_run.Deadlines.from_env()
    clock = f.Clock()
    budget = long_run.Budget(d, clock=clock.now)
    budget.start_case(1)
    clock.t = research
    budget.spend("research", research)
    assert d.judge_policy == "shared_remaining"
    assert budget.remaining("judge") == 390 - research
    clock.t = 390
    assert budget.remaining("judge") == 0


def test_explicit_stage_limit_is_still_honoured(monkeypatch):
    monkeypatch.setenv("RF_EVAL_JUDGE_DEADLINE_S", "140")
    budget = long_run.Budget(long_run.Deadlines.from_env(), clock=lambda: 171, started=0)
    assert budget.remaining("judge") == 140


def test_multi_case_reserves_future_cases_and_rolls_unused_time_forward():
    clock = f.Clock()
    b = long_run.Budget(long_run.Deadlines.from_env(case_count=2), clock=clock.now)
    b.start_case(2)
    assert b.remaining("judge") == 345
    clock.t = 200
    b.start_case(1)
    assert b.remaining("judge") == 490
    clock.t = 691
    assert b.remaining("judge") == 0


@pytest.mark.parametrize("shared", [False, True])
async def test_identical_fixed_workload_uses_unused_budget_without_extra_calls(
    tmp_path, monkeypatch, shared
):
    clock = f.Clock()
    ctx, store = saved_context(tmp_path, clock)
    report = "# Findings\n" + "\n".join(f"Fact item {n} is documented [1]." for n in range(74))
    claims = [c for c in metrics.claim_lines(report) if metrics.CITE_RE.search(c)]
    assert len(claims) == 74
    doc = f.draft_doc()
    doc.update(
        draft_report=report,
        claim_lines=claims,
        report_sha256=long_run.sha256_text(report),
        claims_sha256=long_run.canonical_hash(claims),
    )
    store.save_draft(f.CASE_ID, doc)
    ctx.budget.deadlines = replace(ctx.deadlines, judge_s=390 if shared else 140)
    clock.t = 171
    ctx.budget.spend("research", 171)
    judge = f.ScriptedJudge(on_call=lambda _: setattr(clock, "t", clock.t + 10))
    monkeypatch.setattr(harness, "_judge_llm", lambda: judge)
    with long_run.active(ctx):
        _, rows = await harness.judge_citation_support(report, f.SOURCES)
    assert sum(r["judged"] for r in rows) == (74 if shared else 56)
    assert judge.calls == (19 if shared else 14)
    assert clock.t <= 390


def test_os_lock_prevents_second_process_from_claiming_same_batch(tmp_path):
    code = "from pathlib import Path;from evals.agent_eval_ext.batch_ledger import BatchLease,BatchBusy;import sys\ntry:\n with BatchLease(Path(sys.argv[1]),'same'): pass\nexcept BatchBusy: sys.exit(17)"
    with BatchLease(tmp_path, "same"):
        child = subprocess.run([sys.executable, "-c", code, str(tmp_path)], capture_output=True)
        assert child.returncode == 17, child.stderr
    assert (
        subprocess.run([sys.executable, "-c", code, str(tmp_path)], capture_output=True).returncode
        == 0
    )


def test_process_death_releases_ownership_but_discloses_uncertain_charge(tmp_path):
    code = "from pathlib import Path;from evals.agent_eval_ext.batch_ledger import BatchLease;import sys,os\nl=BatchLease(Path(sys.argv[1]),'crash');l.__enter__();l.record('response_received');os._exit(7)"
    assert (
        subprocess.run([sys.executable, "-c", code, str(tmp_path)], capture_output=True).returncode
        == 7
    )
    with BatchLease(tmp_path, "crash") as lease:
        assert lease.uncertain_previous_request and lease.committed is None


async def test_committed_batch_survives_crash_before_local_copy(tmp_path, monkeypatch):
    clock = f.Clock()
    ctx, source = saved_context(tmp_path, clock, name="original")
    lease = source.batch_lease(1)
    with lease:
        source.lease = lease
        original = long_run.atomic_create

        def fail_local(path, payload):
            if path.parent == source.batches_dir:
                raise OSError("disk failed after ledger commit")
            original(path, payload)

        monkeypatch.setattr(long_run, "atomic_create", fail_local)
        with pytest.raises(OSError):
            source.save_batch(1, f.complete_batch_doc(1))
        monkeypatch.setattr(long_run, "atomic_create", original)
    resumed, store = saved_context(tmp_path, clock, name="resumed", resume_root=ctx.root)
    store.load(case_id=f.CASE_ID)
    store.adopt_resume()
    judge = f.ScriptedJudge()
    monkeypatch.setattr(harness, "_judge_llm", lambda: judge)
    with long_run.active(resumed):
        _, rows = await harness.judge_citation_support(f.REPORT, f.SOURCES)
    assert all(r["judged"] for r in rows) and judge.calls == 2
    assert len(list(store.batches_dir.glob("batch-????.json"))) == 3


@pytest.mark.parametrize(
    "field,value",
    [
        ("evidence", [{"text": "changed"}]),
        ("sources", [{"index": 1, "snippet": "changed"}]),
        ("draft_report", "changed"),
    ],
)
def test_changed_saved_inputs_refuse_resume(tmp_path, field, value):
    _, store = saved_context(tmp_path, f.Clock())
    p = store.draft_path(f.CASE_ID)
    doc = json.loads(p.read_text())
    doc[field] = value
    p.write_text(json.dumps(doc))
    with pytest.raises(long_run.ResumeRejected):
        store.load(case_id=f.CASE_ID)


@pytest.mark.parametrize(
    "field,value",
    [("judge_route", "custom:other"), ("judge_prompt_sha256", "other"), ("judge_batch_size", 8)],
)
def test_changed_judge_contract_refuses_saved_scores(tmp_path, field, value):
    _, store = saved_context(tmp_path, f.Clock())
    cfg = dict(store.config)
    cfg[field] = value
    with pytest.raises(long_run.ResumeRejected):
        long_run.ProgressStore(tmp_path / "new", config=cfg, source_dir=store.dir).load(
            case_id=f.CASE_ID
        )


async def test_concurrent_resumes_do_not_duplicate_calls_and_later_reuse_commits(
    tmp_path, monkeypatch
):
    origin, _ = saved_context(tmp_path, f.Clock(), name="origin")
    contexts = []
    for name in ["first", "second"]:
        ctx, store = f.make_context(tmp_path, f.Clock(), name=name, resume_root=origin.root)
        store.load(case_id=f.CASE_ID)
        store.adopt_resume()
        contexts.append(ctx)
    entered, release = asyncio.Event(), asyncio.Event()

    class BlockingJudge(f.ScriptedJudge):
        async def ainvoke(self, messages):
            if self.calls == 0:
                entered.set()
                await release.wait()
            return await super().ainvoke(messages)

    judge = BlockingJudge()
    monkeypatch.setattr(harness, "_judge_llm", lambda: judge)

    async def run(ctx):
        with long_run.active(ctx):
            return await harness.judge_citation_support(f.REPORT, f.SOURCES)

    task = asyncio.create_task(run(contexts[0]))
    try:
        await asyncio.wait_for(entered.wait(), timeout=2)
        _, busy_rows = await run(contexts[1])
        assert not any(r["judged"] for r in busy_rows)
        assert contexts[1].last_judge_status == long_run.JUDGE_PARTIAL
    finally:
        release.set()
        await task
    assert judge.calls == 3
    later, store = f.make_context(tmp_path, f.Clock(), name="later", resume_root=origin.root)
    store.load(case_id=f.CASE_ID)
    store.adopt_resume()
    _, rows = await run(later)
    assert all(r["judged"] for r in rows) and judge.calls == 3


async def test_empty_judge_reply_is_attributed_to_scoring_not_recovered_research_errors(
    tmp_path, monkeypatch
):
    from app.services.evaluation_diagnostics import analyze

    base = tmp_path / "pipeline/artifacts/report-benchmark"
    ctx, _ = f.make_context(
        tmp_path, f.Clock(), name="pipeline/artifacts/report-benchmark/reliability"
    )
    monkeypatch.setenv("RF_JUDGE_BATCH_TRACE_DIR", str(base / "judge-batches"))
    monkeypatch.setattr(harness, "_judge_llm", lambda: f.ScriptedJudge({1: ""}))
    pipe = "pipe-" + "a" * 24
    with long_run.active(ctx):
        row = await harness.run_one(f.QUERY, trace_run_id=pipe)
    (base / "raw_result.json").write_text(json.dumps({"results": [row]}), encoding="utf8")
    diagnostic = analyze(
        {"pipeline_id": pipe, "case_id": f.CASE_ID},
        {"dataset_id": "report-benchmark", "state": "failed", "reason": "judge_partial"},
        base,
    )
    assert diagnostic["failure_phase"] == "judge"
    assert "4 条" in diagnostic["timeline"]["summary"]
    assert "空回复" in diagnostic["candidates"][0]["cause"]
    assert "空或非文本" in diagnostic["report_states"][0]["stop_reason"]
    assert diagnostic["report_states"][0]["quality_gate"] == "not_passed"
