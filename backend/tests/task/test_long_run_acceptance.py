"""Acceptance gaps: repeated resume, candidate isolation and real process-group stops."""

import asyncio
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest
import test_long_run_reliability as fixtures

from evals import harness
from evals.agent_eval_ext import long_run

isolated_env = fixtures.isolated_env


def test_worker_grace_is_inside_the_original_hard_cap(monkeypatch, tmp_path):
    from evals.agent_eval_ext import evaluation_pipeline_worker_v13 as worker

    limits = []

    class Process:
        pid = 4242

        def wait(self, timeout=None):
            limits.append(timeout)
            if len(limits) == 1:
                raise subprocess.TimeoutExpired("scripted", timeout)
            return 3

    monkeypatch.setattr(worker.subprocess, "Popen", lambda *a, **k: Process())
    monkeypatch.setattr(worker, "_signal_group", lambda _: None)
    target = tmp_path / "pipelines/p/artifacts/result.json"
    target.parent.mkdir(parents=True)
    assert worker._run_fixed("report-benchmark", target, 1) == 124
    assert limits == [405, 15]
    assert sum(limits) == 420


def test_multi_case_budget_uses_existing_worker_scope(monkeypatch):
    monkeypatch.setenv("RF_EVAL_WORKER_HARD_CAP_S", "720")
    monkeypatch.setenv("RF_EVAL_WORKER_CASES", "2")
    limits = long_run.Deadlines.from_env()
    assert limits.research_s == 480
    assert limits.judge_s == 690
    assert limits.total_s == 690
    assert limits.total_s + limits.stop_grace_s <= limits.hard_cap_s == 720


def test_cli_candidate_budget_accounts_for_both_arms():
    limits = long_run.Deadlines.from_env(case_count=2)
    assert limits.hard_cap_s == 720
    assert limits.research_s == 480
    assert limits.judge_s == 690


@pytest.mark.parametrize("suffix", ["", "/nested"])
def test_resume_never_writes_inside_the_source(tmp_path, suffix):
    source = tmp_path / "saved"
    with pytest.raises(long_run.ResumeRejected, match="separate_output"):
        long_run.build_context(
            root=Path(str(source) + suffix), judge_provider="fixture", resume_root=source
        )
    assert not source.exists()


async def test_long_retry_after_stops_instead_of_sending_too_early(monkeypatch, tmp_path):
    ctx, _ = fixtures.make_context(tmp_path, fixtures.Clock())
    monkeypatch.setenv("RF_EVAL_JUDGE_RETRY_AFTER_MAX_S", "9")

    class RateLimited(Exception):
        status_code = 429
        headers = {"Retry-After": "120"}

    judge = fixtures.ScriptedJudge(faults={1: RateLimited()})
    monkeypatch.setattr(harness, "_judge_llm", lambda: judge)
    with long_run.active(ctx):
        row = await harness.run_one(fixtures.QUERY)
    assert judge.calls == 1
    assert row["judge_status"] == long_run.JUDGE_PARTIAL
    assert ctx.throttle.blocked_reason == "retry_after_exceeds_wait_limit"


async def test_cooperative_research_stop_awaits_child_cleanup(tmp_path):
    ctx, _ = fixtures.make_context(tmp_path, fixtures.Clock())
    entered, cleaned = asyncio.Event(), asyncio.Event()

    async def research():
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            cleaned.set()

    with long_run.active(ctx):
        task = asyncio.create_task(
            long_run.with_deadline(research(), budget=ctx.budget, stage="research")
        )
        await entered.wait()
        ctx.stop.request("signal_SIGTERM")
        with pytest.raises(long_run.CooperativeStop):
            await task
    assert cleaned.is_set()


async def test_baseline_and_candidate_keep_separate_progress(monkeypatch, tmp_path):
    ctx, baseline = fixtures.make_context(tmp_path, fixtures.Clock())
    judge = fixtures.ScriptedJudge()
    monkeypatch.setattr(harness, "_judge_llm", lambda: judge)
    with long_run.active(ctx):
        first = await harness.run_one(fixtures.QUERY)
        before = {
            p: hashlib.sha256(p.read_bytes()).hexdigest() for p in baseline.dir.rglob("*.json")
        }
        second = await harness.run_one(fixtures.QUERY, prompt_overrides={"planner": "candidate"})
    assert first["judge_status"] == second["judge_status"] == long_run.JUDGE_COMPLETED
    assert len(ctx.cases) == 2
    assert all(hashlib.sha256(p.read_bytes()).hexdigest() == h for p, h in before.items())
    assert len({s.fingerprint for s in ctx.cases.values()}) == 2


async def test_second_resume_owns_a_draft_and_never_repeats_research(monkeypatch, tmp_path):
    previous = None
    graph_calls = []
    monkeypatch.setattr(harness, "build_graph", lambda *_: fixtures.FakeGraph(graph_calls))
    for n in range(3):
        ctx, store = fixtures.make_context(
            tmp_path, fixtures.Clock(), name=f"run{n}", resume_root=previous
        )
        judge = fixtures.ScriptedJudge(
            on_call=(lambda _, context=ctx: context.stop.request("checkpoint")) if n < 2 else None
        )
        monkeypatch.setattr(harness, "_judge_llm", lambda model=judge: model)
        with long_run.active(ctx):
            row = await harness.run_one(fixtures.QUERY)
        assert store.draft_path(fixtures.CASE_ID).is_file()
        assert judge.calls == 1
        previous = ctx.root
    assert graph_calls == ["ainvoke"]
    assert row["judge_status"] == long_run.JUDGE_COMPLETED


async def test_failed_progress_preflight_makes_no_model_calls(monkeypatch, tmp_path):
    ctx, _ = fixtures.make_context(tmp_path, fixtures.Clock())
    calls = []
    monkeypatch.setattr(harness, "build_graph", lambda *_: fixtures.FakeGraph(calls))
    monkeypatch.setattr(
        long_run.ProgressStore,
        "preflight",
        lambda _: (_ for _ in ()).throw(PermissionError("read only")),
    )
    monkeypatch.setattr(
        harness, "_judge_llm", lambda: pytest.fail("preflight failure called a judge")
    )
    with long_run.active(ctx):
        row = await harness.run_one(fixtures.QUERY)
    assert calls == []
    assert row["completed"] is False
    assert row["research_state"] == long_run.ARTIFACT_SAVE_FAILED
    assert ctx.exit_code == long_run.EXIT_ARTIFACT_SAVE_FAILED


async def test_existing_progress_is_never_overwritten_by_a_fresh_run(monkeypatch, tmp_path):
    ctx, store = fixtures.make_context(tmp_path, fixtures.Clock())
    monkeypatch.setattr(harness, "_judge_llm", lambda: fixtures.ScriptedJudge())
    with long_run.active(ctx):
        await harness.run_one(fixtures.QUERY)
    before = {p: p.read_bytes() for p in store.dir.rglob("*.json")}
    fresh = long_run.build_context(root=ctx.root, judge_provider="fake-provider")
    monkeypatch.setattr(
        harness, "build_graph", lambda *_: pytest.fail("overwriting run called research")
    )
    with long_run.active(fresh):
        row = await harness.run_one(fixtures.QUERY)
    assert row["research_state"] == long_run.ARTIFACT_SAVE_FAILED
    assert all(p.read_bytes() == data for p, data in before.items())


@pytest.mark.parametrize(
    "field,value", [("supported", "false"), ("judged_count", 0), ("yes_count", 0)]
)
async def test_resume_rejects_corrupt_verdicts_before_any_request(
    monkeypatch, tmp_path, field, value
):
    ctx, store = fixtures.make_context(tmp_path, fixtures.Clock())
    monkeypatch.setattr(harness, "_judge_llm", lambda: fixtures.ScriptedJudge())
    with long_run.active(ctx):
        await harness.run_one(fixtures.QUERY)
    path = store.batches_dir / "batch-0001.json"
    row = json.loads(path.read_text())
    if field == "supported":
        row["verdicts"]["0"][field] = value
    else:
        row[field] = value
    path.write_text(json.dumps(row))
    resumed, _ = fixtures.make_context(
        tmp_path, fixtures.Clock(), name="resume", resume_root=ctx.root
    )
    monkeypatch.setattr(harness, "build_graph", lambda *_: pytest.fail("resume called research"))
    monkeypatch.setattr(harness, "_judge_llm", lambda: pytest.fail("corrupt resume called judge"))
    with long_run.active(resumed), pytest.raises(long_run.ResumeRejected):
        await harness.run_one(fixtures.QUERY)


@pytest.mark.skipif(os.name != "posix", reason="production worker uses POSIX process-group signals")
def test_real_suite_and_compiled_graph_flush_after_sigterm(tmp_path):
    child = tmp_path / "harness_child.py"
    child.write_text("""import asyncio, sys
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from langchain_core.messages import AIMessage
from research_engine import local
local.load_env_file = lambda *a, **kw: False
from evals import harness
from research_engine import graph as compiled_graph
from research_engine.runconfig import get_run_config, set_run_config, reset_run_config
real_model = compiled_graph.get_llm
class Synthesizer:
    async def ainvoke(self, messages):
        report = "# Report\\n\\n## Findings\\n" + "\\n".join("- A citable fact is present [1]." for _ in range(9))
        return AIMessage(content=report)
compiled_graph.get_llm = lambda role: Synthesizer() if role == "synthesizer" else real_model(role)
original = harness._pipeline_scope
@contextmanager
def scripted_research(overrides):
    with original(overrides):
        token = set_run_config(replace(get_run_config(), llm_mode="fake"))
        try: yield
        finally: reset_run_config(token)
harness._pipeline_scope = scripted_research
marker_path = Path(sys.argv[-1])
class Judge:
    calls = 0
    async def ainvoke(self, messages):
        self.calls += 1
        if self.calls == 2:
            marker_path.write_text("judge_waiting")
            await asyncio.sleep(120)
        n = messages[-1].content.count("\\nEvidence ")
        return AIMessage(content="\\n".join(f"Claim {i}: YES" for i in range(1, n+1)))
judge = Judge()
harness._judge_llm = lambda: judge
sys.argv = sys.argv[:-1]
asyncio.run(harness.main())
""")
    marker, result = tmp_path / "ready", tmp_path / "result.json"
    wrapper = tmp_path / "suite.py"
    wrapper.write_text("""import sys
from pathlib import Path
from evals.agent_eval_ext import evaluation_pipeline_suite_v13 as suite
original_run = suite.subprocess.run
def run(cmd, **kwargs):
    return original_run([sys.executable, sys.argv[1], *cmd[3:], sys.argv[3]], **kwargs)
suite.subprocess.run = run
suite._execute("report-benchmark", Path(sys.argv[2]), 1, case_id="quantum-error-correction", judge_route="custom:scripted-judge")
""")
    env = {
        **os.environ,
        "LLM_MODE": "real",
        "RF_EVAL_ONLINE_ENABLED": "1",
        "PYTHONPATH": str(Path.cwd()),
        "RF_EVAL_TRACE_RUN_ID": "offline-signal-acceptance",
        "GOOGLE_API_KEY": "fixture-only-not-a-real-key",
        "CUSTOM_API_KEY": "fixture-only-not-a-real-key",
        "CUSTOM_BASE_URL": "http://127.0.0.1:1/v1",
    }
    log = tmp_path / "process.log"
    with log.open("w") as stream:
        proc = subprocess.Popen(
            [sys.executable, str(wrapper), str(child), str(result), str(marker)],
            env=env,
            stdout=stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            end = time.monotonic() + 20
            while not marker.exists() and proc.poll() is None and time.monotonic() < end:
                time.sleep(0.02)
            assert marker.exists(), log.read_text()
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                pytest.fail("Suite/harness did not flush and exit after SIGTERM")
            status = json.loads((tmp_path / "reliability/final-status.json").read_text())
            assert status["status"] == long_run.JUDGE_PARTIAL
            assert result.exists()
            assert len(list((tmp_path / "reliability").rglob("batch-0001.json"))) == 1
            assert len(list((tmp_path / "reliability").rglob("drafts/*.json"))) == 1
        finally:
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGKILL)
            proc.wait(timeout=5)
