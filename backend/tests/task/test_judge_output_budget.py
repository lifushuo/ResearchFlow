"""独立输出预算、转换文本及跨预算重评均在本地完成，不请求真实模型。"""

import asyncio
import hashlib
import json
import sys
from dataclasses import replace

import pytest
import test_long_run_reliability as fixtures
from langchain_core.messages import AIMessage

from evals import benchmark, harness
from evals.agent_eval_ext import long_run
from evals.judge_config import judge_client_config, judge_max_output_tokens
from research_engine import llm_factory
from research_engine.runconfig import get_run_config, reset_run_config, set_run_config


@pytest.mark.parametrize("value", ["0", "-1", "true", "1.5", "32769", "", "４０９６"])
def test_invalid_budget_refuses_before_model_construction(monkeypatch, value):
    monkeypatch.setenv("EVAL_JUDGE_MAX_OUTPUT_TOKENS", value)
    with pytest.raises(ValueError, match="1..32768"):
        with judge_client_config("custom:judge"):
            pytest.fail("invalid budget reached factory")


@pytest.mark.parametrize("entry", ["harness", "benchmark"])
def test_both_entries_use_independent_budget_and_restore_critic(monkeypatch, entry):
    monkeypatch.delenv("EVAL_JUDGE_MAX_OUTPUT_TOKENS", raising=False)
    monkeypatch.delenv("EVAL_JUDGE_BASE_URL", raising=False)
    monkeypatch.delenv("EVAL_JUDGE_API_KEY", raising=False)
    base = get_run_config()
    token = set_run_config(
        replace(
            base,
            llm_mode="real",
            enforce_ssrf_guards=False,
            models={role: "custom:research" for role in base.models},
            provider_keys={"custom": "offline", "custom_base_url": "http://localhost:9999/v1"},
        )
    )
    try:
        monkeypatch.setattr(harness, "JUDGE_ROUTE", "custom:judge")
        model = (
            harness._judge_llm()
            if entry == "harness"
            else benchmark._build_judge("custom:judge")[0]
        )
        assert judge_max_output_tokens() == model.max_tokens == 4096
        assert llm_factory.get_llm("critic").max_tokens == 1000
        with pytest.raises(RuntimeError):
            with llm_factory.output_limit("critic", 8192):
                raise RuntimeError("construction failed")
        assert llm_factory.get_llm("critic").max_tokens == 1000
    finally:
        reset_run_config(token)


async def test_generation_context_does_not_leak_to_concurrent_task():
    entered, release = asyncio.Event(), asyncio.Event()

    async def judge():
        with llm_factory.output_limit("critic", 4096):
            entered.set()
            await release.wait()
            assert llm_factory._OUTPUT_LIMITS.get() == {"critic": 4096}

    task = asyncio.create_task(judge())
    await entered.wait()
    assert not llm_factory._OUTPUT_LIMITS.get()
    release.set()
    await task


async def test_nontext_response_is_not_invented_into_a_verdict(monkeypatch):
    calls = []

    class Judge:
        async def ainvoke(self, messages):
            calls.append(1)
            return AIMessage(
                content=[
                    {"type": "reasoning", "text": "Claim 1: NO"},
                    {"type": "image", "data": "offline"},
                ]
            )

    monkeypatch.setattr(harness, "_judge_llm", Judge)
    rate, rows = await harness.judge_citation_support(
        "A documented finding is present [1].", fixtures.SOURCES
    )
    assert rate is None
    assert rows[0]["judged"] is False
    assert len(calls) == 1


async def test_rejudge_reuses_draft_but_scores_all_batches(monkeypatch, tmp_path):
    # 先按旧预算保留完整研究和评分，再切换预算：普通续评必须拒绝，显式重评不能混入旧判定。
    fixtures.isolated_env.__wrapped__(monkeypatch)
    monkeypatch.setenv("EVAL_JUDGE_MAX_OUTPUT_TOKENS", "1000")
    old_ctx, source = fixtures.make_context(tmp_path, fixtures.Clock(), name="old")
    monkeypatch.setattr(harness, "_judge_llm", fixtures.ScriptedJudge)
    with long_run.active(old_ctx):
        await harness.run_one(fixtures.QUERY)
    before = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in source.dir.rglob("*.json")}
    monkeypatch.setenv("EVAL_JUDGE_MAX_OUTPUT_TOKENS", "4096")
    ctx, store = fixtures.make_context(
        tmp_path, fixtures.Clock(), name="new", resume_root=old_ctx.root
    )
    with pytest.raises(long_run.ResumeRejected, match="configuration_changed"):
        store.load(case_id=fixtures.CASE_ID)
    ctx.rejudge = True
    judge = fixtures.ScriptedJudge()
    monkeypatch.setattr(harness, "_judge_llm", lambda: judge)
    monkeypatch.setattr(harness, "build_graph", lambda *_: pytest.fail("research was repeated"))
    with long_run.active(ctx):
        row = await harness.run_one(fixtures.QUERY)
    assert row["judge_status"] == long_run.JUDGE_COMPLETED
    assert judge.calls == fixtures.TOTAL_BATCHES
    assert ctx.batches_restored == 0 and ctx.research_reused == 1
    assert all(hashlib.sha256(p.read_bytes()).hexdigest() == digest for p, digest in before.items())
    loaded = long_run.ProgressStore(store.dir, config=store.config).load(case_id=fixtures.CASE_ID)
    assert len(loaded.batches) == fixtures.TOTAL_BATCHES
    assert loaded.draft["draft_report"] == fixtures.REPORT
    changed = {**store.config, "judge_route": "custom:different"}
    with pytest.raises(long_run.ResumeRejected, match="more_than_output_budget"):
        long_run.ProgressStore(
            tmp_path / "invalid", config=changed, source_dir=source.dir
        ).load_for_rejudge(case_id=fixtures.CASE_ID)


async def test_cli_rejudge_runs_only_judge_and_saves_new_result(monkeypatch, tmp_path):
    fixtures.isolated_env.__wrapped__(monkeypatch)
    monkeypatch.setattr(harness, "JUDGE_ROUTE", "custom:offline-judge")
    monkeypatch.setenv("EVAL_JUDGE_MAX_OUTPUT_TOKENS", "1000")
    ctx, _ = fixtures.make_context(tmp_path, fixtures.Clock(), name="original")
    monkeypatch.setattr(harness, "_judge_llm", fixtures.ScriptedJudge)
    with long_run.active(ctx):
        await harness.run_one(fixtures.QUERY)
    queries = tmp_path / "input"
    queries.mkdir()
    (queries / "queries.json").write_text(json.dumps({"queries": [fixtures.QUERY]}))
    monkeypatch.setattr(harness, "EVALS_DIR", queries)
    monkeypatch.setenv("EVAL_JUDGE_MAX_OUTPUT_TOKENS", "4096")
    judge = fixtures.ScriptedJudge()
    monkeypatch.setattr(harness, "_judge_llm", lambda: judge)
    monkeypatch.setattr(harness, "build_graph", lambda *_: pytest.fail("unexpected graph"))
    result = tmp_path / "new-result.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "harness",
            "--mode",
            "report",
            "--case-id",
            fixtures.CASE_ID,
            "--judge",
            "custom:offline-judge",
            "--rejudge-from",
            str(ctx.root),
            "--progress-dir",
            str(tmp_path / "new-progress"),
            "--out",
            str(result),
        ],
    )
    await harness.main()
    assert judge.calls == fixtures.TOTAL_BATCHES
    assert json.loads(result.read_text())["results"][0]["judge_status"] == long_run.JUDGE_COMPLETED
