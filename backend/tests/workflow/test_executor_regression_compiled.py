"""Actual compiled graph cases; no node, tool validation, SSRF or snippet-verifier stubs."""

import pytest

from evals.agent_eval_ext.executor_regression import CASES, run_case
from research_engine import graph
from research_engine.runconfig import get_run_config


@pytest.mark.parametrize("case_id", CASES)
async def test_executor_compiled_contract(tmp_path, case_id):
    original, config = graph.get_llm, get_run_config()
    result = await run_case(case_id, tmp_path)
    assert result["passed"], result["assertions"]
    assert result["live_agent_accuracy_measured"] is False
    assert graph.get_llm is original and get_run_config() is config


async def test_unknown_case_does_not_mutate_runtime(tmp_path):
    original = graph.get_llm
    with pytest.raises(ValueError):
        await run_case("unknown", tmp_path)
    assert graph.get_llm is original


async def test_cancel_during_case_restores_factory_and_children(tmp_path, monkeypatch):
    import asyncio

    import httpx

    original = graph.get_llm
    entered = asyncio.Event()
    gate = asyncio.Event()
    original_invoke = graph.traced_ainvoke

    async def pause(model, messages, **kwargs):
        if kwargs.get("role") == "executor":
            entered.set()
            await gate.wait()
        return await original_invoke(model, messages, **kwargs)

    monkeypatch.setattr(graph, "traced_ainvoke", pause)
    task = asyncio.create_task(run_case("cancel", tmp_path))
    await asyncio.wait_for(entered.wait(), 5)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert graph.get_llm is original
    assert httpx.AsyncClient.__module__ == "httpx"


def test_process_suite_verification_and_closeout_html(tmp_path):
    import json

    from evals.agent_eval_ext.executor_regression import main, verify_suite
    from evals.agent_eval_ext.interview_closeout import generate
    from evals.agent_eval_ext.unified_eval import build_unified

    suite_root = tmp_path / "executor"
    assert main(["--out-dir", str(suite_root)]) == 0
    source = suite_root / "suite.json"
    suite = json.loads(source.read_text())
    assert len(verify_suite(source, suite)) == 7
    malformed = json.loads(json.dumps(suite))
    malformed["cases"][0]["process_exit_code"] = False
    with pytest.raises(ValueError, match="exit code"):
        verify_suite(source, malformed)
    report = tmp_path / "report.json"
    report.write_text(
        json.dumps(
            {
                "mode": "report",
                "results": [{"id": "controlled", "completed": False, "claim_verdicts": []}],
            }
        )
    )
    unified = tmp_path / "unified.json"
    unified.write_text(json.dumps(build_unified(report=report)))
    generate(unified_path=unified, executor_path=source, html_out=tmp_path / "report.html")
    assert "7 / 7" in (tmp_path / "report.html").read_text(encoding="utf-8")
    damaged = suite_root / suite["cases"][0]["result_path"]
    damaged.write_text("{}")
    with pytest.raises(ValueError, match="bytes"):
        verify_suite(source, suite)
