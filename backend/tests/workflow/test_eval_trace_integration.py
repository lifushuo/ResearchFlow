"""Run the real compiled graph and production tool/parser boundaries without paid providers."""

import asyncio
import json
import os
import sys
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import MemorySaver

from research_engine import eval_trace as trace
from research_engine import graph
from research_engine.runconfig import RunConfig, reset_run_config, set_run_config
from research_engine.runner import initial_state, resume, run

_ENV_BEFORE = dict(os.environ)
from evals import harness  # noqa: E402 — the harness's dotenv loading is scoped out of test state

os.environ.clear()
os.environ.update(_ENV_BEFORE)


def read_events(writer_or_path):
    path = writer_or_path.path if hasattr(writer_or_path, "path") else Path(writer_or_path)
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


@pytest.fixture
def fake_harness(monkeypatch):
    monkeypatch.setattr(harness, "RUN_CONFIG", RunConfig(llm_mode="fake", skip_plan_gate=True))
    monkeypatch.delenv("RF_EVAL_TRACE", raising=False)
    monkeypatch.delenv("RF_EVAL_TRACE_DIR", raising=False)


async def test_compiled_graph_reaches_both_gates_and_finalizer(tmp_path):
    saver = MemorySaver()
    config = RunConfig(llm_mode="fake", skip_plan_gate=False)
    with trace.eval_trace_run(
        case_id="gates", run_id="r", thread_id="gates", directory=tmp_path, enabled=True
    ) as writer:
        planned = await run(
            checkpointer=saver,
            session_id="gates",
            user_id="u",
            query="research memory",
            run_config=config,
        )
        assert planned.status == "awaiting_plan"
        drafted = await resume(checkpointer=saver, session_id="gates", plan={}, run_config=config)
        assert drafted.status == "awaiting_approval"
        rejected = await resume(
            checkpointer=saver,
            session_id="gates",
            approved=False,
            feedback="revise",
            run_config=config,
        )
        assert rejected.status == "awaiting_approval"
        final = await resume(
            checkpointer=saver, session_id="gates", approved=True, run_config=config
        )
        assert final.status == "completed"
    events = read_events(writer)
    nodes = {event["name"] for event in events if event["kind"] == "graph_node"}
    assert {
        "planner",
        "plan_gate",
        "executor",
        "critic",
        "contradiction_detector",
        "synthesizer",
        "hitl_gate",
        "finalizer",
    } <= nodes
    assert any(event["status"] == "interrupt" for event in events)
    assert not any(event["status"] == "error" for event in events)
    assert not writer.bad_case_path.exists()
    assert {event["thread_id"] for event in events} == {"gates"}


async def test_harness_tracing_keeps_scripted_results_equivalent(fake_harness, tmp_path):
    query = {"id": "same", "query": "research memory", "depth": "fast"}
    before = await harness.run_one(query)
    after = await harness.run_one(query, trace_dir=tmp_path)
    for key in (
        "completed",
        "error",
        "source_count",
        "tokens",
        "word_count",
        "resolution_rate",
        "uncited_claim_count",
        "cost_usd",
    ):
        assert before[key] == after[key], key
    assert "runtime_trace" not in before
    assert after["runtime_trace_status"] == "recorded"
    events = read_events(after["runtime_trace"])
    assert any(event["kind"] == "llm" for event in events)
    assert any(event["kind"] == "tool" and event["name"] == "submit_evidence" for event in events)
    assert all("configured_model" not in event["metadata"] for event in events)
    assert trace.active_trace() is None


async def test_actual_planner_parse_failures_record_two_attempts(
    fake_harness, monkeypatch, tmp_path
):
    class InvalidPlanner:
        async def ainvoke(self, messages):
            return AIMessage(content='{"tasks": []}')

    monkeypatch.setattr(graph, "get_llm", lambda role: InvalidPlanner())
    row = await harness.run_one({"id": "invalid", "query": "research memory"}, trace_dir=tmp_path)
    assert not row["completed"]
    events = read_events(row["runtime_trace"])
    attempts = [
        event for event in events if event["kind"] == "attempt" and event["status"] != "start"
    ]
    assert [event["metadata"]["attempt"] for event in attempts] == [1, 2]
    assert any(event["error_type"] == "ValidationError" for event in events)
    bad = json.loads(Path(row["runtime_bad_case"]).read_text())
    assert bad["root_cause"] == "unknown" and not bad["root_cause_confirmed"]


async def test_provider_429_does_not_grow_retry_policy(monkeypatch, tmp_path):
    class QuotaError(Exception):
        status_code = 429

    class Provider:
        calls = 0

        def with_structured_output(self, *args, **kwargs):
            return self

        async def ainvoke(self, messages):
            self.calls += 1
            raise QuotaError("429 quota exhausted PRIVATE_KEY")

    model = Provider()
    monkeypatch.setattr(graph, "get_llm", lambda role: model)
    monkeypatch.setattr(harness, "RUN_CONFIG", RunConfig(llm_mode="real"))
    row = await harness.run_one({"id": "quota", "query": "research memory"}, trace_dir=tmp_path)
    assert model.calls == 1 and not row["completed"]
    text = Path(row["runtime_trace"]).read_text()
    assert "PRIVATE_KEY" not in text
    assert any(
        event["metadata"].get("http_status") == 429 for event in read_events(row["runtime_trace"])
    )


async def test_actual_tool_exception_keeps_existing_observation(monkeypatch, tmp_path):
    class Model:
        def __init__(self):
            self.calls = 0

        def bind_tools(self, *args, **kwargs):
            return self

        async def ainvoke(self, messages):
            self.calls += 1
            if self.calls == 1:
                return AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "read_webpage",
                            "args": {"url": "https://example.com/private"},
                            "id": "call-read",
                        }
                    ],
                )
            return AIMessage(content="")

    class Tool:
        async def ainvoke(self, args):
            raise TimeoutError("PRIVATE_PROVIDER_BODY")

    monkeypatch.setattr(graph, "get_llm", lambda role: Model())
    monkeypatch.setitem(graph._TOOLS_BY_NAME, "read_webpage", Tool())
    token = set_run_config(RunConfig(llm_mode="fake"))
    try:
        state = initial_state(session_id="tool", user_id="u", query="research memory", depth="fast")
        with trace.eval_trace_run(
            case_id="tool", run_id="r", directory=tmp_path, enabled=True
        ) as writer:
            result = await graph._research_one(
                state,
                {"id": 1, "query": "memory", "status": "pending"},
                graph._BudgetGuard(0.0, 0.0),
            )
        assert result["evidence"] == []
    finally:
        reset_run_config(token)
    events = read_events(writer)
    error = next(
        event for event in events if event["kind"] == "tool" and event["status"] == "error"
    )
    assert error["error_type"] == "TimeoutError"
    assert error["metadata"]["tool_call_id"] == "call-read"
    assert "PRIVATE_PROVIDER_BODY" not in writer.path.read_text()
    assert "example.com/private" not in writer.path.read_text()


async def test_cancelled_compiled_graph_preserves_cancellation(monkeypatch, tmp_path):
    entered = asyncio.Event()

    class Blocked:
        async def ainvoke(self, messages):
            entered.set()
            await asyncio.Event().wait()

    monkeypatch.setattr(graph, "get_llm", lambda role: Blocked())
    writers = []

    async def drive():
        with trace.eval_trace_run(
            case_id="cancel", run_id="r", directory=tmp_path, enabled=True
        ) as writer:
            writers.append(writer)
            await run(
                checkpointer=MemorySaver(),
                session_id="cancel",
                user_id="u",
                query="memory",
                run_config=RunConfig(llm_mode="fake"),
            )

    task = asyncio.create_task(drive())
    await asyncio.wait_for(entered.wait(), 5)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert any(event["status"] == "cancelled" for event in read_events(writers[0]))
    assert writers[0].bad_case_path.exists()
    assert trace.active_trace() is None


async def test_parallel_compiled_cases_do_not_share_identity(fake_harness, tmp_path):
    rows = await asyncio.gather(
        *(
            harness.run_one(
                {"id": f"case-{i}", "query": "research memory", "depth": "fast"},
                trace_dir=tmp_path,
                trace_run_id="experiment",
            )
            for i in range(2)
        )
    )
    identities = []
    for i, row in enumerate(rows):
        assert row["completed"]
        events = read_events(row["runtime_trace"])
        assert {event["case_id"] for event in events} == {f"case-{i}"}
        identities.append(events[0]["trace_id"])
    assert len(set(identities)) == 2


async def test_unwritable_trace_does_not_fail_research(fake_harness, monkeypatch, tmp_path):
    original = Path.open

    def open_path(path, *args, **kwargs):
        if path.suffix == ".jsonl":
            raise PermissionError("read-only traces")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", open_path)
    row = await harness.run_one(
        {"id": "disk", "query": "memory", "depth": "fast"}, trace_dir=tmp_path
    )
    assert row["completed"]
    assert row["runtime_trace_status"] == "unavailable"
    assert "runtime_trace" not in row


async def test_trace_cli_places_output_beside_out_and_separates_candidate(
    fake_harness, monkeypatch, tmp_path
):
    spec = tmp_path / "spec.json"
    spec.write_text(json.dumps({"planner": "Outline a research plan"}))
    output = tmp_path / "report.json"
    monkeypatch.setattr(
        sys,
        "argv",
        ["harness", "--trace", "--limit", "1", "--candidate", str(spec), "--out", str(output)],
    )
    await harness.main()
    payload = json.loads(output.read_text())
    baseline = payload["results"][0]
    candidate = payload["candidate"]["results"][0]
    assert Path(baseline["runtime_trace"]).parent == Path("traces")
    assert baseline["runtime_trace"] != candidate["runtime_trace"]
    assert (
        read_events(tmp_path / baseline["runtime_trace"])[0]["run_id"]
        == read_events(tmp_path / candidate["runtime_trace"])[0]["run_id"]
    )
