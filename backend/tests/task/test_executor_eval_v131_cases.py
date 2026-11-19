"""Seven fixed, model-free Executor regression cases for the Evaluation Center.

Uses the actual ResearchFlow Executor research loop and real URL/tool validation,
but scripted LLM turns and mock tool observations. No provider requests or real HTTP.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from pathlib import Path

import httpx
import pytest
from langchain_core.messages import AIMessage

from research_engine import graph
from research_engine import tools as engine_tools
from research_engine.events import reset_emitter, set_emitter
from research_engine.runconfig import RunConfig, reset_run_config, set_run_config

URL = "https://example.org/evaluation-source"
PAGE = "The research study reports reliable source provenance checks and verifiable evidence."
TASK = {"id": 1, "query": "Summarise verifiable research evidence"}


@pytest.fixture(autouse=True)
def evaluation_case_trace(request):
    directory = os.environ.get("RF_EXECUTOR_CASE_TRACE_DIR")
    if not directory:
        yield
        return
    from research_engine.eval_trace import eval_trace_run, trace_span

    manifest = (
        Path(__file__).resolve().parents[2]
        / "evals/agent_eval_ext/fixtures/executor_suite_v131.json"
    )
    rows = json.loads(manifest.read_text(encoding="utf8"))["cases"]
    case_id = next(
        row["case_id"] for row in rows if row["test"].split("::")[-1] == request.node.name
    )
    with eval_trace_run(
        directory=directory,
        run_id=os.environ["RF_EVAL_TRACE_RUN_ID"],
        case_id=case_id,
        enabled=True,
        capture_content=True,
    ):
        with trace_span("executor.contract", kind="controlled_executor_case"):
            yield


def _call(name: str, args: dict, key: str) -> dict:
    return {"name": name, "args": args, "id": key}


def _turn(*calls: dict) -> AIMessage:
    return AIMessage(content="", tool_calls=list(calls))


class ScriptedModel:
    def __init__(self, turns: list[AIMessage]):
        self.turns = list(turns)
        self.calls = 0

    def bind_tools(self, tools, tool_choice=None):
        return self

    async def ainvoke(self, messages):
        self.calls += 1
        return self.turns.pop(0) if self.turns else _turn()


class FakeTool:
    def __init__(self, fn):
        self.ainvoke = fn


def _state() -> dict:
    return {
        "session_id": "v131-executor-eval",
        "tasks": [TASK],
        "evidence": [],
        "verdicts": {},
        "retries": {},
        "research_round": 0,
        "cost_usd": 0.0,
        "tokens_input": 0,
        "tokens_output": 0,
        "started_at": time.time(),
    }


async def _run(monkeypatch, turns: list[AIMessage], *, tools: dict | None = None, events=None):
    llm = ScriptedModel(turns)
    monkeypatch.setattr(graph, "get_llm", lambda role: llm)
    if tools is not None:
        monkeypatch.setattr(graph, "_TOOLS_BY_NAME", tools)
    ctx = set_run_config(RunConfig(llm_mode="real"))

    async def receive(_sid, event):
        if events is not None:
            events.append(event)

    emit = set_emitter(receive)
    try:
        return await graph._research_one(_state(), TASK, graph._BudgetGuard(0.0, 0.0))
    finally:
        reset_emitter(emit)
        reset_run_config(ctx)


def _submit(key: str = "evidence-1") -> AIMessage:
    return _turn(
        _call(
            "submit_evidence",
            {
                "evidence": [
                    {
                        "source_url": URL,
                        "source_title": "Research source",
                        "snippet": PAGE,
                        "key_fact": "Evidence provenance is verifiable.",
                    }
                ]
            },
            key,
        )
    )


@pytest.mark.asyncio
async def test_search_read_submit(monkeypatch):
    actions = []

    async def search(_args):
        actions.append("search")
        return [{"title": "Research source", "url": URL, "snippet": PAGE}]

    async def read(_args):
        actions.append("read")
        return {"url": URL, "text": PAGE, "error": None}

    result = await _run(
        monkeypatch,
        [
            _turn(_call("web_search", {"query": TASK["query"]}, "search-1")),
            _turn(_call("read_webpage", {"url": URL}, "read-1")),
            _submit(),
        ],
        tools={"web_search": FakeTool(search), "read_webpage": FakeTool(read)},
    )
    assert actions == ["search", "read"]
    assert len(result["evidence"]) == 1
    assert result["evidence"][0]["snippet"] == PAGE


@pytest.mark.asyncio
async def test_malformed_url(monkeypatch):
    events = []
    result = await _run(
        monkeypatch,
        [
            _turn(_call("read_webpage", {"url": [URL, URL]}, "bad-url")),
            _submit(),
        ],
        events=events,
    )
    assert isinstance(result["evidence"], list)
    observations = [
        str((e.get("detail") or {}).get("observation", ""))
        for e in events
        if (e.get("detail") or {}).get("tool") == "read_webpage"
    ]
    assert observations and "tool error" in observations[0]


@pytest.mark.asyncio
async def test_ssrf_blocked():
    ctx = set_run_config(RunConfig(llm_mode="real"))
    try:
        result = await engine_tools.read_webpage.ainvoke({"url": "http://127.0.0.1/admin"})
    finally:
        reset_run_config(ctx)
    assert "blocked" in result.get("error", "").lower()
    assert not result.get("text")


@pytest.mark.asyncio
async def test_tool_timeout(monkeypatch):
    class TimeoutClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def get(self, url, headers):
            raise httpx.ConnectTimeout("controlled timeout")

    monkeypatch.setattr(engine_tools, "validate_url", lambda _url: None)
    monkeypatch.setattr(engine_tools.httpx, "AsyncClient", lambda **_k: TimeoutClient())
    ctx = set_run_config(RunConfig(llm_mode="real"))
    try:
        result = await engine_tools.read_webpage.ainvoke({"url": URL})
    finally:
        reset_run_config(ctx)
    assert "timeout" in result.get("error", "").lower()
    assert not result.get("text")


@pytest.mark.asyncio
async def test_duplicate_read(monkeypatch):
    fetched = []

    async def read(args):
        fetched.append(args["url"])
        return {"url": URL, "text": PAGE, "error": None}

    first = _turn(
        _call("read_webpage", {"url": URL}, "read-1"), _call("read_webpage", {"url": URL}, "read-2")
    )
    result = await _run(monkeypatch, [first, _submit()], tools={"read_webpage": FakeTool(read)})
    assert fetched == [URL]
    assert len(result["evidence"]) == 1


@pytest.mark.asyncio
async def test_duplicate_submit(monkeypatch):
    async def search(_args):
        return [{"title": "Research source", "url": URL, "snippet": PAGE}]

    submit = _submit().tool_calls[0]
    duplicate = dict(submit, id="evidence-duplicate")
    result = await _run(
        monkeypatch,
        [
            _turn(_call("web_search", {"query": TASK["query"]}, "search-1")),
            _turn(submit, duplicate),
        ],
        tools={"web_search": FakeTool(search)},
    )
    assert len(result["evidence"]) == 1


@pytest.mark.asyncio
async def test_cancellation_is_not_success(monkeypatch):
    started = asyncio.Event()

    async def blocked(_args):
        started.set()
        await asyncio.Future()

    llm = [_turn(_call("read_webpage", {"url": URL}, "read-waits"))]
    task = asyncio.create_task(_run(monkeypatch, llm, tools={"read_webpage": FakeTool(blocked)}))
    try:
        await asyncio.wait_for(started.wait(), timeout=5)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    finally:
        if not task.done():
            task.cancel()
