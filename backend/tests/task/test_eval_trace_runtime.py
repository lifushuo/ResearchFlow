"""Phase2 tracer tests: no LangGraph, internet, keys, or DB required."""

from __future__ import annotations

import asyncio
import json

import pytest
from langgraph.errors import GraphInterrupt

from evals.agent_eval_ext.trace_inspect import inspect_trace
from research_engine import eval_trace as t


def events_of(writer):
    return [json.loads(x) for x in writer.path.read_text("utf-8").splitlines()]


def test_disabled_trace_is_an_exact_noop(tmp_path):
    with t.eval_trace_run(case_id="x", run_id="r", directory=tmp_path, enabled=False) as writer:
        assert writer is None
        assert t.active_trace() is None
        with t.trace_span("planner", kind="graph_node"):
            pass
    assert list(tmp_path.iterdir()) == []


@pytest.mark.asyncio
async def test_real_async_node_generates_parent_child_spans(tmp_path):
    async def fake_planner(state):
        with t.trace_span(
            "tool.read_webpage",
            kind="tool",
            metadata={"tool_name": "read_webpage", "tool_call_id": "call-1", "args": "SECRET"},
        ):
            await asyncio.sleep(0)
        return {"tasks": [1]}

    wrapped = t.traced_node("planner", fake_planner)
    with t.eval_trace_run(
        case_id="llm-memory", run_id="run1", directory=tmp_path, enabled=True
    ) as w:
        assert await wrapped({}) == {"tasks": [1]}
    ev = events_of(w)
    assert [e["status"] for e in ev] == ["start", "start", "ok", "ok"]
    assert ev[1]["parent_span_id"] == ev[0]["span_id"]
    assert ev[2]["metadata"]["tool_call_id"] == "call-1"
    assert "SECRET" not in w.path.read_text("utf-8")
    assert not w.bad_case_path.exists()


def test_disabled_inner_scope_cannot_borrow_outer_case(tmp_path):
    with t.eval_trace_run(case_id="outer", run_id="r", directory=tmp_path, enabled=True) as outer:
        with t.eval_trace_run(case_id="disabled", run_id="r", directory=tmp_path, enabled=False):
            with t.trace_span("private-inner", kind="graph_node"):
                assert t.active_trace() is None
        assert t.active_trace() is outer
    assert "private-inner" not in outer.path.read_text()


def test_failed_inner_scope_cannot_borrow_outer_case(tmp_path):
    blocked = tmp_path / "file-not-directory"
    blocked.write_text("preserve")
    with t.eval_trace_run(case_id="outer", run_id="r", directory=tmp_path, enabled=True) as outer:
        with t.eval_trace_run(
            case_id="broken", run_id="r", directory=blocked, enabled=True
        ) as failed:
            assert failed is None and t.active_trace() is None
        assert t.active_trace() is outer


async def test_inspection_keeps_unknown_usage_unmeasured(tmp_path):
    class Model:
        async def ainvoke(self, messages):
            return object()

    with t.eval_trace_run(case_id="usage", run_id="r", directory=tmp_path, enabled=True) as writer:
        await t.traced_ainvoke(Model(), [], role="critic")
    result = inspect_trace(writer.path)
    assert result["llm_input_tokens_observed"] is None
    assert result["llm_output_tokens_observed"] is None
    assert result["llm_calls_without_input_usage"] == 1
    assert result["trace_complete"]


async def test_response_metadata_failure_cannot_replace_model_result(tmp_path):
    class Response:
        @property
        def usage_metadata(self):
            raise RuntimeError("PRIVATE")

    response = Response()

    class Model:
        async def ainvoke(self, messages):
            return response

    with t.eval_trace_run(case_id="meta", run_id="r", directory=tmp_path, enabled=True) as writer:
        assert await t.traced_ainvoke(Model(), [], role="critic") is response
    assert "PRIVATE" not in writer.path.read_text()


@pytest.mark.parametrize("alias", ["auto/best", "custom:auto/best", "provider/auto"])
async def test_router_alias_is_not_disclosed_as_a_served_model(tmp_path, alias):
    class Response:
        response_metadata = {"model_name": alias}

    class Model:
        async def ainvoke(self, messages):
            return Response()

    with t.eval_trace_run(case_id="alias", run_id="r", directory=tmp_path, enabled=True) as writer:
        await t.traced_ainvoke(Model(), [], role="critic")
    terminal = events_of(writer)[-1]
    assert "served_model" not in terminal["metadata"]
    assert terminal["metadata"]["disclosed_alias"] == alias


def test_exception_metadata_failure_preserves_original_exception(tmp_path):
    class Error(Exception):
        @property
        def status_code(self):
            raise RuntimeError("PRIVATE")

    with t.eval_trace_run(case_id="meta", run_id="r", directory=tmp_path, enabled=True) as writer:
        with pytest.raises(Error):
            with t.trace_span("planner", kind="graph_node"):
                raise Error()
    assert "PRIVATE" not in writer.path.read_text()


def test_inspection_detects_incomplete_or_mixed_trace(tmp_path):
    with t.eval_trace_run(
        case_id="partial", run_id="r", directory=tmp_path, enabled=True
    ) as writer:
        writer.emit(name="planner", kind="graph_node", status="start", span_id="unfinished")
    assert not inspect_trace(writer.path)["trace_complete"]
    events = events_of(writer)
    events.append({**events[0], "seq": 2, "case_id": "other"})
    writer.path.write_text("\n".join(json.dumps(event) for event in events))
    with pytest.raises(ValueError, match="Mixed"):
        inspect_trace(writer.path)


@pytest.mark.asyncio
async def test_business_failure_creates_bad_case_without_inventing_root_cause(tmp_path):
    async def fake_planner(_state):
        return {"error": "planner: could not produce a valid task list"}

    with t.eval_trace_run(
        case_id="quantum-error-correction", run_id="r", directory=tmp_path, enabled=True
    ) as w:
        result = await t.traced_node("planner", fake_planner)({})
        assert "error" in result
    bad = json.loads(w.bad_case_path.read_text("utf-8"))
    assert bad["root_cause"] == "unknown"
    assert bad["root_cause_confirmed"] is False
    assert bad["failure_symptoms"][0]["symptom"] == "business_error"


@pytest.mark.asyncio
async def test_exception_is_reraised_not_hidden(tmp_path):
    async def broken(_state):
        raise ValueError("sensitive full upstream payload")

    with t.eval_trace_run(case_id="fail", run_id="r", directory=tmp_path, enabled=True) as w:
        with pytest.raises(ValueError):
            await t.traced_node("planner", broken)({})
    raw = w.path.read_text("utf-8")
    assert "ValueError" in raw
    assert "sensitive full upstream payload" not in raw
    assert json.loads(w.bad_case_path.read_text("utf-8"))["root_cause"] == "unknown"


@pytest.mark.asyncio
async def test_concurrent_cases_are_isolated(tmp_path):
    async def one(i):
        with t.eval_trace_run(
            case_id=f"case-{i}", run_id="run", directory=tmp_path, enabled=True
        ) as w:
            with t.trace_span("executor", kind="graph_node"):
                await asyncio.sleep(0.001)
                with t.trace_span(
                    "web_search", kind="tool", metadata={"tool_call_id": f"call-{i}"}
                ):
                    await asyncio.sleep(0)
            return w

    a, b = await asyncio.gather(one(1), one(2))
    assert a.path != b.path
    for writer, i in ((a, 1), (b, 2)):
        all_events = events_of(writer)
        assert {e["case_id"] for e in all_events} == {f"case-{i}"}
        assert [e["seq"] for e in all_events] == list(range(1, len(all_events) + 1))
        assert {e["trace_id"] for e in all_events} == {writer.trace_id}


@pytest.mark.asyncio
async def test_real_llm_call_counts_and_missing_usage(tmp_path):
    class Model:
        async def ainvoke(self, _messages):
            class Response:
                usage_metadata = {"input_tokens": 11, "output_tokens": 5}

            return Response()

    class UnknownUsage:
        async def ainvoke(self, _messages):
            return object()

    with t.eval_trace_run(case_id="t", run_id="r", directory=tmp_path, enabled=True) as w:
        await t.traced_ainvoke(Model(), ["SECRET_PROMPT"], role="executor")
        await t.traced_ainvoke(UnknownUsage(), ["SECRET_PROMPT"], role="critic")
    finished = [e for e in events_of(w) if e["kind"] == "llm" and e["status"] == "ok"]
    assert finished[0]["metadata"]["input_tokens"] == 11
    assert "input_tokens" not in finished[1]["metadata"]
    assert "SECRET_PROMPT" not in w.path.read_text("utf-8")


def test_sync_node_also_works_and_new_files_never_overwrite(tmp_path):
    wrapped = t.traced_node("finalizer", lambda s: {"final_report": s["draft_report"]})
    with t.eval_trace_run(case_id="x", run_id="r", directory=tmp_path, enabled=True) as first:
        assert wrapped({"draft_report": "ok"})["final_report"] == "ok"
    with t.eval_trace_run(case_id="x", run_id="r", directory=tmp_path, enabled=True) as second:
        assert wrapped({"draft_report": "ok"})["final_report"] == "ok"
    assert first.path != second.path
    assert first.path.read_text("utf-8") != ""


@pytest.mark.asyncio
async def test_trace_io_failure_does_not_break_agent(monkeypatch, tmp_path):
    with t.eval_trace_run(case_id="x", run_id="r", directory=tmp_path, enabled=True) as w:

        def failing_write(_line):
            raise OSError("disk full")

        monkeypatch.setattr(w._stream, "write", failing_write, raising=False)
        with t.trace_span("planner", kind="graph_node"):
            pass
        assert w._disabled


@pytest.mark.asyncio
async def test_graphinterrupt_is_control_flow_not_bad_case(tmp_path):
    async def interrupt(_state):
        raise GraphInterrupt()

    with t.eval_trace_run(case_id="x", run_id="r", directory=tmp_path, enabled=True) as w:
        with pytest.raises(GraphInterrupt):
            await t.traced_node("plan_gate", interrupt)({})
    assert [e["status"] for e in events_of(w)] == ["start", "interrupt"]
    assert not w.bad_case_path.exists()
