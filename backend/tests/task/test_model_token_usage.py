"""模型用量的真实图、落库、恢复和缺失口径回归；无外网或收费调用。"""

import asyncio
import uuid
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import MemorySaver
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.models.agent_log import AgentLog
from app.services.evaluation_usage import model_token_usage
from app.services.model_usage import load_model_usage
from research_engine import events, runner
from research_engine.eval_trace import traced_ainvoke
from research_engine.runconfig import RunConfig, reset_run_config, set_run_config
from research_engine.token_usage import response_usage, summarize_usage, usage_session


def test_missing_zero_and_invalid_values_are_distinct():
    summary = summarize_usage(
        [
            {"input_tokens": 0, "output_tokens": 0},
            {"input_tokens": True, "output_tokens": "20"},
            {"input_tokens": -1, "output_tokens": None},
        ]
    )
    assert summary["calls"] == 3
    assert summary["known_total_tokens"] == 0
    assert summary["missing_usage_calls"] == 2
    assert summarize_usage([{}])["known_total_tokens"] is None
    assert summarize_usage([])["known_total_tokens"] is None


def test_structured_responses_and_sdk_fallback_preserve_actual_model():
    raw = AIMessage(
        content="",
        usage_metadata={"input_tokens": 10, "output_tokens": 4, "total_tokens": 14},
        response_metadata={"model_name": "served-model"},
    )
    assert response_usage({"raw": raw, "parsed": None}) == {
        "served_model": "served-model",
        "input_tokens": 10,
        "output_tokens": 4,
    }
    legacy = SimpleNamespace(
        response_metadata={"token_usage": {"prompt_tokens": 7, "completion_tokens": 3}}
    )
    assert response_usage(legacy)["input_tokens"] == 7
    summary = summarize_usage([{"configured_model": "custom:requested", **response_usage(raw)}])
    assert summary["models"][0]["served_model"] == "served-model"
    assert summary["models"][0]["configured_model"] == "custom:requested"


def test_aliases_missing_identity_and_fake_models_are_not_named_as_actual():
    summary = summarize_usage(
        [
            {"served_model": "auto/best", "configured_model": "custom:requested"},
            {
                "llm_mode": "fake",
                "served_model": "should-not-be-called",
                "configured_model": "custom:real",
            },
        ]
    )
    assert all(row["served_model"] is None for row in summary["models"])
    assert (
        next(row for row in summary["models"] if row["llm_mode"] == "fake")["configured_model"]
        is None
    )


def test_deduplicate_call_events_without_merging_research_and_judge():
    first = {
        "call_id": "one",
        "served_model": "shared",
        "role": "planner",
        "input_tokens": 5,
        "output_tokens": 2,
    }
    second = {**first, "call_id": "two", "role": "judge"}
    summary = summarize_usage([first, first, second])
    assert summary["calls"] == 2 and summary["known_total_tokens"] == 14
    assert {row["kind"] for row in summary["models"]} == {"research", "judge"}


async def test_cancelled_call_is_recorded_with_unknown_usage():
    entered = asyncio.Event()
    captured = []

    class HangingModel:
        async def ainvoke(self, _):
            entered.set()
            await asyncio.Event().wait()

    async def sink(session, event):
        captured.append((session, event))

    token = events.set_emitter(sink)
    try:
        with usage_session("cancelled-run"):
            task = asyncio.create_task(traced_ainvoke(HangingModel(), [], role="executor"))
        await entered.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    finally:
        events.reset_emitter(token)
    assert len(captured) == 1 and captured[0][0] == "cancelled-run"
    row = captured[0][1]["detail"]
    assert row["status"] == "cancelled" and row["input_tokens"] is None


async def test_persistence_error_does_not_mask_provider_failure():
    class FailedModel:
        async def ainvoke(self, _):
            raise ValueError("original provider failure")

    async def sink(*_):
        raise OSError("event store unavailable")

    token = events.set_emitter(sink)
    try:
        with usage_session("failed-run"), pytest.raises(ValueError, match="original provider"):
            await traced_ainvoke(FailedModel(), [], role="planner")
    finally:
        events.reset_emitter(token)


async def test_concurrent_calls_keep_their_run_and_model_identity():
    captured = []

    class Model:
        def __init__(self, name):
            self.name = name

        async def ainvoke(self, _):
            await asyncio.sleep(0)
            return AIMessage(content="", response_metadata={"model_name": self.name})

    async def one(name):
        config = set_run_config(RunConfig(llm_mode="real"))
        try:
            with usage_session(name):
                await traced_ainvoke(Model(name), [], role="planner")
        finally:
            reset_run_config(config)

    async def sink(sid, event):
        captured.append((sid, event["detail"]["served_model"]))

    token = events.set_emitter(sink)
    try:
        await asyncio.gather(one("first"), one("second"))
    finally:
        events.reset_emitter(token)
    assert sorted(captured) == [("first", "first"), ("second", "second")]


async def test_actual_graph_usage_survives_gate_and_is_filtered_by_run(tmp_path, monkeypatch):
    from app import adapters

    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'usage.sqlite'}")
    async with engine.begin() as conn:
        await conn.run_sync(AgentLog.__table__.create)

    async def publish(*_):
        pass

    monkeypatch.setattr(adapters, "publish_event", publish)
    ident = uuid.uuid4()
    try:
        async with async_sessionmaker(engine, expire_on_commit=False)() as db:
            sink = adapters.agent_log_sink(db, str(ident))
            saver = MemorySaver()
            config = RunConfig(llm_mode="fake", skip_plan_gate=False)
            first = await runner.run(
                checkpointer=saver,
                session_id=str(ident),
                user_id="user",
                query="Compare retrieval and fine tuning",
                run_config=config,
                event_sink=sink,
            )
            assert first.status == "awaiting_plan"
            before = await load_model_usage(db, ident)
            assert before["calls"] == 1
            resumed = await runner.resume(
                checkpointer=saver,
                session_id=str(ident),
                plan={},
                run_config=config,
                event_sink=sink,
            )
            assert resumed.status == "awaiting_approval"
            after = await load_model_usage(db, ident)
            assert after["calls"] > before["calls"] and after["known_total_tokens"] > 0
            assert all(row["llm_mode"] == "fake" for row in after["models"])
            assert (await load_model_usage(db, uuid.uuid4()))["calls"] == 0
    finally:
        await engine.dispose()


def test_evaluation_counts_llm_end_only_and_excludes_restored_batches():
    meta = {
        "role": "planner",
        "input_tokens": 10,
        "output_tokens": 2,
        "served_model": "research-model",
    }
    graph = [
        {
            "trace_id": "one",
            "trace_status": "recorded",
            "events": [
                {"span_id": "p", "kind": "graph_node", "status": "ok", "metadata": meta},
                {"span_id": "a", "kind": "llm", "status": "start", "metadata": {}},
                {"span_id": "a", "kind": "llm", "status": "ok", "metadata": meta},
            ],
        }
    ]
    judges = [
        {
            "trace_complete": True,
            "judge_route": "custom:judge",
            "batch_measurements": [
                {"tokens_input": 3, "tokens_output": 1, "served_model": "judge-model"},
            ],
            "events": [{"event": "batch_restored", "tokens_input": 99999}],
        }
    ]
    summary = model_token_usage(graph, judges)
    assert summary["calls"] == 2 and summary["known_total_tokens"] == 16
    assert {row["served_model"] for row in summary["models"]} == {"research-model", "judge-model"}


def test_historical_missing_usage_stays_unknown_and_open_call_is_counted():
    assert model_token_usage([], [])["known_total_tokens"] is None
    result = model_token_usage(
        [
            {
                "trace_id": "a",
                "events": [
                    {
                        "span_id": "open",
                        "kind": "llm",
                        "status": "start",
                        "metadata": {"role": "executor"},
                    },
                ],
            }
        ],
        [],
    )
    assert result["calls"] == result["missing_usage_calls"] == 1
    assert result["known_total_tokens"] is None and result["collection_incomplete"] is True


def test_calibration_and_grounding_calls_in_execution_trace_are_counted():
    trace = {
        "trace_id": "adapter",
        "trace_status": "recorded",
        "events": [
            {
                "kind": "llm",
                "span_id": "calibration",
                "status": "ok",
                "metadata": {
                    "role": "judge",
                    "input_tokens": 50,
                    "output_tokens": 10,
                    "served_model": "calibration-model",
                    "llm_mode": "real",
                },
            },
        ],
    }
    result = model_token_usage([], [], [trace])
    assert result["calls"] == 1 and result["known_total_tokens"] == 60
    assert result["models"][0]["kind"] == "judge"


def test_open_judge_request_is_unknown_usage_and_restored_batch_is_not_new():
    result = model_token_usage(
        [],
        [
            {
                "trace_complete": False,
                "events": [
                    {"event": "batch_started", "batch_no": 2},
                    {"event": "batch_restored", "batch_no": 1},
                ],
                "batch_measurements": [],
            }
        ],
    )
    assert result["calls"] == result["missing_usage_calls"] == 1
    assert result["known_total_tokens"] is None


def test_controlled_executor_real_config_is_not_claimed_as_paid_model():
    trace = {
        "trace_id": "controlled",
        "trace_status": "recorded",
        "events": [
            {
                "kind": "llm",
                "span_id": "call",
                "status": "ok",
                "metadata": {
                    "role": "executor",
                    "configured_model": "custom:real-looking-route",
                    "llm_mode": "real",
                    "input_tokens": 100,
                    "output_tokens": 50,
                },
            },
        ],
    }
    row = model_token_usage([trace], [], scripted=True)["models"][0]
    assert row["llm_mode"] == "fake" and row["configured_model"] is None
    assert row["known_total_tokens"] == 150
