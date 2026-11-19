"""Drive the pinned SDK conversion and real scoring entry with content-free observations."""

import asyncio
import json
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from openai.types.chat import ChatCompletion

from evals import harness
from evals.agent_eval_ext.judge_runtime import judge_trace_artifacts, trace_judge_case
from evals.agent_eval_ext.judge_runtime_inspect import summarize
from research_engine.eval_trace import eval_trace_run, traced_ainvoke
from research_engine.observed_openai import ObservedChatOpenAI
from research_engine.response_diagnostics import (
    ResponseObservation,
    current_observation,
    diagnostic_models,
    response_observation,
    safe_response_diagnostic,
)

PRIVATE = "NEVER_RECORD_RESPONSE_OR_REASONING_913785"


def completion(content="Claim 1: YES", *, finish="stop", usage=True, tools=None):
    result = {
        "id": "chatcmpl-offline",
        "object": "chat.completion",
        "created": 1,
        "model": "offline-model",
        "choices": [
            {
                "index": 0,
                "finish_reason": finish,
                "message": {"role": "assistant", "content": content, "reasoning_content": PRIVATE},
            }
        ],
    }
    if tools:
        result["choices"][0]["message"]["tool_calls"] = tools
    if usage:
        result["usage"] = {
            "prompt_tokens": 12,
            "completion_tokens": 1000,
            "total_tokens": 1012,
            "completion_tokens_details": {"reasoning_tokens": 997},
        }
    return result


def model_for(payload, *, typed=False):
    calls = []
    if typed:
        payload = ChatCompletion.model_validate(payload)
        payload._request_id = "req-offline-123"

    async def create(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(parse=lambda: payload, headers={"authorization": PRIVATE})

    client = SimpleNamespace(with_raw_response=SimpleNamespace(create=create))
    return ObservedChatOpenAI(
        model="offline-model",
        api_key="offline-key",
        max_tokens=1000,
        max_retries=0,
        async_client=client,
    ), calls


@pytest.mark.parametrize(
    "content,finish,category",
    [
        ("Claim 1: YES", "stop", "normal"),
        ("", "length", "output_truncated"),
        ("Claim 1: YES", "length", "output_truncated"),
        ("", "stop", "provider_empty"),
        (None, "stop", "provider_empty"),
        ("", None, "unknown"),
        ([{"type": "image", "data": PRIVATE}], "stop", "nontext_response"),
    ],
)
async def test_real_sdk_boundary_categories(content, finish, category):
    model, calls = model_for(completion(content, finish=finish))
    with response_observation("batch-1") as obs:
        message = await traced_ainvoke(model, [HumanMessage(content="offline")], role="judge")
        chars = len(message.content) if isinstance(message.content, str) else 0
    doc = obs.snapshot(harness_text_chars=chars)
    assert doc["category"] == category
    assert doc["classification_confirmed"] == (category != "unknown")
    assert doc["provider"]["output_tokens"] == 1000
    assert doc["provider"]["reasoning_tokens"] == 997
    assert doc["request_parameters"]["max_completion_tokens"] == 1000
    assert doc["request_parameters"]["max_tokens"] == "unavailable"
    assert PRIVATE not in json.dumps(doc)
    assert len(calls) == 1 and not doc["root_cause_confirmed"]
    assert current_observation() is None


async def test_typed_provider_request_id_and_actual_request_override():
    model, calls = model_for(completion(), typed=True)
    with response_observation("batch-typed") as obs:
        await model.ainvoke([HumanMessage(content="offline")], max_tokens=27)
    doc = obs.snapshot(harness_text_chars=12)
    assert doc["provider"]["request_id"] == "req-offline-123"
    assert doc["provider"]["response_id"] == "chatcmpl-offline"
    assert doc["request_parameters"]["max_completion_tokens"] == 27
    assert calls[0]["max_completion_tokens"] == 27


@pytest.mark.parametrize("raises", [False, True])
async def test_sdk_conversion_loss_or_exception_preserves_behavior(monkeypatch, raises):
    import langchain_openai.chat_models.base as sdk

    def damaged(_):
        if raises:
            raise ValueError(PRIVATE)
        return AIMessage(content="")

    monkeypatch.setattr(sdk, "_convert_dict_to_message", damaged)
    model, calls = model_for(completion(PRIVATE))
    with response_observation("batch-conversion") as obs:
        if raises:
            with pytest.raises(ValueError, match=PRIVATE):
                await model.ainvoke([HumanMessage(content="offline")])
        else:
            assert (await model.ainvoke([HumanMessage(content="offline")])).content == ""
    doc = obs.snapshot(harness_text_chars="unavailable" if raises else 0)
    assert doc["category"] == "sdk_conversion_empty"
    assert doc["provider"]["text_chars"] == len(PRIVATE)
    assert doc["conversion_error"] == raises
    assert PRIVATE not in json.dumps(doc) and len(calls) == 1


async def test_missing_metadata_never_inferred_from_equal_token_cap():
    model, _ = model_for(completion("", finish=None, usage=False))
    with response_observation("batch-missing") as obs:
        await model.ainvoke([HumanMessage(content="offline")])
    doc = obs.snapshot(harness_text_chars=0)
    assert doc["provider"]["finish_reason"] == "unavailable"
    assert doc["provider"]["output_tokens"] == "unavailable"
    assert doc["provider"]["reasoning_tokens"] == "unavailable"
    assert doc["provider"]["request_id"] == "unavailable"
    assert doc["category"] == "unknown"
    empty = ResponseObservation("batch-other-provider").snapshot()
    assert all(v == "unavailable" for v in empty["provider"].values())


async def test_harness_links_trace_badcase_and_keeps_unjudged(monkeypatch, tmp_path):
    model, calls = model_for(completion("", finish="length"), typed=True)
    monkeypatch.setattr(harness, "_judge_llm", lambda: model)
    report = "A sufficiently long supported finding [1]."
    with trace_judge_case(
        case_id="offline", trace_dir=tmp_path, total_claims=1, capture_content=True
    ) as judge:
        with eval_trace_run(
            case_id="offline",
            run_id="offline-run",
            directory=tmp_path,
            enabled=True,
            capture_content=True,
        ) as graph:
            rate, rows = await harness.judge_citation_support(
                report, [{"index": 1, "snippet": "public"}]
            )
    assert rate is None and not rows[0]["judged"] and len(calls) == 1
    events = [json.loads(line) for line in judge.path.read_text().splitlines()]
    batch = next(e for e in events if e["event"] == "batch_finished")
    doc = batch["response_diagnostic"]
    graph_events = [json.loads(line) for line in graph.path.read_text().splitlines()]
    graph_doc = next(e["response_diagnostic"] for e in graph_events if e.get("response_diagnostic"))
    assert doc == graph_doc
    assert doc["batch_id"] and doc["observation_id"]
    bad = json.loads(judge.response_bad_case_path.read_text())
    assert bad["response_problems"][0]["seq"] == batch["seq"]
    assert bad["response_problems"][0]["response_diagnostic"] == doc
    assert summarize(judge.path)["problem_batches"][0]["response_diagnostic"] == doc
    artifacts = judge_trace_artifacts(judge, requested=True, artifact_root=tmp_path)
    assert artifacts["judge_response_bad_case"] == judge.response_bad_case_path.name
    for path in tmp_path.iterdir():
        assert PRIVATE not in path.read_text()


async def test_no_response_body_even_with_content_capture(monkeypatch, tmp_path):
    model, _ = model_for(completion("Claim 1: YES\n" + PRIVATE))
    monkeypatch.setattr(harness, "_judge_llm", lambda: model)
    with trace_judge_case(case_id="normal", trace_dir=tmp_path, capture_content=True):
        with eval_trace_run(
            case_id="normal",
            run_id="normal",
            directory=tmp_path,
            enabled=True,
            capture_content=True,
        ):
            rate, rows = await harness.judge_citation_support(
                "A sufficiently long finding [1].", []
            )
    assert rate == 1.0 and rows[0]["judged"]
    assert all(PRIVATE not in path.read_text() for path in tmp_path.iterdir())


async def test_concurrent_calls_do_not_mix_observations():
    async def run(name, text):
        model, _ = model_for(completion(text))
        with response_observation(name) as obs:
            await model.ainvoke([HumanMessage(content="offline")])
        return obs.snapshot(harness_text_chars=len(text))

    a, b = await asyncio.gather(run("batch-a", "one"), run("batch-b", "longer"))
    assert a["batch_id"] == "batch-a" and b["batch_id"] == "batch-b"
    assert a["provider"]["text_chars"] == 3 and b["provider"]["text_chars"] == 6
    assert a["observation_id"] != b["observation_id"]


def test_factory_opt_in_does_not_change_business_models():
    from langchain_openai import ChatOpenAI

    from research_engine import llm_factory

    assert type(llm_factory._build("ollama", "offline", "critic")) is ChatOpenAI
    with diagnostic_models():
        assert type(llm_factory._build("ollama", "offline", "critic")) is ObservedChatOpenAI
    assert type(llm_factory._build("ollama", "offline", "critic")) is ChatOpenAI


async def test_openai_http_mock_preserves_response_and_header_request_id():
    import httpx2
    from openai import AsyncOpenAI

    requests = []

    def endpoint(request):
        requests.append(json.loads(request.content))
        return httpx2.Response(
            200,
            json=completion("", finish="length"),
            headers={"x-request-id": "req-http-123", "set-cookie": PRIVATE},
        )

    async with httpx2.AsyncClient(transport=httpx2.MockTransport(endpoint)) as http:
        client = AsyncOpenAI(
            api_key="offline-key", base_url="https://offline.invalid/v1", http_client=http
        )
        model = ObservedChatOpenAI(
            model="offline-model",
            api_key="offline-key",
            max_tokens=1000,
            async_client=client.chat.completions,
            root_async_client=client,
        )
        with response_observation("batch-http") as obs:
            response = await model.ainvoke([HumanMessage(content="offline")])
    doc = obs.snapshot(harness_text_chars=0)
    assert response.content == "" and len(requests) == 1
    assert doc["provider"]["request_id"] == "req-http-123"
    assert doc["provider"]["boundary"] == "http_response_json"
    assert doc["provider_parsed"]["boundary"] == "openai_parsed_chat_completion"
    assert doc["wire_response"] == "json_metadata_observed"
    assert doc["provider"]["finish_reason"] == "length"
    assert (
        doc["request_parameters"]["max_completion_tokens"] == requests[0]["max_completion_tokens"]
    )
    assert PRIVATE not in json.dumps(doc)


async def test_http_json_detects_loss_during_openai_sdk_parse():
    import httpx2

    model, _ = model_for(completion(""))
    body = completion(PRIVATE)
    raw_http = httpx2.Response(200, json=body, headers={"x-request-id": "req-parse-loss"})

    async def create(**_):
        return SimpleNamespace(http_response=raw_http, parse=lambda: completion(""))

    model.async_client.with_raw_response.create = create
    # Install the delegating observer around this controlled SDK response boundary.
    from research_engine.observed_openai import _ObservedRawCompletions

    model.async_client.with_raw_response = _ObservedRawCompletions(
        model.async_client.with_raw_response
    )
    with response_observation("batch-parse-loss") as obs:
        response = await model.ainvoke([HumanMessage(content="offline")])
    doc = obs.snapshot(harness_text_chars=0)
    assert response.content == "" and doc["category"] == "sdk_conversion_empty"
    assert doc["provider"]["text_chars"] == len(PRIVATE)
    assert doc["provider_parsed"]["text_chars"] == 0
    assert PRIVATE not in json.dumps(doc)


async def test_non_json_http_does_not_leak_error_preview(monkeypatch, tmp_path, capsys):
    import httpx2
    from openai import AsyncOpenAI

    def endpoint(_):
        return httpx2.Response(200, text=PRIVATE, headers={"x-request-id": "req-non-json"})

    async with httpx2.AsyncClient(transport=httpx2.MockTransport(endpoint)) as http:
        client = AsyncOpenAI(
            api_key="offline-key", base_url="https://offline.invalid/v1", http_client=http
        )
        model = ObservedChatOpenAI(
            model="offline-model",
            api_key="offline-key",
            max_tokens=1000,
            async_client=client.chat.completions,
            root_async_client=client,
        )
        monkeypatch.setattr(harness, "_judge_llm", lambda: model)
        with trace_judge_case(case_id="non-json", trace_dir=tmp_path) as trace:
            rate, rows = await harness.judge_citation_support(
                "A sufficiently long finding [1].", []
            )
    assert rate is None and not rows[0]["judged"]
    event = next(
        json.loads(line)
        for line in trace.path.read_text().splitlines()
        if json.loads(line).get("response_diagnostic")
    )
    doc = event["response_diagnostic"]
    assert doc["provider"]["request_id"] == "req-non-json"
    assert doc["wire_response"] == "unavailable"
    assert doc["category"] == "sdk_conversion_error"
    assert PRIVATE not in capsys.readouterr().out
    assert all(PRIVATE not in p.read_text() for p in tmp_path.iterdir())


async def test_opaque_actual_judge_key_echo_in_identifiers_is_redacted():
    import httpx2
    from openai import AsyncOpenAI

    key = "opaquejudgecredential94327"

    def endpoint(_):
        payload = completion()
        payload["id"] = key
        return httpx2.Response(200, json=payload, headers={"x-request-id": key})

    async with httpx2.AsyncClient(transport=httpx2.MockTransport(endpoint)) as http:
        client = AsyncOpenAI(api_key=key, base_url="https://offline.invalid/v1", http_client=http)
        model = ObservedChatOpenAI(
            model="offline-model",
            api_key=key,
            max_tokens=1000,
            async_client=client.chat.completions,
            root_async_client=client,
        )
        with response_observation("batch-opaque") as obs:
            await model.ainvoke([HumanMessage(content="offline")])
    doc = obs.snapshot(harness_text_chars=12)
    assert doc["provider"]["request_id"] == "unavailable"
    assert doc["provider"]["response_id"] == "unavailable"
    assert doc["provider_parsed"]["request_id"] == "unavailable"
    assert key not in json.dumps(doc) and key not in repr(obs)


async def test_tool_only_reply_is_nontext_and_carries_no_arguments():
    tools = [
        {
            "id": "tool-1",
            "type": "function",
            "function": {"name": "anything", "arguments": json.dumps({"data": PRIVATE})},
        }
    ]
    model, _ = model_for(completion(None, finish="tool_calls", tools=tools))
    with response_observation("batch-tool") as obs:
        response = await model.ainvoke([HumanMessage(content="offline")])
    doc = obs.snapshot(harness_text_chars=0)
    assert response.tool_calls and doc["category"] == "nontext_response"
    assert PRIVATE not in json.dumps(doc) and "anything" not in json.dumps(doc)


async def test_conversion_exception_does_not_leak_into_trace_or_stdout(
    monkeypatch, tmp_path, capsys
):
    import langchain_openai.chat_models.base as sdk

    def broken(_):
        raise ValueError(PRIVATE)

    monkeypatch.setattr(sdk, "_convert_dict_to_message", broken)
    model, calls = model_for(completion(PRIVATE))
    monkeypatch.setattr(harness, "_judge_llm", lambda: model)
    with trace_judge_case(case_id="broken", trace_dir=tmp_path, capture_content=True) as judge:
        with eval_trace_run(
            case_id="broken",
            run_id="broken",
            directory=tmp_path,
            enabled=True,
            capture_content=True,
        ):
            rate, rows = await harness.judge_citation_support(
                "A sufficiently long finding [1].", []
            )
    assert rate is None and not rows[0]["judged"] and len(calls) == 1
    assert PRIVATE not in capsys.readouterr().out
    assert all(PRIVATE not in p.read_text() for p in tmp_path.iterdir())
    doc = next(
        json.loads(line)["response_diagnostic"]
        for line in judge.path.read_text().splitlines()
        if json.loads(line).get("response_diagnostic")
    )
    assert doc["category"] == "sdk_conversion_empty"
    assert doc["harness_text_chars"] == "unavailable"


def test_untrusted_artifact_projection_is_strict_and_recomputes_category():
    doc = ResponseObservation("batch-project").snapshot()
    doc.update(
        {
            "category": "output_truncated",
            "classification_confirmed": True,
            "root_cause_confirmed": True,
            "headers": {"authorization": PRIVATE},
            "body": PRIVATE,
        }
    )
    doc["provider"].update(
        {
            "finish_reason": PRIVATE,
            "reasoning_content": PRIVATE,
            "request_id": "sk-abcdefghijsecret",
            "output_tokens": True,
        }
    )
    clean = safe_response_diagnostic(doc)
    assert clean["category"] == "unknown" and not clean["root_cause_confirmed"]
    assert clean["provider"]["request_id"] == "unavailable"
    assert clean["provider"]["output_tokens"] == "unavailable"
    assert PRIVATE not in json.dumps(clean) and "sk-" not in json.dumps(clean)


async def test_diagnostic_reaches_api_projection(monkeypatch, tmp_path):
    from app.services import evaluation_diagnostics as diagnostics

    base = tmp_path / "artifacts/report-benchmark"
    model, _ = model_for(completion("", finish="length"))
    monkeypatch.setattr(harness, "_judge_llm", lambda: model)
    with trace_judge_case(case_id="api", trace_dir=base / "judge-batches", run_id="pipe-api"):
        await harness.judge_citation_support("A sufficiently long finding [1].", [])
    row = diagnostics.analyze(
        {"pipeline_id": "pipe-api", "case_id": None},
        {
            "dataset_id": "report-benchmark",
            "state": "failed",
            "reason": "judge_failed",
            "metrics": None,
        },
        base,
    )
    events = row["judge_traces"][0]["events"]
    doc = next(e["response_diagnostic"] for e in events if e.get("response_diagnostic"))
    assert doc["category"] == "output_truncated"
    assert PRIVATE not in json.dumps(row)


async def test_observer_failure_does_not_change_model_response(monkeypatch):
    def fail(*_):
        raise RuntimeError(PRIVATE)

    model, calls = model_for(completion())
    monkeypatch.setattr(ResponseObservation, "before_conversion", fail)
    with response_observation("batch-observer-failure") as obs:
        response = await model.ainvoke([HumanMessage(content="offline")])
    assert response.content == "Claim 1: YES" and len(calls) == 1
    assert (
        obs.snapshot(harness_text_chars=len(response.content))["provider"]["boundary"]
        == "unavailable"
    )


async def test_cancelled_judge_propagates_and_releases_scope(monkeypatch, tmp_path):
    entered, cleaned = asyncio.Event(), asyncio.Event()

    async def block(**_):
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            cleaned.set()

    model, _ = model_for(completion())
    model.async_client.with_raw_response.create = block
    monkeypatch.setattr(harness, "_judge_llm", lambda: model)

    async def run():
        with trace_judge_case(case_id="cancel", trace_dir=tmp_path):
            await harness.judge_citation_support("A sufficiently long finding [1].", [])

    task = asyncio.create_task(run())
    await asyncio.wait_for(entered.wait(), 2)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert cleaned.is_set() and current_observation() is None
    events = [json.loads(line) for line in next(tmp_path.glob("*.jsonl")).read_text().splitlines()]
    batch = next(e for e in events if e["event"] == "batch_finished")
    assert batch["response_diagnostic"]["provider"]["boundary"] == "unavailable"
    assert batch["response_diagnostic"]["category"] == "unknown"
    assert batch["status"] == "cancelled"


async def test_committed_observations_restore_without_rescoring(monkeypatch, tmp_path):
    import test_long_run_reliability as fixtures

    from evals.agent_eval_ext import long_run

    for name in fixtures._DEADLINE_ENV:
        monkeypatch.delenv(name, raising=False)
    ctx, store = fixtures.make_context(tmp_path, fixtures.Clock())
    store.save_draft(fixtures.CASE_ID, fixtures.draft_doc())
    text = "\n".join(f"Claim {i}: YES" for i in range(1, 5))
    model, calls = model_for(completion(text))
    monkeypatch.setattr(harness, "_judge_llm", lambda: model)
    with long_run.active(ctx):
        first = await harness.judge_citation_support(fixtures.REPORT, fixtures.SOURCES)
    saved = sorted(store.batches_dir.glob("batch-*.json"))
    assert len(saved) == 3 and len(calls) == 3
    snapshots = {p.name: p.read_bytes() for p in saved}
    for path in saved:
        batch = json.loads(path.read_text())
        assert batch["batch_id"] == batch["response_diagnostic"]["batch_id"]
        store.validate_batch(batch, batch["batch_no"], fixtures.draft_doc())
    resumed, resume_store = fixtures.make_context(
        tmp_path, fixtures.Clock(), name="resume", resume_root=ctx.root
    )
    resume_store.load(case_id=fixtures.CASE_ID)
    with long_run.active(resumed):
        with trace_judge_case(case_id=fixtures.CASE_ID, trace_dir=tmp_path / "restored") as trace:
            second = await harness.judge_citation_support(fixtures.REPORT, fixtures.SOURCES)
    assert first == second and len(calls) == 3
    assert all(p.read_bytes() == snapshots[p.name] for p in saved)
    events = [json.loads(line) for line in trace.path.read_text().splitlines()]
    restored = [e for e in events if e["event"] == "batch_restored"]
    assert len(restored) == 3
    assert all(e["application_level_calls"] == 0 and e["response_diagnostic"] for e in restored)
