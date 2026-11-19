"""Actual graph/call instrumentation and bounded content delivery without provider traffic."""

import asyncio
import json
import os
import subprocess
import sys

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from app.services import evaluation_diagnostics as d
from evals.agent_eval_ext.execution_observer import execution_scope, subprocess_output
from evals.agent_eval_ext.judge_runtime import (
    trace_judge_batch,
    trace_judge_case,
    trace_judge_stage,
)
from research_engine import eval_trace as t
from research_engine.runconfig import RunConfig, reset_run_config, set_run_config
from research_engine.trace_content import content_snapshot


def read(path):
    return [json.loads(s) for s in path.read_text(encoding="utf8").splitlines()]


def test_default_trace_still_excludes_content_and_opt_in_captures_redacted_arguments(tmp_path):
    cfg = set_run_config(
        RunConfig(llm_mode="fake", provider_keys={"custom": "opaque-provider-credential"})
    )
    try:
        for detailed in [False, True]:
            with t.eval_trace_run(
                directory=tmp_path,
                case_id="case",
                run_id="run",
                enabled=True,
                capture_content=detailed,
            ) as rec:
                with t.trace_span(
                    "read_webpage",
                    kind="tool",
                    inputs={
                        "url": "https://example.org/page?signature=private",
                        "api_key": "opaque-provider-credential",
                    },
                ) as span:
                    span.observe(
                        output={
                            "text": "source text",
                            "error": "HTTP 403 opaque-provider-credential",
                        }
                    )
            raw = rec.path.read_text()
            assert "opaque-provider-credential" not in raw and "signature=private" not in raw
            assert ("source text" in raw) is detailed
            assert ("details" in read(rec.path)[0]) is detailed
    finally:
        reset_run_config(cfg)


@pytest.mark.parametrize(
    "payload",
    [
        {"headers": {"x-custom": "private"}, "authorization": "private"},
        {"body": "Bearer opaque123 password=opaque456 person@example.org"},
        {"url": "https://user:private@example.org/path?arbitrary=private"},
        {"image_url": "data:image/png;base64,private"},
    ],
)
def test_content_redacts_headers_credentials_urls_email_and_binary(payload):
    value = json.dumps(content_snapshot(payload))
    assert "private" not in value and "opaque123" not in value and "opaque456" not in value
    assert "person@example.org" not in value


def test_trace_read_rejects_symlink_content_sources(tmp_path):
    root = tmp_path / "job"
    root.mkdir()
    elsewhere = tmp_path / "elsewhere"
    with execution_scope(elsewhere, "pipe-p", "private"):
        pass
    (root / "execution-traces").symlink_to(elsewhere / "execution-traces", target_is_directory=True)
    assert d.traces(root, [], journal=True) == []


def test_content_reports_truncation_without_leaking_a_cut_secret():
    secret = "opaque-" + "x" * 5000
    row = content_snapshot({"text": secret + "z" * 15000}, secrets=(secret,))
    assert row["truncated"] and row["omitted_paths"]
    assert secret[:100] not in json.dumps(row)
    assert len(json.dumps(row).encode()) < 20000


async def test_actual_model_message_projection_and_parser_error_details(tmp_path):
    calls = []

    class Model:
        async def ainvoke(self, messages):
            calls.append(messages)
            return {
                "raw": AIMessage(
                    content="invalid JSON",
                    response_metadata={"headers": {"Authorization": "hidden"}},
                ),
                "parsed": None,
                "parsing_error": ValueError("missing tasks field"),
            }

    with t.eval_trace_run(
        directory=tmp_path, case_id="case", run_id="run", enabled=True, capture_content=True
    ) as rec:
        result = await t.traced_ainvoke(
            Model(),
            [SystemMessage(content="instructions"), HumanMessage(content="question")],
            role="planner",
        )
    text = rec.path.read_text()
    assert len(calls) == 1 and result["parsed"] is None
    assert all(
        x in text for x in ["question", "instructions", "invalid JSON", "missing tasks field"]
    )
    assert "hidden" not in text and '"headers"' not in text


async def test_cancelled_call_preserves_details_and_propagates_cancellation(tmp_path):
    class Model:
        async def ainvoke(self, _):
            raise asyncio.CancelledError()

    with t.eval_trace_run(
        directory=tmp_path, case_id="case", run_id="run", enabled=True, capture_content=True
    ) as rec:
        with pytest.raises(asyncio.CancelledError):
            await t.traced_ainvoke(Model(), [HumanMessage(content="question")], role="executor")
    assert read(rec.path)[-1]["status"] == "cancelled"
    assert "CancelledError" in rec.path.read_text() and t.active_trace() is None


def test_journal_and_judge_join_into_unified_trace_and_model_packet(tmp_path, monkeypatch):
    base = tmp_path / "artifacts/report-benchmark"
    base.mkdir(parents=True)
    monkeypatch.setenv("EVAL_JUDGE_API_KEY", "opaque-judge-credential")
    with execution_scope(base, "pipe-fixture", "adapter.run"):
        with subprocess_output("process") as (stream, span):
            os.write(stream, b"specific failure opaque-judge-credential")
            span.observe(exit_code=1)
        with trace_judge_case(
            case_id="case",
            trace_dir=base / "judge-batches",
            run_id="pipe-fixture",
            capture_content=True,
            redaction_secrets=("opaque-judge-credential",),
        ):
            trace_judge_stage(
                "batch_started",
                batch_no=1,
                batch_start_zero_based=0,
                expected_count=1,
                details={"input": [HumanMessage(content="claim evidence")]},
            )
            trace_judge_batch(
                batch_no=1,
                batch_start=0,
                expected_count=1,
                verdict_by_claim={},
                response=AIMessage(content="not parseable opaque-judge-credential"),
                text="not parseable",
                error=None,
                elapsed_ms=1,
            )
    req = {"pipeline_id": "pipe-fixture", "case_id": None}
    step = {
        "dataset_id": "report-benchmark",
        "state": "failed",
        "reason": "suite_execution_failed",
        "metrics": None,
    }
    row = d.analyze(req, step, base)
    assert row["unified_trace"]["coverage"]["content_events"] >= 2
    assert {e["source"] for e in row["unified_trace"]["events"]} == {"E", "J"}
    text = json.dumps(row)
    assert "specific failure" in text and "claim evidence" in text
    assert "opaque-judge-credential" not in text
    assert not d.analyze({"pipeline_id": "other"}, step, base)["execution_traces"]


def test_actual_compiled_graph_harness_cli_captures_content_and_result_save(tmp_path):
    output = tmp_path / "new.json"
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "evals.harness",
            "--case-id",
            "quantum-error-correction",
            "--trace-content",
            "--out",
            str(output),
        ],
        env={
            **os.environ,
            "LLM_MODE": "fake",
            "RF_EVAL_CONTENT_TRACE": "1",
            "RF_EVAL_EXECUTION_ROOT": str(tmp_path),
            "RF_EVAL_TRACE_RUN_ID": "pipe-offline",
        },
        capture_output=True,
        timeout=60,
    )
    assert proc.returncode == 0, proc.stderr.decode()
    assert [r["id"] for r in json.loads(output.read_text())["results"]] == [
        "quantum-error-correction"
    ]
    events = [e for p in tmp_path.glob("traces/*.jsonl") for e in read(p)]
    assert any(e["kind"] == "llm" and e["details"]["value"].get("output") for e in events)
    assert any(e["kind"] == "tool" and e["details"]["value"].get("input") for e in events)
    assert any(e["name"] == "hitl_gate" and e["status"] == "interrupt" for e in events)
    journals = [e for p in tmp_path.glob("execution-traces/*.jsonl") for e in read(p)]
    assert any(e["name"] == "result.save" and e["status"] == "ok" for e in journals)


def test_historical_trace_stays_without_content_and_no_judge_is_invented(tmp_path):
    with t.eval_trace_run(
        directory=tmp_path / "traces", case_id="case", run_id="run", enabled=True
    ):
        with t.trace_span("planner", kind="graph_node"):
            pass
    step = {
        "dataset_id": "planner-replay-v1",
        "state": "failed",
        "reason": "runner_timeout",
        "metrics": None,
    }
    row = d.analyze({"pipeline_id": "pipe-test"}, step, tmp_path)
    assert row["unified_trace"]["coverage"]["content_events"] == 0
    assert row["judge_traces"] == []


def test_content_budget_is_explicit_and_does_not_silence_span_end(tmp_path):
    with t.eval_trace_run(
        directory=tmp_path, case_id="c", run_id="r", enabled=True, capture_content=True
    ) as rec:
        rec._content_bytes = 4 * 1024 * 1024
        with t.trace_span("tool", kind="tool", inputs="large") as span:
            span.observe(output="done")
    rows = read(rec.path)
    assert [r["status"] for r in rows] == ["start", "ok"]
    assert rows[-1]["details"]["omitted_paths"] == ["$.trace_content_budget"]


def test_process_output_is_drained_bounded_and_redacted_without_raw_log_files(tmp_path):
    with execution_scope(tmp_path, "pipe-p", "worker.run"):
        with subprocess_output("process") as (fd, _):
            result = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    "import sys;sys.stdout.write('x'*300000 + ' password=private')",
                ],
                stdout=fd,
                stderr=fd,
                timeout=10,
            )
    assert result.returncode == 0
    files = list((tmp_path / "execution-traces").glob("*.jsonl"))
    text = files[0].read_text()
    assert "private" not in text and "log_tail_truncated" in text
    assert len(text) < 70000


def test_pipe_stop_after_eagain_still_drains_the_final_write(monkeypatch):
    from evals.agent_eval_ext import execution_observer as observer

    class Stop:
        stopped = False

        def is_set(self):
            return self.stopped

    stop = Stop()
    replies = iter([None, b"the final error", b""])

    def read(_fd, _size):
        value = next(replies)
        if value is None:
            stop.stopped = True
            raise BlockingIOError()
        return value

    monkeypatch.setattr(observer.os, "read", read)
    tail = bytearray()
    assert observer._drain_pipe(-1, stop, tail) == len(b"the final error")
    assert tail == b"the final error"
