"""Regress cross-phase attribution, evidence delivery and legacy-answer preservation."""

import json
from datetime import UTC, datetime, timedelta

import pytest

from app.services import evaluation_diagnostics as d
from app.services.evaluation_timeline import timestamp


def at(value):
    return (datetime(2026, 11, 14, tzinfo=UTC) + timedelta(seconds=value)).isoformat()


def graph_event(seq, span, status, when, **kw):
    return {
        "event_version": 2,
        "trace_id": "t",
        "run_id": "run",
        "case_id": "case",
        "seq": seq,
        "timestamp": at(when),
        "span_id": span,
        "parent_span_id": None,
        "kind": "graph_node",
        "name": "executor",
        "status": status,
        "duration_ms": None if status == "start" else 1000,
        "error_type": None,
        "metadata": {},
        **kw,
    }


def evidence(tmp_path, *, explicit=None, closed=False, run="run"):
    base = tmp_path / "artifacts"
    (base / "traces").mkdir(parents=True)
    (base / "judge-batches").mkdir()
    graph = [
        graph_event(1, "root", "start", 1),
        graph_event(2, "read", "start", 2, parent_span_id="root", kind="tool", name="read_webpage"),
        graph_event(
            3, "read", "business_error", 3, parent_span_id="root", kind="tool", name="read_webpage"
        ),
        graph_event(4, "root", "ok", 179),
        graph_event(5, "gate", "start", 179.5, name="hitl_gate"),
        graph_event(6, "gate", "interrupt", 180, name="hitl_gate"),
    ]
    (base / "traces/trace.jsonl").write_text("\n".join(json.dumps(e) for e in graph))
    events = []

    def add(kind, when, **kw):
        events.append(
            {
                "schema_version": 1,
                "trace_id": "judge",
                "run_id": run,
                "case_id": "case",
                "phase": "baseline",
                "seq": len(events) + 1,
                "timestamp": at(when),
                "event": kind,
                **kw,
            }
        )

    add("case_start", 180)
    for i in range(9):
        add(
            "batch_finished",
            190 + i * 25,
            batch_no=i + 1,
            batch_start_zero_based=i * 4,
            expected_count=4,
            judged_count=0 if i == 2 else 4,
            unjudged_count=4 if i == 2 else 0,
            yes_count=0 if i == 2 else 4,
            no_count=0,
            status="unjudged" if i == 2 else "complete",
            reason_hint="empty_or_nontext_response" if i == 2 else None,
            elapsed_ms=10000,
            application_level_calls=1,
            sdk_retry_attempts=None,
        )
    if explicit == "wait":
        add("wait_started", 390.01, batch_no=9, wait_seconds=15)
    if explicit == "request":
        add("batch_started", 394, batch_no=10, batch_start_zero_based=36, expected_count=4)
    if closed:
        add("case_end", 391, trace_available=True, status="finished")
    (base / "judge-batches/judge.jsonl").write_text("\n".join(json.dumps(e) for e in events))
    (base / "execution-start.json").write_text(
        json.dumps({"started_at": at(0), "case_ids": ["case"]})
    )
    (base / "execution-end.json").write_text(json.dumps({"finished_at": at(395)}))
    step = {
        "dataset_id": "report-benchmark",
        "state": "failed",
        "reason": "runner_timeout",
        "elapsed_seconds": 395,
        "metrics": None,
    }
    request = {"pipeline_id": "pipe-fixture", "case_id": "case"}
    return base, request, step


def test_recovered_web_error_is_not_terminal_phase_and_legacy_wait_is_only_inferred(tmp_path):
    base, request, step = evidence(tmp_path)
    before = {p: p.read_bytes() for p in base.rglob("*") if p.is_file()}
    row = d.analyze(request, step, base)
    assert row["failure_phase"] == "judge_wait_inferred"
    assert row["timeline"]["certainty"] == "phase_observed_wait_inferred"
    assert row["timeline"]["stages"][0]["duration_s"] == 179
    judge = row["timeline"]["stages"][1]
    assert judge["duration_s"] == 215 and judge["completed_request_s"] == 90
    assert judge["outside_completed_request_s"] == 125 and judge["closed"] is False
    assert len(row["candidates"]) == 1 and "读取" not in row["candidates"][0]["cause"]
    assert row["timeline"]["symptoms"][0]["workflow_continued"] is True
    assert row["quality_measurement"] == "unmeasured"
    assert all(p.read_bytes() == raw for p, raw in before.items())


@pytest.mark.parametrize(
    ("explicit", "phase"), [("wait", "judge_wait"), ("request", "judge_request")]
)
def test_explicit_judge_phase_is_observed_not_guessed(tmp_path, explicit, phase):
    base, req, step = evidence(tmp_path, explicit=explicit)
    row = d.analyze(req, step, base)
    assert row["failure_phase"] == phase
    assert row["timeline"]["certainty"] == "observed"
    assert row["timeline"]["evidence_ids"][-1].startswith("J0:")


def test_finished_judge_points_to_postprocessing_not_recovered_tool(tmp_path):
    base, req, step = evidence(tmp_path, closed=True)
    assert d.analyze(req, step, base)["failure_phase"] == "postprocessing"


def test_judge_from_different_run_is_not_correlated(tmp_path):
    base, req, step = evidence(tmp_path, run="another-run")
    row = d.analyze(req, step, base)
    assert row["failure_phase"] == "after_graph"
    assert row["judge_traces"][0]["trace_complete"] is False
    assert row["timeline"]["certainty"] == "insufficient"


@pytest.mark.parametrize(
    "bad", ["not-a-date", "2026-01-01T00:00:00", False, None, "2026-02-30T00:00:00+00:00"]
)
def test_bad_timestamps_do_not_become_measured_times(bad):
    assert timestamp(bad) is None


def test_future_or_clock_reversed_judge_is_not_terminal_evidence(tmp_path):
    base, req, step = evidence(tmp_path)
    file = base / "judge-batches/judge.jsonl"
    rows = [json.loads(x) for x in file.read_text().splitlines()]
    rows[-1]["timestamp"] = at(500)
    file.write_text("\n".join(json.dumps(x) for x in rows))
    assert d.analyze(req, step, base)["failure_phase"] == "after_graph"


def test_phase_from_earlier_case_does_not_override_later_open_graph(tmp_path):
    base, req, step = evidence(tmp_path, closed=True)
    file = base / "traces/later.jsonl"
    row = graph_event(1, "next", "start", 393)
    row.update(case_id="second", trace_id="second-trace", run_id="run")
    file.write_text(json.dumps(row) + "\n")
    (base / "execution-start.json").write_text(
        json.dumps({"started_at": at(0), "case_ids": ["case", "second"]})
    )
    result = d.analyze(req, step, base)
    assert result["failure_phase"] == "executor"
    assert result["timeline"]["certainty"] == "observed_open_span"


def test_expected_errors_in_passing_contract_do_not_become_a_failed_run(tmp_path):
    base, req, step = evidence(tmp_path)
    step.update(state="completed", reason=None, metrics={"gate_passed": True})
    row = d.analyze(req, step, base)
    assert row["has_issue"] is False and row["candidates"] == []
    assert row["failure_phase"] is None and row["timeline"]["certainty"] == "no_incident"
