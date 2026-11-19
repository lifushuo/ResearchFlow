"""Execution evidence, incident routing and optional model assistance without paid calls."""

import json
import subprocess
from uuid import uuid4

import pytest

from app.api.v1 import evaluation
from app.services import evaluation_diagnostics as d
from app.services import evaluation_pipeline_v13 as p
from evals.agent_eval_ext import evaluation_pipeline_worker_v13 as worker


@pytest.fixture
def root(tmp_path, monkeypatch):
    monkeypatch.setenv("RF_EVAL_JOB_ROOT", str(tmp_path))
    monkeypatch.setenv("RF_EVAL_ONLINE_ENABLED", "1")
    monkeypatch.setenv("EVAL_JUDGE_MODEL", "custom:fixture")
    return tmp_path


def failed(root, dataset="report-benchmark", *, reason="runner_timeout", owner="alice", case=None):
    pipe = p.create_pipeline(
        root,
        owner,
        [dataset],
        str(uuid4()),
        1,
        p.PAID_CONFIRM if p.DATASETS[dataset] == "online" else None,
        **({"case_id": case} if case else {}),
    )
    folder, request, index = p.claim_next(root, p.DATASETS[dataset])
    p.finish_step(
        root,
        folder,
        index,
        state="failed",
        reason=reason,
        metrics=None,
        case_count=None,
        elapsed_seconds=420,
    )
    return pipe["pipeline_id"], folder, request


def event(seq, span, status, **kwargs):
    return {
        "event_version": 2,
        "trace_id": "t",
        "run_id": "run",
        "case_id": "quantum-error-correction",
        "seq": seq,
        "span_id": span,
        "parent_span_id": None,
        "name": "planner",
        "kind": "llm",
        "status": status,
        "metadata": {},
        **kwargs,
    }


def trace(folder, rows, suffix=""):
    base = folder / "artifacts/report-benchmark/traces"
    base.mkdir(parents=True, exist_ok=True)
    file = base / f"trace{suffix}.jsonl"
    file.write_text("\n".join(json.dumps(x) for x in rows) + "\n")
    return file


def test_historical_timeout_is_actionable_without_fabricating_trace_or_scores(root):
    pid, folder, _ = failed(root, case="quantum-error-correction")
    before = (folder / "status.json").read_bytes()
    result = d.read_diagnostics(root, "alice", pid, "report-benchmark")
    assert result["reason"] == "runner_timeout"
    assert result["collection"] == "saved_status_analysis"
    assert result["elapsed_seconds"] == 420
    assert result["quality_measurement"] == "unmeasured"
    assert result["traces"] == [] and result["cases"] == []
    assert result["candidates"] and result["next_actions"] and result["missing_evidence"]
    assert result["root_cause_confirmed"] is False
    assert (folder / "status.json").read_bytes() == before


def test_large_report_has_cases_errors_and_case_scoped_evidence(root):
    pid, folder, _ = failed(root)
    base = folder / "artifacts/report-benchmark"
    base.mkdir(parents=True)
    artifact = base / "raw_result.json"
    artifact.write_text(
        json.dumps(
            {
                "padding": "x" * 300000,
                "results": [
                    {
                        "id": "bad-case",
                        "completed": False,
                        "error": "planner: could not produce a valid task list",
                        "latency_s": 9.15,
                    },
                    {"id": "good-case", "completed": True, "latency_s": 10},
                ],
            }
        )
    )
    trace(
        folder,
        [event(1, "p", "start", case_id="bad-case"), event(2, "p", "error", case_id="bad-case")],
    )
    trace(
        folder,
        [event(1, "q", "start", case_id="good-case"), event(2, "q", "ok", case_id="good-case")],
        "other",
    )
    before = artifact.read_bytes()
    selected = d.read_diagnostics(root, "alice", pid, "report-benchmark", "bad-case")
    assert selected["cases"][0]["error"].startswith("planner:")
    assert selected["available_case_ids"] == ["bad-case", "good-case"]
    assert selected["failure_phase"] == "planner"
    assert all(t.get("case_id") == "bad-case" for t in selected["traces"])
    assert not any(e["source"] == "E" for e in selected["unified_trace"]["events"])
    incidents = d.incident_queue(root, "alice", pid, "report-benchmark")
    assert [row["case_id"] for row in incidents] == ["bad-case"]
    assert d.incident_queue(root, "bob", pid, "report-benchmark") == []
    assert artifact.read_bytes() == before
    with pytest.raises(KeyError):
        d.read_diagnostics(root, "alice", pid, "report-benchmark", "missing-case")


def test_oversized_diagnostic_artifact_is_still_rejected(tmp_path):
    (tmp_path / "raw_result.json").write_bytes(b"x" * (d.MAX_BYTES + 1))
    assert d.case_observations(tmp_path, "report-benchmark") == []


@pytest.mark.parametrize("dataset", p.DATASETS)
def test_all_eight_failed_adapters_enter_owner_scoped_incident_queue(root, dataset):
    pid, _, _ = failed(root, dataset)
    assert d.read_diagnostics(root, "alice", pid, dataset)["has_issue"] is True
    assert d.incident_queue(root, "alice")[0]["pipeline_id"] == pid
    assert d.incident_queue(root, "bob") == []
    with pytest.raises(KeyError):
        d.read_diagnostics(root, "bob", pid, dataset)


def test_partial_trace_preserves_open_llm_and_filters_secrets(root):
    pid, folder, _ = failed(root, case="quantum-error-correction")
    trace(
        folder,
        [
            event(
                1,
                "p",
                "start",
                metadata={
                    "role": "planner",
                    "api_key": "private",
                    "prompt": "private",
                    "input_tokens": 20,
                },
            )
        ],
    )
    result = d.read_diagnostics(root, "alice", pid, "report-benchmark")
    assert result["failure_phase"] == "planner"
    assert result["traces"][0]["trace_status"] == "partial"
    assert result["traces"][0]["open_spans"] == ["p"]
    assert "private" not in json.dumps(result)
    assert len(result["traces"][0]["events"]) == 1


def test_per_case_metrics_expose_numeric_contract_checks_without_arguments(root):
    _, folder, _ = failed(root, "tool-contract-v1")
    base = folder / "artifacts/tool-contract-v1/artifacts"
    base.mkdir(parents=True)
    (base / "tool-result.json").write_text(
        json.dumps(
            {
                "cases": [
                    {
                        "case_id": "tool-case",
                        "pass": False,
                        "selection_correct": True,
                        "arguments_correct": False,
                        "args": {"query": "private query"},
                        "metrics": {"precision": 0.5, "api_key": 12345, "body": "private body"},
                    }
                ]
            }
        )
    )
    result = d.case_observations(folder / "artifacts/tool-contract-v1", "tool-contract-v1")
    assert result[0]["metrics"]["arguments_correct"] is False
    assert result[0]["metrics"]["selection_correct"] is True
    assert "private" not in json.dumps(result) and "12345" not in json.dumps(result)


@pytest.mark.parametrize(
    "dataset,body",
    [
        ("corpus-retrieval", {"precision": {"@5": 0.4}, "recall": {"@5": None}}),
        ("selection-retrieval", {"with_cap": {"precision": {"@5": 0.4}, "recall": {"@5": None}}}),
    ],
)
def test_retrieval_metric_layouts_preserve_at_k_and_missing_values(root, dataset, body):
    _, folder, _ = failed(root, dataset)
    base = folder / "artifacts" / dataset
    base.mkdir(parents=True)
    (base / "raw_result.json").write_text(json.dumps({"queries": [{"id": "query-1", **body}]}))
    result = d.case_observations(base, dataset)[0]
    observed = result["metrics"].get("with_cap", result["metrics"])
    assert observed["precision"]["@5"] == 0.4
    assert observed["recall"]["@5"] is None


def test_frontend_case_verdicts_reuse_canonical_worker_adapter(root):
    _, folder, _ = failed(root, "grounding-v1")
    base = folder / "artifacts/grounding-v1"
    base.mkdir(parents=True)
    (base / "case-observations.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "dataset_id": "grounding-v1",
                "cases": [{"case_id": "g-1", "status": "failed", "metrics": {"supported": False}}],
            }
        )
    )
    assert d.case_observations(base, "grounding-v1")[0]["status"] == "failed"


@pytest.mark.parametrize(
    "error,http,expected",
    [
        ("ValidationError", None, "结构"),
        ("RateLimitError", 429, "限流"),
        ("ReadTimeout", None, "超时"),
        ("AuthenticationError", 401, "权限"),
    ],
)
def test_exception_evidence_produces_specific_hypotheses(root, error, http, expected):
    pid, folder, _ = failed(root, reason="suite_execution_failed")
    trace(
        folder,
        [
            event(1, "p", "start"),
            event(2, "p", "error", error_type=error, metadata={"http_status": http}),
        ],
    )
    result = d.read_diagnostics(root, "alice", pid, "report-benchmark")
    assert expected in result["candidates"][0]["cause"]
    assert result["traces"][0]["trace_status"] == "recorded"
    assert result["candidates"][0]["evidence_ids"]


@pytest.mark.parametrize(
    "change", ["mixed", "sequence", "truncated", "duplicate", "boolean_sequence"]
)
def test_broken_trace_never_becomes_complete(root, change):
    pid, folder, _ = failed(root)
    rows = [event(1, "p", "start"), event(2, "p", "ok")]
    if change == "mixed":
        rows[1]["case_id"] = "other"
    elif change == "sequence":
        rows[1]["seq"] = 3
    elif change == "duplicate":
        rows[1]["status"] = "start"
    elif change == "boolean_sequence":
        rows[0]["seq"] = True
    file = trace(folder, rows)
    if change == "truncated":
        file.write_text(json.dumps(rows[0]) + '\n{"event_version":')
    result = d.read_diagnostics(root, "alice", pid, "report-benchmark")
    assert result["traces"][0]["trace_status"] == "partial"


def test_worker_start_and_finish_observation_automatically_collect_timeout(root, monkeypatch):
    monkeypatch.setenv("LLM_MODE", "real")
    pipe = p.create_pipeline(
        root,
        "alice",
        ["report-benchmark"],
        str(uuid4()),
        1,
        p.PAID_CONFIRM,
        case_id="quantum-error-correction",
    )
    monkeypatch.setattr(
        worker,
        "_execute_step",
        lambda *_: {
            "state": "failed",
            "reason": "runner_timeout",
            "metrics": None,
            "case_count": None,
        },
    )
    assert worker.run_once(root, "online")
    diagnostic = d.read_diagnostics(root, "alice", pipe["pipeline_id"], "report-benchmark")
    assert diagnostic["collection"] == "worker_observation"
    assert diagnostic["input_signature"]
    assert diagnostic["quality_measurement"] == "unmeasured"
    assert d.incident_queue(root, "alice")
    from evals.agent_eval_ext.closure_observer import store_at

    store = store_at(root)
    cases = store.list_bad_cases("alice")
    assert len(cases) == 1 and cases[0]["case_id"] == "quantum-error-correction"
    gate = store.quality_gate("alice", pipe["pipeline_id"] + "--report-benchmark")
    assert gate["verdict"] != "passed" and gate["measured"] == 0
    assert worker.run_once(root, "online") is False


def test_preexisting_step_artifacts_cannot_dispatch_or_be_overwritten(root, monkeypatch):
    pipe = p.create_pipeline(root, "alice", ["report-benchmark"], str(uuid4()), 1, p.PAID_CONFIRM)
    folder = p._folder(root, pipe["pipeline_id"])
    base = folder / "artifacts/report-benchmark"
    base.mkdir(parents=True)
    file = base / "existing-evidence.json"
    file.write_text('{"preserved":true}')

    def forbidden(*args):
        pytest.fail("preexisting execution directory reached model dispatch")

    monkeypatch.setattr(worker, "_execute_step", forbidden)
    assert worker.run_once(root, "online")
    assert file.read_text() == '{"preserved":true}'
    assert (
        p.get_pipeline(root, "alice", pipe["pipeline_id"])["steps"][0]["reason"]
        == "execution_artifacts_already_exist"
    )


def test_unexpected_diagnostic_error_never_exposes_private_details():
    from fastapi import HTTPException

    for error in [ValueError("opaque private provider error"), OSError("secret path")]:

        def reject(error=error):
            raise error

        with pytest.raises(HTTPException) as caught:
            evaluation._diagnostic_call(reject)
        assert "private" not in caught.value.detail and "secret path" not in caught.value.detail


def test_compiled_report_failure_reaches_frontend_diagnostics_with_real_schema_trace(root):
    import os
    import sys

    pipe = p.create_pipeline(
        root,
        "alice",
        ["report-benchmark"],
        str(uuid4()),
        1,
        p.PAID_CONFIRM,
        case_id="quantum-error-correction",
    )
    folder, _, index = p.claim_next(root, "online")
    base = folder / "artifacts/report-benchmark"
    base.mkdir(parents=True)
    output = base / "raw_result.json"
    args = [
        "harness",
        "--case-id",
        "quantum-error-correction",
        "--trace",
        "--trace-dir",
        str(base / "traces"),
        "--out",
        str(output),
    ]
    program = (
        "import asyncio,sys;from evals import harness;from research_engine import graph;"
        "from evals.agent_eval_ext.replay import _FaultPlanner;original=graph.get_llm;"
        "planner=_FaultPlanner('planner_invalid_schema');"
        "graph.get_llm=lambda role: planner if role=='planner' else original(role);"
        "sys.argv=" + repr(args) + ";asyncio.run(harness.main())"
    )
    proc = subprocess.run(
        [sys.executable, "-c", program],
        env={**os.environ, "LLM_MODE": "fake"},
        capture_output=True,
        timeout=45,
    )
    assert proc.returncode == 0, proc.stderr.decode()
    summary = worker._summary("report-benchmark", output, case_id="quantum-error-correction")
    p.finish_step(root, folder, index, reason=None, elapsed_seconds=1, **summary)
    result = d.read_diagnostics(root, "alice", pipe["pipeline_id"], "report-benchmark")
    assert result["has_issue"] and result["cases"][0]["status"] == "failed"
    assert result["traces"][0]["trace_status"] == "recorded"
    assert any(e["error_type"] == "ValidationError" for e in result["traces"][0]["events"])
    assert any("结构" in c["cause"] for c in result["candidates"])


@pytest.mark.skipif(
    not hasattr(__import__("os"), "killpg"),
    reason="Process-group timeout is a Linux worker boundary",
)
def test_real_subprocess_timeout_preserves_already_flushed_partial_trace(root, monkeypatch):
    import os
    import sys

    pid, folder, _ = failed(root)
    base = folder / "artifacts/report-benchmark"
    base.mkdir(parents=True)
    program = (
        "from pathlib import Path;import json,time; p=Path("
        + repr(str(base / "traces"))
        + ");p.mkdir();f=(p/'partial.jsonl').open('w');f.write("
        + repr(json.dumps(event(1, "p", "start")) + "\n")
        + ");f.flush();time.sleep(30)"
    )
    original = subprocess.Popen

    class Child:
        def __init__(self, *args, **kwargs):
            self.child = original([sys.executable, "-c", program], **kwargs)
            self.pid = self.child.pid

        def wait(self, timeout=None):
            return self.child.wait(timeout=1 if timeout else None)

    monkeypatch.setattr(worker.subprocess, "Popen", Child)
    assert worker._run_fixed("report-benchmark", base / "raw_result.json", 1) == 124
    assert (
        d.read_diagnostics(root, "alice", pid, "report-benchmark")["traces"][0]["trace_status"]
        == "partial"
    )
    assert not (base / "raw_result.json").exists()
    assert os.getpid() > 0
