"""Offline extension tests: python -m pytest -q tests/task/test_agent_eval_ext.py."""

import asyncio
import json
import os
import sys
from copy import deepcopy
from pathlib import Path

import pytest

from evals.agent_eval_ext import artifacts, calibrate_live, cli
from evals.agent_eval_ext.audit import classify_error, load_run, report_audit
from evals.agent_eval_ext.calibration import load_calibration, score_calibration
from evals.agent_eval_ext.compare import compare_runs
from evals.agent_eval_ext.gates import evaluate_gate
from evals.agent_eval_ext.trace import TraceWriter
from evals.agent_eval_ext.trajectory import check_trajectory


def make_run(*, yes=True, judge="custom:kimi-k3", completed=True):
    return {
        "mode": "report",
        "llm_mode": "real",
        "models": {"planner": "custom:deepseek-chat"},
        "judge": {"route": judge},
        "method": {
            "metrics_version": 4,
            "retriever": "tavily",
            "query_set": "evals/queries.json",
            "limits": {"max_parallel_tasks": 1},
        },
        "aggregate": {"completion_rate": 1 if completed else 0},
        "results": [
            {
                "id": "a",
                "completed": completed,
                "error": None if completed else "planner: could not produce a valid task list",
                "tokens": 300,
                "cost_usd": 0.0,
                "claim_verdicts": [
                    {"judged": True, "supported": yes},
                    {"judged": False, "supported": None},
                ],
            }
        ],
    }


def test_unjudged_is_not_no():
    a = report_audit(make_run())
    assert a["judge"]["yes"] == 1 and a["judge"]["no"] == 0 and a["judge"]["unjudged"] == 1
    assert a["judge"]["coverage"] == 0.5
    assert a["judge"]["support_among_judged"] == 1.0
    assert a["judge"]["supported_among_all_cited_claims"] == 0.5


def test_false_is_counted_no():
    a = report_audit(make_run(yes=False))
    assert a["judge"]["no"] == 1 and a["judge"]["support_among_judged"] == 0


def test_zero_cost_is_not_asserted_free():
    assert any("0.0 cost" in w for w in report_audit(make_run())["warnings"])


def test_mismatch_judge_prevents_controlled_claim():
    r = compare_runs(make_run(), make_run(judge="anthropic:claude-sonnet"))
    assert not r["is_controlled_comparison"]
    assert any("judge model" in b for b in r["blockers"])


def test_failed_case_is_badcase():
    a = report_audit(make_run(completed=False))
    assert a["bad_cases"][0]["category"] == "structured_output_or_validation"


def test_badcase_unknown_is_not_invented():
    assert classify_error("planner refused task") == "unknown"
    assert (
        classify_error("gateway returned 522", http_status=522) == "upstream_or_gateway_unavailable"
    )


def test_gate_measures_coverage_not_just_yes():
    gate = evaluate_gate(report_audit(make_run()))
    assert not gate["passed"]
    assert gate["checks"]["judge_coverage"]["status"] == "fail"


def test_trajectory_has_order_and_forbidden_check():
    a = [{"name": "search", "args": {"q": 1}}, {"name": "read", "args": {"id": 1}}]
    assert check_trajectory(a, a)["passed"]
    assert not check_trajectory(a, list(reversed(a)))["passed"]
    assert check_trajectory(a, list(reversed(a)), mode="unordered")["passed"]
    assert not check_trajectory(a, a, forbidden_tools=["read"])["passed"]
    assert not check_trajectory(a, [{"name": "search", "args": {"q": 2}}], mode="subset")["passed"]


def test_calibration_fixture_has_distinct_positive_and_negative():
    p = (
        Path(__file__).resolve().parents[2]
        / "evals/agent_eval_ext/fixtures/judge_calibration.jsonl"
    )
    rows = load_calibration(p)
    assert len(rows) >= 20
    assert sum(r["expected_supported"] for r in rows) >= 10
    assert sum(not r["expected_supported"] for r in rows) >= 10
    score = score_calibration(rows)
    assert score["accuracy_on_judged"] is None
    assert score["unjudged"] == len(rows)


def test_calibration_confounders():
    rows = [
        {"expected_supported": False, "predicted_supported": True},
        {"expected_supported": True, "predicted_supported": True},
        {"expected_supported": False, "predicted_supported": None},
    ]
    s = score_calibration(rows)
    assert s["confusion"]["fp"] == 1 and s["confusion"]["tp"] == 1
    assert s["coverage"] == 2 / 3


def test_trace_file_no_plaintext_secrets(tmp_path):
    p = tmp_path / "trace.jsonl"
    tw = TraceWriter(p, run_id="run1", case_id="case1")
    with tw.span("planner"):
        pass
    rows = [json.loads(x) for x in p.read_text("utf-8").splitlines()]
    assert [x["status"] for x in rows] == ["start", "ok"]
    assert rows[0]["seq"] == 1 and rows[1]["seq"] == 2
    assert "prompt" not in rows[0]


def test_reject_bad_duplicate_ids(tmp_path):
    d = make_run()
    d["results"].append(d["results"][0])
    p = tmp_path / "bad.json"
    p.write_text(json.dumps(d))
    with pytest.raises(ValueError):
        load_run(p)


def test_all_unjudged_stays_unmeasured():
    run = make_run()
    run["results"][0]["claim_verdicts"] = [{"judged": False, "supported": None}]
    audit = report_audit(run)
    assert audit["judge"]["support_among_judged"] is None
    assert audit["judge"]["coverage"] == 0
    assert evaluate_gate(audit)["checks"]["judge_support"]["status"] == "unmeasured"


def test_missing_records_do_not_masquerade_as_full_coverage():
    run = make_run()
    run["results"][0]["claim_verdicts"] = [{"judged": True, "supported": True}]
    run["results"].append({"id": "missing", "completed": True})
    audit = report_audit(run)
    assert audit["judge"]["coverage"] is None
    assert audit["judge"]["cases_missing_claim_records"] == ["missing"]
    assert not evaluate_gate(audit)["passed"]


@pytest.mark.parametrize(
    "claim",
    [
        {"judged": "true", "supported": True},
        {"judged": True, "supported": 1},
        {"judged": False, "supported": True},
    ],
)
def test_ambiguous_verdicts_are_not_measurements(claim):
    run = make_run()
    run["results"][0]["claim_verdicts"] = [claim]
    audit = report_audit(run)
    assert audit["judge"]["yes"] == 0
    assert audit["judge"]["unjudged"] == 1


@pytest.mark.parametrize("mutation", ["duplicate", "non_object", "non_boolean", "memory"])
def test_comparison_validates_inputs_even_without_load_run(mutation):
    bad = make_run()
    if mutation == "duplicate":
        bad["results"].append(deepcopy(bad["results"][0]))
    elif mutation == "non_object":
        bad["results"] = [1]
    elif mutation == "non_boolean":
        bad["results"][0]["completed"] = "true"
    else:
        bad["mode"] = "memory"
    with pytest.raises(ValueError):
        compare_runs(make_run(), bad)


def test_current_dataset_hash_cannot_prove_historical_control(tmp_path):
    dataset = tmp_path / "queries.json"
    dataset.write_text('{"queries": []}', encoding="utf-8")
    result = compare_runs(
        make_run(), make_run(), baseline_dataset=dataset, candidate_dataset=dataset
    )
    assert not result["is_controlled_comparison"]
    assert result["signatures"]["baseline"]["query_set_sha256"] is None
    assert result["signatures"]["baseline"]["supplied_dataset_sha256"]


def test_recorded_fingerprints_and_configurations_can_match():
    run = make_run()
    run["method"]["query_set_sha256"] = "a" * 64
    run["method"]["prompt_snapshot_sha256"] = "b" * 64
    assert compare_runs(run, deepcopy(run))["is_controlled_comparison"]


def test_dataset_mismatch_is_rejected(tmp_path):
    dataset = tmp_path / "queries.json"
    dataset.write_text("{}", encoding="utf-8")
    run = make_run()
    run["method"]["query_set_sha256"] = "a" * 64
    with pytest.raises(ValueError, match="does not match"):
        report_audit(run, dataset_path=dataset)


def test_comparison_exposes_missing_and_regressed_cases():
    baseline = make_run()
    candidate = make_run(completed=False)
    baseline["results"].append({"id": "baseline-only", "completed": True})
    result = compare_runs(baseline, candidate)
    assert not result["is_controlled_comparison"]
    assert result["only_in_baseline"] == ["baseline-only"]
    assert result["case_changes"][0]["status"] == "regressed"


def test_gate_cli_exit_code_and_write_once(monkeypatch, tmp_path):
    source, output = tmp_path / "run.json", tmp_path / "gate.json"
    source.write_text(json.dumps(make_run()), encoding="utf-8")
    original = source.read_bytes()
    monkeypatch.setattr(
        sys, "argv", ["agent-eval", "gate", "--input", str(source), "--out", str(output)]
    )
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code == 1
    assert json.loads(output.read_text())["checks"]["judge_coverage"]["status"] == "fail"
    assert source.read_bytes() == original
    saved = output.read_bytes()
    with pytest.raises(FileExistsError):
        cli.main()
    assert output.read_bytes() == saved


def test_offline_cli_does_not_import_live_judge(monkeypatch, tmp_path, capsys):
    source = tmp_path / "run.json"
    source.write_text(json.dumps(make_run()), encoding="utf-8")
    monkeypatch.setenv("LLM_MODE", "real")
    monkeypatch.setitem(sys.modules, "evals.harness", None)
    monkeypatch.setattr(sys, "argv", ["agent-eval", "audit", "--input", str(source)])
    cli.main()
    assert json.loads(capsys.readouterr().out)["judge"]["coverage"] == 0.5


@pytest.mark.parametrize("allow_paid,real_mode", [(False, True), (True, False)])
def test_live_calibration_requires_both_opt_ins(monkeypatch, tmp_path, allow_paid, real_mode):
    argv = [
        "calibrate",
        "--input",
        "unused.jsonl",
        "--judge",
        "custom:kimi-k3",
        "--out",
        str(tmp_path / "new.json"),
    ]
    if allow_paid:
        argv.append("--allow-paid")
    monkeypatch.setenv("LLM_MODE", "real" if real_mode else "fake")
    monkeypatch.setattr(sys, "argv", argv)
    monkeypatch.setattr(calibrate_live, "run", lambda *args: pytest.fail("paid runner reached"))
    with pytest.raises(SystemExit) as exc:
        calibrate_live.main()
    assert exc.value.code == 2
    assert not (tmp_path / "new.json").exists()


def test_live_preflight_actually_writes_before_paid_runner(monkeypatch, tmp_path):
    def deny_creation(*args, **kwargs):
        raise PermissionError("read-only mount")

    monkeypatch.setenv("LLM_MODE", "real")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "calibrate",
            "--input",
            "unused.jsonl",
            "--judge",
            "custom:kimi-k3",
            "--out",
            str(tmp_path / "new.json"),
            "--allow-paid",
        ],
    )
    monkeypatch.setattr(artifacts.tempfile, "NamedTemporaryFile", deny_creation)
    monkeypatch.setattr(calibrate_live, "run", lambda *args: pytest.fail("paid runner reached"))
    with pytest.raises(SystemExit) as exc:
        calibrate_live.main()
    assert exc.value.code == 2


def test_output_name_race_cannot_overwrite_evidence(tmp_path):
    output = tmp_path / "new.json"
    artifacts.check_output_path(output)
    assert not output.exists()
    assert not list(tmp_path.iterdir())
    output.write_text("another run", encoding="utf-8")
    with pytest.raises(FileExistsError):
        artifacts.save_artifact({"value": 1}, output)
    assert output.read_text() == "another run"


def test_trajectory_counts_duplicates_and_enforces_arguments():
    search = {"name": "search", "args": {"query": "paper"}}
    assert not check_trajectory([search], [search, search], mode="subset")["passed"]
    assert not check_trajectory([search, search], [search], mode="superset")["passed"]
    assert check_trajectory([search], [search, search], mode="superset")["passed"]
    other = {"name": "search", "args": {"query": "different"}}
    assert not check_trajectory([search], [other])["passed"]
    assert check_trajectory([search], [other], check_args=False)["passed"]


def test_calibration_rejects_non_boolean_prediction():
    with pytest.raises(ValueError):
        score_calibration([{"expected_supported": False, "predicted_supported": "YES"}])


def test_trace_does_not_append_to_historical_file(tmp_path):
    path = tmp_path / "trace.jsonl"
    path.write_text("historical trace", encoding="utf-8")
    with pytest.raises(FileExistsError):
        TraceWriter(path, run_id="new", case_id="case")
    assert path.read_text() == "historical trace"


def test_trace_failure_cannot_mask_business_error(monkeypatch, tmp_path, caplog):
    writer = TraceWriter(tmp_path / "trace.jsonl", run_id="run", case_id="case")

    def deny_open(*args, **kwargs):
        raise PermissionError("secret must not be logged")

    monkeypatch.setattr(Path, "open", deny_open)
    with pytest.raises(ValueError, match="business failure"):
        with writer.span("planner"):
            raise ValueError("business failure")
    assert "agent_eval_trace_write_failed" in caplog.text
    assert "secret must not be logged" not in caplog.text


async def test_manual_trace_concurrency_does_not_mix_parent_spans(tmp_path):
    path = tmp_path / "trace.jsonl"
    writer = TraceWriter(path, run_id="run", case_id="case")

    async def worker(name):
        with writer.span(name):
            await asyncio.sleep(0)
            with writer.span(name + "-tool"):
                await asyncio.sleep(0)

    await asyncio.gather(worker("left"), worker("right"))
    events = [json.loads(line) for line in path.read_text().splitlines()]
    starts = {event["node"]: event for event in events if event["status"] == "start"}
    assert starts["left"]["parent_id"] is None
    assert starts["right"]["parent_id"] is None
    assert starts["left-tool"]["parent_id"] == starts["left"]["span_id"]
    assert starts["right-tool"]["parent_id"] == starts["right"]["span_id"]
    assert [event["seq"] for event in events] == list(range(1, 9))


def test_trace_preserves_cancellation_and_omits_exception_text(tmp_path):
    path = tmp_path / "trace.jsonl"
    writer = TraceWriter(path, run_id="run", case_id="case")
    with pytest.raises(asyncio.CancelledError):
        with writer.span("executor"):
            raise asyncio.CancelledError("private credential")
    events = [json.loads(line) for line in path.read_text().splitlines()]
    assert events[-1]["status"] == "cancelled"
    assert events[-1]["error_type"] == "CancelledError"
    assert "private credential" not in path.read_text()


async def test_live_calibration_uses_existing_parser_and_discloses_actual_judge(monkeypatch):
    environment_before = dict(os.environ)
    try:
        from evals import harness
    finally:
        os.environ.clear()
        os.environ.update(environment_before)
    from dataclasses import replace
    from types import SimpleNamespace

    replies = iter(["Claim 1: YES", "Claim 1: NO", "No parseable ruling"])
    requests = []

    class Judge:
        async def ainvoke(self, messages):
            requests.append(messages)
            return SimpleNamespace(
                content=next(replies),
                response_metadata={"model_name": "actual-test-judge"},
            )

    async def no_sleep(*args):
        pass

    monkeypatch.setenv("LLM_MODE", "real")
    # The live adapter configures a module-global route; keep later tests independent.
    monkeypatch.setattr(harness, "JUDGE_ROUTE", harness.JUDGE_ROUTE)
    monkeypatch.setattr(harness, "RUN_CONFIG", replace(harness.RUN_CONFIG, llm_mode="real"))
    monkeypatch.setattr(harness, "_judge_llm", lambda: Judge())
    monkeypatch.setattr(harness.asyncio, "sleep", no_sleep)
    fixture = (
        Path(__file__).resolve().parents[2]
        / "evals/agent_eval_ext/fixtures/judge_calibration.jsonl"
    )
    rows = await calibrate_live.run(fixture, "custom:kimi-k3", 3, "dev")
    assert len(requests) == 3
    assert [row["predicted_supported"] for row in rows] == [True, False, None]
    assert rows[0]["claim_verdicts"][0]["judged_by"] == "actual-test-judge"
    assert rows[2]["claim_verdicts"][0]["judged"] is False
    assert "The batch contains three files." in requests[1][-1].content
