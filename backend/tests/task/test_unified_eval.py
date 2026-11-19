"""Pure offline tests for Unified Evaluation v0.7; no API or database."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from evals.agent_eval_ext import unified_cli
from evals.agent_eval_ext.unified_eval import (
    adapt_grounding,
    adapt_replay,
    adapt_report,
    adapt_retrieval,
    build_unified,
    compare_unified,
    fingerprint,
    load_dataset,
    metric,
    save_new,
)

FIXTURES = Path(__file__).parents[2] / "evals" / "agent_eval_ext" / "fixtures"


def put(path: Path, value: dict) -> Path:
    if "_fixture_dataset" in value:
        dataset = value.pop("_fixture_dataset")
        if path.name != "suite.json":
            path = path.parent / path.stem / "suite.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        materialize_suite(path.parent, value, dataset)
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
    return path


def sample_report() -> dict:
    return {
        "mode": "report",
        "llm_mode": "real",
        "models": {"planner": "custom:demo"},
        "method": {"metrics_version": 4, "query_set": "evals/queries.json"},
        "judge": {"route": "custom:judge"},
        "aggregate": {"completion_rate": 0.5, "avg_cost_usd": 0},
        "results": [
            {
                "id": "a",
                "completed": True,
                "error": None,
                "latency_s": 21,
                "tokens": 200,
                "cost_usd": 0.0,
                "claim_verdicts": [
                    {"judged": True, "supported": True},
                    {"judged": True, "supported": False},
                    {"judged": False, "supported": None},
                ],
            },
            {
                "id": "b",
                "completed": False,
                "error": "planner bad",
                "latency_s": 2,
                "claim_verdicts": [],
            },
        ],
    }


def sample_retrieval(flavor: str = "corpus") -> dict:
    agg = (
        {"overall": {"queries": 2, "recall@5": 0.75, "precision@5": 0.50}}
        if flavor == "corpus"
        else {"queries": 2, "hit_cap": {"after (per file)": {"recall@5": 0.80, "ndcg@5": 0.90}}}
    )
    return {
        "status": "measured",
        "retrieval_metrics_version": 1,
        "dataset_fingerprint": "recorded_sha",
        "retriever": "test",
        "top_k": 5,
        "aggregate": agg,
        "queries": [{"id": "a"}, {"id": "b"}],
    }


def sample_grounding(status: str = "measured") -> dict:
    return {
        "eval": "grounding-benchmark",
        "set_version": 1,
        "reference_labels": {"rubric_sha256": "gold"},
        "critic_routing": "local",
        "result": {
            "status": status,
            "claims": 8,
            "false_accept": 0.125,
            "false_reject": 0.25,
            "unmeasured_share": 0.0,
        },
    }


def sample_suite(dataset: dict, *, verdicts: list[bool] | None = None) -> dict:
    cases = dataset["cases"]
    verdicts = [True] * len(cases) if verdicts is None else verdicts
    return {
        "_fixture_dataset": dataset,
        "schema_version": 1,
        "kind": "researchflow_replay_suite",
        "replay_type": "controlled_simulation",
        "historical_replay": False,
        "manifest_sha256": dataset["manifest_sha256"],
        "revision_supplied": "synthetic-unit-fixture",
        "revision_verified": False,
        "total": len(cases),
        "passed_count": sum(verdicts),
        "all_passed": all(verdicts),
        "cases": [
            {
                **c,
                "passed": ok,
                "result_verified": True,
                "result_passed": ok,
                "process_exit_code": 0 if ok else 1,
                "harness_failure": None,
            }
            for c, ok in zip(cases, verdicts, strict=True)
        ],
    }


def materialize_suite(root, summary, dataset):
    """Synthetic child artifacts exercise the real verifier, without running an Agent."""
    from evals.agent_eval_ext.replay import load_case
    from evals.agent_eval_ext.replay_suite import _verify_result
    from evals.agent_eval_ext.trace_inspect import inspect_trace

    inputs = root / "inputs"
    inputs.mkdir()
    (inputs / "manifest.json").write_bytes(dataset["_manifest_bytes"])
    specs = {}
    for name, content in dataset["_case_bytes"].items():
        (inputs / name).write_bytes(content)
        case, digest = load_case(inputs / name)
        specs[case.case_id] = (case, digest)
    for row in summary["cases"]:
        case, digest = specs[row["case_id"]]
        passed = row["passed"]
        status = case.expected_status if passed or case.expected_status == "failed" else "failed"
        calls = case.expected_planner_calls or 1
        if not passed and case.expected_planner_calls is not None:
            calls += 1
        case_dir = root / case.case_id
        traces = case_dir / "traces"
        traces.mkdir(parents=True)
        run_id, trace_id = "unit-" + case.case_id, "trace-" + case.case_id
        events = []
        terminal = "failer" if status == "failed" else "hitl_gate"
        for n in range(calls + 1):
            for phase in (
                "start",
                "ok" if n < calls else "business_error" if status == "failed" else "interrupt",
            ):
                events.append(
                    {
                        "event_version": 2,
                        "seq": len(events) + 1,
                        "trace_id": trace_id,
                        "run_id": run_id,
                        "case_id": case.case_id,
                        "thread_id": run_id,
                        "span_id": str(n),
                        "status": phase,
                        "kind": "llm" if n < calls else "graph_node",
                        "name": "llm.planner" if n < calls else terminal,
                        "metadata": {"role": "planner"},
                    }
                )
        trace = traces / "unit.jsonl"
        trace.write_text("\n".join(json.dumps(e) for e in events), encoding="utf-8")
        bad_path = "traces/unit.badcase.json" if status == "failed" else None
        if bad_path:
            (case_dir / bad_path).write_text(
                json.dumps({"case_id": case.case_id, "run_id": run_id, "trace_id": trace_id}),
                encoding="utf-8",
            )
        detail = {
            "schema_version": 1,
            "case_id": case.case_id,
            "case_file_sha256": digest,
            "case_spec_sha256": hashlib.sha256(
                json.dumps(case.as_dict(), sort_keys=True).encode()
            ).hexdigest(),
            "replay_type": "controlled_simulation",
            "historical_replay": False,
            "llm_mode": "fake",
            "fault_kind": case.fault_kind,
            "run_id": run_id,
            "passed": passed,
            "status": status,
            "planner_calls": calls,
            "assertions": {
                "status": status == case.expected_status,
                "planner_calls": None
                if case.expected_planner_calls is None
                else calls == case.expected_planner_calls,
                "error_contains": True if case.error_contains else None,
                "trace_available": True,
                "terminal_trace": True,
            },
            "trace_path": "traces/unit.jsonl",
            "bad_case_path": bad_path,
            "trace_status": "recorded",
            "trace_inspection": inspect_trace(trace),
        }
        result = case_dir / "result.json"
        result.write_text(json.dumps(detail), encoding="utf-8")
        hashes = _verify_result(case, digest, detail, case_dir)
        row.update(
            status=status,
            fault_kind=case.fault_kind,
            result_path=f"{case.case_id}/result.json",
            artifact_sha256={**hashes, "result.json": fingerprint(result)},
        )


def test_real_v06_dataset_validates_and_has_4_cases():
    data = load_dataset(FIXTURES / "unified_dataset_v1.json")
    assert data["dataset_id"] == "rf-interview-regression"
    assert len(data["cases"]) == 4
    assert len(set(x["case_sha256"] for x in data["cases"])) == 4


@pytest.mark.parametrize(
    "bad",
    [
        {
            "schema_version": 1,
            "dataset_id": "x",
            "dataset_version": 1,
            "split": "regression",
            "replay_manifest": "../secret.json",
        },
        {
            "schema_version": 1,
            "dataset_id": "x",
            "dataset_version": True,
            "split": "regression",
            "replay_manifest": "ok.json",
        },
        {
            "schema_version": 1,
            "dataset_id": "bad/id",
            "dataset_version": 1,
            "split": "regression",
            "replay_manifest": "ok.json",
        },
        {
            "schema_version": 1,
            "dataset_id": "x",
            "dataset_version": 1,
            "split": "random",
            "replay_manifest": "ok.json",
        },
    ],
)
def test_dataset_rejects_unsafe_or_malformed(tmp_path, bad):
    with pytest.raises(ValueError):
        load_dataset(put(tmp_path / "dataset.json", bad))


def test_metric_missing_is_not_zero_or_false_pass():
    assert metric(None)["status"] == "unmeasured"
    assert metric(0)["status"] == "measured"
    assert metric(True)["status"] == "unmeasured"
    assert metric(float("nan"))["status"] == "unmeasured"
    assert metric(1.5)["status"] == "unmeasured"
    assert metric(0.5, valid=False)["status"] == "invalid"


def test_report_claim_denominators_and_cost(tmp_path):
    normalized = adapt_report(put(tmp_path / "report.json", sample_report()))
    m = normalized["metrics"]
    assert m["completion_rate"]["value"] == 0.5
    assert m["citation_support_among_judged"]["value"] == 0.5
    assert m["judge_coverage"]["value"] == pytest.approx(2 / 3)
    assert m["latency_p95_completed_s"]["value"] == 21
    assert m["api_cost_usd"]["status"] == "unmeasured"
    assert normalized["judge_counts"] == {"yes": 1, "no": 1, "unjudged": 1}
    assert normalized["cases"][0]["test_result"] == "unmeasured"
    assert normalized["cases"][1]["agent_status"] == "failed"


def test_report_empty_claims_keep_unknown_judge_coverage(tmp_path):
    doc = sample_report()
    doc["results"] = [doc["results"][1]]
    normalized = adapt_report(put(tmp_path / "r.json", doc))
    assert normalized["metrics"]["judge_coverage"]["status"] == "unmeasured"


def test_report_rejects_memory_mode(tmp_path):
    doc = sample_report()
    doc["mode"] = "memory"
    with pytest.raises(ValueError):
        adapt_report(put(tmp_path / "r.json", doc))


def test_report_rejects_non_boolean_completed(tmp_path):
    doc = sample_report()
    doc["results"][0]["completed"] = 1
    with pytest.raises(ValueError):
        adapt_report(put(tmp_path / "r.json", doc))


@pytest.mark.parametrize(
    "flavor,key,expected",
    [
        ("corpus", "precision_at_5", 0.5),
        ("selection", "ndcg_at_5", 0.9),
    ],
)
def test_retrieval_two_official_layouts(tmp_path, flavor, key, expected):
    normalized = adapt_retrieval(put(tmp_path / "retrieval.json", sample_retrieval(flavor)))
    assert normalized["status"] == "measured"
    assert normalized["metrics"][key]["value"] == expected
    assert normalized["signature"]["dataset_fingerprint_recorded"] == "recorded_sha"


def test_retrieval_rejects_wrong_shape(tmp_path):
    with pytest.raises(ValueError):
        adapt_retrieval(put(tmp_path / "not-retrieval.json", {"results": []}))


@pytest.mark.parametrize("status,expected", [("measured", "measured"), ("invalid", "invalid")])
def test_grounding_source_validity_is_not_gate_pass(tmp_path, status, expected):
    output = adapt_grounding(put(tmp_path / "grounding.json", sample_grounding(status)))
    assert output["status"] == expected
    assert output["metrics"]["false_accept"]["status"] == expected
    assert output["metrics"]["false_accept"]["value"] == 0.125


def test_replay_measured_only_as_fault_contract(tmp_path):
    dataset = load_dataset(FIXTURES / "unified_dataset_v1.json")
    data = adapt_replay(put(tmp_path / "suite.json", sample_suite(dataset)), dataset)
    assert data["metrics"]["case_pass_rate"]["value"] == 1
    assert data["contract"] == "controlled_fault_handling_not_task_completion"
    assert sum(c["agent_status"] == "failed" for c in data["cases"]) == 3
    assert data["cases"][0]["agent_status"] == "awaiting_approval"


def test_replay_rejects_wrong_case_bytes(tmp_path):
    dataset = load_dataset(FIXTURES / "unified_dataset_v1.json")
    bad = sample_suite(dataset)
    bad["cases"][0]["case_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="do not match"):
        adapt_replay(put(tmp_path / "suite.json", bad), dataset)


def test_replay_rejects_inconsistent_totals(tmp_path):
    dataset = load_dataset(FIXTURES / "unified_dataset_v1.json")
    bad = sample_suite(dataset)
    bad["passed_count"] = -1
    with pytest.raises(ValueError, match="inconsistent"):
        adapt_replay(put(tmp_path / "suite.json", bad))


def test_unified_missing_dimensions_do_not_count_as_zero(tmp_path):
    output = build_unified(report=put(tmp_path / "r.json", sample_report()))
    assert output["dimensions"]["outcome"]["status"] == "measured"
    assert output["dimensions"]["reliability"]["status"] == "unmeasured"
    assert output["dimensions"]["grounding"]["metrics"] == {}
    assert output["not_a_single_score"] is True


def test_unified_multiple_read_only_sources_and_provenance(tmp_path):
    paths = [
        put(tmp_path / "report.json", sample_report()),
        put(tmp_path / "retrieval.json", sample_retrieval()),
        put(tmp_path / "grounding.json", sample_grounding()),
    ]
    before = [fingerprint(p) for p in paths]
    output = build_unified(report=paths[0], retrieval=paths[1], grounding=paths[2])
    assert [fingerprint(p) for p in paths] == before
    assert output["dimensions"]["retrieval"]["source"]["sha256"] == before[1]
    assert output["dimensions"]["grounding"]["status"] == "measured"


def test_compare_detects_a_real_regression_in_contract(tmp_path):
    ds = load_dataset(FIXTURES / "unified_dataset_v1.json")
    source = tmp_path / "r.json"
    baseline = build_unified(dataset=ds, replay_suite=put(source, sample_suite(ds)))
    source2 = tmp_path / "candidate.json"
    candidate = build_unified(
        dataset=ds, replay_suite=put(source2, sample_suite(ds, verdicts=[True, False, True, True]))
    )
    result = compare_unified(baseline, candidate)
    assert result["comparable"] is True
    assert result["regressions"] == 1
    assert result["quality_gate_passed"] is False
    assert result["code_revision_verified"] is False


def test_compare_all_pass_and_no_fake_release_claim(tmp_path):
    ds = load_dataset(FIXTURES / "unified_dataset_v1.json")
    a = build_unified(dataset=ds, replay_suite=put(tmp_path / "suite.json", sample_suite(ds)))
    result = compare_unified(a, a)
    assert result["quality_gate_passed"]
    assert result["other_dimensions"].startswith("not_paired")


def test_compare_rejects_changed_case(tmp_path):
    ds = load_dataset(FIXTURES / "unified_dataset_v1.json")
    p = tmp_path / "baseline.json"
    a = build_unified(dataset=ds, replay_suite=put(p, sample_suite(ds)))
    candidate = json.loads(json.dumps(a))
    candidate["dimensions"]["reliability"]["cases"][0]["case_sha256"] = "1" * 64
    assert compare_unified(a, candidate)["reason"] == "replay_case_bytes_mismatch"


def test_compare_fails_closed_when_no_replay(tmp_path):
    x = build_unified(report=put(tmp_path / "r.json", sample_report()))
    result = compare_unified(x, x)
    assert not result["comparable"]
    assert result["quality_gate_passed"] is None


def test_save_is_exclusive(tmp_path):
    output = tmp_path / "new.json"
    save_new(output, {"ok": True})
    before = output.read_bytes()
    with pytest.raises(FileExistsError):
        save_new(output, {"ok": False})
    assert before == output.read_bytes()


def test_cli_collect_and_compare_exit_codes(tmp_path, capsys):
    ds = load_dataset(FIXTURES / "unified_dataset_v1.json")
    base = put(tmp_path / "suite.json", sample_suite(ds))
    output = tmp_path / "unified.json"
    assert (
        unified_cli.main(
            [
                "collect",
                "--dataset",
                str(FIXTURES / "unified_dataset_v1.json"),
                "--replay-suite",
                str(base),
                "--out",
                str(output),
            ]
        )
        == 0
    )
    assert (
        unified_cli.main(
            [
                "collect",
                "--dataset",
                str(FIXTURES / "unified_dataset_v1.json"),
                "--replay-suite",
                str(base),
                "--out",
                str(output),
            ]
        )
        == 2
    )
    assert unified_cli.main(["compare", "--baseline", str(output), "--candidate", str(output)]) == 0
    capsys.readouterr()


def test_cli_run_preflight_refuses_old_dir_before_execution(tmp_path, monkeypatch):
    from evals.agent_eval_ext import replay_suite

    called = []
    monkeypatch.setattr(replay_suite, "run_suite", lambda *a, **k: called.append(True))
    existing = tmp_path / "already"
    existing.mkdir()
    assert (
        unified_cli.main(
            [
                "run",
                "--dataset",
                str(FIXTURES / "unified_dataset_v1.json"),
                "--out-dir",
                str(existing),
            ]
        )
        == 2
    )
    assert not called


def test_cli_run_with_stub_suite_and_no_paid_api(tmp_path, monkeypatch):
    from evals.agent_eval_ext import replay_suite

    ds = load_dataset(FIXTURES / "unified_dataset_v1.json")
    calls = []

    def fake_suite(manifest, output_dir, **kwargs):
        calls.append((manifest, kwargs))
        target = Path(output_dir)
        target.mkdir()
        put(target / "suite.json", sample_suite(ds))

    monkeypatch.setattr(replay_suite, "run_suite", fake_suite)
    dest = tmp_path / "new"
    status = unified_cli.main(
        ["run", "--dataset", str(FIXTURES / "unified_dataset_v1.json"), "--out-dir", str(dest)]
    )
    assert status == 0
    assert len(calls) == 1
    result = json.loads((dest / "unified.json").read_text())
    assert result["dimensions"]["reliability"]["status"] == "measured"
    assert result["dimensions"]["outcome"]["status"] == "unmeasured"


def test_cli_preflight_report_error_does_not_spawn_replay(tmp_path, monkeypatch):
    from evals.agent_eval_ext import replay_suite

    called = []
    monkeypatch.setattr(replay_suite, "run_suite", lambda *a, **k: called.append(True))
    bad = put(tmp_path / "bad.json", {"mode": "memory", "results": []})
    dest = tmp_path / "not-created"
    rc = unified_cli.main(
        [
            "run",
            "--dataset",
            str(FIXTURES / "unified_dataset_v1.json"),
            "--out-dir",
            str(dest),
            "--report",
            str(bad),
        ]
    )
    assert rc == 2
    assert not dest.exists()
    assert not called


def test_cli_compare_detects_candidate_regression_and_returns_failure(tmp_path):
    dataset = load_dataset(FIXTURES / "unified_dataset_v1.json")
    old_suite = put(tmp_path / "old-suite.json", sample_suite(dataset))
    new_suite = put(
        tmp_path / "new-suite.json", sample_suite(dataset, verdicts=[False, True, True, True])
    )
    before = build_unified(dataset=dataset, replay_suite=old_suite)
    after = build_unified(dataset=dataset, replay_suite=new_suite)
    baseline = put(tmp_path / "baseline.json", before)
    candidate = put(tmp_path / "candidate.json", after)
    diff_path = tmp_path / "comparison.json"
    rc = unified_cli.main(
        [
            "compare",
            "--baseline",
            str(baseline),
            "--candidate",
            str(candidate),
            "--out",
            str(diff_path),
        ]
    )
    assert rc == 1
    result = json.loads(diff_path.read_text())
    assert result["regressions"] == 1
    assert result["quality_gate_passed"] is False


def test_cli_run_returns_one_when_case_contract_fails(tmp_path, monkeypatch):
    from evals.agent_eval_ext import replay_suite

    ds = load_dataset(FIXTURES / "unified_dataset_v1.json")

    def fake_failed_suite(manifest, output_dir, **kwargs):
        dest = Path(output_dir)
        dest.mkdir()
        put(dest / "suite.json", sample_suite(ds, verdicts=[True, True, True, False]))

    monkeypatch.setattr(replay_suite, "run_suite", fake_failed_suite)
    out = tmp_path / "new-output"
    rc = unified_cli.main(
        ["run", "--dataset", str(FIXTURES / "unified_dataset_v1.json"), "--out-dir", str(out)]
    )
    assert rc == 1
    unified = json.loads((out / "unified.json").read_text())
    assert unified["dimensions"]["reliability"]["metrics"]["case_pass_rate"]["value"] == 0.75


@pytest.mark.parametrize("value", [None, [], {}, True])
def test_compare_rejects_missing_or_empty_case_identity(tmp_path, value):
    ds = load_dataset(FIXTURES / "unified_dataset_v1.json")
    doc = build_unified(dataset=ds, replay_suite=put(tmp_path / "suite.json", sample_suite(ds)))
    doc["dimensions"]["reliability"]["cases"] = value
    with pytest.raises(ValueError):
        compare_unified(doc, doc)


def test_compare_rejects_metric_inconsistent_with_cases(tmp_path):
    ds = load_dataset(FIXTURES / "unified_dataset_v1.json")
    doc = build_unified(dataset=ds, replay_suite=put(tmp_path / "suite.json", sample_suite(ds)))
    doc["dimensions"]["reliability"]["metrics"]["case_pass_rate"]["value"] = 0.75
    with pytest.raises(ValueError, match="pass rate"):
        compare_unified(doc, doc)


def test_collect_rechecks_trace_bytes(tmp_path):
    ds = load_dataset(FIXTURES / "unified_dataset_v1.json")
    source = put(tmp_path / "suite.json", sample_suite(ds))
    trace = next(tmp_path.glob("*/traces/unit.jsonl"))
    trace.write_text("corrupt", encoding="utf-8")
    with pytest.raises(ValueError):
        adapt_replay(source, ds)


def test_compare_requires_versioned_dataset(tmp_path):
    ds = load_dataset(FIXTURES / "unified_dataset_v1.json")
    doc = build_unified(replay_suite=put(tmp_path / "suite.json", sample_suite(ds)))
    assert compare_unified(doc, doc)["reason"] == "unified_dataset_not_supplied"


def test_grounding_uses_conditional_denominators(tmp_path):
    doc = sample_grounding()
    doc["result"].update(
        false_accept_counts={"negatives_measured": 5},
        false_reject_counts={"positives_measured": 3},
        confusion={
            "contradicted": {"supported": 0, "not_supported": 1, "contradicted": 2, "unmeasured": 4}
        },
    )
    metrics = adapt_grounding(put(tmp_path / "g.json", doc))["metrics"]
    assert metrics["false_accept"]["n"] == 5
    assert metrics["false_reject"]["n"] == 3
    assert metrics["contradicted_recall"]["n"] == 3
    assert metrics["unmeasured_share"]["n"] == 8


def test_unknown_grounding_status_is_unmeasured(tmp_path):
    doc = sample_grounding()
    del doc["result"]["status"]
    adapted = adapt_grounding(put(tmp_path / "g.json", doc))
    assert adapted["status"] == "unmeasured"
    assert adapted["metrics"]["false_accept"]["value"] is None
    assert metric(None, valid=False)["status"] == "invalid"


def test_report_missing_completed_latency_does_not_shrink_population(tmp_path):
    doc = sample_report()
    doc["results"][1].update(completed=True, latency_s=None)
    adapted = adapt_report(put(tmp_path / "r.json", doc))
    assert adapted["completed_latency_missing"] == 1
    assert adapted["metrics"]["latency_p95_completed_s"]["status"] == "unmeasured"
    assert adapted["metrics"]["citation_support_among_judged"]["n"] == 2


def test_retrieval_metric_n_excludes_missing_observations(tmp_path):
    doc = sample_retrieval("selection")
    doc["aggregate"]["hit_cap"]["after (per file)"]["passage_recall"] = 1
    doc["queries"][0]["with_cap"] = {"passage_recall": 1}
    doc["queries"][1]["with_cap"] = {"passage_recall": None}
    adapted = adapt_retrieval(put(tmp_path / "r.json", doc))
    assert adapted["sample_count"] == 2
    assert adapted["metrics"]["passage_recall"]["n"] == 1


@pytest.mark.parametrize("change", ["duplicate", "count", "empty"])
def test_retrieval_invalid_population(tmp_path, change):
    doc = sample_retrieval()
    if change == "duplicate":
        doc["queries"][1]["id"] = "a"
    if change == "count":
        doc["aggregate"]["overall"]["queries"] = 3
    if change == "empty":
        doc["queries"] = []
    path = put(tmp_path / "r.json", doc)
    if change == "empty":
        assert adapt_retrieval(path)["status"] == "unmeasured"
    else:
        with pytest.raises(ValueError):
            adapt_retrieval(path)


def test_serialization_failure_leaves_no_partial_file(tmp_path):
    target = tmp_path / "invalid.json"
    with pytest.raises(ValueError):
        save_new(target, {"rate": float("nan")})
    assert not target.exists()


def test_run_trial_write_failure_precedes_process(tmp_path, monkeypatch):
    from evals.agent_eval_ext import replay_suite

    called = []
    monkeypatch.setattr(replay_suite, "run_suite", lambda *a, **k: called.append(True))

    def denied(path):
        raise PermissionError("controlled write failure")

    monkeypatch.setattr(unified_cli, "check_output_path", denied)
    assert (
        unified_cli.main(
            [
                "run",
                "--dataset",
                str(FIXTURES / "unified_dataset_v1.json"),
                "--out-dir",
                str(tmp_path / "new"),
            ]
        )
        == 2
    )
    assert not called


def test_run_uses_frozen_dataset_and_historical_source(tmp_path, monkeypatch):
    from evals.agent_eval_ext import replay_suite

    ds = load_dataset(FIXTURES / "unified_dataset_v1.json")
    original = put(tmp_path / "report.json", sample_report())
    old_hash = fingerprint(original)

    def runner(manifest, output_dir, **kwargs):
        assert Path(manifest).parent == tmp_path / "output" / "dataset"
        original.write_text("corrupt-after-preflight", encoding="utf-8")
        Path(output_dir).mkdir()
        put(Path(output_dir) / "suite.json", sample_suite(ds))

    monkeypatch.setattr(replay_suite, "run_suite", runner)
    assert (
        unified_cli.main(
            [
                "run",
                "--dataset",
                str(FIXTURES / "unified_dataset_v1.json"),
                "--out-dir",
                str(tmp_path / "output"),
                "--report",
                str(original),
            ]
        )
        == 0
    )
    result = json.loads((tmp_path / "output/unified.json").read_text())
    assert result["dimensions"]["outcome"]["source"]["sha256"] == old_hash


def test_cli_actual_four_process_graph_run(tmp_path):
    out = tmp_path / "actual"
    assert (
        unified_cli.main(
            ["run", "--dataset", str(FIXTURES / "unified_dataset_v1.json"), "--out-dir", str(out)]
        )
        == 0
    )
    result = json.loads((out / "unified.json").read_text())
    cases = result["dimensions"]["reliability"]["cases"]
    assert len(cases) == 4 and all(c["test_result"] == "pass" for c in cases)
    assert sum(c["agent_status"] == "failed" for c in cases) == 3
    assert all(c["result_verified"] for c in cases)
    assert result["cross_source_pairing_verified"] is False
