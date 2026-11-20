"""v1.3.1 dataset execution contracts without paid model calls.

The registry also carries planner-evolution-reviewed (collection=cases): a
planner-evolution training set scored by the CLI, not a runnable pipeline dataset,
so the eight runnable contracts below are pinned separately from its inventory row.
"""

from __future__ import annotations

import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from app.services import evaluation_pipeline_v13 as pipeline
from app.services.evaluation_dataset_registry import list_datasets
from evals.agent_eval_ext import evaluation_pipeline_worker_v13 as worker
from evals.agent_eval_ext import executor_regression_v131 as executor_regression
from evals.agent_eval_ext.judge_calibration_runner_v131 import validate_source

REQUEST = "d27c880d-84dc-4200-8888-c4eb79162fb9"


def test_runnable_dataset_verified_source_counts():
    inventory = list_datasets()
    assert {x["id"]: x["verified_count"] for x in inventory["datasets"]} == {
        "report-benchmark": 10,
        "corpus-retrieval": 12,
        "selection-retrieval": 16,
        "grounding-v1": 425,
        "judge-calibration": 24,
        "planner-replay-v1": 4,
        "executor-regression-v1": 7,
        "tool-contract-v1": 4,
        "planner-evolution-reviewed": 30,
    }
    assert all(x["source_status"] == "available" for x in inventory["datasets"])
    assert all(x["count_basis"] == "source_verified" for x in inventory["datasets"])


def test_judge_original_data_integrity():
    items = validate_source()
    assert len(items) == 24
    assert len({x["case_id"] for x in items}) == 24
    assert sum(x["split"] == "holdout" for x in items) == 8
    assert sum(x["expected_supported"] for x in items) == 12


def test_executor_manifest_is_exactly_seven():
    rows = executor_regression._cases()
    assert tuple(x["case_id"] for x in rows) == executor_regression.EXPECTED
    assert all(x["test"].startswith("tests/task/test_executor_eval_v131_cases.py::") for x in rows)


def test_eight_adapters_ready_when_configured(monkeypatch):
    monkeypatch.setenv("RF_EVAL_ONLINE_ENABLED", "1")
    monkeypatch.setenv("RF_EVAL_JUDGE_ROUTE", "custom:test-independent-judge")
    assert all(x["ready"] for x in pipeline.capabilities()["datasets"])


def test_full_suite_requires_stronger_consent(monkeypatch, tmp_path):
    monkeypatch.setenv("RF_EVAL_ONLINE_ENABLED", "1")
    monkeypatch.setenv("RF_EVAL_JUDGE_ROUTE", "custom:test-independent-judge")
    with pytest.raises(pipeline.PipelineError, match="explicit_full_dataset_confirmation_required"):
        pipeline.create_pipeline(
            tmp_path, "alice", ["grounding-v1"], REQUEST, max_cases=0, consent=pipeline.PAID_CONFIRM
        )
    rec = pipeline.create_pipeline(
        tmp_path,
        "alice",
        ["grounding-v1"],
        REQUEST,
        max_cases=0,
        consent=pipeline.PAID_FULL_CONFIRM,
    )
    assert rec["max_cases"] == 0 and rec["paid_execution_authorized"] is True
    assert rec["steps"][0]["state"] == "pending"


def test_offline_full_needs_no_paid_confirmation(tmp_path, monkeypatch):
    monkeypatch.delenv("RF_EVAL_ONLINE_ENABLED", raising=False)
    rec = pipeline.create_pipeline(
        tmp_path, "alice", ["executor-regression-v1"], REQUEST, max_cases=0
    )
    assert rec["state"] == "queued"
    assert rec["paid_execution_authorized"] is False


def test_judge_summary_rejects_unmeasured(tmp_path):
    raw = tmp_path / "judge.json"
    raw.write_text(
        json.dumps(
            {
                "dataset_id": "judge-calibration",
                "status": "unmeasured",
                "metrics": {"cases": 2, "judged": 0, "unjudged": 2},
                "cases": [{"claim": "PRIVATE"}, {"claim": "PRIVATE2"}],
            }
        ),
        encoding="utf-8",
    )
    result = worker._summary("judge-calibration", raw)
    assert result["state"] == "failed"
    assert "PRIVATE" not in json.dumps(result)


def test_judge_summary_redacts_raw_case_text(tmp_path):
    raw = tmp_path / "judge.json"
    raw.write_text(
        json.dumps(
            {
                "dataset_id": "judge-calibration",
                "status": "measured",
                "metrics": {
                    "cases": 2,
                    "judged": 1,
                    "unjudged": 1,
                    "coverage": 0.5,
                    "accuracy_on_judged": 1.0,
                },
                "split_counts": {"dev": 2, "holdout": 0},
                "cases": [
                    {"claim": "PRIVATE", "predicted_supported": True},
                    {"evidence": "HIDDEN", "predicted_supported": None},
                ],
            }
        ),
        encoding="utf-8",
    )
    result = worker._summary("judge-calibration", raw)
    assert result["case_count"] == 2
    assert result["metrics"]["coverage"] == 0.5
    assert "PRIVATE" not in json.dumps(result) and "HIDDEN" not in json.dumps(result)


def test_executor_junit_parser_checks_failures(tmp_path):
    cases = executor_regression._cases()
    root = ET.Element("testsuite")
    for index, case in enumerate(cases):
        name = case["test"].split("::")[-1]
        node = ET.SubElement(root, "testcase", name=name)
        if index == 2:
            ET.SubElement(node, "failure", message="controlled failure")
    junit = tmp_path / "junit.xml"
    ET.ElementTree(root).write(junit, encoding="utf-8")
    results = executor_regression._parse_xml(junit, cases)
    assert len(results) == 7
    assert sum(row["passed"] for row in results) == 6
    assert results[2]["case_id"] == "blocked-url" and results[2]["passed"] is False


def test_executor_junit_parser_refuses_missing_case(tmp_path):
    junit = tmp_path / "junit.xml"
    junit.write_text('<testsuite><testcase name="test_search_read_submit" /></testsuite>')
    with pytest.raises(ValueError, match="case_set_mismatch"):
        executor_regression._parse_xml(junit, executor_regression._cases())


def test_no_provider_keys_enter_offline_executor(monkeypatch):
    from evals.agent_eval_ext.evaluation_job_worker import _env

    monkeypatch.setenv("DEEPSEEK_API_KEY", "DO_NOT_LEAK")
    assert "DO_NOT_LEAK" not in json.dumps(_env())


def test_calibration_checksum_matches_known_archive():
    original = (
        Path(__file__).resolve().parents[2]
        / "evals/agent_eval_ext/fixtures/judge_calibration.jsonl"
    )
    assert hashlib.sha256(original.read_bytes()).hexdigest() == (
        "78a778a8315688568c7845ada2099d1c422f1bce65b1e89448c9174cae526800"
    )


def test_invalid_grounding_is_not_quality_pass(tmp_path):
    raw = tmp_path / "grounding.json"
    raw.write_text(
        json.dumps(
            {
                "claims": [{"id": "g001"}, {"id": "g002"}],
                "result": {
                    "claims": 2,
                    "status": "invalid",
                    "unmeasured_share": 0.5,
                    "meets_threshold": None,
                },
            }
        ),
        encoding="utf-8",
    )
    result = worker._summary("grounding-v1", raw)
    assert result["case_count"] == 2
    assert result["metrics"]["invalid_metric"] is True
    assert result["metrics"]["meets_threshold"] is False
