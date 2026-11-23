"""Local tests for the GEPA v3 holdout purpose guard — no model calls.

Covers: denied purposes, allowed one-shot final Planner A/B, fingerprint
tamper detection, report A/B entry-point rejection, and search entry-point
rejection. Nothing here reads or scores the holdout cases.
"""

from __future__ import annotations

import json

import pytest

from evals import holdout_usage_guard as guard
from evals.holdout_usage_guard import (
    CONTENT_SHA256,
    FILE_SHA256,
    HOLDOUT_ID,
    HoldoutUsageError,
    assert_holdout_usage,
    shares_questions_with_holdout,
    verify_fingerprints,
)


def test_fingerprints_verify():
    fp = verify_fingerprints()
    assert fp["content_sha256"] == CONTENT_SHA256
    assert fp["file_sha256"] == FILE_SHA256
    assert fp["case_count"] == 9


@pytest.mark.parametrize(
    "purpose",
    [
        "report_ab",
        "search",
        "candidate_selection",
        "prompt_adjustment",
        "calibration",
    ],
)
def test_denied_purposes_fail_closed(purpose):
    with pytest.raises(HoldoutUsageError) as exc:
        assert_holdout_usage(purpose, dataset_id=HOLDOUT_ID)
    assert purpose in exc.value.reason
    assert "DENIED" in exc.value.reason


def test_final_ab_requires_candidate_frozen():
    with pytest.raises(HoldoutUsageError) as exc:
        assert_holdout_usage("planner_final_ab", dataset_id=HOLDOUT_ID)
    assert "candidate_frozen" in exc.value.reason


def test_final_ab_allowed_once_when_frozen():
    decision = assert_holdout_usage(
        "planner_final_ab",
        dataset_id=HOLDOUT_ID,
        candidate_frozen=True,
        allow_single_final_ab=True,
    )
    assert decision.allowed is True
    assert decision.fingerprint_verified is True


def test_non_holdout_dataset_unaffected():
    decision = assert_holdout_usage("report_ab", dataset_id="some-other-dataset")
    assert decision.allowed is True
    assert decision.fingerprint_verified is False


def test_audit_log_records_denial(tmp_path):
    log = tmp_path / "audit.jsonl"
    with pytest.raises(HoldoutUsageError):
        assert_holdout_usage("report_ab", dataset_id=HOLDOUT_ID, audit_log=log)
    rows = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
    assert rows[-1]["allowed"] is False
    assert "report_ab" in rows[-1]["reason"]


def test_shares_questions_opt_in():
    data = json.loads(guard.HOLDOUT_PATH.read_text(encoding="utf-8"))
    texts = [c["query"] for c in data["cases"]]
    assert shares_questions_with_holdout(texts) is True
    assert shares_questions_with_holdout(["unrelated question"]) is False


def test_report_ab_entrypoint_rejects_holdout():
    from evals.report_ab_dataset import fixed_report_queries

    with pytest.raises(HoldoutUsageError) as exc:
        fixed_report_queries(guard.HOLDOUT_PATH)
    assert "report_ab" in exc.value.reason


def test_search_entrypoint_rejects_holdout(monkeypatch):
    from evals.agent_eval_ext.planner_evolution.__main__ import _optimize

    class Args:
        dataset = str(guard.HOLDOUT_PATH)
        live = False
        baseline = None
        proposals = 1
        max_evals = 2
        seed = 0
        min_gain = 0.05
        timeout = 1
        metric_version = "v1"
        judge = None
        reflector = None
        out_dir = None

    with pytest.raises(HoldoutUsageError) as exc:
        _optimize(Args())
    assert "search" in exc.value.reason
