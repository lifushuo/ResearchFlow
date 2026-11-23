"""Immutable report A/B case selection for Planner GEPA experiments.

The 9-case set excludes ``llm-memory`` because it entered GEPA train as
an audited bad case. Never silently substitute another benchmark.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

EVALS = Path(__file__).resolve().parent
ORIGINAL_REPORT_SET = EVALS / "queries.json"
INDEPENDENT_REPORT_SET = EVALS / "queries-planner-gepa-ab-v1.json"
EXCLUDED_CASE_ID = "llm-memory"


def _read_rows(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if (
        not isinstance(payload, dict)
        or payload.get("version") != 1
        or not isinstance(payload.get("queries"), list)
        or not payload["queries"]
        or any(not isinstance(row, dict) for row in payload["queries"])
    ):
        raise ValueError("invalid_report_dataset_format")
    return payload["queries"]


def fixed_report_queries(path: Path) -> list[dict[str, Any]]:
    """Allow only the original fixed set or the exact audited nine-case subset.

    A different file path is fine, but its cases MUST exactly match an allowed
    set, including order and all original fields. The original file is never edited.

    The frozen Planner holdout is explicitly rejected: those nine questions were
    historically report-benchmarked, so reusing them here would make the final
    Planner acceptance dual-use. See evals/holdout_usage_guard.py.
    """
    from evals.holdout_usage_guard import assert_holdout_usage

    assert_holdout_usage("report_ab", path=path)
    baseline = _read_rows(ORIGINAL_REPORT_SET)
    if len(baseline) != 10 or len({row.get("id") for row in baseline}) != 10:
        raise ValueError("original_report_dataset_changed")
    without_exposed = [row for row in baseline if row.get("id") != EXCLUDED_CASE_ID]
    if len(without_exposed) != 9:
        raise ValueError("exposed_report_case_missing_or_duplicated")
    selected = _read_rows(path)
    if selected != baseline and selected != without_exposed:
        raise ValueError("report_dataset_not_an_approved_fixed_set")
    return selected


def fixed_set_hash(rows: list[dict[str, Any]]) -> str:
    """Match evaluation_closure.input_hash for JSON-only report query rows."""
    encoded = json.dumps(
        rows, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def selected_report_dataset_for_experiment(experiment: dict) -> tuple[Path, list[dict]]:
    """Pin report A/B by the hash recorded when optimization began.

    Supports old 10-case experiments for backwards compatibility, but new
    live experiments should use the independent 9-case set by default.
    """
    saved_hash = experiment.get("report_dataset_sha256")
    for path in (INDEPENDENT_REPORT_SET, ORIGINAL_REPORT_SET):
        rows = fixed_report_queries(path)
        if saved_hash == fixed_set_hash(rows):
            return path, rows
    raise ValueError("experiment_report_dataset_hash_not_recognized")
