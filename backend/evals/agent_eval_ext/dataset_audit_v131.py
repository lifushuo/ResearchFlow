"""Strict, read-only source/Gold/executor contract audit of all eight registered datasets.

Does not call an LLM or any external service. A valid source is not a valid score;
execution results are verified separately by the respective runners.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from app.services.evaluation_dataset_registry import list_datasets
from evals.agent_eval_ext.executor_regression_v131 import EXPECTED, _cases
from evals.agent_eval_ext.judge_calibration_runner_v131 import validate_source

ROOT = Path(__file__).resolve().parents[2]
EXPECTED_COUNTS = {
    "report-benchmark": 10,
    "corpus-retrieval": 12,
    "selection-retrieval": 16,
    "grounding-v1": 425,
    "judge-calibration": 24,
    "planner-replay-v1": 4,
    "executor-regression-v1": 7,
    "tool-contract-v1": 4,
}


def _json(relative: str) -> dict:
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def _unique(rows: list[dict], field: str) -> None:
    ids = [x[field] for x in rows]
    if any(not isinstance(cid, str) or not cid for cid in ids) or len(ids) != len(set(ids)):
        raise ValueError("missing_or_duplicate_id")


def _source_validation(dataset_id: str) -> None:
    if dataset_id == "report-benchmark":
        rows = _json("evals/queries.json")["queries"]
        _unique(rows, "id")
        if any(not x.get("query") for x in rows):
            raise ValueError("empty_report_query")
    elif dataset_id in ("corpus-retrieval", "selection-retrieval"):
        prefix = (
            "evals/datasets/retrieval"
            if dataset_id == "corpus-retrieval"
            else "evals/datasets/selection-retrieval"
        )
        rows = _json(prefix + "/queries.json")["queries"]
        _unique(rows, "id")
        for row in rows:
            if not row.get("query") or not isinstance(row.get("grades"), dict):
                raise ValueError("missing_retrieval_query_gold")
            # Selection's negative-control s11 intentionally has no relevant gold
            # in the selected corpus; it measures leakage, not answer recall.
            if dataset_id == "corpus-retrieval" and not row["grades"]:
                raise ValueError("missing_retrieval_query_gold")
        if dataset_id == "corpus-retrieval":
            corpus = prefix + "/corpus"
            for row in rows:
                for name in row["grades"]:
                    if not (ROOT / corpus / name).is_file():
                        raise ValueError("retrieval_gold_corpus_missing")
        else:
            for row in rows:
                if not row.get("selection") or not set(row["grades"]).issubset(
                    set(row["selection"])
                ):
                    raise ValueError("selection_gold_not_in_scope")
    elif dataset_id == "grounding-v1":
        rows = _json("evals/datasets/grounding/v1/items.json")["items"]
        _unique(rows, "id")
        gold = _json("evals/datasets/grounding/v1/labels.json")
        if set(gold.get("labels", {})) != {x["id"] for x in rows}:
            raise ValueError("grounding_missing_gold")
        if set(gold["labels"].values()) - {"supported", "not_supported", "contradicted"}:
            raise ValueError("grounding_invalid_gold")
        if any(not x.get("claim") or not x.get("snippets") for x in rows):
            raise ValueError("grounding_case_incomplete")
    elif dataset_id == "judge-calibration":
        validate_source()
    elif dataset_id == "planner-replay-v1":
        manifest = _json("evals/agent_eval_ext/fixtures/replay_suite_v1.json")
        for name in manifest["cases"]:
            if "/" in name or "\\" in name or not name.endswith(".json"):
                raise ValueError("invalid_replay_manifest")
            case = _json("evals/agent_eval_ext/fixtures/" + name)
            if not case.get("case_id"):
                raise ValueError("invalid_replay_case")
    elif dataset_id == "executor-regression-v1":
        if tuple(c["case_id"] for c in _cases()) != EXPECTED:
            raise ValueError("invalid_executor_suite")
    elif dataset_id == "tool-contract-v1":
        fixture = "evals/agent_eval_ext/fixtures/"
        gold = _json(fixture + "tool_gold_v1.json")["cases"]
        good = _json(fixture + "tool_observed_pass_v1.json")["cases"]
        bad = _json(fixture + "tool_observed_fail_v1.json")["cases"]
        _unique(gold, "case_id")
        if not (
            {r["case_id"] for r in gold}
            == {r["case_id"] for r in good}
            == {r["case_id"] for r in bad}
        ):
            raise ValueError("tool_control_missing_case")
    else:
        raise ValueError("unknown_dataset")


def audit() -> dict:
    inventory = list_datasets()["datasets"]
    if {x["id"] for x in inventory} != set(EXPECTED_COUNTS):
        raise ValueError("dataset_registry_mismatch")
    report = []
    for row in inventory:
        entry = {
            "dataset_id": row["id"],
            "case_count": row["verified_count"],
            "source_sha256": row["source_sha256"],
            "ok": False,
            "reason": None,
        }
        try:
            if row["source_status"] != "available" or row["count_basis"] != "source_verified":
                raise ValueError("source_unavailable_or_not_verified")
            if row["verified_count"] != EXPECTED_COUNTS[row["id"]]:
                raise ValueError("source_count_mismatch")
            _source_validation(row["id"])
            entry["ok"] = True
        except (KeyError, ValueError, TypeError, OSError) as exc:
            entry["reason"] = str(exc)[:120]
        report.append(entry)
    return {
        "schema_version": 1,
        "kind": "dataset_source_audit",
        "all_sources_ready": all(row["ok"] for row in report),
        "note": "Dataset source and Gold integrity only; model capability is NOT established",
        "datasets": report,
    }


def main() -> int:
    result = audit()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["all_sources_ready"] else 1


if __name__ == "__main__":
    sys.exit(main())
