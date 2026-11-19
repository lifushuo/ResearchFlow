"""Shared pre-call selection over the original query rows; never rewrites a dataset."""

import json
from pathlib import Path


def select_queries(
    queries: list[dict], *, case_id: str | None = None, limit: int | None = None
) -> list[dict]:
    if case_id is None:
        return queries[:limit] if limit else queries
    if limit is not None:
        raise ValueError("case_id_conflicts_with_limit")
    if not isinstance(case_id, str) or not case_id or case_id != case_id.strip():
        raise ValueError("invalid_report_case_id")
    if not isinstance(queries, list) or any(
        not isinstance(q, dict) or not isinstance(q.get("id"), str) for q in queries
    ):
        raise ValueError("invalid_report_queries")
    ids = [q["id"] for q in queries]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate_report_case_id")
    selected = [q for q in queries if q["id"] == case_id]
    if len(selected) != 1:
        raise ValueError("unknown_report_case_id")
    return selected


def load_report_queries(case_id: str | None = None, *, path: Path | None = None) -> list[dict]:
    source = path if path is not None else Path(__file__).with_name("queries.json")
    queries = json.loads(source.read_text(encoding="utf-8-sig"))["queries"]
    return select_queries(queries, case_id=case_id)


def validate_report_selection(datasets: list[str], max_cases: int, case_id: str | None) -> None:
    if case_id is None:
        return
    if datasets != ["report-benchmark"]:
        raise ValueError("case_id_requires_report_benchmark_only")
    if type(max_cases) is not int or max_cases != 1:
        raise ValueError("case_id_requires_max_cases_1")
    load_report_queries(case_id)
