"""把真实 Planner 任务证据填入 Judge 校准表；不伪造、不改标签、不动维度。

接受两类来源：
- ``--kind archive``：评测卷中 report-benchmark 的 ``raw_result.json``（含签名校验）
- ``--kind probe``：``run_planner_only_probe.py`` 的输出（必须是 real 模式）

只写 ``planner_task_queries`` 与 ``planner_trace_reference``；17 个必答维度、
human/judge 标签一律不碰。已填过任务或已有人工标注的题目拒绝覆盖。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1


def _canonical(value: Any) -> str:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    )


def verified_tasks_from_archive(doc: dict, match_id: str) -> dict:
    """Read one row's ``planner_evidence`` from an archived raw_result.json, integrity-checked."""
    rows = doc.get("results") if isinstance(doc, dict) else None
    if not isinstance(rows, list):
        raise ValueError("evidence file is not a raw_result document")
    row = next((r for r in rows if isinstance(r, dict) and r.get("id") == match_id), None)
    if row is None:
        raise ValueError(f"no archived result row for case {match_id}")
    evidence = row.get("planner_evidence")
    if not isinstance(evidence, dict) or evidence.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("archived row has no schema-1 planner evidence")
    signature = evidence.get("evidence_sha256")
    unsigned = {k: v for k, v in evidence.items() if k != "evidence_sha256"}
    expected = hashlib.sha256(_canonical(unsigned).encode("utf-8")).hexdigest()
    if not isinstance(signature, str) or signature != expected:
        raise ValueError("archived planner evidence failed its content fingerprint")
    return evidence


def verified_tasks_from_probe(doc: dict, match_id: str) -> dict:
    """Read a planner-only probe output; fake-mode probes are refused outright."""
    if not isinstance(doc, dict) or doc.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("probe file is not a schema-1 planner probe")
    if doc.get("case_id") != match_id:
        raise ValueError("probe case_id does not match the requested calibration case")
    if doc.get("llm_mode") != "real":
        raise ValueError("probe was recorded in fake mode; it cannot calibrate a real Judge")
    signature = doc.get("probe_evidence_sha256")
    unsigned = {k: v for k, v in doc.items() if k != "probe_evidence_sha256"}
    expected = hashlib.sha256(_canonical(unsigned).encode("utf-8")).hexdigest()
    if not isinstance(signature, str) or signature != expected:
        raise ValueError("probe failed its content fingerprint")
    return doc


def normalized_tasks(evidence: dict) -> list[str]:
    tasks = evidence.get("tasks")
    if not isinstance(tasks, list) or not 1 <= len(tasks) <= 30:
        raise ValueError("evidence has no usable task list (1–30 tasks)")
    queries = []
    for task in tasks:
        query = task.get("query") if isinstance(task, dict) else None
        if not isinstance(query, str) or not 5 <= len(query.strip()) <= 500:
            raise ValueError("evidence contains a malformed task query")
        queries.append(query.strip())
    return queries


def fill_case(
    sheet: dict,
    *,
    case_id: str,
    tasks: list[str],
    trace_reference: str,
) -> dict:
    """Return a new sheet with the case's task evidence filled; everything else untouched."""
    if sheet.get("purpose") != "manual_planner_judge_calibration_only":
        raise ValueError("wrong_calibration_sheet_purpose")
    target = next((c for c in sheet.get("cases", []) if c.get("case_id") == case_id), None)
    if target is None:
        raise ValueError(f"calibration sheet has no case {case_id}")
    if target.get("planner_task_queries"):
        raise ValueError("case already carries planner tasks; refusing to overwrite evidence")
    if any(
        a.get("human_covered") is not None or a.get("judge_covered") is not None
        for a in target["aspects"]
    ):
        raise ValueError("case already carries human or judge labels; refusing to modify it")
    filled = json.loads(json.dumps(sheet, ensure_ascii=False))
    case = next(c for c in filled["cases"] if c["case_id"] == case_id)
    case["planner_task_queries"] = tasks
    case["planner_trace_reference"] = trace_reference
    return filled


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sheet", type=Path, required=True)
    parser.add_argument("--case-id", required=True, help="calibration sheet case_id")
    parser.add_argument("--source", type=Path, required=True, help="raw_result.json or probe JSON")
    parser.add_argument("--kind", choices=("archive", "probe"), required=True)
    parser.add_argument("--match-id", required=True, help="archived row id / probe case_id")
    parser.add_argument("--run-id", default=None, help="archived run id (archive kind)")
    args = parser.parse_args(argv)

    sheet = json.loads(args.sheet.read_text(encoding="utf-8-sig"))
    doc = json.loads(args.source.read_text(encoding="utf-8-sig"))
    if args.kind == "archive":
        if not args.run_id:
            print("archive evidence requires --run-id", file=sys.stderr)
            return 2
        evidence = verified_tasks_from_archive(doc, args.match_id)
        sheet_case = next(
            (c for c in sheet.get("cases", []) if c.get("case_id") == args.case_id), None
        )
        if sheet_case is None:
            print(f"calibration sheet has no case {args.case_id}", file=sys.stderr)
            return 2
        # 证据必须属于这道题：问题指纹与校准表里的原题一致才允许回填。
        question_hash = hashlib.sha256(
            str(sheet_case.get("original_query", "")).encode("utf-8")
        ).hexdigest()
        if evidence.get("query_sha256") != question_hash:
            raise ValueError("evidence question fingerprint does not match the calibration case")
        tasks = normalized_tasks(evidence)
        trace_reference = f"auto:{args.run_id}:{evidence.get('evidence_sha256')}"
    else:
        evidence = verified_tasks_from_probe(doc, args.match_id)
        sheet_case = next(
            (c for c in sheet.get("cases", []) if c.get("case_id") == args.case_id), None
        )
        if sheet_case is None:
            print(f"calibration sheet has no case {args.case_id}", file=sys.stderr)
            return 2
        question_hash = hashlib.sha256(
            str(sheet_case.get("original_query", "")).encode("utf-8")
        ).hexdigest()
        if evidence.get("query_sha256") != question_hash:
            raise ValueError("evidence question fingerprint does not match the calibration case")
        tasks = normalized_tasks(evidence)
        trace_reference = f"planner-probe:{args.match_id}:{evidence.get('probe_evidence_sha256')}"

    filled = fill_case(sheet, case_id=args.case_id, tasks=tasks, trace_reference=trace_reference)
    tmp = args.sheet.with_suffix(args.sheet.suffix + ".tmp")
    tmp.write_text(json.dumps(filled, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, args.sheet)
    print(f"filled {len(tasks)} real planner tasks into {args.case_id}; trace: {trace_reference}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValueError, OSError) as exc:
        print(f"fill refused: {exc}; the sheet was not modified", file=sys.stderr)
        sys.exit(2)
