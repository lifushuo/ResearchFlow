"""Create an empty human-vs-Judge annotation sheet from TRAIN cases only.

Never calls an LLM. Labels default to null, not false.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def _fingerprint(value: Any) -> str:
    raw = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def make_sheet(dataset: dict, *, max_cases: int = 5) -> dict:
    if type(max_cases) is not int or not 1 <= max_cases <= 30:
        raise ValueError("max_cases_must_be_between_1_and_30")
    cases = dataset.get("cases") if isinstance(dataset, dict) else None
    if not isinstance(cases, list):
        raise ValueError("invalid_planner_dataset")
    train = [case for case in cases if isinstance(case, dict) and case.get("split") == "train"]
    if not train:
        raise ValueError("no_train_cases")
    if any(
        not isinstance(case.get("case_id"), str)
        or not isinstance(case.get("query"), str)
        or not isinstance(case.get("required_aspects"), list)
        or not case["required_aspects"]
        for case in train
    ):
        raise ValueError("invalid_train_case")
    # Include the audited backflow case first if it exists. 坏例通道回流带有
    # source_bad_case_id；评测案例通道（evaluation_case）没有该字段，但前端生成的
    # case_id 以 planner-review- 开头——两种真实回流都必须优先进入校准表。
    audited = [
        case
        for case in train
        if case.get("source_bad_case_id")
        or str(case.get("case_id", "")).startswith("planner-review-")
    ]
    rest = [case for case in train if case not in audited]
    selected = (audited[:1] + rest)[:max_cases]
    return {
        "schema_version": 1,
        "purpose": "manual_planner_judge_calibration_only",
        "dataset_sha256": _fingerprint(dataset),
        "created_at": datetime.now(UTC).isoformat(),
        "warning": "Null means unmeasured; it must never be scored as a negative.",
        "cases": [
            {
                "case_id": case["case_id"],
                "source_bad_case_id": case.get("source_bad_case_id"),
                "original_query": case["query"],
                "planner_trace_reference": None,
                "planner_task_queries": [],
                "aspects": [
                    {
                        "aspect_index": i,
                        "required_aspect": aspect,
                        "human_covered": None,
                        "human_matching_task_ids": [],
                        "human_reason": "",
                        "judge_covered": None,
                        "judge_matching_task_ids": [],
                        "judge_reason": "",
                        "agreement": None,
                    }
                    for i, aspect in enumerate(case["required_aspects"], 1)
                ],
                "reviewer": "",
                "review_note": "",
            }
            for case in selected
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-cases", type=int, default=5)
    args = parser.parse_args()
    dataset = json.loads(args.dataset.read_text(encoding="utf-8"))
    sheet = make_sheet(dataset, max_cases=args.max_cases)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(sheet, ensure_ascii=False, indent=2) + "\n")
    print(f"Generated {len(sheet['cases'])} TRAIN-only blank calibration cases: {args.out}")


if __name__ == "__main__":
    main()
