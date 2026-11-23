"""对校准表中一道已填任务证据的题目，做一次独立 Judge 覆盖建议。

复用 ``app.services.planner_review_judge.suggest_planner_coverage``——与评测中心
Judge 建议同一提示词、同一结构化契约、同一独立性检查，不重新实现 Judge。
建议写入独立文件（``automatic_label: false``）；校准表本身不被修改，
``judge_covered`` 由人工核对建议后自行填写。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

REQUIRED_CONSENT = "I_CONFIRM_JUDGE_API_COST"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sheet", type=Path, required=True)
    parser.add_argument("--case-id", required=True)
    parser.add_argument(
        "--run-id", required=True, help="real run or probe reference recorded in the sheet"
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--consent", default=None, help=f"must equal {REQUIRED_CONSENT}")
    parser.add_argument(
        "--metric-version",
        choices=("v1", "v2"),
        default="v1",
        help="v2 returns per-task specificity and per-aspect plan quality suggestions",
    )
    args = parser.parse_args(argv)

    if args.consent != REQUIRED_CONSENT:
        print(f"refusing: a real Judge call requires --consent {REQUIRED_CONSENT}", file=sys.stderr)
        return 2
    sheet = json.loads(args.sheet.read_text(encoding="utf-8-sig"))
    case = next((c for c in sheet.get("cases", []) if c.get("case_id") == args.case_id), None)
    if case is None:
        print(f"calibration sheet has no case {args.case_id}", file=sys.stderr)
        return 2
    if not case.get("planner_task_queries"):
        print(
            "case has no real planner tasks yet; run the evidence fill step first", file=sys.stderr
        )
        return 2

    async def run() -> dict:
        from app.services import planner_review_judge

        evidence = {
            "run_id": args.run_id,
            "case_id": args.case_id,
            "evidence_reference": case.get("planner_trace_reference") or "",
            "tasks": [
                {"id": str(index), "query": query}
                for index, query in enumerate(case["planner_task_queries"], 1)
            ],
        }
        suggest = (
            planner_review_judge.suggest_planner_coverage_v2
            if args.metric_version == "v2"
            else planner_review_judge.suggest_planner_coverage
        )
        return await suggest(
            evidence=evidence,
            question=case["original_query"],
            aspects=[aspect["required_aspect"] for aspect in case["aspects"]],
        )

    try:
        suggestion = asyncio.run(run())
    except ValueError as exc:
        print(f"judge suggestion refused: {exc}", file=sys.stderr)
        return 2
    except TimeoutError:
        print("Planner Judge timed out; nothing was recorded", file=sys.stderr)
        return 3
    except Exception as exc:  # noqa: BLE001 — provider errors may embed credentials
        print(f"judge call failed: {type(exc).__name__}; check server logs", file=sys.stderr)
        return 3

    payload = {
        "schema_version": 1,
        "purpose": (
            "manual_planner_judge_calibration_suggestion"
            if args.metric_version == "v1"
            else "manual_planner_judge_calibration_suggestion_v2"
        ),
        "metric_version": args.metric_version,
        "sheet_case_id": args.case_id,
        "run_id": args.run_id,
        "suggestion": suggestion,
        "warning": "Suggestions only; copy nothing automatically into judge_covered. Review every aspect yourself.",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    print(
        f"suggestion saved: {args.out} "
        f"(judge={suggestion.get('served_judge') or suggestion.get('judge_route')}, "
        f"automatic_label={suggestion.get('automatic_label')})"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
