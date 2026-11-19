"""Explicitly paid live calibration using ResearchFlow's EXISTING judge implementation.

Runs only with --allow-paid and LLM_MODE=real. A claim is evaluated against exactly
one synthetic source so it exercises the real judge prompt/response parser.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from datetime import UTC, datetime
from pathlib import Path

from .artifacts import check_output_path, save_artifact
from .audit import sha256_file
from .calibration import load_calibration, score_calibration


async def run(path: Path, judge: str, limit: int | None, split: str | None) -> list[dict]:
    if os.environ.get("LLM_MODE") != "real":
        raise ValueError("Live calibration requires explicit LLM_MODE=real")
    if limit is not None and limit <= 0:
        raise ValueError("limit must be positive")
    rows = load_calibration(path)
    if split is not None:
        rows = [r for r in rows if r.get("split") == split]
    if limit is not None:
        rows = rows[:limit]
    if not rows:
        raise ValueError("No calibration cases selected")

    from evals import harness

    if harness.RUN_CONFIG.llm_mode != "real":
        raise ValueError("Judge harness is already initialized in fake mode; use a new process")
    harness.configure_judge(judge)
    for i, row in enumerate(rows, 1):
        # Let the original report judge extract its [1] citation and evidence.
        report = f"{row['claim'].rstrip('.!? ')} [1]."
        sources = [{"index": 1, "snippets": [row["evidence"]]}]
        from research_engine.eval_trace import trace_span

        with trace_span(
            "calibration.case",
            kind="evaluation_case",
            inputs={"case_id": row["case_id"], "report": report, "sources": sources},
        ) as span:
            _rate, verdicts = await harness.judge_citation_support(report, sources)
            span.observe(output={"rate": _rate, "verdicts": verdicts})
        if len(verdicts) == 1 and verdicts[0].get("judged"):
            row["predicted_supported"] = verdicts[0]["supported"]
        else:
            row["predicted_supported"] = None
        row["claim_verdicts"] = verdicts
        print(f"[{i}/{len(rows)}] {row['case_id']}: {row['predicted_supported']}")
    return rows


def main() -> None:
    p = argparse.ArgumentParser(description="Paid independent Judge calibration")
    p.add_argument("--input", required=True)
    p.add_argument("--judge", required=True, help="e.g. custom:kimi-k3")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument(
        "--split", choices=["dev", "holdout"], help="evaluate only selected labeled split"
    )
    p.add_argument("--out", required=True, help="MUST be a new path")
    p.add_argument(
        "--allow-paid", action="store_true", help="explicit confirmation: real model API charges"
    )
    args = p.parse_args()
    if not args.allow_paid:
        p.error("Refusing real API calls without --allow-paid")
    if os.environ.get("LLM_MODE") != "real":
        p.error("Refusing real calibration without LLM_MODE=real")
    if args.limit is not None and args.limit <= 0:
        p.error("--limit must be positive")
    path = Path(args.out)
    try:
        check_output_path(path)
    except OSError as exc:
        p.error(f"Cannot write calibration output: {exc}; no model calls were made")
    rows = asyncio.run(run(Path(args.input), args.judge, args.limit, args.split))
    result = {
        "judge": args.judge,
        "generated_at": datetime.now(UTC).isoformat(),
        "fixture_sha256": sha256_file(args.input),
        "split": args.split,
        "metrics": score_calibration(rows),
        "cases": rows,
    }
    save_artifact(result, path)
    print("Wrote:", path)
    print(json.dumps(result["metrics"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
