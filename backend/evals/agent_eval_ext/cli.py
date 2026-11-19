"""Python standard-library CLI; zero paid API calls.

Usage (from backend/): python -m evals.agent_eval_ext.cli audit --input <result.json>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .artifacts import save_artifact
from .audit import load_run, report_audit, sha256_file
from .calibration import load_calibration, score_calibration
from .compare import compare_runs
from .gates import evaluate_gate
from .judge_batch_audit import analyze_report
from .judge_runtime_inspect import summarize
from .trace_inspect import inspect_trace
from .trajectory import check_trajectory


def _save(result: dict, output: str | None) -> None:
    if output:
        path = Path(output)
        save_artifact(result, path)
        print(f"Created new artifact: {path}")
    else:
        print(json.dumps({"extension_schema_version": 1, **result}, indent=2, ensure_ascii=False))


def main() -> None:
    p = argparse.ArgumentParser(description="ResearchFlow additive agent eval tools (offline)")
    sub = p.add_subparsers(dest="command", required=True)
    audit = sub.add_parser("audit", help="recount the existing 10-query JSON and list bad cases")
    audit.add_argument("--input", required=True)
    audit.add_argument(
        "--dataset", help="hash a supplied dataset; does not prove historical provenance"
    )
    audit.add_argument("--out")
    batches = sub.add_parser("judge-batches", help="infer batch coverage from saved claim verdicts")
    batches.add_argument("--input", required=True)
    batches.add_argument("--out")
    batches.add_argument("--batch-size", type=int, default=4, help="Assumed historical batch size")
    gate = sub.add_parser("gate", help="proposed stricter quality gate including judge coverage")
    gate.add_argument("--input", required=True)
    gate.add_argument("--out")
    compare = sub.add_parser(
        "compare", help="paired case-id comparison, with comparability warnings"
    )
    compare.add_argument("--baseline", required=True)
    compare.add_argument("--candidate", required=True)
    compare.add_argument("--baseline-dataset")
    compare.add_argument("--candidate-dataset")
    compare.add_argument("--out")
    calibration = sub.add_parser(
        "calibration", help="score a labeled JSONL; null predictions remain unmeasured"
    )
    calibration.add_argument("--input", required=True)
    calibration.add_argument("--out")
    calibration.add_argument("--split", choices=["dev", "holdout"])
    trajectory = sub.add_parser(
        "trajectory", help="check tool-call fixture JSON (does NOT call the Agent)"
    )
    trajectory.add_argument("--input", required=True)
    trajectory.add_argument("--out")
    trace = sub.add_parser("trace", help="inspect runtime JSONL metadata; no model calls")
    trace.add_argument("--input", required=True)
    trace.add_argument("--out")
    judge_trace = sub.add_parser(
        "judge-runtime", help="inspect actual application-level judge batches"
    )
    judge_trace.add_argument("--input", required=True)
    judge_trace.add_argument("--out")
    args = p.parse_args()
    if args.command == "audit":
        result = report_audit(load_run(args.input), dataset_path=args.dataset)
        result["input_sha256"] = sha256_file(args.input)
    elif args.command == "judge-batches":
        result = analyze_report(load_run(args.input), batch_size=args.batch_size)
        result["input_sha256"] = sha256_file(args.input)
    elif args.command == "gate":
        result = evaluate_gate(report_audit(load_run(args.input)))
        result["input_sha256"] = sha256_file(args.input)
    elif args.command == "compare":
        result = compare_runs(
            load_run(args.baseline),
            load_run(args.candidate),
            baseline_dataset=args.baseline_dataset,
            candidate_dataset=args.candidate_dataset,
        )
        result["inputs_sha256"] = {
            "baseline": sha256_file(args.baseline),
            "candidate": sha256_file(args.candidate),
        }
    elif args.command == "calibration":
        rows = load_calibration(args.input)
        if args.split:
            rows = [r for r in rows if r.get("split") == args.split]
        result = score_calibration(rows)
    elif args.command == "trace":
        result = inspect_trace(args.input)
    elif args.command == "judge-runtime":
        result = summarize(args.input)
    else:
        spec = json.loads(Path(args.input).read_text(encoding="utf-8"))
        result = check_trajectory(
            spec["expected"],
            spec["observed"],
            mode=spec.get("mode", "strict"),
            check_args=spec.get("check_args", True),
            forbidden_tools=spec.get("forbidden_tools", []),
        )
    _save(result, args.out)
    if args.command in {"gate", "trajectory"} and not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
