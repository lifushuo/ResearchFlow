"""Controlled tool-call scoring; no network and no paid model execution."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .artifacts import check_output_path
from .tool_safety import (
    compare,
    evaluate,
    load_gold,
    load_observed,
    save_new,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="ResearchFlow Tool/Safety contract v0.8")
    sub = parser.add_subparsers(dest="command", required=True)
    check = sub.add_parser("score")
    check.add_argument("--gold", required=True)
    check.add_argument("--observed", required=True)
    check.add_argument("--out")
    comp = sub.add_parser("compare")
    comp.add_argument("--baseline", required=True)
    comp.add_argument("--candidate", required=True)
    comp.add_argument("--out")
    args = parser.parse_args(argv)
    try:
        if args.out:
            check_output_path(Path(args.out))
        if args.command == "score":
            gold, gold_sha = load_gold(args.gold)
            obs, obs_sha = load_observed(args.observed)
            result = evaluate(gold, obs, gold_sha256=gold_sha, observed_sha256=obs_sha)
            passed = result["gate_passed"]
        else:
            result = compare(
                json.loads(Path(args.baseline).read_text(encoding="utf-8-sig")),
                json.loads(Path(args.candidate).read_text(encoding="utf-8-sig")),
            )
            passed = result["comparable"] and result["gate_passed"]
        if args.out:
            save_new(args.out, result)
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
        return 0 if passed else 1
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"tool_safety: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
