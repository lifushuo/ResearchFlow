"""不调用模型，仅检查完成结果，并将门禁策略写入审计。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .core import ClosureError, EvaluationClosure, NotFound


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="ResearchFlow offline quality-gate checker (audited)")
    p.add_argument("--db", required=True, help="Mounted evaluation closure SQLite file")
    p.add_argument("--owner", required=True, help="Verified owner ID used by trusted worker")
    p.add_argument("--run-id", required=True, help="Exact dataset-step run ID")
    p.add_argument("--min-measured", type=int, default=1)
    p.add_argument("--min-pass-rate", type=float, default=0.9)
    p.add_argument("--max-p95-ms", type=int, default=None)
    args = p.parse_args(argv)
    try:
        if not Path(args.db).is_file():
            raise ClosureError("Closure database does not exist")
        store = EvaluationClosure(args.db)
        report = store.quality_gate(
            args.owner,
            args.run_id,
            min_measured=args.min_measured,
            min_pass_rate=args.min_pass_rate,
            max_p95_latency_ms=args.max_p95_ms,
        )
    except (ClosureError, NotFound) as ex:
        print(json.dumps({"verdict": "error", "message": str(ex)}, ensure_ascii=False))
        return 3
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return {"passed": 0, "failed": 2, "blocked": 3}[report["verdict"]]


if __name__ == "__main__":
    sys.exit(main())
