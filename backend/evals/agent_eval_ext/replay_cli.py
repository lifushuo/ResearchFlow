"""CLI: python -m evals.agent_eval_ext.replay_cli --case FILE --out FILE --trace-dir DIR"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from evals.agent_eval_ext.artifacts import check_output_path
from evals.agent_eval_ext.replay import load_case, run_replay, write_result_exclusive


def main() -> int:
    parser = argparse.ArgumentParser(description="Offline compiled-graph controlled fault replay")
    parser.add_argument("--case", required=True, help="Versioned JSON fixture")
    parser.add_argument("--out", required=True, help="New result JSON; no overwrite")
    parser.add_argument("--trace-dir", required=True, help="New Trace artifacts directory")
    args = parser.parse_args()
    case, sha256 = load_case(args.case)
    out = Path(args.out)
    # Refuse before graph execution when its result cannot be preserved (docs/08 §5).
    try:
        check_output_path(out)
    except OSError as exc:
        parser.error(f"Cannot create replay output: {exc}; no graph was run")
    result = asyncio.run(run_replay(case, trace_dir=args.trace_dir, artifact_root=out.parent))
    result["case_file_sha256"] = sha256
    write_result_exclusive(out, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
