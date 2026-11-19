"""Fixed adapters for the original ResearchFlow evaluation implementations.

This module is invoked by the dedicated online worker, not by FastAPI. It accepts
ONLY known dataset IDs, writes new output to a worker-owned job directory and
never overwrites old evals/results files. Report invokes the existing harness;
retrieval and grounding directly call their existing `run()` implementations.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import signal
import subprocess
import sys
from pathlib import Path

from app.services.evaluation_pipeline_v13 import judge_route, online_lane_enabled

RUNNABLE = frozenset(
    {
        "report-benchmark",
        "corpus-retrieval",
        "selection-retrieval",
        "grounding-v1",
        "judge-calibration",
    }
)


def _assert_new_file(path: Path) -> None:
    if path.is_symlink() or not path.parent.is_dir() or path.exists():
        raise ValueError("output_not_new_or_unsafe")


def _hold_stop_signals() -> dict[int, object]:
    """Keep the suite alive long enough to let the harness flush its progress.

    The worker sends SIGTERM to the whole process group, so the harness receives
    the same signal as this suite and its own `install_stop_handlers` turns it
    into a cooperative stop. Without a handler here the suite would die from the
    default action and orphan `subprocess.run` before `progress.json` /
    `final-status.json` could be written. Keeping `subprocess.run` also preserves
    the existing adapter mock contract used by the case-selection tests.
    """
    previous: dict[int, object] = {}

    def hold(_signum, _frame):  # noqa: ANN001
        # Deliberately a no-op: the child owns the cooperative stop and the flush.
        pass

    for name in ("SIGTERM", "SIGINT"):
        sig = getattr(signal, name, None)
        if not isinstance(sig, int):
            continue
        try:
            previous[sig] = signal.getsignal(sig)
            signal.signal(sig, hold)
        except ValueError, OSError:
            previous.pop(sig, None)
    return previous


def _restore_stop_signals(previous: dict[int, object]) -> None:
    for sig, handler in previous.items():
        try:
            signal.signal(sig, handler)
        except ValueError, OSError:
            pass


async def _evaluate(dataset_id: str, limit: int) -> dict:
    if dataset_id == "corpus-retrieval":
        from evals.retrieval import run

        return await run()
    if dataset_id == "selection-retrieval":
        from evals.selection_retrieval import run

        return await run()
    if dataset_id == "grounding-v1":
        from evals.grounding_benchmark import run

        return await run(limit=limit or None)
    raise ValueError("unsupported_async_dataset")


def _execute(
    dataset_id: str,
    output: Path,
    max_cases: int,
    *,
    judge_route: str | None = None,
    case_id: str | None = None,
) -> None:
    from evals.query_selection import validate_report_selection

    validate_report_selection([dataset_id], max_cases, case_id)
    if (
        dataset_id not in RUNNABLE
        or type(max_cases) is not int
        or not (max_cases == 0 or 1 <= max_cases <= 20)
    ):
        raise ValueError("unsupported_dataset_or_case_limit")
    if not online_lane_enabled() or os.getenv("LLM_MODE") != "real":
        raise ValueError("real_eval_not_explicitly_enabled")
    _assert_new_file(output)
    if dataset_id == "judge-calibration":
        if not judge_route:
            raise ValueError("independent_pinned_judge_route_required")
        from evals.agent_eval_ext.judge_calibration_runner_v131 import execute as judge_execute

        result = asyncio.run(judge_execute(limit=max_cases, judge_route=judge_route))
        with output.open("x", encoding="utf-8") as stream:
            json.dump(result, stream, ensure_ascii=False, allow_nan=False)
            stream.write("\n")
        return
    if dataset_id == "report-benchmark":
        if not judge_route or ":" not in judge_route or len(judge_route) > 128:
            raise ValueError("independent_pinned_judge_route_required")
        # Exact existing ResearchFlow harness; never call it through shell=True.
        cmd = [
            sys.executable,
            "-m",
            "evals.harness",
            "--mode",
            "report",
            *(
                ["--case-id", case_id]
                if case_id is not None
                else ["--limit", str(min(max_cases or 10, 10))]
            ),
            "--judge",
            judge_route,
            "--trace",
            "--trace-dir",
            str(output.parent / "traces"),
            "--out",
            str(output),
        ]
        # The worker controls the timeout and process-group termination. Only the
        # report lane gets a reliability root; the other datasets keep their exact
        # existing environment and code path.
        from evals.agent_eval_ext.execution_observer import subprocess_output

        with subprocess_output("adapter.report_harness") as (logs, span):
            previous = _hold_stop_signals()
            try:
                proc = subprocess.run(
                    cmd,
                    env={
                        **os.environ,
                        "RF_JUDGE_BATCH_TRACE_DIR": str(output.parent / "judge-batches"),
                        "RF_EVAL_CONTENT_TRACE": "1",
                        "RF_EVAL_EXECUTION_ROOT": str(output.parent),
                        "RF_EVAL_RELIABILITY_DIR": str(output.parent / "reliability"),
                    },
                    check=False,
                    stdin=subprocess.DEVNULL,
                    stdout=logs,
                    stderr=subprocess.STDOUT,
                )
            finally:
                _restore_stop_signals(previous)
            returncode = proc.returncode
            span.observe(exit_code=returncode)
            if returncode:
                span.status = "business_error"
        if returncode:
            raise RuntimeError("research_harness_failed")
        if not output.is_file():
            raise RuntimeError("report_output_missing")
        if case_id is not None:
            rows = json.loads(output.read_text(encoding="utf-8"))["results"]
            if len(rows) != 1 or rows[0].get("id") != case_id:
                raise ValueError("report_case_identity_mismatch")
        return
    result = asyncio.run(_evaluate(dataset_id, max_cases))
    with output.open("x", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, allow_nan=False)
        f.write("\n")


def execute(
    dataset_id: str,
    output: Path,
    max_cases: int,
    *,
    judge_route: str | None = None,
    case_id: str | None = None,
) -> None:
    from evals.agent_eval_ext.execution_observer import execution_scope
    from evals.query_selection import validate_report_selection

    validate_report_selection([dataset_id], max_cases, case_id)
    if dataset_id not in RUNNABLE or not online_lane_enabled() or os.getenv("LLM_MODE") != "real":
        raise ValueError("real_eval_not_explicitly_enabled")
    _assert_new_file(output)
    # Dedicated evaluation only. No web request or normal research run enables content capture.
    with execution_scope(
        output.parent,
        os.environ.get("RF_EVAL_TRACE_RUN_ID") or output.parents[2].name,
        "adapter." + dataset_id,
        inputs={"case_id": case_id, "max_cases": max_cases},
    ):
        _execute(dataset_id, output, max_cases, judge_route=judge_route, case_id=case_id)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-id", required=True, choices=sorted(RUNNABLE))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-cases", required=True, type=int)
    parser.add_argument("--case-id")
    args = parser.parse_args()
    execute(
        args.dataset_id,
        args.output,
        args.max_cases,
        judge_route=judge_route(),
        **({"case_id": args.case_id} if args.case_id is not None else {}),
    )


if __name__ == "__main__":
    main()
