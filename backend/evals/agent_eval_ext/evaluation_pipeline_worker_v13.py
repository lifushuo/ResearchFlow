"""Two physical lanes for one v1.3 pipeline queue.

Launch with --lane offline in a network:none container without keys, or --lane
online in a separate trusted container, deliberately enabled with explicit env.
Only fixed module commands execute. Raw results NEVER go to the public API.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import structlog

from app.services.evaluation_jobs import job_root
from app.services.evaluation_pipeline_v13 import (
    claim_next,
    finish_step,
    online_lane_enabled,
    recover_interrupted,
)
from evals.agent_eval_ext import long_run
from evals.agent_eval_ext.evaluation_job_worker import (
    _execute as legacy_offline_execute,
)
from evals.agent_eval_ext.evaluation_job_worker import (
    _run_command as fixed_offline_command,
)

BACKEND = Path(__file__).resolve().parents[2]
STOP = False
_ACTIVE_PROCESS: subprocess.Popen | None = None
#: The P0 cooperative-stop path is deliberately limited to the report lane;
#: the other seven datasets keep their historical termination behaviour.
_ACTIVE_DATASET: str | None = None


def _stop_grace_s() -> float:
    """The same grace the harness reads; bounded so a stop never outlives the cap."""
    return long_run.stop_grace_seconds()


def _signal_group(name: str) -> None:
    """Signal the whole child session (Linux online worker). Windows has no process
    groups/catchable SIGTERM, so it falls back to terminating the direct child;
    the online evaluation lane is Linux-only, so cooperative saving is preserved
    everywhere a paid run can actually happen.
    """
    proc = _ACTIVE_PROCESS
    if proc is None:
        return
    if not hasattr(os, "killpg"):
        # Windows has neither SIGKILL nor process groups. Popen.kill() is the
        # equivalent last resort; SIGTERM only reaches the direct child.
        sig = getattr(signal, name, None)
        try:
            if sig is None:
                proc.kill()
            else:
                proc.send_signal(sig)
        except ProcessLookupError, OSError:
            pass
        return
    sig = getattr(signal, name, None)
    if not isinstance(sig, int):
        return
    try:
        os.killpg(proc.pid, sig)
    except ProcessLookupError, OSError:
        pass


def _stop(*_args: object):
    """For the report lane, stop cooperatively before any forced termination.

    SIGKILL is scheduled only as a fallback after the stop grace, and becomes a
    no-op when the child has already exited and `_ACTIVE_PROCESS` was cleared.
    Other datasets keep their historical worker-cancellation behaviour.
    """
    global STOP
    STOP = True
    if _ACTIVE_DATASET != "report-benchmark":
        return
    _signal_group("SIGTERM")
    timer = threading.Timer(_stop_grace_s(), lambda: _signal_group("SIGKILL"))
    timer.daemon = True
    timer.start()


def _clean_metrics(value: object, *, depth: int = 0) -> object:
    """Expose numerical aggregates only, never raw queries, claims, tool args or model output."""
    if depth > 4:
        return None
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        if abs(value) < 1e12 and (not isinstance(value, float) or value == value):
            return value
        return None
    if isinstance(value, dict):
        return {
            str(k)[:60]: _clean_metrics(v, depth=depth + 1)
            for k, v in value.items()
            if isinstance(k, str)
            and len(k) <= 60
            and isinstance(v, (int, float, bool, type(None), dict))
        }
    return None


def _summary(dataset: str, path: Path, *, case_id: str | None = None) -> dict:
    if path.stat().st_size > 8 * 1024 * 1024 or path.is_symlink():
        raise ValueError("unsafe_or_oversized_artifact")
    doc = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(doc, dict):
        raise ValueError("invalid_artifact_root")
    if dataset == "judge-calibration":
        metrics = doc.get("metrics")
        if (
            doc.get("dataset_id") != dataset
            or not isinstance(metrics, dict)
            or type(metrics.get("cases")) is not int
            or not isinstance(doc.get("cases"), list)
            or len(doc["cases"]) != metrics["cases"]
            or metrics["cases"] < 1
            or metrics.get("judged", 0) + metrics.get("unjudged", -1) != metrics["cases"]
        ):
            raise ValueError("invalid_judge_calibration_result")
        if doc.get("status") != "measured" or metrics["judged"] == 0:
            return {
                "state": "failed",
                "reason": "judge_all_unmeasured",
                "case_count": metrics["cases"],
                "metrics": {"judged": 0, "unjudged": metrics["unjudged"]},
            }
        public = {**metrics, "split_counts": doc.get("split_counts", {})}
        return {
            "state": "completed",
            "case_count": metrics["cases"],
            "metrics": _clean_metrics(public),
        }
    if dataset == "report-benchmark":
        agg = doc.get("aggregate", {})
        if not isinstance(agg, dict) or not isinstance(doc.get("results"), list):
            raise ValueError("invalid_report")
        rows = doc["results"]
        if not rows or not all(isinstance(r, dict) for r in rows):
            raise ValueError("invalid_report_cases")
        if case_id is not None and (len(rows) != 1 or rows[0].get("id") != case_id):
            raise ValueError("report_case_identity_mismatch")
        return {
            "state": "completed",
            "case_count": len(rows),
            "metrics": {
                "task_completion": sum(r.get("completed") is True for r in rows) / len(rows),
                "completed": sum(r.get("completed") is True for r in rows),
                "meets_threshold": all(r.get("completed") is True for r in rows),
                "judge": _clean_metrics(doc.get("judge", {}).get("baseline", {})),
                "aggregate": _clean_metrics(agg),
            },
        }
    if dataset != "grounding-v1" and doc.get("status") != "measured":
        raise ValueError("run_not_measured")
    rows = doc.get("queries") if dataset != "grounding-v1" else doc.get("claims")
    # Grounding benchmark stores its verdict metrics under `result`, not `aggregate`.
    agg = doc.get("result") if dataset == "grounding-v1" else doc.get("aggregate")
    if not isinstance(agg, dict):
        raise ValueError("invalid_aggregate")
    if dataset == "grounding-v1" and agg.get("status") not in ("measured", "invalid"):
        raise ValueError("grounding_invalid_or_unmeasured")
    count = (
        agg.get("claims")
        if dataset == "grounding-v1"
        else (len(rows) if isinstance(rows, list) else None)
    )
    if type(count) is not int or count < 1:
        raise ValueError("missing_case_count")
    if dataset == "grounding-v1" and (not isinstance(rows, list) or len(rows) != count):
        raise ValueError("grounding_count_mismatch")
    if dataset == "grounding-v1" and agg.get("status") == "invalid":
        # A checker that left too many claims unjudged has an INVALID metric;
        # preserve the observed denominator, but never turn it into a PASS.
        return {
            "state": "completed",
            "case_count": count,
            "metrics": {
                **(_clean_metrics(agg) or {}),
                "meets_threshold": False,
                "invalid_metric": True,
            },
        }
    return {"state": "completed", "case_count": count, "metrics": _clean_metrics(agg)}


def _run_fixed(dataset_id: str, result: Path, max_cases: int, *, case_id: str | None = None) -> int:
    global _ACTIVE_PROCESS, _ACTIVE_DATASET
    if case_id is not None:
        from evals.query_selection import validate_report_selection

        validate_report_selection([dataset_id], max_cases, case_id)
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONSTARTUP", "PYTHONINSPECT")}
    env["LLM_MODE"] = "real"
    env["RF_EVAL_CONTENT_TRACE"] = "1"
    env["RF_EVAL_TRACE_RUN_ID"] = result.parents[2].name
    hard_cap_s = long_run.worker_hard_cap(max_cases)
    grace_s = _stop_grace_s() if dataset_id == "report-benchmark" else 0
    if dataset_id == "report-benchmark":
        env["RF_EVAL_BATCH_LEDGER_ROOT"] = str(
            Path(os.environ.get("RF_EVAL_JOB_ROOT") or result.parent) / "judge-batch-ledger"
        )
        env["RF_EVAL_WORKER_HARD_CAP_S"] = str(hard_cap_s)
        env["RF_EVAL_WORKER_CASES"] = str(1 if case_id is not None else min(max_cases or 10, 10))
    cmd = [
        sys.executable,
        "-m",
        "evals.agent_eval_ext.evaluation_pipeline_suite_v13",
        "--dataset-id",
        dataset_id,
        "--output",
        str(result),
        "--max-cases",
        str(max_cases),
    ]
    if case_id is not None:
        cmd.extend(["--case-id", case_id])
    from evals.agent_eval_ext.execution_observer import subprocess_output

    with subprocess_output("worker.evaluator_process") as (output, span):
        process = subprocess.Popen(
            cmd,
            cwd=str(BACKEND),
            env=env,
            shell=False,
            stdin=subprocess.DEVNULL,
            stdout=output,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        # Keep the existing 420 s hard cap untouched. The reliability layer stops
        # *inside* that budget; this is only the worker's last-resort boundary.
        _ACTIVE_PROCESS = process
        _ACTIVE_DATASET = dataset_id
        stop_mode = None
        try:
            # The grace is inside the cap, rather than extending a 420 s job to 435 s.
            code = process.wait(timeout=hard_cap_s - grace_s)
        except subprocess.TimeoutExpired:
            if dataset_id == "report-benchmark":
                outcome = long_run.stop_child(
                    send=lambda name: _signal_group(name),
                    wait=long_run.child_waiter(process),
                    grace_s=grace_s,
                )
                stop_mode = outcome["mode"]
            else:
                # The other seven datasets keep their exact historical timeout path.
                if hasattr(os, "killpg"):
                    os.killpg(process.pid, signal.SIGKILL)
                else:
                    process.kill()
                process.wait()
                stop_mode = "legacy_forced"
            code = 124
        finally:
            _ACTIVE_PROCESS = None
            _ACTIVE_DATASET = None
        span.observe(exit_code=code, timed_out=code == 124)
        if code:
            span.status = "business_error"
    reliability = long_run.read_root_status(result.parent / "reliability") or {}
    try:
        from app.services.evaluation_jobs import update_status

        update_status(
            result.parent / "process.json",
            {
                "exit_code": code,
                "timed_out": code == 124,
                "stop_mode": stop_mode,
                "reliability_status": reliability.get("status"),
                "reliability_reason": reliability.get("reason"),
            },
        )
    except OSError as exc:
        structlog.get_logger().warning(
            "evaluation_process_observation_failed", error_type=type(exc).__name__
        )
    return code


def _execute_step(folder: Path, request: dict, index: int, lane: str) -> dict:
    case_id = request.get("case_id")
    if case_id is not None:
        from evals.query_selection import validate_report_selection

        validate_report_selection(request["datasets"], request["max_cases"], case_id)
    dataset_id = request["datasets"][index]
    out = folder / "artifacts" / dataset_id
    out.mkdir(mode=0o700, parents=True, exist_ok=True)
    if lane == "offline":
        if dataset_id == "executor-regression-v1":
            result_path = out / "executor-result.json"
            code = fixed_offline_command(
                [
                    sys.executable,
                    "-m",
                    "evals.agent_eval_ext.executor_regression_v131",
                    "--output",
                    str(result_path),
                ]
            )
            if code not in (0, 1) or not result_path.is_file():
                return {
                    "state": "failed",
                    "reason": "executor_suite_execution_failed",
                    "metrics": None,
                    "case_count": None,
                }
            record = json.loads(result_path.read_text(encoding="utf-8"))
            from evals.agent_eval_ext.executor_regression_v131 import EXPECTED

            verdicts = record.get("cases", [])
            if (
                record.get("evidence_kind") != "controlled_executor_regression"
                or record.get("real_model_executed") is not False
                or record.get("case_count") != 7
                or not isinstance(verdicts, list)
                or len(verdicts) != 7
                or tuple(c.get("case_id") for c in verdicts) != EXPECTED
                or any(type(c.get("passed")) is not bool for c in verdicts)
                or record.get("passed_count") != sum(c["passed"] for c in verdicts)
                or record.get("gate_passed") is not (record["passed_count"] == 7)
                or (code == 0) is not record["gate_passed"]
            ):
                return {
                    "state": "failed",
                    "reason": "executor_result_validation_failed",
                    "metrics": None,
                    "case_count": None,
                }
            return {
                "state": "completed",
                "reason": None,
                "metrics": {
                    "passed_count": record["passed_count"],
                    "gate_passed": record["gate_passed"],
                    "controlled_regression": True,
                },
                "case_count": 7,
            }
        if dataset_id not in ("planner-replay-v1", "tool-contract-v1"):
            return {
                "state": "failed",
                "reason": "unsupported_offline_adapter",
                "metrics": None,
                "case_count": None,
            }
        # Reuse the v1.2 certified concrete implementation; no new scorer.
        record = legacy_offline_execute(
            out, {"dataset_id": dataset_id, "job_id": request["pipeline_id"]}
        )
        if record.get("state") != "completed":
            return {
                "state": "failed",
                "reason": record.get("failure_code", "offline_run_failed"),
                "metrics": None,
                "case_count": None,
            }
        return {
            "state": "completed",
            "reason": None,
            "metrics": {
                "passed_count": record.get("passed_count"),
                "gate_passed": record.get("gate_passed"),
            },
            "case_count": record.get("case_count"),
        }
    if dataset_id not in (
        "report-benchmark",
        "corpus-retrieval",
        "selection-retrieval",
        "grounding-v1",
        "judge-calibration",
    ):
        return {
            "state": "blocked",
            "reason": "unlinked_or_unverified_adapter",
            "metrics": None,
            "case_count": None,
        }
    if not online_lane_enabled() or os.getenv("LLM_MODE") != "real":
        return {
            "state": "failed",
            "reason": "real_model_not_enabled",
            "metrics": None,
            "case_count": None,
        }
    result = out / "raw_result.json"
    case_kwargs = {"case_id": case_id} if case_id is not None else {}
    code = _run_fixed(dataset_id, result, request["max_cases"], **case_kwargs)
    if code != 0:
        # The historical `runner_timeout` attribution is preserved for a hard cap.
        # A cooperative stop ends *before* it, with a named partial/failed state
        # from `final-status.json` instead of an anonymous subprocess failure.
        reason = "runner_timeout" if code == 124 else "suite_execution_failed"
        if code != 124:
            status = long_run.read_root_status(out / "reliability")
            if status is not None and status.get("status") in (
                long_run.RESEARCH_FAILED,
                long_run.JUDGE_PARTIAL,
                long_run.JUDGE_FAILED,
                long_run.ARTIFACT_SAVE_FAILED,
            ):
                reason = status["status"]
        return {
            "state": "failed",
            "reason": reason,
            "metrics": None,
            "case_count": None,
        }
    if not result.is_file() or result.stat().st_size > 8 * 1024 * 1024:
        return {
            "state": "failed",
            "reason": "result_missing_or_oversized",
            "metrics": None,
            "case_count": None,
        }
    try:
        from research_engine.eval_trace import trace_span

        with trace_span("result.validate", kind="artifact_read") as span:
            summary = _summary(dataset_id, result, **case_kwargs)
            span.observe(output=summary)
        return {**summary, "reason": None}
    except (OSError, ValueError, TypeError, KeyError) as _exc:
        return {
            "state": "failed",
            "reason": "result_semantics_invalid",
            "metrics": None,
            "case_count": None,
        }


def _run_claimed(root: Path, lane: str, folder: Path, request: dict, index: int) -> bool:
    # Observation now creates the directory before dispatch. Preserve the old
    # no-reexecution guard before creating that observer-owned marker.
    existing = folder / "artifacts" / request["datasets"][index]
    if existing.exists() or existing.is_symlink() or existing.parent.is_symlink():
        finish_step(
            root,
            folder,
            index,
            state="failed",
            reason="execution_artifacts_already_exist",
            metrics=None,
            case_count=None,
            elapsed_seconds=0,
        )
        return True
    capture = None
    closure_status = "unavailable"
    try:
        from evals.agent_eval_ext.closure_observer import start

        capture = start(root, folder, request, index)
    except Exception as exc:
        structlog.get_logger().warning(
            "evaluation_closure_start_failed", error_type=type(exc).__name__
        )
    try:
        from app.services.evaluation_diagnostics import begin_observation

        begin_observation(folder, request, index, capture)
    except Exception as exc:
        structlog.get_logger().warning(
            "evaluation_observation_start_failed", error_type=type(exc).__name__
        )
    t = time.monotonic()
    try:
        from research_engine.eval_trace import trace_span

        with trace_span(
            "worker.dispatch",
            kind="evaluation_stage",
            inputs={
                "dataset": request["datasets"][index],
                "case_id": request.get("case_id"),
                "max_cases": request["max_cases"],
            },
        ) as span:
            result = _execute_step(folder, request, index, lane)
            span.observe(output=result)
    except Exception:  # noqa: BLE001 - diagnostic span keeps a sanitized private error
        result = {
            "state": "failed",
            "reason": "worker_internal_error",
            "metrics": None,
            "case_count": None,
        }
    if capture is not None:
        try:
            from evals.agent_eval_ext.closure_observer import finish

            closure_status = finish(capture, result)
        except Exception as exc:
            structlog.get_logger().warning(
                "evaluation_closure_finish_failed", error_type=type(exc).__name__
            )
    finish_step(
        root,
        folder,
        index,
        elapsed_seconds=time.monotonic() - t,
        closure_run_id=capture.run_id if capture else None,
        closure_status=closure_status,
        **result,
    )
    try:
        from app.services.evaluation_diagnostics import finish_observation

        finish_observation(folder, request["datasets"][index], result, time.monotonic() - t)
    except Exception as exc:
        structlog.get_logger().warning(
            "evaluation_observation_finish_failed", error_type=type(exc).__name__
        )
    return True


def run_once(root: Path, lane: str) -> bool:
    work = claim_next(root, lane)
    if work is None:
        return False
    folder, request, index = work
    base = folder / "artifacts" / request["datasets"][index]
    # Preserve the existing no-reexecution check before the journal creates its directory.
    if base.exists() or base.is_symlink() or base.parent.is_symlink():
        return _run_claimed(root, lane, folder, request, index)
    # The journal lives beside adapter outputs, leaving their exclusive-directory guard intact.
    from evals.agent_eval_ext.execution_observer import execution_scope

    with execution_scope(
        folder / "execution-observations" / request["datasets"][index],
        request["pipeline_id"],
        "worker.run",
    ) as span:
        result = _run_claimed(root, lane, folder, request, index)
        span.observe(output={"step_finished": result})
        return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lane", choices=("offline", "online"), required=True)
    args = parser.parse_args()
    if args.lane == "offline":
        if os.getenv("LLM_MODE") != "fake":
            raise RuntimeError("offline worker requires LLM_MODE=fake")
        if any(any(word in k for word in ("API_KEY", "SECRET", "TOKEN")) for k in os.environ):
            raise RuntimeError("offline lane cannot contain provider credentials")
    elif not online_lane_enabled() or os.getenv("LLM_MODE") != "real":
        raise RuntimeError("online worker not explicitly enabled")
    root = job_root()
    # IMPORTANT: only start one replica per lane. Deploying multiple replicas while
    # one is already running would falsely classify its in-flight job as interrupted.
    recover_interrupted(root, args.lane)
    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)
    while not STOP:
        worked = run_once(root, args.lane)
        if not worked:
            time.sleep(1)


if __name__ == "__main__":
    main()
