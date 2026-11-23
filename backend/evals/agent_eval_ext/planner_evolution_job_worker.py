"""Dedicated GEPA queue worker: offline/no network and authorized online lanes.

Unlike the ordinary Celery research worker, this worker consumes only the bounded
GEPA queue. It does not re-run uncertain paid jobs after a crash.
"""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from app.services.evaluation_jobs import _read, job_root
from app.services.planner_evolution_jobs import (
    claim_next,
    finish_job,
    mark_phase,
    recover_interrupted,
)

BACKEND = Path(__file__).resolve().parents[2]
STOP = False
ACTIVE: subprocess.Popen | None = None


def _stop(*_args: object) -> None:
    global STOP
    STOP = True
    _terminate()


def _terminate() -> None:
    proc = ACTIVE
    if proc is None or proc.poll() is not None:
        return
    try:
        if hasattr(os, "killpg"):
            os.killpg(proc.pid, signal.SIGTERM)
        else:
            proc.terminate()
    except OSError, ProcessLookupError:
        pass


def _command(folder: Path, req: dict) -> list[str]:
    if req.get("task_kind") == "planner_final_ab":
        return [
            sys.executable,
            "-m",
            "evals.agent_eval_ext.planner_final_ab",
            "--job-dir",
            str(folder),
            "--out-dir",
            str(folder.parent.parent / "experiments" / req["job_id"]),
            "--strict-budget",
            "--budget-config",
            str(folder / "budget.json"),
            "--live" if req["mode"] == "live" else "--fake",
        ]
    # Only server-approved flags, paths and options; no browser-supplied shell fragments.
    out = folder.parent.parent / "experiments" / req["job_id"]
    args = [
        sys.executable,
        "-m",
        "evals.agent_eval_ext.planner_evolution",
        "optimize",
        "--out-dir",
        str(out),
        "--max-evals",
        str(req["max_evals"]),
        "--proposals",
        str(req["proposals"]),
        "--seed",
        str(req["seed"]),
        "--metric-version",
        req.get("metric_version", "v1"),
    ]
    if req.get("search_only"):
        timeouts = req["frozen"].get(
            "request_timeouts", {"planner": 90, "judge": 90, "reflection": 90}
        )
        args += ["--timeout", str(timeouts["planner"]), "--judge-timeout", str(timeouts["judge"])]
        args += [
            "--strict-budget",
            "--budget-config",
            str(folder / "budget.json"),
            "--frozen-config",
            str(folder / "frozen-config.json"),
            "--baseline",
            str(folder / "baseline.json"),
            "--no-holdout",
            "--min-gain",
            "0.05",
        ]
    if req["mode"] == "demo":
        if req.get("search_only"):
            args += ["--fake", "--dataset", str(folder / "dataset.json")]
        else:
            args.append("--demo")
    else:
        args += ["--live", "--dataset", str(folder / "dataset.json")]
        if req.get("frozen"):
            args += [
                "--judge",
                req["frozen"]["routes"]["judge"],
                "--reflector",
                req["frozen"]["routes"]["reflection"],
            ]
    return args


def run_claimed(root: Path, folder: Path, req: dict) -> None:
    global ACTIVE
    out = folder.parent.parent / "experiments" / req["job_id"]
    if out.exists() or out.is_symlink():
        finish_job(root, folder, state="failed", error_type="experiment_output_already_exists")
        return
    is_final = req.get("task_kind") == "planner_final_ab"
    if is_final:
        try:
            from app.services.planner_final_jobs import validate_inputs

            validate_inputs(root, folder, req)
        except OSError, ValueError, KeyError:
            finish_job(root, folder, state="failed", error_type="final_frozen_inputs_changed")
            return
    if req["mode"] == "live" and not req.get("search_only") and not is_final:
        finish_job(
            root, folder, state="failed", error_type="legacy_live_job_requires_budget_resubmission"
        )
        return
    if req.get("search_only"):
        try:
            from app.services.evaluation_closure import input_hash
            from evals.agent_eval_ext.long_run import sha256_text
            from evals.agent_eval_ext.planner_evolution.metrics_v3 import rules_hash

            if (
                any(
                    (folder / p).is_symlink()
                    for p in ("budget.json", "frozen-config.json", "baseline.json", "dataset.json")
                )
                or _read(folder / "budget.json") != req["budget"]
                or _read(folder / "frozen-config.json") != req["frozen"]
                or sha256_text(_read(folder / "baseline.json")["planner"])
                != req["frozen"]["baseline_sha256"]
                or input_hash(_read(folder / "dataset.json")) != req["dataset_sha256"]
                or req["frozen"]["metric_rules_sha256"] != rules_hash()
                or any(
                    c["split"] not in ("train", "validation")
                    for c in _read(folder / "dataset.json")["cases"]
                )
            ):
                raise ValueError("changed frozen inputs")
        except OSError, ValueError, KeyError:
            finish_job(root, folder, state="failed", error_type="frozen_inputs_changed")
            return
    if req["mode"] == "live" and not is_final:
        # Check the paid-job authorization and the immutable dataset snapshot again in
        # the executing process, before any model request can be made.
        from app.services.evaluation_closure import input_hash
        from app.services.planner_evolution_jobs import _paid_allowed

        dataset_path = folder / "dataset.json"
        if (
            not _paid_allowed(req["owner"])
            or dataset_path.is_symlink()
            or not dataset_path.is_file()
        ):
            finish_job(root, folder, state="failed", error_type="worker_paid_authorization_invalid")
            return
        import json

        if (
            input_hash(json.loads(dataset_path.read_text(encoding="utf-8")))
            != req["dataset_sha256"]
        ):
            finish_job(root, folder, state="failed", error_type="dataset_snapshot_changed")
            return
    env = dict(os.environ)
    if req.get("frozen"):
        for role, route in req["frozen"]["routes"]["pipeline"].items():
            env["MODEL_" + role.upper()] = route
        env["EVAL_JUDGE_MAX_OUTPUT_TOKENS"] = str(req["frozen"]["judge_settings"]["output_tokens"])
    # Offline lane must stay fake even if someone left real-mode env in its context.
    if req["mode"] == "demo":
        env["LLM_MODE"] = "fake"
    mark_phase(root, folder, "starting_optimizer")
    log_stream = None
    try:
        log_stream = (folder / "optimizer.log").open("x", encoding="utf-8")
        ACTIVE = subprocess.Popen(
            _command(folder, req),
            cwd=BACKEND,
            env=env,
            stdout=log_stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        stopped_at = None
        started_at = time.monotonic()
        while ACTIVE.poll() is None:
            status = _read(folder / "status.json")
            if STOP or status["state"] == "cancelling" or time.monotonic() - started_at > 5400:
                if stopped_at is None:
                    stopped_at = time.monotonic()
                    _terminate()
                elif time.monotonic() - stopped_at > 5:
                    if hasattr(os, "killpg"):
                        os.killpg(ACTIVE.pid, signal.SIGKILL)
                    else:
                        ACTIVE.kill()
            else:
                journal = out / "journal"
                events = sum(1 for _ in journal.glob("event-*.json")) if journal.is_dir() else 0
                if events:
                    mark_phase(root, folder, f"running_journal_events_{events}")
            time.sleep(0.4)
        status = _read(folder / "status.json")
        if STOP or status["state"] == "cancelling":
            finish_job(root, folder, state="cancelled")
            return
        result_path = out / "experiment.json"
        if not result_path.is_file() or result_path.is_symlink():
            from app.services.planner_evolution_insights import project_journal

            audit = project_journal(out)
            error = next(
                (
                    e.get("error_type")
                    for e in reversed(audit["events"])
                    if e["kind"] == "interrupted"
                ),
                None,
            )
            finish_job(root, folder, state="failed", error_type=error or "optimizer_no_result")
            return
        if result_path.stat().st_size > 8 * 1024 * 1024:
            finish_job(root, folder, state="failed", error_type="result_oversized")
            return
        import json

        result = json.loads(result_path.read_text(encoding="utf-8"))
        from app.services.planner_evolution_jobs import _verify_result_binding

        _verify_result_binding(req, result)
        gate = result["holdout"]["gate"]["passed"]
        if req.get("search_only") and ACTIVE.returncode == 0 and gate is None:
            if result.get("search", {}).get("validation_gain") is None:
                finish_job(root, folder, state="failed", error_type="validation_unmeasured")
            else:
                accepted = result.get("search", {}).get("accepted_candidate") is True
                finish_job(
                    root, folder, state="pending_final" if accepted else "no_candidate", gate=None
                )
            return
        if ACTIVE.returncode == 0 and gate is True:
            finish_job(root, folder, state="succeeded", gate=True)
        elif ACTIVE.returncode == 1 and gate is False:
            finish_job(root, folder, state="quality_rejected", gate=False)
        else:
            finish_job(root, folder, state="failed", error_type="optimizer_exit_or_gate_mismatch")
    except Exception:
        # Never surface raw provider exception or prompt content in the public job record.
        finish_job(root, folder, state="failed", error_type="worker_internal_error")
    finally:
        if log_stream is not None:
            log_stream.close()
        ACTIVE = None


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("demo", "live"), required=True)
    mode = parser.parse_args(argv).mode
    if mode == "demo":
        if os.getenv("LLM_MODE") != "fake":
            raise RuntimeError("demo worker requires LLM_MODE=fake")
        if any("API_KEY" in key or "SECRET" in key for key in os.environ):
            raise RuntimeError("demo worker must not receive model credentials")
    elif os.getenv("LLM_MODE") != "real" or os.getenv("RF_GEPA_LIVE_ENABLED") != "1":
        raise RuntimeError("live GEPA worker must be explicitly enabled")
    root = job_root()
    # Deploy one worker replica per lane. Uncertain in-flight paid work must NEVER be retried.
    recover_interrupted(root, mode)
    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)
    while not STOP:
        claim = claim_next(root, mode)
        if claim is None:
            time.sleep(1)
        else:
            run_claimed(root, *claim)


if __name__ == "__main__":
    main()
