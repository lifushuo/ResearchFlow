"""Fixed, credential-free offline adapters used by the unified pipeline worker."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
from pathlib import Path

from app.services.evaluation_jobs import RUNNABLE, _read

FIXTURES = Path(__file__).resolve().parent / "fixtures"


def _env() -> dict[str, str]:
    from research_engine.eval_trace import active_trace

    trace = active_trace()
    # Explicit allowlist, not parent's potentially key-bearing env.
    return {
        "PATH": os.environ.get("PATH", "/venv/bin:/usr/local/bin:/usr/bin:/bin"),
        "PYTHONPATH": str(Path(__file__).resolve().parents[2]),
        "LLM_MODE": "fake",
        "HOME": "/tmp",
        "PYTHONUNBUFFERED": "1",
        "RF_EVAL_CONTENT_TRACE": "1",
        **({"RF_EVAL_TRACE_RUN_ID": trace.run_id} if trace is not None else {}),
    }


def _run_command(command: list[str]) -> int:
    from evals.agent_eval_ext.execution_observer import subprocess_output

    with subprocess_output("worker.offline_process") as (output, span):
        child = subprocess.Popen(
            command,
            cwd=str(Path(__file__).resolve().parents[2]),
            env=_env(),
            stdout=output,
            stderr=subprocess.STDOUT,
            shell=False,
            start_new_session=True,
        )
        try:
            code = child.wait(timeout=360)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGKILL)
            child.wait()
            code = 124
        span.observe(exit_code=code, timed_out=code == 124)
        if code:
            span.status = "business_error"
        return code


def _execute(folder: Path, request: dict[str, object]) -> dict[str, object]:
    dataset_id = request["dataset_id"]
    if dataset_id not in RUNNABLE:
        return {"state": "failed", "failure_code": "unsupported_dataset"}
    artifacts = folder / "artifacts"
    artifacts.mkdir(exist_ok=False)
    if dataset_id == "planner-replay-v1":
        command = [
            sys.executable,
            "-m",
            "evals.agent_eval_ext.unified_cli",
            "run",
            "--dataset",
            str(FIXTURES / "unified_dataset_v1.json"),
            "--out-dir",
            str(artifacts / "result"),
            "--timeout",
            "60",
            "--revision",
            "browser-offline-v1.2",
        ]
        code = _run_command(command)
        suite_path = artifacts / "result" / "replay" / "suite.json"
        unified = artifacts / "result" / "unified.json"
        if not suite_path.is_file() or not unified.is_file():
            return {
                "state": "failed",
                "failure_code": "result_missing" if code != 124 else "runner_timeout",
            }
        suite = _read(suite_path)
        summary = _read(unified)
        if (
            summary.get("kind") != "researchflow_unified_evaluation"
            or suite.get("kind") != "researchflow_replay_suite"
            or suite.get("replay_type") != "controlled_simulation"
            or suite.get("historical_replay") is not False
            or type(suite.get("total")) is not int
            or suite["total"] != 4
            or type(suite.get("passed_count")) is not int
            or suite.get("all_passed") is not (suite["passed_count"] == 4)
            or code not in (0, 1)
        ):
            return {"state": "failed", "failure_code": "result_validation_failed"}
        case_ids = {
            x["case_id"]
            for x in suite["cases"]
            if isinstance(x, dict) and isinstance(x.get("case_id"), str)
        }
        if len(case_ids) != 4:
            return {"state": "failed", "failure_code": "case_identity_failed"}
        return {
            "state": "completed",
            "gate_passed": suite["all_passed"],
            "case_count": 4,
            "passed_count": suite["passed_count"],
        }
    command = [
        sys.executable,
        "-m",
        "evals.agent_eval_ext.tool_safety_cli",
        "score",
        "--gold",
        str(FIXTURES / "tool_gold_v1.json"),
        "--observed",
        str(FIXTURES / "tool_observed_pass_v1.json"),
        "--out",
        str(artifacts / "tool-result.json"),
    ]
    code = _run_command(command)
    file = artifacts / "tool-result.json"
    if not file.is_file():
        return {
            "state": "failed",
            "failure_code": "result_missing" if code != 124 else "runner_timeout",
        }
    data = _read(file)
    if (
        data.get("kind") != "researchflow_tool_safety_evaluation"
        or data.get("evidence_kind") != "controlled_fixture"
        or data.get("actual_agent_executed") is not False
        or data.get("cases_total") != 4
        or code not in (0, 1)
    ):
        return {"state": "failed", "failure_code": "result_validation_failed"}
    return {
        "state": "completed",
        "gate_passed": data.get("gate_passed") is True,
        "case_count": 4,
        "passed_count": sum(x.get("pass") is True for x in data.get("cases", [])),
    }
