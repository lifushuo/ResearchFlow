"""有数量边界的统一评测 Pipeline，使用固定的数据集适配器。

浏览器只发 ID，不发命令、路径、提示词、模型或密钥。离线与在线 Worker 分离；缺少执行前提时保留 BLOCKED，不能误报通过。"""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

from app.services.evaluation_jobs import _admission, _read, now, update_status

PIPE_RE = re.compile(r"^pipe-[a-f0-9]{24}$")
REQUEST_RE = re.compile(
    r"^[a-f0-9]{8}-[a-f0-9]{4}-[1-8][a-f0-9]{3}-[89ab][a-f0-9]{3}-[a-f0-9]{12}$"
)
DATASETS = {
    "report-benchmark": "online",
    "corpus-retrieval": "online",  # 实际向量化，可使用本地 Ollama。
    "selection-retrieval": "online",
    "grounding-v1": "online",
    "judge-calibration": "online",  # 必须具备可核对的来源关联。
    "planner-replay-v1": "offline",
    "executor-regression-v1": "offline",  # 必须具备可执行契约。
    "tool-contract-v1": "offline",
}
ORDER = tuple(DATASETS)
TERMINALS = frozenset({"completed", "completed_with_issues", "failed"})
STEP_TERMINALS = frozenset({"completed", "failed", "blocked"})
MAX_PIPELINES = 60
PAID_CONFIRM = "I_CONFIRM_REAL_MODEL_API_COST"
PAID_FULL_CONFIRM = "I_CONFIRM_FULL_DATASET_API_COST"


class PipelineError(ValueError):
    pass


class PipelineBusy(PipelineError):
    pass


def judge_route() -> str:
    """优先沿用部署 EVAL_JUDGE_MODEL 指定的独立 Judge，避免要求重复配置。

    RF_EVAL_JUDGE_ROUTE 仅作可选显式覆盖，与 harness 和 judge_config 使用的默认来源一致。"""
    return (os.getenv("RF_EVAL_JUDGE_ROUTE") or "").strip() or (
        os.getenv("EVAL_JUDGE_MODEL") or ""
    ).strip()


def online_lane_enabled() -> bool:
    """解析在线通道开关。

    非空 RF_EVAL_ONLINE_ENABLED 优先，0 强制关闭、1 强制开启；未设置时沿用部署的 real 模式和独立 Judge 配置。"""
    explicit = (os.getenv("RF_EVAL_ONLINE_ENABLED") or "").strip()
    if explicit:
        return explicit == "1"
    return (os.getenv("LLM_MODE") or "").strip().lower() == "real" and bool(judge_route())


def _online_enabled() -> bool:
    """保留旧版兼容别名，避免 _preflight 的参数名遮蔽新函数名。"""
    return online_lane_enabled()


def _config_sources() -> dict[str, str | bool]:
    """提供不含密钥的配置来源信息，便于界面解释在线评测为何启用或不可用。"""
    explicit = (os.getenv("RF_EVAL_ONLINE_ENABLED") or "").strip()
    judge_override = (os.getenv("RF_EVAL_JUDGE_ROUTE") or "").strip()
    judge_env = (os.getenv("EVAL_JUDGE_MODEL") or "").strip()
    return {
        "online_config_source": (
            "rf_eval_override" if explicit else ("env" if _online_enabled() else "disabled")
        ),
        "judge_config_source": (
            "rf_eval_override" if judge_override else ("env" if judge_env else "none")
        ),
        "judge_route_configured": bool(judge_route()),
    }


def _preflight(dataset_id: str, *, online_enabled: bool | None = None) -> dict[str, str | bool]:
    if dataset_id not in DATASETS:
        raise PipelineError("unknown_dataset")
    lane = DATASETS[dataset_id]
    enabled = _online_enabled() if online_enabled is None else online_enabled
    if lane == "online" and not enabled:
        return {"ready": False, "reason": "online_runner_disabled", "lane": lane}
    # 原始来源或执行契约缺失时，不得把该数据集宣传为可运行。
    backend = Path(__file__).resolve().parents[2]
    if dataset_id == "judge-calibration":
        try:
            from evals.agent_eval_ext.judge_calibration_runner_v131 import validate_source

            validate_source()
        except (ImportError, OSError, ValueError) as _exc:
            return {"ready": False, "reason": "judge_original_source_invalid", "lane": lane}
        if not judge_route():
            return {"ready": False, "reason": "judge_route_not_configured", "lane": lane}
    if dataset_id == "executor-regression-v1":
        manifest = backend / "evals/agent_eval_ext/fixtures/executor_suite_v131.json"
        test_file = backend / "tests/task/test_executor_eval_v131_cases.py"
        if manifest.is_symlink() or test_file.is_symlink() or not test_file.is_file():
            return {"ready": False, "reason": "executor_suite_missing", "lane": lane}
        try:
            from evals.agent_eval_ext.executor_regression_v131 import _cases

            if len(_cases()) != 7:
                raise ValueError("invalid count")
        except (ImportError, OSError, ValueError, KeyError, TypeError) as _exc:
            return {"ready": False, "reason": "executor_suite_invalid", "lane": lane}
    if dataset_id == "report-benchmark" and not judge_route():
        return {"ready": False, "reason": "judge_route_not_configured", "lane": lane}
    return {"ready": True, "reason": "ready", "lane": lane}


def capabilities() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "online_enabled": _online_enabled(),
        "paid_confirm_token_required": True,
        "datasets": [{"dataset_id": item, **_preflight(item)} for item in ORDER],
        "not_a_single_score": True,
        **_config_sources(),
    }


def _folder(root: Path, pipe_id: str) -> Path:
    if not PIPE_RE.fullmatch(pipe_id):
        raise PipelineError("invalid_pipeline_id")
    base = root / "pipelines"
    path = base / pipe_id
    if base.is_symlink() or path.is_symlink():
        raise PipelineError("unsafe_pipeline_path")
    return path


def _rows(root: Path) -> list[tuple[Path, dict[str, Any], dict[str, Any]]]:
    base = root / "pipelines"
    if not base.is_dir() or base.is_symlink():
        return []
    result = []
    for path in base.iterdir():
        if path.is_symlink() or not path.is_dir() or not PIPE_RE.fullmatch(path.name):
            continue
        try:
            request, status = _read(path / "request.json"), _read(path / "status.json")
            if request.get("pipeline_id") != path.name or status.get("pipeline_id") != path.name:
                continue
            if request.get("datasets") and all(x in DATASETS for x in request["datasets"]):
                result.append((path, request, status))
        except (ValueError, OSError, KeyError, TypeError) as _exc:
            continue
    return sorted(result, key=lambda x: x[1].get("created_at", ""), reverse=True)


def _public(request: dict[str, Any], status: dict[str, Any]) -> dict[str, Any]:
    return {
        **({"case_id": request["case_id"]} if request.get("case_id") is not None else {}),
        "pipeline_id": request["pipeline_id"],
        "datasets": request["datasets"],
        "max_cases": request["max_cases"],
        "state": status["state"],
        "steps": status["steps"],
        "created_at": request["created_at"],
        "updated_at": status.get("updated_at"),
        "paid_execution_authorized": request["paid_execution_authorized"],
        "not_a_single_score": True,
    }


def _read_owned(root: Path, user_id: str, pipe_id: str):
    path = _folder(root, pipe_id)
    if not path.is_dir():
        raise KeyError("pipeline_not_found")
    request = _read(path / "request.json")
    if request.get("owner") != user_id or request.get("pipeline_id") != pipe_id:
        raise KeyError("pipeline_not_found")
    return path, request, _read(path / "status.json")


def _valid_input(datasets: list[str], request_id: str, max_cases: int, consent: str | None):
    if not isinstance(datasets, list) or not 1 <= len(datasets) <= len(DATASETS):
        raise PipelineError("invalid_dataset_selection")
    if len(set(datasets)) != len(datasets) or any(x not in DATASETS for x in datasets):
        raise PipelineError("invalid_dataset_selection")
    if not isinstance(request_id, str) or not REQUEST_RE.fullmatch(request_id):
        raise PipelineError("invalid_request_id")
    if type(max_cases) is not int or not (max_cases == 0 or 1 <= max_cases <= 20):
        raise PipelineError("max_cases_out_of_range_1_to_20")
    online_requested = any(DATASETS[x] == "online" for x in datasets)
    ready_online = any(DATASETS[x] == "online" and _preflight(x)["ready"] for x in datasets)
    if (
        online_requested
        and not _online_enabled()
        and not any(DATASETS[x] == "offline" for x in datasets)
    ):
        raise PipelineError("online_runner_disabled")
    # 阻塞步骤没有执行收费调用；混合 Pipeline 仍可执行离线项，在线项明确保持 BLOCKED。
    required_consent = PAID_FULL_CONFIRM if max_cases == 0 else PAID_CONFIRM
    if ready_online and consent != required_consent:
        raise PipelineError(
            "explicit_full_dataset_confirmation_required"
            if max_cases == 0
            else "explicit_paid_confirmation_required"
        )
    if not online_requested and consent not in (None, ""):
        raise PipelineError("unexpected_paid_confirmation")
    if online_requested and consent not in (None, "", PAID_CONFIRM, PAID_FULL_CONFIRM):
        raise PipelineError("invalid_paid_confirmation")
    return ready_online


def create_pipeline(
    root: Path,
    user_id: str,
    datasets: list[str],
    request_id: str,
    max_cases: int = 1,
    consent: str | None = None,
    *,
    case_id: str | None = None,
) -> dict[str, Any]:
    if case_id is not None:
        from evals.query_selection import validate_report_selection

        try:
            validate_report_selection(datasets, max_cases, case_id)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise PipelineError(str(exc)) from exc
    online = _valid_input(datasets, request_id, max_cases, consent)
    pipe_id = "pipe-" + hashlib.sha256((user_id + "/" + request_id).encode()).hexdigest()[:24]
    base = root / "pipelines"
    if base.is_symlink():
        raise PipelineError("unsafe_pipeline_root")
    base.mkdir(mode=0o700, exist_ok=True)
    path = _folder(root, pipe_id)
    with _admission(root):
        if path.exists():
            p, previous, status = _read_owned(root, user_id, pipe_id)
            if (
                previous["datasets"] != datasets
                or previous["max_cases"] != max_cases
                or previous["paid_execution_authorized"] != online
                or previous.get("case_id") != case_id
            ):
                raise PipelineError("idempotency_conflict")
            return _public(previous, status)
        rows = _rows(root)
        if len(rows) >= MAX_PIPELINES:
            raise PipelineBusy("pipeline_history_full")
        if any(r.get("owner") == user_id and s["state"] not in TERMINALS for _, r, s in rows):
            raise PipelineBusy("one_active_pipeline_per_user")
        if sum(s["state"] not in TERMINALS for _, _, s in rows) >= 2:
            raise PipelineBusy("pipeline_queue_full")
        t = now()
        request = {
            **({"case_id": case_id} if case_id is not None else {}),
            "pipeline_id": pipe_id,
            "request_id": request_id,
            "owner": user_id,
            "datasets": datasets,
            "max_cases": max_cases,
            "paid_execution_authorized": online,
            "created_at": t,
            "schema_version": 1,
        }
        steps = []
        for dataset_id in datasets:
            pre = _preflight(dataset_id)
            steps.append(
                {
                    "dataset_id": dataset_id,
                    "lane": DATASETS[dataset_id],
                    "state": "pending" if pre["ready"] else "blocked",
                    "reason": None if pre["ready"] else pre["reason"],
                    "metrics": None,
                    "case_count": None,
                    "elapsed_seconds": None,
                }
            )
        initial_state = (
            "completed_with_issues" if all(x["state"] == "blocked" for x in steps) else "queued"
        )
        status = {"pipeline_id": pipe_id, "state": initial_state, "steps": steps, "updated_at": t}
        path.mkdir(mode=0o700)
        try:
            with (path / "request.json").open("x", encoding="utf-8") as out:
                json.dump(request, out, ensure_ascii=False)
            update_status(path / "status.json", status)
        except BaseException:
            for child in path.iterdir():
                if child.is_file():
                    child.unlink()
            path.rmdir()
            raise
        return _public(request, status)


def list_pipelines(root: Path, user_id: str) -> list[dict[str, Any]]:
    return [_public(r, s) for _, r, s in _rows(root) if r.get("owner") == user_id][:MAX_PIPELINES]


def get_pipeline(root: Path, user_id: str, pipe_id: str) -> dict[str, Any]:
    _, request, status = _read_owned(root, user_id, pipe_id)
    return _public(request, status)


def _advance(status: dict[str, Any]) -> None:
    if any(step["state"] == "running" for step in status["steps"]):
        status["state"] = "running"
    elif any(step["state"] == "pending" for step in status["steps"]):
        status["state"] = "queued"
    elif all(step["state"] == "completed" for step in status["steps"]):
        # 执行结束不等于质量门禁通过，合成契约与真实质量评测都要保留此区别。
        quality_failed = any(
            isinstance(s.get("metrics"), dict)
            and (
                s["metrics"].get("gate_passed") is False
                or s["metrics"].get("meets_threshold") is False
            )
            for s in status["steps"]
        )
        status["state"] = "completed_with_issues" if quality_failed else "completed"
    else:
        status["state"] = "completed_with_issues"
    status["updated_at"] = now()


def claim_next(root: Path, lane: str):
    """跨 Worker 原子领取任务，不能越过前置步骤乱序领取。"""
    if lane not in ("offline", "online"):
        raise PipelineError("bad_lane")
    with _admission(root):
        for path, request, status in reversed(_rows(root)):
            if status["state"] in TERMINALS:
                continue
            if any(s["state"] == "running" for s in status["steps"]):
                continue
            next_step = next(
                (i for i, s in enumerate(status["steps"]) if s["state"] == "pending"), None
            )
            if next_step is None or status["steps"][next_step]["lane"] != lane:
                continue
            status["steps"][next_step]["state"] = "running"
            _advance(status)
            update_status(path / "status.json", status)
            return path, request, next_step
    return None


def finish_step(
    root: Path,
    path: Path,
    index: int,
    *,
    state: str,
    reason: str | None,
    metrics: dict[str, Any] | None,
    case_count: int | None,
    elapsed_seconds: float,
    closure_run_id: str | None = None,
    closure_status: str | None = None,
) -> None:
    if state not in STEP_TERMINALS:
        raise PipelineError("bad_step_state")
    with _admission(root):
        status = _read(path / "status.json")
        if status["steps"][index]["state"] != "running":
            raise PipelineError("step_not_owned")
        status["steps"][index].update(
            {
                "state": state,
                "reason": reason,
                "metrics": metrics,
                "case_count": case_count,
                "elapsed_seconds": round(elapsed_seconds, 3),
            }
        )
        if closure_status is not None:
            status["steps"][index].update(
                closure_run_id=closure_run_id, closure_status=closure_status
            )
        _advance(status)
        update_status(path / "status.json", status)


def recover_interrupted(root: Path, lane: str) -> int:
    """Worker 重启后将中断步骤标为失败，绝不静默重跑付费调用。"""
    changed = 0
    with _admission(root):
        for path, _req, status in _rows(root):
            for step in status["steps"]:
                if step["state"] == "running" and step["lane"] == lane:
                    step.update(state="failed", reason="worker_restarted_result_uncertain")
                    changed += 1
            if changed:
                _advance(status)
                update_status(path / "status.json", status)
    return changed
