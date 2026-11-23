"""GEPA P1 作业队列：离线验证隔离、去重、取消、恢复和费用门禁。"""

from __future__ import annotations

import uuid
from pathlib import Path

import pytest

from app.services.planner_evolution_jobs import (
    EvolutionJobBusy,
    EvolutionJobError,
    cancel_job,
    claim_next,
    create_job,
    finish_job,
    get_job,
    list_jobs,
    recover_interrupted,
)


def _start(root: Path, owner: str = "alice", **kwargs):
    return create_job(
        root,
        owner,
        request_id=kwargs.pop("request_id", str(uuid.uuid4())),
        mode=kwargs.pop("mode", "demo"),
        max_evals=kwargs.pop("max_evals", 64),
        proposals=kwargs.pop("proposals", 4),
        seed=kwargs.pop("seed", 42),
        consent=kwargs.pop("consent", None),
        **kwargs,
    )


def test_demo_queue_and_owner_isolation(tmp_path):
    a = _start(tmp_path)
    assert a["state"] == "queued"
    assert [x["job_id"] for x in list_jobs(tmp_path, "alice")] == [a["job_id"]]
    assert list_jobs(tmp_path, "bob") == []
    with pytest.raises(KeyError):
        get_job(tmp_path, "bob", a["job_id"])
    folder, request = claim_next(tmp_path, "demo")
    assert folder.name == a["job_id"] and request["mode"] == "demo"
    assert claim_next(tmp_path, "demo") is None
    finish_job(tmp_path, folder, state="quality_rejected", gate=False)
    assert get_job(tmp_path, "alice", a["job_id"])["quality_gate_passed"] is False


def test_metric_version_v3_accepted_and_demo_runs_v3(tmp_path):
    """v3 通过版本校验进入请求；demo 任务按契约强制 v1（脚本模拟无质量子分）。"""
    demo = _start(tmp_path, metric_version="v3")
    assert demo["metric_version"] == "v3"
    # 每用户同时只允许一个活跃任务，v4 校验用另一用户触发。
    with pytest.raises(EvolutionJobError, match="invalid_metric_version"):
        _start(tmp_path, owner="bob", metric_version="v4")


def test_no_paid_execution_without_admin_flag_and_explicit_consent(tmp_path, monkeypatch):
    monkeypatch.delenv("RF_GEPA_LIVE_ENABLED", raising=False)
    monkeypatch.delenv("RF_GEPA_ALLOWED_OWNERS", raising=False)
    with pytest.raises(EvolutionJobError, match="not_authorized"):
        _start(tmp_path, mode="live", consent="I_CONFIRM_GEPA_MODEL_API_COST")
    monkeypatch.setenv("RF_GEPA_LIVE_ENABLED", "1")
    monkeypatch.setenv("RF_GEPA_ALLOWED_OWNERS", "alice")
    with pytest.raises(EvolutionJobError, match="not_authorized"):
        _start(tmp_path, mode="live")
    with pytest.raises(EvolutionJobError, match="not_authorized"):
        _start(tmp_path, owner="bob", mode="live", consent="I_CONFIRM_GEPA_MODEL_API_COST")
    assert list_jobs(tmp_path, "alice") == []


def test_retry_idempotency_and_one_active_job(tmp_path):
    request_id = str(uuid.uuid4())
    first = _start(tmp_path, request_id=request_id)
    assert _start(tmp_path, request_id=request_id)["job_id"] == first["job_id"]
    with pytest.raises(EvolutionJobError, match="idempotency_conflict"):
        _start(tmp_path, request_id=request_id, proposals=3)
    with pytest.raises(EvolutionJobBusy, match="one_active"):
        _start(tmp_path)


def test_cancelled_queue_never_claimed_and_running_cancel_stops(tmp_path):
    queued = _start(tmp_path)
    assert cancel_job(tmp_path, "alice", queued["job_id"])["state"] == "cancelled"
    assert claim_next(tmp_path, "demo") is None
    running = _start(tmp_path)
    folder, _ = claim_next(tmp_path, "demo")
    assert cancel_job(tmp_path, "alice", running["job_id"])["state"] == "cancelling"
    finish_job(tmp_path, folder, state="succeeded", gate=True)
    assert get_job(tmp_path, "alice", running["job_id"])["state"] == "cancelled"


def test_restart_does_not_replay_uncertain_job(tmp_path):
    item = _start(tmp_path)
    claim_next(tmp_path, "demo")
    assert recover_interrupted(tmp_path, "demo") == 1
    assert (
        get_job(tmp_path, "alice", item["job_id"])["error_type"] == "worker_restarted_no_auto_retry"
    )
    assert claim_next(tmp_path, "demo") is None
