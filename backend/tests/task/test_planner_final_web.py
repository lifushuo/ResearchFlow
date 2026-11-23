"""Actual API -> shared Worker -> final CLI, with no formal holdout or network."""

import json
import os
import threading
import time
import uuid
from pathlib import Path

import pytest
from test_gepa_web_v3_e2e import start
from test_gepa_web_v3_e2e import web as web

from app.api.v1 import evaluation
from app.services import planner_final_jobs as final
from app.services.planner_evolution_jobs import claim_next
from evals.agent_eval_ext import planner_evolution_job_worker as worker
from evals.paid_call_budget import BudgetConfig, PaidCallBudget, UnknownCallBlocked

URL = "/api/v1/evaluation/evolution/jobs"
HEADERS = {"X-ResearchFlow-Eval-Action": "planner-final-ab-v1"}


def source(web):
    job, folder, req = start(web)
    worker.run_claimed(web[1], folder, req)
    assert web[0].get(f"{URL}/{job['job_id']}").json()["state"] == "pending_final"
    return job


def enqueue(web, parent, **options):
    response = web[0].post(
        f"{URL}/{parent['job_id']}/final",
        headers=HEADERS,
        json={"request_id": str(uuid.uuid4()), "mode": "demo", **options},
    )
    assert response.status_code == 202, response.text
    folder, req = claim_next(web[1], "demo")
    return response.json(), folder, req


def state(root, job):
    folder = root / "planner-evolution/experiments" / job["job_id"] / "paid-budget"
    return PaidCallBudget(
        folder, BudgetConfig(**json.loads((folder / "budget_config.json").read_text()))
    ).state()


def test_final_fake_complete_and_parent_history_unchanged(web):
    client, root = web
    parent = source(web)
    path = root / "planner-evolution/experiments" / parent["job_id"] / "experiment.json"
    before = path.read_bytes()
    preview = client.get(f"{URL}/{parent['job_id']}/final/preview").json()
    assert preview["fake_enabled"] and preview["request_cap"] == 90 and not preview["live_enabled"]
    job, folder, req = enqueue(web, parent)
    assert req["budget"]["quotas"] == {"planner": 72, "judge": 18, "reflection": 0}
    assert not (folder / "dataset.json").exists()  # API does not snapshot/read formal questions
    assert "planner_final_ab" in worker._command(folder, req)[2]
    worker.run_claimed(root, folder, req)
    status = client.get(f"{URL}/{job['job_id']}").json()
    assert status["state"] == "succeeded", status
    result = client.get(f"{URL}/{job['job_id']}/result").json()
    assert result["quality_gate_passed"] is True and result["release_ready"] is False
    assert (
        result["task_kind"] == "planner_final_ab" and result["comparison"]["quality_delta"] > 0.05
    )
    audit = client.get(f"{URL}/{job['job_id']}/audit").json()
    assert audit["result_integrity"] == "verified" and audit["journal"]["observation_count"] == 18
    assert audit["journal"]["reflection_count"] == 0 and audit["budget"]["used_total"] == 36
    assert audit["budget"]["unknown"] == [] and audit["holdout_sha256"] is None
    assert not (root / "planner-evolution/final-exposures").exists()
    raw_result = final.read_artifact(
        folder.parent.parent / "experiments" / job["job_id"] / "experiment.json"
    )
    for field in ("source_job_id", "dataset_sha256", "release_ready"):
        with pytest.raises(ValueError):
            final.verify_result(req, {**raw_result, field: "changed"})
    assert (
        path.read_bytes() == before
        and client.get(f"{URL}/{parent['job_id']}").json()["state"] == "pending_final"
    )
    if os.getenv("GEPA_FINAL_ACCEPTANCE_ARTIFACT"):
        Path(os.environ["GEPA_FINAL_ACCEPTANCE_ARTIFACT"]).write_text(
            json.dumps({"job": status, "audit": audit, "result": result}), "utf-8"
        )


@pytest.mark.parametrize(
    "scenario,expected",
    [
        ("no_improvement", "quality_rejected"),
        ("coverage_regression", "quality_rejected"),
        ("model_error", "failed"),
    ],
)
def test_final_rejection_and_model_error(web, monkeypatch, scenario, expected):
    parent = source(web)
    monkeypatch.setenv("RF_GEPA_FAKE_SCENARIO", scenario)
    job, folder, req = enqueue(web, parent)
    worker.run_claimed(web[1], folder, req)
    status = web[0].get(f"{URL}/{job['job_id']}").json()
    assert status["state"] == expected, status
    if scenario == "model_error":
        assert state(web[1], job)["unknown"] == ["planner-000001"]
        ledger = folder.parent.parent / "experiments" / job["job_id"] / "paid-budget"
        budget = PaidCallBudget(ledger, BudgetConfig(**req["budget"]))
        with pytest.raises(UnknownCallBlocked):
            budget.reserve("judge", "no retry")
    else:
        result = web[0].get(f"{URL}/{job['job_id']}/result").json()
        assert result["quality_gate_passed"] is False
        assert (
            "previously_covered_aspects_lost"
            if scenario == "coverage_regression"
            else "quality_gain_below_threshold"
        ) in result["holdout"]["reasons"]


def test_final_budget_exhaustion_and_frozen_tampering(web):
    parent = source(web)
    job, folder, req = enqueue(
        web, parent, request_quotas={"planner": 1, "judge": 0, "reflection": 0}
    )
    worker.run_claimed(web[1], folder, req)
    status = web[0].get(f"{URL}/{job['job_id']}").json()
    assert status["state"] == "failed" and status["error_type"] == "BudgetExhausted", status
    assert state(web[1], job)["used_total"] == 1
    job, folder, req = enqueue(web, parent)
    (folder / "source-experiment.json").write_text("{}")
    worker.run_claimed(web[1], folder, req)
    assert (
        web[0].get(f"{URL}/{job['job_id']}").json()["error_type"] == "final_frozen_inputs_changed"
    )
    assert not (folder.parent.parent / "experiments" / job["job_id"]).exists()


def test_final_cancel_running_child(web, monkeypatch):
    parent = source(web)
    monkeypatch.setenv("RF_GEPA_FAKE_SCENARIO", "slow")
    job, folder, req = enqueue(web, parent)
    thread = threading.Thread(target=worker.run_claimed, args=(web[1], folder, req))
    thread.start()
    slot = (
        folder.parent.parent
        / "experiments"
        / job["job_id"]
        / "paid-budget/slots/planner-000001.json"
    )
    deadline = time.monotonic() + 15
    while not slot.exists() and time.monotonic() < deadline:
        time.sleep(0.05)
    try:
        assert slot.exists()
    finally:
        web[0].post(
            f"{URL}/{job['job_id']}/cancel",
            headers={"X-ResearchFlow-Eval-Action": "planner-gepa-cancel-v1"},
        )
        thread.join(12)
    assert not thread.is_alive()
    assert web[0].get(f"{URL}/{job['job_id']}").json()["state"] == "cancelled"


def test_final_illegal_unauthorized_and_cross_owner(web):
    client, _ = web
    parent = source(web)
    url = f"{URL}/{parent['job_id']}/final"
    body = {"request_id": str(uuid.uuid4()), "mode": "demo"}
    assert client.post(url, json=body).status_code == 403
    assert client.post(url, headers=HEADERS, json={**body, "min_gain": 0.01}).status_code == 422
    assert (
        client.post(
            url, headers=HEADERS, json={**body, "mode": "live", "consent": final.CONSENT}
        ).status_code
        == 409
    )
    assert (
        client.post(
            url,
            headers=HEADERS,
            json={**body, "request_quotas": {"planner": 73, "judge": 18, "reflection": 0}},
        ).status_code
        == 409
    )
    client.app.dependency_overrides[evaluation._closure_owner] = lambda: "bob"
    assert client.get(url + "/preview").status_code == 404
    assert client.post(url, headers=HEADERS, json=body).status_code == 404


def test_final_idempotency_and_no_automatic_publish(web):
    parent = source(web)
    client, _ = web
    url = f"{URL}/{parent['job_id']}/final"
    body = {"request_id": str(uuid.uuid4()), "mode": "demo"}
    first = client.post(url, headers=HEADERS, json=body)
    again = client.post(url, headers=HEADERS, json=body)
    assert (
        first.status_code == again.status_code == 202
        and first.json()["job_id"] == again.json()["job_id"]
    )
    assert (
        client.post(
            url,
            headers=HEADERS,
            json={**body, "request_quotas": {"planner": 0, "judge": 0, "reflection": 0}},
        ).status_code
        == 409
    )
