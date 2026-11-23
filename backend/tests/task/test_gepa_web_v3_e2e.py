"""Actual API -> queue -> subprocess CLI -> GEPA -> ledger -> result; no paid models."""

import json
import os
import threading
import time
import uuid

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.v1 import evaluation
from app.services.planner_evolution_jobs import claim_next
from evals.agent_eval_ext import planner_evolution_job_worker as worker
from evals.paid_call_budget import BudgetConfig, PaidCallBudget

URL = "/api/v1/evaluation/evolution/jobs"
HEADERS = {"X-ResearchFlow-Eval-Action": "planner-gepa-optimize-v1"}


@pytest.fixture
def web(tmp_path, monkeypatch):
    monkeypatch.setenv("LLM_MODE", "fake")
    monkeypatch.delenv("RF_GEPA_FAKE_SCENARIO", raising=False)
    app = FastAPI()
    app.include_router(evaluation.router, prefix="/api/v1")
    app.dependency_overrides[evaluation._evaluation_root] = lambda: tmp_path
    app.dependency_overrides[evaluation._closure_owner] = lambda: "alice"
    return TestClient(app), tmp_path


def start(web, **options):
    client, root = web
    response = client.post(
        URL,
        headers=HEADERS,
        json={
            "request_id": str(uuid.uuid4()),
            "mode": "demo",
            "metric_version": "v3",
            "max_evals": 8,
            "proposals": 1,
            "seed": 42,
            "dataset_preset": "smoke",
            **options,
        },
    )
    assert response.status_code == 202, response.text
    job = response.json()
    folder, req = claim_next(root, "demo")
    return job, folder, req


def ledger(root, job):
    path = root / "planner-evolution/experiments" / job["job_id"] / "paid-budget"
    return PaidCallBudget(
        path, BudgetConfig(**json.loads((path / "budget_config.json").read_text("utf-8")))
    ).state()


def test_api_worker_cli_v3_candidate_budget_and_result(web):
    client, root = web
    job, folder, req = start(web)
    assert (folder / "budget.json").exists()
    assert job["request_cap"] == 41
    assert req["frozen"]["request_timeouts"] == {"planner": 90, "judge": 240, "reflection": 90}
    command = worker._command(folder, req)
    assert command[command.index("--judge-timeout") + 1] == "240"
    assert set(
        c["split"] for c in json.loads((folder / "dataset.json").read_text("utf-8"))["cases"]
    ) == {"train", "validation"}
    worker.run_claimed(root, folder, req)
    experiment = json.loads(
        (root / "planner-evolution/experiments" / job["job_id"] / "experiment.json").read_text(
            "utf-8"
        )
    )
    assert experiment["report_dataset_sha256"] is None
    assert experiment["request_timeouts"] == req["frozen"]["request_timeouts"]
    status = client.get(f"{URL}/{job['job_id']}").json()
    assert status["state"] == "pending_final", status
    result = client.get(f"{URL}/{job['job_id']}/result").json()
    assert result["comparison"]["quality_delta"] > 0.05
    assert result["comparison"]["same_prompt"] is False
    assert result["prompt_diff"] and result["quality_gate_passed"] is None
    audit = client.get(f"{URL}/{job['job_id']}/audit").json()
    assert audit["journal"]["reflection_count"] == 1
    assert audit["journal"]["holdout_observed"] is False
    assert audit["journal"]["model_routes"]["planner"] == "fake:planner"
    assert audit["result_integrity"] == "verified"
    assert audit["budget"]["used_total"] <= 41 and audit["budget"]["unknown"] == []
    assert audit["journal"]["observed_tokens"]["reflection_input"] is None
    rows = [e for e in audit["journal"]["events"] if e["kind"] == "observation"]
    assert rows and all(
        set(e["scores"]) == {"coverage", "specificity", "plan_quality", "redundancy", "quality"}
        for e in rows
    )
    assert ledger(root, job)["used_total"] == 17
    if os.environ.get("GEPA_ACCEPTANCE_ARTIFACT"):
        from pathlib import Path

        Path(os.environ["GEPA_ACCEPTANCE_ARTIFACT"]).write_text(
            json.dumps(
                {"job": status, "result": result, "audit": audit}, ensure_ascii=False, indent=2
            ),
            encoding="utf-8",
        )


@pytest.mark.parametrize(
    "scenario,expected",
    [
        ("no_improvement", "no_candidate"),
        ("coverage_regression", "no_candidate"),
        ("model_error", "failed"),
    ],
)
def test_no_candidate_coverage_regression_and_model_exception(web, monkeypatch, scenario, expected):
    client, root = web
    monkeypatch.setenv("RF_GEPA_FAKE_SCENARIO", scenario)
    job, folder, req = start(web)
    worker.run_claimed(root, folder, req)
    status = client.get(f"{URL}/{job['job_id']}").json()
    assert status["state"] == expected, status
    if scenario == "coverage_regression":
        result = client.get(f"{URL}/{job['job_id']}/result").json()
        assert "validation_coverage_regression" in result["rejection_reasons"]
    if scenario == "model_error":
        assert ledger(root, job)["unknown"]


def test_exhausted_budget_stops_before_an_unbudgeted_judge(web):
    client, root = web
    job, folder, req = start(web, request_quotas={"planner": 1, "judge": 0, "reflection": 1})
    worker.run_claimed(root, folder, req)
    status = client.get(f"{URL}/{job['job_id']}").json()
    assert status["state"] == "failed" and status["error_type"] == "BudgetExhausted", status
    state = ledger(root, job)
    assert state["used_by_role"] == {"planner": 1, "judge": 0, "reflection": 0}
    assert state["used_total"] == 1


def test_actual_running_subprocess_can_be_cancelled(web, monkeypatch):
    client, root = web
    monkeypatch.setenv("RF_GEPA_FAKE_SCENARIO", "slow")
    job, folder, req = start(web)
    thread = threading.Thread(target=worker.run_claimed, args=(root, folder, req))
    thread.start()
    slot = (
        root
        / "planner-evolution/experiments"
        / job["job_id"]
        / "paid-budget/slots/planner-000001.json"
    )
    deadline = time.monotonic() + 15
    while not slot.exists() and time.monotonic() < deadline:
        time.sleep(0.05)
    try:
        assert slot.exists()
        response = client.post(
            f"{URL}/{job['job_id']}/cancel",
            headers={"X-ResearchFlow-Eval-Action": "planner-gepa-cancel-v1"},
        )
        assert response.status_code == 200
    finally:
        if thread.is_alive():
            client.post(
                f"{URL}/{job['job_id']}/cancel",
                headers={"X-ResearchFlow-Eval-Action": "planner-gepa-cancel-v1"},
            )
        thread.join(12)
    assert not thread.is_alive()
    assert client.get(f"{URL}/{job['job_id']}").json()["state"] == "cancelled"


def test_illegal_requests_and_live_consent_fail_without_calls(web):
    client, _ = web
    body = {
        "request_id": str(uuid.uuid4()),
        "mode": "demo",
        "metric_version": "v3",
        "max_evals": 8,
        "proposals": 1,
        "dataset_preset": "smoke",
    }
    assert (
        client.post(URL, headers=HEADERS, json={**body, "metric_version": "v2"}).status_code == 422
    )
    assert client.post(URL, headers=HEADERS, json={**body, "min_gain": 0.01}).status_code == 422
    assert client.post(URL, headers=HEADERS, json={**body, "mode": "live"}).status_code == 409
    assert client.post(URL, json=body).status_code == 403


def test_mutated_frozen_input_is_rejected_before_any_subprocess(web):
    client, root = web
    job, folder, req = start(web)
    (folder / "baseline.json").write_text('{"planner":"tampered"}', encoding="utf-8")
    worker.run_claimed(root, folder, req)
    assert client.get(f"{URL}/{job['job_id']}").json()["error_type"] == "frozen_inputs_changed"
    assert not (root / "planner-evolution/experiments" / job["job_id"]).exists()
