"""Paid admission/one-exposure tests create metadata only, never run a model."""

import json
import uuid

import pytest
from test_gepa_web_v3_e2e import web as web
from test_planner_final_web import HEADERS, URL, source

from app.services import planner_final_jobs as final


@pytest.mark.parametrize("simulated,expected", [(True, False), (False, True)])
def test_loaded_formal_cases_are_audited_even_without_a_successful_score(
    tmp_path, simulated, expected
):
    from app.services.planner_evolution_insights import project_journal
    from evals.agent_eval_ext.planner_evolution.journal import ExperimentJournal

    ExperimentJournal(tmp_path / "journal")(
        "final_cases_loaded", {"simulated": simulated, "case_count": 9}
    )
    audit = project_journal(tmp_path)
    assert audit["error"] is None and audit["observation_count"] == 0
    assert audit["holdout_observed"] is expected


def test_paid_final_one_exposure_is_not_released_by_cancel(web, monkeypatch):
    client, root = web
    parent = source(web)
    folder = root / "planner-evolution/jobs" / parent["job_id"]
    result_path = root / "planner-evolution/experiments" / parent["job_id"] / "experiment.json"
    request = json.loads((folder / "request.json").read_text(encoding="utf-8"))
    result = json.loads(result_path.read_text(encoding="utf-8"))
    # Synthetic metadata fixture only: no actual real-mode model execution.
    request["mode"] = "live"
    result["evidence_kind"] = "live_planner"
    (folder / "request.json").write_text(json.dumps(request), encoding="utf-8")
    result_path.write_text(json.dumps(result), encoding="utf-8")
    monkeypatch.setenv("RF_GEPA_LIVE_ENABLED", "1")
    monkeypatch.setenv("RF_GEPA_ALLOWED_OWNERS", "alice")
    monkeypatch.setenv("RF_GEPA_FINAL_LIVE_ENABLED", "1")
    monkeypatch.setattr(final, "protocol_metadata", lambda: {"dataset_id": final.guard.HOLDOUT_ID})
    from evals.agent_eval_ext.planner_evolution import __main__ as cli

    monkeypatch.setattr(cli, "_live_routes", lambda *_: (None, result["routes"]))
    # If admission ever attempts to load/verify formal questions, fail immediately.
    monkeypatch.setattr(
        final.guard,
        "verify_fingerprints",
        lambda: (_ for _ in ()).throw(AssertionError("admission read formal holdout")),
    )
    url = f"{URL}/{parent['job_id']}/final"
    body = {"request_id": str(uuid.uuid4()), "mode": "live", "consent": final.CONSENT}
    assert client.post(url, headers=HEADERS, json={**body, "consent": None}).status_code == 409
    first = client.post(url, headers=HEADERS, json=body)
    assert first.status_code == 202, first.text
    assert client.post(url, headers=HEADERS, json=body).json()["job_id"] == first.json()["job_id"]
    claim = final.exposure_path(root)
    assert json.loads(claim.read_text(encoding="utf-8"))["job_id"] == first.json()["job_id"]
    cancelled = client.post(
        f"{URL}/{first.json()['job_id']}/cancel",
        headers={"X-ResearchFlow-Eval-Action": "planner-gepa-cancel-v1"},
    )
    assert cancelled.json()["state"] == "cancelled"
    repeat = client.post(url, headers=HEADERS, json={**body, "request_id": str(uuid.uuid4())})
    assert repeat.status_code == 409 and "formal_holdout_already_reserved" in repeat.text
    assert not (root / "planner-evolution/experiments" / first.json()["job_id"]).exists()
