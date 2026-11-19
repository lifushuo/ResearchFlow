"""No model or network calls; direct in-memory/temp-file v1.3 contract tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.services import evaluation_pipeline_v13 as p
from evals.agent_eval_ext import evaluation_pipeline_worker_v13 as worker

REQ = "e6cde1d0-d318-4b5c-b31d-382f1c60262c"
REQ2 = "b01644c6-d591-4233-a34d-091e0bbd56cf"


@pytest.fixture()
def root(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("RF_EVAL_ONLINE_ENABLED", raising=False)
    monkeypatch.delenv("RF_EVAL_JUDGE_ROUTE", raising=False)
    return tmp_path


def test_all_eight_in_registry(root):
    c = p.capabilities()
    assert len(c["datasets"]) == 8
    assert {x["dataset_id"] for x in c["datasets"]} == set(p.DATASETS)


def test_restored_executor_ready_but_online_judge_disabled(root):
    c = {x["dataset_id"]: x for x in p.capabilities()["datasets"]}
    assert c["judge-calibration"]["ready"] is False
    assert c["executor-regression-v1"]["ready"] is True


def test_online_explicitly_disabled(root):
    with pytest.raises(p.PipelineError, match="online_runner_disabled"):
        p.create_pipeline(root, "alice", ["report-benchmark"], REQ, 1, p.PAID_CONFIRM)


def test_online_consent_required(root, monkeypatch):
    monkeypatch.setenv("RF_EVAL_ONLINE_ENABLED", "1")
    with pytest.raises(p.PipelineError, match="explicit_paid_confirmation_required"):
        p.create_pipeline(root, "alice", ["corpus-retrieval"], REQ, 1, None)


def test_non_executable_input_rejected(root):
    for suites in ([], ["arbitrary"], ["tool-contract-v1", "tool-contract-v1"]):
        with pytest.raises(p.PipelineError):
            p.create_pipeline(root, "alice", suites, REQ)
    with pytest.raises(p.PipelineError, match="max_cases"):
        p.create_pipeline(root, "alice", ["planner-replay-v1"], REQ, 200)


def test_create_idempotent_and_conflict(root):
    first = p.create_pipeline(root, "alice", ["planner-replay-v1"], REQ)
    assert first["state"] == "queued"
    assert p.create_pipeline(root, "alice", ["planner-replay-v1"], REQ) == first
    with pytest.raises(p.PipelineError, match="idempotency_conflict"):
        p.create_pipeline(root, "alice", ["tool-contract-v1"], REQ)


def test_other_user_cannot_read(root):
    first = p.create_pipeline(root, "alice", ["tool-contract-v1"], REQ)
    with pytest.raises(KeyError):
        p.get_pipeline(root, "bob", first["pipeline_id"])
    assert p.list_pipelines(root, "bob") == []


def test_single_user_concurrent_limit(root):
    p.create_pipeline(root, "alice", ["planner-replay-v1"], REQ)
    with pytest.raises(p.PipelineBusy, match="one_active_pipeline"):
        p.create_pipeline(root, "alice", ["tool-contract-v1"], REQ2)


def test_lane_order_and_finish(root, monkeypatch):
    monkeypatch.setenv("RF_EVAL_ONLINE_ENABLED", "1")
    response = p.create_pipeline(
        root, "alice", ["tool-contract-v1", "corpus-retrieval"], REQ, 1, p.PAID_CONFIRM
    )
    assert p.claim_next(root, "online") is None
    claim = p.claim_next(root, "offline")
    assert claim is not None
    assert p.claim_next(root, "online") is None
    folder, request, index = claim
    p.finish_step(
        root,
        folder,
        index,
        state="completed",
        reason=None,
        metrics={"passed_count": 4},
        case_count=4,
        elapsed_seconds=1.0,
    )
    second = p.claim_next(root, "online")
    assert second and second[2] == 1
    p.finish_step(
        root,
        second[0],
        second[2],
        state="failed",
        reason="no_embedder",
        metrics=None,
        case_count=None,
        elapsed_seconds=1.0,
    )
    assert (
        p.get_pipeline(root, "alice", response["pipeline_id"])["state"] == "completed_with_issues"
    )


def test_completed_all_steps(root):
    response = p.create_pipeline(root, "alice", ["tool-contract-v1", "planner-replay-v1"], REQ)
    for _ in range(2):
        claim = p.claim_next(root, "offline")
        assert claim
        p.finish_step(
            root,
            claim[0],
            claim[2],
            state="completed",
            reason=None,
            metrics={"gate_passed": True},
            case_count=4,
            elapsed_seconds=0.2,
        )
    assert p.get_pipeline(root, "alice", response["pipeline_id"])["state"] == "completed"


def test_blocked_steps_not_pass(root):
    resp = p.create_pipeline(root, "alice", ["executor-regression-v1", "judge-calibration"], REQ)
    assert resp["state"] == "queued"
    assert [s["state"] for s in resp["steps"]] == ["pending", "blocked"]
    folder, _, index = p.claim_next(root, "offline")
    p.finish_step(
        root,
        folder,
        index,
        state="completed",
        reason=None,
        metrics={"gate_passed": True},
        case_count=7,
        elapsed_seconds=1.0,
    )
    assert p.get_pipeline(root, "alice", resp["pipeline_id"])["state"] == "completed_with_issues"


def test_interrupted_not_replayed(root):
    resp = p.create_pipeline(root, "alice", ["planner-replay-v1"], REQ)
    assert p.claim_next(root, "offline")
    assert p.recover_interrupted(root, "offline") == 1
    assert p.claim_next(root, "offline") is None
    state = p.get_pipeline(root, "alice", resp["pipeline_id"])
    assert state["steps"][0]["reason"] == "worker_restarted_result_uncertain"


def test_summary_grounding_result_shape(root, tmp_path):
    file = tmp_path / "ground.json"
    file.write_text(
        json.dumps(
            {
                "result": {"status": "measured", "claims": 3, "false_accept": 0.1},
                "claims": [{"id": "g001", "claim": "SECRET"}, {"id": "g002"}, {"id": "g003"}],
            }
        )
    )
    summary = worker._summary("grounding-v1", file)
    assert summary["case_count"] == 3
    assert "SECRET" not in json.dumps(summary)


def test_summary_report_filters_raw_text(root, tmp_path):
    file = tmp_path / "report.json"
    file.write_text(
        json.dumps(
            {
                "results": [{"completed": True, "report": "SECRET"}, {"completed": False}],
                "aggregate": {"count": 2, "completed": 1, "message": "SECRET"},
            }
        )
    )
    summary = worker._summary("report-benchmark", file)
    assert summary["case_count"] == 2
    assert summary["metrics"]["task_completion"] == 0.5
    assert "SECRET" not in json.dumps(summary)


def test_path_traversal_never_reads(root):
    with pytest.raises(p.PipelineError):
        p.get_pipeline(root, "alice", "../../etc/passwd")


def test_no_paid_key_in_public_metadata(root, monkeypatch):
    monkeypatch.setenv("RF_EVAL_ONLINE_ENABLED", "1")
    monkeypatch.setenv("GOOGLE_API_KEY", "dont-disclose")
    result = p.create_pipeline(root, "alice", ["corpus-retrieval"], REQ, 1, p.PAID_CONFIRM)
    assert "dont-disclose" not in json.dumps(result)


def test_workflow_has_no_shell_user_input(root):
    # Fixed adapter allowlist, client cannot influence process args or file names.
    assert (
        "evals.agent_eval_ext.evaluation_pipeline_suite_v13" in worker._run_fixed.__code__.co_consts
    )


def test_quality_gate_failure_is_not_all_pass(root):
    response = p.create_pipeline(root, "alice", ["tool-contract-v1"], REQ)
    folder, _, index = p.claim_next(root, "offline")
    p.finish_step(
        root,
        folder,
        index,
        state="completed",
        reason=None,
        metrics={"gate_passed": False},
        case_count=4,
        elapsed_seconds=0.5,
    )
    assert (
        p.get_pipeline(root, "alice", response["pipeline_id"])["state"] == "completed_with_issues"
    )


def test_missing_online_runner_mixed_selection_can_proceed_offline(root):
    response = p.create_pipeline(root, "alice", ["tool-contract-v1", "corpus-retrieval"], REQ)
    assert response["steps"][0]["state"] == "pending"
    assert response["steps"][1]["state"] == "blocked"
    assert response["paid_execution_authorized"] is False


def test_report_judge_tally_is_counted(root, tmp_path):
    file = tmp_path / "report.json"
    file.write_text(
        json.dumps(
            {
                "results": [{"completed": True}],
                "aggregate": {"queries": 1},
                "judge": {
                    "baseline": {
                        "claims_judged": 4,
                        "claims_unjudged": 1,
                        "served_models": ["dont-expose-route"],
                    }
                },
            }
        )
    )
    summary = worker._summary("report-benchmark", file)
    assert summary["metrics"]["judge"]["claims_judged"] == 4
    assert summary["metrics"]["judge"]["claims_unjudged"] == 1
    assert "dont-expose-route" not in json.dumps(summary)


# ── v1.3.2: online lane reuses the model/judge config already in .env ───────────


def test_judge_route_reuses_existing_env_config(root, monkeypatch):
    monkeypatch.setenv("EVAL_JUDGE_MODEL", "custom:test-independent-judge")
    assert p.judge_route() == "custom:test-independent-judge"


def test_rf_eval_judge_route_override_still_wins(root, monkeypatch):
    monkeypatch.setenv("EVAL_JUDGE_MODEL", "custom:from-env")
    monkeypatch.setenv("RF_EVAL_JUDGE_ROUTE", "custom:from-override")
    assert p.judge_route() == "custom:from-override"


def test_online_enabled_from_existing_real_mode_env(root, monkeypatch):
    monkeypatch.setenv("LLM_MODE", "real")
    monkeypatch.setenv("EVAL_JUDGE_MODEL", "custom:test-independent-judge")
    assert p.online_lane_enabled() is True
    caps = p.capabilities()
    assert caps["online_enabled"] is True
    assert caps["online_config_source"] == "env"
    assert caps["judge_config_source"] == "env"
    assert caps["judge_route_configured"] is True
    assert all(x["ready"] for x in caps["datasets"])


def test_online_disabled_in_fake_mode(root, monkeypatch):
    monkeypatch.setenv("LLM_MODE", "fake")
    monkeypatch.setenv("EVAL_JUDGE_MODEL", "custom:test-independent-judge")
    assert p.online_lane_enabled() is False
    assert p.capabilities()["online_config_source"] == "disabled"


def test_explicit_disable_overrides_env(root, monkeypatch):
    monkeypatch.setenv("LLM_MODE", "real")
    monkeypatch.setenv("EVAL_JUDGE_MODEL", "custom:test-independent-judge")
    monkeypatch.setenv("RF_EVAL_ONLINE_ENABLED", "0")
    assert p.online_lane_enabled() is False
    assert p.capabilities()["online_config_source"] == "rf_eval_override"
    with pytest.raises(p.PipelineError, match="online_runner_disabled"):
        p.create_pipeline(root, "alice", ["report-benchmark"], REQ, 1, p.PAID_CONFIRM)


def test_env_config_enables_real_pipeline_with_consent(root, monkeypatch):
    monkeypatch.setenv("LLM_MODE", "real")
    monkeypatch.setenv("EVAL_JUDGE_MODEL", "custom:test-independent-judge")
    rec = p.create_pipeline(root, "alice", ["report-benchmark"], REQ, 1, p.PAID_CONFIRM)
    assert rec["paid_execution_authorized"] is True
    assert rec["steps"][0]["state"] == "pending"


def test_capabilities_never_expose_judge_route_or_keys(root, monkeypatch):
    monkeypatch.setenv("LLM_MODE", "real")
    monkeypatch.setenv("EVAL_JUDGE_MODEL", "custom:secret-route")
    monkeypatch.setenv("EVAL_JUDGE_API_KEY", "DO_NOT_LEAK")
    payload = json.dumps(p.capabilities())
    assert "custom:secret-route" not in payload
    assert "DO_NOT_LEAK" not in payload
