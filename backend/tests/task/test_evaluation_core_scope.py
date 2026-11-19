"""The reduced surface retains authenticated evidence and gates, with no paid RCA path."""

from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.v1 import evaluation
from app.dependencies import get_current_user
from app.services.evaluation_closure.storage import store_at


def test_unified_surface_rejects_retired_routes(tmp_path, monkeypatch):
    monkeypatch.setenv("RF_EVAL_JOB_ROOT", str(tmp_path))
    app = FastAPI()
    app.include_router(evaluation.router, prefix="/api/v1")
    paths = set(app.openapi()["paths"])
    assert "/api/v1/evaluation/pipelines" in paths
    assert "/api/v1/evaluation/datasets" in paths
    assert not any(
        path.endswith(
            ("/model", "/model-evidence", "/triage", "/promote", "/review-decision", "/jobs")
        )
        for path in paths
    )
    with TestClient(app) as client:
        for path in ["/datasets", "/pipelines", "/closure/runs", "/closure/execution-incidents"]:
            assert client.get("/api/v1/evaluation" + path).status_code == 401
        app.dependency_overrides[get_current_user] = lambda: SimpleNamespace(id="alice")
        assert client.get("/api/v1/evaluation/datasets").status_code == 200
        assert client.get("/api/v1/evaluation/pipelines").json() == {"pipelines": []}
        assert (
            client.post(
                "/api/v1/evaluation/pipelines/any/diagnostics/report-benchmark/model", json={}
            ).status_code
            == 404
        )


def test_saved_partial_results_block_gate_and_remain_readable(tmp_path, monkeypatch):
    monkeypatch.setenv("RF_EVAL_JOB_ROOT", str(tmp_path))
    store = store_at(tmp_path)
    store.register_run(
        run_id="partial",
        owner_id="alice",
        dataset_id="report-benchmark",
        execution_mode="real",
        sample_set_hash="frozen",
        metrics_version="5",
        model_route="fixture",
        expected_case_ids=["a", "b"],
    )
    store.ingest_case(
        run_id="partial",
        owner_id="alice",
        case_id="a",
        status="passed",
        metrics={"full_judge_coverage": 1.0},
    )
    store.ingest_case(
        run_id="partial",
        owner_id="alice",
        case_id="b",
        status="unmeasured",
        error_code="judge_partial",
    )
    store.finish_run("alice", "partial", complete=False)
    assert store.quality_gate("alice", "partial")["verdict"] != "passed"
    assert store.runs("alice")[0]["run_id"] == "partial"
    assert store.runs("bob") == []
