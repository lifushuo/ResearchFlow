"""Closure guards and the current pipeline's actual offline worker callback."""

import json
import sqlite3
from uuid import uuid4

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app.api.v1 import evaluation
from app.api.v1.evaluation_closure import GateBody
from app.services import evaluation_pipeline_v13 as pipeline
from app.services.evaluation_closure import ClosureError, EvaluationClosure, input_hash
from app.services.evaluation_closure.storage import store_at
from evals.agent_eval_ext import closure_observer as observer
from evals.agent_eval_ext import evaluation_pipeline_worker_v13 as worker


def register(store, rid, expected=None, **kwargs):
    return store.register_run(
        run_id=rid,
        owner_id="alice",
        dataset_id="planner-replay-v1",
        execution_mode="fake",
        sample_set_hash="f" * 64,
        metrics_version="fixture-v1",
        model_route="scripted:fixture",
        expected_case_ids=expected,
        **kwargs,
    )


def ingest(store, rid, status="passed", ident="a", latency=1, fixture=None):
    store.ingest_case(
        run_id=rid,
        owner_id="alice",
        case_id=ident,
        status=status,
        input_sha256=input_hash(fixture or {"a": 1}),
        latency_ms=latency,
        metrics={"score": 1},
    )


@pytest.fixture
def store(tmp_path):
    return EvaluationClosure(tmp_path / "closure.sqlite3")


def test_sealing_expected_set_blocks_partial_and_additional_results(store):
    register(store, "r1", ["a", "b"])
    ingest(store, "r1")
    assert store.quality_gate("alice", "r1")["verdict"] == "blocked"
    store.finish_run("alice", "r1")
    assert store.runs("alice")[0]["state"] == "incomplete"
    with pytest.raises(ClosureError, match="sealed"):
        ingest(store, "r1", ident="b")


def test_empty_gate_with_required_metric_blocks_instead_of_dividing_by_zero(store):
    register(store, "r1")
    assert (
        store.quality_gate("alice", "r1", required_metric_minimums={"score": 1})["verdict"]
        == "blocked"
    )


@pytest.mark.parametrize(
    "payload", [{"min_pass_rate": True}, {"required_metric_minimums": {"score": True}}]
)
def test_api_gate_does_not_coerce_booleans_to_numeric_thresholds(payload):
    with pytest.raises(ValidationError):
        GateBody.model_validate(payload)


@pytest.mark.parametrize(
    "options",
    [
        {"min_measured": True},
        {"min_pass_rate": float("nan")},
        {"min_pass_rate": True},
        {"max_p95_latency_ms": True},
        {"required_metric_minimums": {"score": True}},
    ],
)
def test_invalid_gate_configuration_is_not_a_measurement(store, options):
    register(store, "r1")
    with pytest.raises(ClosureError):
        store.quality_gate("alice", "r1", **options)


def test_missing_latency_keeps_p95_null_and_blocks_latency_gate(store):
    register(store, "r1")
    ingest(store, "r1", latency=100)
    ingest(store, "r1", ident="b", latency=None)
    store.finish_run("alice", "r1")
    result = store.quality_gate("alice", "r1", max_p95_latency_ms=200)
    assert result["p95_latency_ms"] is None and result["verdict"] == "blocked"
    assert store.audit_log("alice")[0]["action"] == "gate.check"


def test_empty_same_and_unfinished_runs_cannot_compare(store):
    register(store, "r1")
    register(store, "r2")
    assert store.compare("alice", "r1", "r2")["comparable"] is False
    ingest(store, "r1")
    ingest(store, "r2")
    assert store.compare("alice", "r1", "r2")["comparable"] is False
    store.finish_run("alice", "r1")
    assert store.compare("alice", "r1", "r1")["comparable"] is False


def test_refuses_to_initialize_over_an_unrelated_database(tmp_path):
    path = tmp_path / "business.sqlite3"
    with sqlite3.connect(path) as db:
        db.execute("CREATE TABLE business(value TEXT)")
    original = path.read_bytes()
    with pytest.raises(ClosureError, match="refusing"):
        EvaluationClosure(path)
    assert path.read_bytes() == original


@pytest.mark.parametrize("dataset", pipeline.ORDER)
def test_eight_original_source_snapshots_preserve_stable_ids(dataset):
    rows, inventory = observer.sources(dataset, 1)
    assert rows and inventory and all(len(value) == 64 for value in inventory.values())
    assert len(rows) == (
        1
        if dataset in {"report-benchmark", "judge-calibration", "grounding-v1"}
        else {
            "corpus-retrieval": 12,
            "selection-retrieval": 16,
            "planner-replay-v1": 4,
            "executor-regression-v1": 7,
            "tool-contract-v1": 4,
        }[dataset]
    )


def test_report_scoring_version_is_recorded_and_unknown_pricing_is_not_a_metric(
    tmp_path, monkeypatch
):
    from evals import metrics

    monkeypatch.setattr(metrics, "METRICS_VERSION", 123)
    folder = tmp_path / "pipeline"
    folder.mkdir()
    capture = observer.start(
        tmp_path,
        folder,
        {
            "datasets": ["report-benchmark"],
            "max_cases": 1,
            "pipeline_id": "fixture-version",
            "owner": "alice",
        },
        0,
    )
    assert capture.store.runs("alice")[0]["metrics_version"].endswith("report-parser-123")
    assert observer.numeric({"cost_usd": 0, "tokens": 10}) == {"tokens": 10}


@pytest.mark.parametrize(
    "dataset", ["planner-replay-v1", "tool-contract-v1", "executor-regression-v1"]
)
def test_actual_worker_creates_new_sealed_case_results(tmp_path, monkeypatch, dataset):
    monkeypatch.setenv("LLM_MODE", "fake")
    job = pipeline.create_pipeline(tmp_path, "alice", [dataset], str(uuid4()), 1, None)
    assert worker.run_once(tmp_path, "offline") is True
    result = pipeline.get_pipeline(tmp_path, "alice", job["pipeline_id"])
    step = result["steps"][0]
    assert step["state"] == "completed" and step["closure_status"] == "recorded"
    db = store_at(tmp_path)
    gate = db.quality_gate("alice", step["closure_run_id"], min_pass_rate=1)
    assert gate["verdict"] == "passed"
    assert (
        gate["measured"]
        == {"planner-replay-v1": 4, "tool-contract-v1": 4, "executor-regression-v1": 7}[dataset]
    )
    assert db.list_bad_cases("alice") == []
    assert db.runs("bob") == []


def test_observer_write_failure_does_not_change_original_worker_verdict(tmp_path, monkeypatch):
    monkeypatch.setattr(
        observer, "start", lambda *args: (_ for _ in ()).throw(OSError("private-secret"))
    )
    monkeypatch.setattr(
        worker,
        "_execute_step",
        lambda *args: {
            "state": "completed",
            "reason": None,
            "metrics": {"gate_passed": True},
            "case_count": 4,
        },
    )
    job = pipeline.create_pipeline(tmp_path, "alice", ["tool-contract-v1"], str(uuid4()), 1, None)
    worker.run_once(tmp_path, "offline")
    step = pipeline.get_pipeline(tmp_path, "alice", job["pipeline_id"])["steps"][0]
    assert step["state"] == "completed" and step["closure_status"] == "unavailable"
    assert "private-secret" not in json.dumps(step)


@pytest.mark.parametrize(
    "dataset",
    [
        "report-benchmark",
        "judge-calibration",
        "grounding-v1",
        "corpus-retrieval",
        "selection-retrieval",
    ],
)
def test_online_adapter_normalization_uses_original_case_oracles_without_model_calls(
    tmp_path, dataset
):
    folder = tmp_path / "pipeline"
    folder.mkdir()
    capture = observer.start(
        tmp_path,
        folder,
        {
            "datasets": [dataset],
            "max_cases": 1,
            "pipeline_id": "fixture-pipeline",
            "owner": "alice",
        },
        0,
    )
    destination = folder / "artifacts" / dataset
    destination.mkdir(parents=True)
    ids = list(capture.inputs)
    if dataset == "report-benchmark":
        doc = {"results": [{"id": ids[0], "completed": False, "latency_s": 0.5}]}
    elif dataset == "judge-calibration":
        doc = {
            "cases": [
                {
                    "case_id": ids[0],
                    "predicted_supported": not capture.inputs[ids[0]]["case"]["expected_supported"],
                }
            ]
        }
    elif dataset == "grounding-v1":
        reference = observer.read(observer.BACKEND / "evals/datasets/grounding/v1/labels.json")[
            "labels"
        ][ids[0]]
        doc = {
            "result": {"status": "measured"},
            "claims": [
                {
                    "id": ids[0],
                    "verdict": "contradicted" if reference != "contradicted" else "supported",
                }
            ],
        }
    else:
        doc = {"queries": [{"id": ident, "precision": {"@5": 0.5}, "leakage": 0} for ident in ids]}
    (destination / "raw_result.json").write_text(json.dumps(doc), encoding="utf-8")
    assert observer.finish(capture, {"state": "completed"}) == "recorded"
    gate = capture.store.quality_gate("alice", capture.run_id)
    if dataset in {"corpus-retrieval", "selection-retrieval"}:
        assert gate["verdict"] == "blocked" and gate["pass_rate"] is None
        assert capture.store.list_bad_cases("alice") == []
    else:
        assert gate["verdict"] == "failed"
        assert len(capture.store.list_bad_cases("alice")) == 1


def test_input_change_during_run_blocks_observation(tmp_path, monkeypatch):
    folder = tmp_path / "pipeline"
    folder.mkdir()
    capture = observer.start(
        tmp_path,
        folder,
        {
            "datasets": ["planner-replay-v1"],
            "max_cases": 1,
            "pipeline_id": "fixture-pipeline",
            "owner": "alice",
        },
        0,
    )
    monkeypatch.setattr(observer, "sources", lambda *args: (capture.inputs, {"changed": "f" * 64}))
    with pytest.raises(ValueError, match="changed_during"):
        observer.finish(capture, {"state": "completed"})
    assert capture.store.quality_gate("alice", capture.run_id)["verdict"] == "blocked"


def test_production_routes_use_signed_cookie_and_real_user_query(tmp_path, monkeypatch):
    import asyncio

    from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

    from app.dependencies import get_db
    from app.models.user import User
    from app.services.tokens import create_access_token

    engine = create_async_engine("sqlite+aiosqlite:///" + str(tmp_path / "auth.sqlite3"))
    sessions = async_sessionmaker(engine, expire_on_commit=False)
    alice, bob = uuid4(), uuid4()

    async def setup():
        async with engine.begin() as conn:
            await conn.run_sync(User.__table__.create)
        async with sessions() as session:
            session.add_all(
                [
                    User(id=alice, email="alice@fixture.invalid", hashed_pw="unused"),
                    User(id=bob, email="bob@fixture.invalid", hashed_pw="unused"),
                ]
            )
            await session.commit()

    asyncio.run(setup())

    async def db_session():
        async with sessions() as session:
            yield session

    app = FastAPI()
    app.include_router(evaluation.router, prefix="/api/v1")
    app.dependency_overrides[get_db] = db_session
    monkeypatch.setenv("RF_EVAL_JOB_ROOT", str(tmp_path))
    store = store_at(tmp_path)
    store.register_run(
        run_id="owner-run",
        owner_id=str(alice),
        dataset_id="fixture",
        execution_mode="fake",
        sample_set_hash="a" * 64,
        metrics_version="1",
        model_route="fake",
    )
    store.ingest_case(run_id="owner-run", owner_id=str(alice), case_id="a", status="failed")
    bid = store.list_bad_cases(str(alice))[0]["bad_case_id"]
    with TestClient(app) as client:
        base = "/api/v1/evaluation/closure"
        assert client.get(base + "/runs", headers={"X-User-Id": str(alice)}).status_code == 401
        client.cookies.set("access_token", create_access_token(bob))
        assert client.get(base + "/bad-cases").json() == []
        assert client.get(base + "/bad-cases/" + bid).status_code == 404
        client.cookies.set("access_token", create_access_token(alice))
        assert len(client.get(base + "/bad-cases").json()) == 1
        endpoint = base + "/runs/owner-run/gate"
        assert client.post(endpoint, json={}).status_code == 403
        header = {"X-ResearchFlow-Eval-Action": "run-pipeline-v1.3"}
        assert client.post(endpoint, headers=header, json={"owner_id": str(bob)}).status_code == 422
        assert client.post(endpoint, headers=header, json={}).status_code == 200
        assert (
            client.post(base + "/ingest", headers=header, json={"status": "passed"}).status_code
            == 404
        )
    asyncio.run(engine.dispose())
