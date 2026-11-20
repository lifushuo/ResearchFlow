"""评测中心坏例回流、活跃数据集组装与实验证据投影的回归；不调用真实模型。"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.services.evaluation_closure import EvaluationClosure, NotFound, input_hash
from app.services.planner_evolution_center import (
    BackflowError,
    backflow,
    benchmark_bad_cases,
    benchmark_cases,
    list_experiments,
    load_benchmark_annotations,
    load_seed_dataset,
    merged_dataset,
    overview,
)
from evals.agent_eval_ext.closure_observer import sources

BENCH_CASE = "llm-memory"


def make_store(tmp_path: Path) -> EvaluationClosure:
    return EvaluationClosure(tmp_path / "closure.sqlite3")


def register_failed_case(
    store: EvaluationClosure,
    owner: str = "alice",
    case_id: str = BENCH_CASE,
    *,
    tamper_hash: bool = False,
) -> str:
    source_cases, _ = sources("report-benchmark", 0)
    run_id = f"pipe-{case_id}--report-benchmark"
    store.register_run(
        run_id=run_id,
        owner_id=owner,
        dataset_id="report-benchmark",
        execution_mode="real",
        sample_set_hash="v1",
        metrics_version="v1",
        model_route="configured-sha256:0" * 4,
        judge_route="custom:judge",
    )
    sha = "0" * 64 if tamper_hash else input_hash(source_cases[case_id])
    store.ingest_case(
        run_id=run_id,
        owner_id=owner,
        case_id=case_id,
        status="failed",
        input_sha256=sha,
        error_code="research_incomplete",
    )
    rows = store.list_bad_cases(owner)
    return next(row["bad_case_id"] for row in rows if row["case_id"] == case_id)


def flow(
    store,
    tmp_path: Path,
    *,
    case_id="planner-gap-01",
    stage="planner",
    group="agent-memory",
    **kwargs,
):
    return backflow(
        store,
        "alice",
        tmp_path,
        case_id=case_id,
        group=group,
        aspects=["记忆写入路径", "检索延迟量级"],
        review_note="Planner 任务列表缺少实验条件核对，轨迹确认。",
        stage=stage,
        **kwargs,
    )


def test_backflow_from_bad_case_appends_train_case(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    bad_id = register_failed_case(store)
    record = flow(store, tmp_path, bad_case_id=bad_id)
    assert record["case"]["split"] == "train"
    assert record["case"]["source_bad_case_id"] == bad_id
    assert record["case"]["query"]
    assert record["review"]["source"]["run_id"].endswith("--report-benchmark")
    assert record["review"]["benchmark_aspects"], "标注预填必须来自基准维度"
    merged = merged_dataset(tmp_path)
    assert merged["seed_case_count"] == 30
    assert merged["backflow_case_count"] == 1
    assert merged["splits"]["train"] == 13
    assert merged["error"] is None


def test_backflow_rejects_tampered_or_foreign_bad_cases(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    bad_id = register_failed_case(store, tamper_hash=True)
    with pytest.raises(BackflowError, match="基准题源不一致"):
        flow(store, tmp_path, bad_case_id=bad_id)
    other = make_store(tmp_path / "other")
    foreign = register_failed_case(other, owner="bob")
    with pytest.raises(NotFound):
        flow(store, tmp_path, bad_case_id=foreign)


def test_backflow_rejects_group_split_conflicts_and_dupes(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    bad_id = register_failed_case(store)
    with pytest.raises(BackflowError, match="跨 split"):
        flow(store, tmp_path, bad_case_id=bad_id, group="databases")
    flow(store, tmp_path, bad_case_id=bad_id)
    with pytest.raises(BackflowError, match="案例 ID 已有回流记录"):
        flow(store, tmp_path, bad_case_id=bad_id)
    with pytest.raises(BackflowError, match="该坏例已回流"):
        flow(store, tmp_path, case_id="planner-gap-02", bad_case_id=bad_id)
    second = register_failed_case(store, case_id="postgres-vs-mysql")
    # 先归因检索环节（只进 reviews/），再改归因 planner：同一坏例不允许二次回流。
    flow(store, tmp_path, case_id="review-gap-02", bad_case_id=second, stage="retrieval")
    with pytest.raises(BackflowError, match="该坏例已回流"):
        flow(store, tmp_path, case_id="planner-gap-02", bad_case_id=second)
    seed_query = load_seed_dataset().cases[0].query
    with pytest.raises(BackflowError, match="相同问题已存在于种子数据集"):
        flow(store, tmp_path, case_id="manual-dupe-01", manual_query=seed_query)


def test_backflow_requires_exactly_one_source(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    with pytest.raises(BackflowError, match="只能选择"):
        flow(store, tmp_path)
    bad_id = register_failed_case(store)
    with pytest.raises(BackflowError, match="只能选择"):
        flow(store, tmp_path, bad_case_id=bad_id, manual_query="这是一个足够长的研究问题。")


def test_non_planner_stage_is_evidence_only(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    bad_id = register_failed_case(store)
    flow(store, tmp_path, bad_case_id=bad_id, stage="retrieval")
    merged = merged_dataset(tmp_path)
    assert merged["backflow_case_count"] == 0
    assert merged["review_record_count"] == 1
    assert (tmp_path / "planner-evolution" / "reviews").is_dir()


def test_manual_backflow_records_audit_event(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    record = flow(store, tmp_path, manual_query="比较两种共识协议在广域网下的延迟与一致性代价。")
    assert record["case"]["source_bad_case_id"] is None
    assert record["review"]["benchmark_aspects"] is None
    actions = [row["action"] for row in store.audit_log("alice")]
    assert actions.count("planner_backflow") == 1


def test_bad_case_listing_prefills_annotations(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    register_failed_case(store)
    result = benchmark_bad_cases(store, "alice")
    row = result["bad_cases"][0]
    assert row["case_id"] == BENCH_CASE
    assert row["suggested_aspects"], "回流预填必须带基准维度标注"
    assert result["annotated_cases"] == 10


def test_benchmark_cases_expose_real_queries_and_annotations() -> None:
    result = benchmark_cases()
    assert result["dataset_id"] == "report-benchmark"
    assert result["case_count"] == result["annotated_case_count"] == 10
    source_cases, _ = sources("report-benchmark", 0)
    rows = {row["case_id"]: row for row in result["cases"]}
    assert set(rows) == set(source_cases)
    for case_id, row in rows.items():
        assert row["query"] == source_cases[case_id]["case"]["query"]
        assert len(row["required_aspects"]) >= 2
        assert row["rationale"]


def test_annotations_cover_every_benchmark_case() -> None:
    annotations = load_benchmark_annotations()
    benchmark_cases, _ = sources("report-benchmark", 0)
    assert set(annotations["aspects"]) == set(benchmark_cases)


def test_experiment_projection_excludes_observation_payloads(tmp_path: Path) -> None:
    base = tmp_path / "planner-evolution" / "experiments" / "planner-01"
    base.mkdir(parents=True)
    doc = {
        "schema_version": 1,
        "created_at": "2026-11-20T00:00:00+00:00",
        "evidence_kind": "scripted_demo",
        "metric_version": "planner-task-coverage/1",
        "optimizer": {"name": "GEPA", "version": "0.1.4"},
        "limits": {"evaluations": 64},
        "routes": {"judge": "custom:judge"},
        "dataset_sha256": "ab" * 32,
        "dataset": {"cases": [1] * 12},
        "search": {"validation_gain": 0.1, "validation_baseline": ["large-payload"]},
        "holdout": {"gate": {"passed": True, "delta": 0.1, "reasons": []}},
        "candidate": {"sha256": "cd" * 32, "override": "secret-prompt"},
        "prompt_diff": "--- baseline\n+++ candidate",
    }
    (base / "experiment.json").write_text(json.dumps(doc), encoding="utf-8")
    broken = tmp_path / "planner-evolution" / "experiments" / "planner-02"
    broken.mkdir(parents=True)
    (broken / "experiment.json").write_text("{broken", encoding="utf-8")
    rows = list_experiments(tmp_path)
    assert rows[0]["name"] == "planner-01"
    projected = rows[0]["experiment"]
    assert projected["holdout_gate"]["passed"] is True
    assert projected["candidate"] == "cd" * 32
    assert "validation_baseline" not in projected
    assert "override" not in json.dumps(projected)
    assert "cases" not in projected
    assert rows[1]["experiment"] is None


def test_overview_reports_loop_stage_counts(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    bad_id = register_failed_case(store)
    flow(store, tmp_path, bad_case_id=bad_id)
    base = tmp_path / "planner-evolution" / "experiments" / "planner-01"
    base.mkdir(parents=True)
    (base / "experiment.json").write_text(
        json.dumps({"holdout": {"gate": {"passed": True}}}), encoding="utf-8"
    )
    state = overview(store, "alice", tmp_path)
    assert state["dataset"]["backflow_case_count"] == 1
    assert state["bad_cases"]["open_benchmark_count"] == 1
    assert state["bad_cases"]["flowed_count"] == 1
    assert state["experiments"]["last_holdout_gate_passed"] is True


def test_overview_ignores_other_owners_bad_cases(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    register_failed_case(store, owner="bob")
    state = overview(store, "alice", tmp_path)
    assert state["bad_cases"]["open_benchmark_count"] == 0


def test_cli_dataset_assembles_seed_and_backflow(tmp_path: Path) -> None:
    from evals.agent_eval_ext.planner_evolution import __main__ as cli
    from evals.agent_eval_ext.planner_evolution.experiment import Dataset

    store = make_store(tmp_path)
    bad_id = register_failed_case(store)
    flow(store, tmp_path, bad_case_id=bad_id)
    cases_dir = tmp_path / "planner-evolution" / "cases"
    (cases_dir / "broken.json").write_text("{oops", encoding="utf-8")
    out = tmp_path / "planner-active.json"
    assert cli.main(["dataset", "--cases-dir", str(cases_dir), "--out", str(out)]) == 0
    dataset = Dataset.model_validate(json.loads(out.read_text(encoding="utf-8")))
    assert len(dataset.cases) == 31
    assert dataset.provenance == "reviewed"


def build_api_app(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, owner: str = "alice"):
    from fastapi import FastAPI

    from app.api.v1.evaluation import router as evaluation_router
    from app.dependencies import get_current_user

    monkeypatch.setenv("RF_EVAL_JOB_ROOT", str(tmp_path))
    app = FastAPI()
    # evaluation_router 自带 /evaluation 前缀；这里只补全局 /api/v1。
    app.include_router(evaluation_router, prefix="/api/v1")
    app.dependency_overrides[get_current_user] = lambda: SimpleNamespace(id=owner)
    # 该 FastAPI 版本在 include_router 时惰性挂载；生成 OpenAPI 文档会强制展开路由表。
    app.openapi()
    return app


@pytest.fixture
def api_store(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> EvaluationClosure:
    """应用内 `_closure_store` 通过 store_at(job_root) 打开同一份数据库。"""
    from app.services.evaluation_closure.storage import store_at

    monkeypatch.setenv("RF_EVAL_JOB_ROOT", str(tmp_path))
    return store_at(tmp_path)


def api_client(app):
    """该 FastAPI 版本下 include 是惰性解析；进入 lifespan 后路由才可用。"""
    from fastapi.testclient import TestClient

    with TestClient(app) as client:
        yield client


def test_api_requires_backflow_action_header(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, api_store: EvaluationClosure
) -> None:
    bad_id = register_failed_case(api_store)
    body = {
        "case_id": "planner-gap-01",
        "group": "agent-memory",
        "aspects": ["记忆写入路径", "检索延迟量级"],
        "review_note": "复核结论",
        "stage": "planner",
        "bad_case_id": bad_id,
    }
    for client in api_client(build_api_app(monkeypatch, tmp_path)):
        denied = client.post("/api/v1/evaluation/evolution/backflow", json=body)
        assert denied.status_code == 403
        ok = client.post(
            "/api/v1/evaluation/evolution/backflow",
            json=body,
            headers={"X-ResearchFlow-Eval-Action": "planner-backflow-v1"},
        )
        assert ok.status_code == 200
        assert ok.json()["case"]["case_id"] == "planner-gap-01"


def test_api_backflow_conflict_and_read_endpoints(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, api_store: EvaluationClosure
) -> None:
    bad_id = register_failed_case(api_store)
    body = {
        "case_id": "planner-gap-01",
        "group": "agent-memory",
        "aspects": ["记忆写入路径", "检索延迟量级"],
        "review_note": "复核结论",
        "stage": "planner",
        "bad_case_id": bad_id,
    }
    headers = {"X-ResearchFlow-Eval-Action": "planner-backflow-v1"}
    for client in api_client(build_api_app(monkeypatch, tmp_path)):
        assert (
            client.post(
                "/api/v1/evaluation/evolution/backflow", json=body, headers=headers
            ).status_code
            == 200
        )
        conflict = client.post("/api/v1/evaluation/evolution/backflow", json=body, headers=headers)
        assert conflict.status_code == 409

        listing = client.get("/api/v1/evaluation/evolution/bad-cases")
        assert listing.status_code == 200
        assert listing.json()["bad_cases"][0]["suggested_aspects"]

        merged = client.get("/api/v1/evaluation/evolution/dataset")
        assert merged.status_code == 200
        assert merged.json()["backflow_case_count"] == 1

        annotations = client.get("/api/v1/evaluation/evolution/annotations")
        assert annotations.status_code == 200
        assert len(annotations.json()["aspects"]) == 10

        benchmark = client.get("/api/v1/evaluation/evolution/benchmark-cases")
        assert benchmark.status_code == 200
        assert benchmark.json()["case_count"] == 10

        state = client.get("/api/v1/evaluation/evolution/overview")
        assert state.status_code == 200
        assert state.json()["benchmark"]["case_count"] == 10

        experiments = client.get("/api/v1/evaluation/evolution/experiments")
        assert experiments.status_code == 200
        assert experiments.json() == []


def test_api_hides_other_owners_bad_cases(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, api_store: EvaluationClosure
) -> None:
    register_failed_case(api_store, owner="bob")
    for client in api_client(build_api_app(monkeypatch, tmp_path)):
        listing = client.get("/api/v1/evaluation/evolution/bad-cases")
        assert listing.status_code == 200
        assert listing.json()["bad_cases"] == []
