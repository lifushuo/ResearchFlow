"""评测中心坏例回流、活跃数据集组装与实验证据投影的回归；不调用真实模型。"""

from __future__ import annotations

import hashlib
import json
import os
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
    planner_evidence_for_bad_case,
)
from evals.agent_eval_ext.closure_observer import sources

# evals.harness 在模块导入时执行 load_env_file()，会把 .env 泄漏进整个测试会话
# （test_evals_result_paths.py 顶部的既有约定）；导入前后快照 os.environ。
_ENV_BEFORE_HARNESS = dict(os.environ)
from evals import harness  # noqa: E402 — 必须跟随上面的快照

os.environ.clear()
os.environ.update(_ENV_BEFORE_HARNESS)

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
    review_evidence = (
        {
            "planner_tasks": ["搜索长程记忆读写框架的内部数据流"],
            "missing_aspects": ["检索延迟量级"],
            "evidence_reference": "pipe-llm-memory--report-benchmark/planner.agent_log#4",
            "review_confirmed": True,
        }
        if stage == "planner"
        else {}
    )
    review_evidence.update(kwargs)
    return backflow(
        store,
        "alice",
        tmp_path,
        case_id=case_id,
        group=group,
        aspects=["记忆写入路径", "检索延迟量级"],
        review_note="Planner 任务列表缺少实验条件核对，轨迹确认。",
        stage=stage,
        **review_evidence,
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
    assert record["review"]["task_evidence_status"] == "reviewer_supplied"
    assert record["review"]["evidence_sha256"]
    merged = merged_dataset(tmp_path)
    assert merged["seed_case_count"] == 30
    assert merged["backflow_case_count"] == 1
    assert merged["splits"]["train"] == 13
    assert merged["error"] is None


def test_planner_feedback_requires_real_tasks_and_missing_aspect(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    bad_id = register_failed_case(store)
    with pytest.raises(BackflowError, match="需要人工确认"):
        flow(store, tmp_path, bad_case_id=bad_id, review_confirmed=False)
    with pytest.raises(BackflowError, match="实际 Planner 任务列表"):
        flow(store, tmp_path, bad_case_id=bad_id, planner_tasks=[])
    with pytest.raises(BackflowError, match="至少一条"):
        flow(store, tmp_path, bad_case_id=bad_id, missing_aspects=[])
    with pytest.raises(BackflowError, match="必须来自"):
        flow(store, tmp_path, bad_case_id=bad_id, missing_aspects=["这个不是必答维度"])
    with pytest.raises(BackflowError, match="证据编号"):
        flow(store, tmp_path, bad_case_id=bad_id, evidence_reference="")
    assert merged_dataset(tmp_path)["backflow_case_count"] == 0


def test_legacy_planner_only_label_does_not_enter_training(tmp_path: Path) -> None:
    from app.services.planner_evolution_center import evolution_root

    folder = evolution_root(tmp_path) / "cases"
    folder.mkdir(parents=True)
    case = (
        load_seed_dataset()
        .cases[0]
        .model_copy(
            update={
                "case_id": "legacy-gap-001",
                "group": "legacy-gap-001",
                "query": "比较两类自主智能体的工具可靠性测试与失效模式。",
                "split": "train",
            }
        )
    )
    (folder / "legacy.json").write_text(
        json.dumps(
            {
                "kind": "planner_backflow_v1",
                "case": case.model_dump(),
                "review": {"stage": "planner", "task_evidence_status": "not_collected"},
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    assert merged_dataset(tmp_path)["backflow_case_count"] == 0


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
        "planner_tasks": ["搜索长程记忆读写框架的内部数据流"],
        "missing_aspects": ["检索延迟量级"],
        "evidence_reference": "pipe-llm-memory--report-benchmark/planner.agent_log#4",
        "review_confirmed": True,
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


def test_api_rejects_unreviewed_planner_feedback(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, api_store: EvaluationClosure
) -> None:
    bad_id = register_failed_case(api_store)
    body = {
        "case_id": "planner-gap-api-02",
        "group": "agent-memory",
        "aspects": ["记忆写入路径", "检索延迟量级"],
        "review_note": "错误报告不代表 Planner 漏项。",
        "stage": "planner",
        "bad_case_id": bad_id,
    }
    for client in api_client(build_api_app(monkeypatch, tmp_path)):
        denied = client.post(
            "/api/v1/evaluation/evolution/backflow",
            json=body,
            headers={"X-ResearchFlow-Eval-Action": "planner-backflow-v1"},
        )
        assert denied.status_code == 409
        assert "人工确认" in denied.json()["detail"]
        assert client.get("/api/v1/evaluation/evolution/dataset").json()["backflow_case_count"] == 0


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
        "planner_tasks": ["搜索长程记忆读写框架的内部数据流"],
        "missing_aspects": ["检索延迟量级"],
        "evidence_reference": "pipe-llm-memory--report-benchmark/planner.agent_log#4",
        "review_confirmed": True,
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


def test_automatic_planner_evidence_is_owner_scoped_and_checked(tmp_path: Path) -> None:
    """从真实评测产物自动取证，只有当前用户能读取，并检查内容指纹。"""
    import hashlib

    from app.services.planner_evolution_center import planner_evidence_for_bad_case

    store = make_store(tmp_path)
    pipeline_id = "pipe-" + "a" * 24
    run_id = pipeline_id + "--report-benchmark"
    source_cases, _ = sources("report-benchmark", 0)
    original_query = source_cases[BENCH_CASE]["case"]["query"]
    store.register_run(
        run_id=run_id,
        owner_id="alice",
        dataset_id="report-benchmark",
        execution_mode="real",
        sample_set_hash="benchmark-v1",
        metrics_version="v1",
        model_route="configured-sha256:abc",
        judge_route="custom:judge",
    )
    store.ingest_case(
        run_id=run_id,
        owner_id="alice",
        case_id=BENCH_CASE,
        status="failed",
        input_sha256=input_hash(source_cases[BENCH_CASE]),
        error_code="research_incomplete",
    )
    bad_id = store.list_bad_cases("alice")[0]["bad_case_id"]
    folder = tmp_path / "pipelines" / pipeline_id
    output = folder / "artifacts" / "report-benchmark" / "raw_result.json"
    output.parent.mkdir(parents=True)
    (folder / "request.json").write_text(
        json.dumps(
            {
                "owner": "alice",
                "pipeline_id": pipeline_id,
                "datasets": ["report-benchmark"],
                "max_cases": 1,
                "paid_execution_authorized": True,
                "created_at": "2026-11-21T00:00:00Z",
            }
        ),
        encoding="utf-8",
    )
    (folder / "status.json").write_text(
        json.dumps(
            {
                "pipeline_id": pipeline_id,
                "state": "completed",
                "steps": [],
                "updated_at": "2026-11-21T00:00:00Z",
            }
        ),
        encoding="utf-8",
    )
    evidence = {
        "schema_version": 1,
        "case_id": BENCH_CASE,
        "query_sha256": hashlib.sha256(original_query.encode()).hexdigest(),
        "tasks": [{"id": "1", "query": "检索长期记忆写入和存储的数据路径"}],
        "prompt_sha256": "b" * 64,
        "trace_id": "trace-01",
    }
    evidence["evidence_sha256"] = hashlib.sha256(
        json.dumps(evidence, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()
    ).hexdigest()
    output.write_text(
        json.dumps({"results": [{"id": BENCH_CASE, "planner_evidence": evidence}]}),
        encoding="utf-8",
    )

    saved = planner_evidence_for_bad_case(store, "alice", tmp_path, bad_id)
    assert saved["status"] == "recorded"
    assert saved["tasks"][0]["query"] == evidence["tasks"][0]["query"]
    assert saved["evidence_reference"].startswith("auto:" + run_id + ":")
    with pytest.raises(NotFound):
        planner_evidence_for_bad_case(store, "bob", tmp_path, bad_id)

    record = backflow(
        store,
        "alice",
        tmp_path,
        case_id="auto-trace-gap-01",
        group="agent-memory",
        aspects=["记忆写入路径", "检索延迟量级"],
        review_note="对照自动归档任务，未安排延迟测量检索。",
        stage="planner",
        bad_case_id=bad_id,
        planner_tasks=[evidence["tasks"][0]["query"]],
        missing_aspects=["检索延迟量级"],
        evidence_reference=saved["evidence_reference"],
        review_confirmed=True,
    )
    assert record["review"]["task_evidence_status"] == "trace_verified"
    assert merged_dataset(tmp_path)["backflow_case_count"] == 1

    # 原始任务改变但证据编号未改变时，不得误认为归档证据仍然可信。
    changed = json.loads(output.read_text(encoding="utf-8"))
    changed["results"][0]["planner_evidence"]["tasks"][0]["query"] = "伪造的模型检索任务内容"
    output.write_text(json.dumps(changed), encoding="utf-8")
    assert (
        planner_evidence_for_bad_case(store, "alice", tmp_path, bad_id)["status"] == "unavailable"
    )


def test_planner_evidence_projection_from_mock_state() -> None:
    """_planner_evidence 的受限投影：任务、提示词指纹、Trace 与证据指纹都来自真实输入。"""
    from research_engine.prompts import PLANNER_PROMPT_V2

    question = "比较两种 TSP 求解器的解质量与运行预算。"
    evidence = harness._planner_evidence(
        {
            "tasks": [
                {"id": "1", "query": "检索两种算法的解质量指标"},
                {"query": "调查 Gap 参考解的定义来源"},
                {"id": "y" * 60, "query": "  核实测试硬件  "},
            ]
        },
        query={"id": "tsp-comparison", "query": question},
        prompt_records={"planner.main": {"effective_prompt": PLANNER_PROMPT_V2}},
        runtime_trace=SimpleNamespace(trace_id="trace-01"),
    )
    assert evidence is not None
    assert evidence["schema_version"] == 1
    assert evidence["case_id"] == "tsp-comparison"
    assert evidence["query_sha256"] == hashlib.sha256(question.encode("utf-8")).hexdigest()
    # id 缺失以序号兜底，超长 id 截断，query 原样去空白——不做任何补造。
    assert evidence["tasks"] == [
        {"id": "1", "query": "检索两种算法的解质量指标"},
        {"id": "2", "query": "调查 Gap 参考解的定义来源"},
        {"id": "y" * 40, "query": "核实测试硬件"},
    ]
    assert (
        evidence["prompt_sha256"] == hashlib.sha256(PLANNER_PROMPT_V2.encode("utf-8")).hexdigest()
    )
    assert evidence["trace_id"] == "trace-01"
    unsigned = {k: v for k, v in evidence.items() if k != "evidence_sha256"}
    assert (
        evidence["evidence_sha256"]
        == hashlib.sha256(
            json.dumps(unsigned, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
                "utf-8"
            )
        ).hexdigest()
    )


def test_planner_evidence_honest_when_inputs_are_missing() -> None:
    """任务缺失或畸形返回 None，不补造；提示词或 Trace 缺失记 null，不伪造指纹。"""
    query = {"id": "case-1", "query": "比较两种 TSP 求解器的解质量与运行预算。"}
    trace = SimpleNamespace(trace_id="trace-01")
    for state in (
        {},
        {"tasks": []},
        {"tasks": "不是列表"},
        {"tasks": [{"query": "太短"}]},
        {"tasks": [{"id": "1", "query": 123}]},
        {"tasks": [{"id": "1", "query": "合法任务内容"}, "混入非字典项"]},
        {"tasks": [{"id": str(i), "query": "检索任务内容"} for i in range(31)]},
    ):
        assert (
            harness._planner_evidence(state, query=query, prompt_records={}, runtime_trace=trace)
            is None
        )

    partial = harness._planner_evidence(
        {"tasks": [{"id": "1", "query": "检索记忆写入路径"}]},
        query=query,
        prompt_records={},
        runtime_trace=trace,
    )
    assert partial is not None
    assert partial["prompt_sha256"] is None
    assert partial["trace_id"] == "trace-01"

    no_trace = harness._planner_evidence(
        {"tasks": [{"id": "1", "query": "检索记忆写入路径"}]},
        query=query,
        prompt_records={"planner.main": {"effective_prompt": "P"}},
        runtime_trace=object(),
    )
    assert no_trace is not None
    assert no_trace["trace_id"] is None


def test_planner_evidence_round_trip_through_raw_result(tmp_path: Path) -> None:
    """harness 行结构写入 raw_result.json 后，取证接口必须原样读回；归档被改则降级。"""
    from research_engine.prompts import PLANNER_PROMPT_V2

    store = make_store(tmp_path)
    pipeline_id = "pipe-" + "b" * 24
    run_id = pipeline_id + "--report-benchmark"
    source_cases, _ = sources("report-benchmark", 0)
    question = source_cases[BENCH_CASE]["case"]["query"]
    store.register_run(
        run_id=run_id,
        owner_id="alice",
        dataset_id="report-benchmark",
        execution_mode="real",
        sample_set_hash="benchmark-v1",
        metrics_version="v1",
        model_route="configured-sha256:abc",
        judge_route="custom:judge",
    )
    store.ingest_case(
        run_id=run_id,
        owner_id="alice",
        case_id=BENCH_CASE,
        status="failed",
        input_sha256=input_hash(source_cases[BENCH_CASE]),
        error_code="research_incomplete",
    )
    bad_id = store.list_bad_cases("alice")[0]["bad_case_id"]

    evidence = harness._planner_evidence(
        {
            "tasks": [
                {"id": "1", "query": "检索长期记忆写入和存储的数据路径"},
                {"id": "2", "query": "测量记忆检索延迟的量级与方差"},
            ]
        },
        query={"id": BENCH_CASE, "query": question},
        prompt_records={"planner.main": {"effective_prompt": PLANNER_PROMPT_V2}},
        runtime_trace=SimpleNamespace(trace_id="trace-01"),
    )
    assert evidence is not None

    folder = tmp_path / "pipelines" / pipeline_id
    output = folder / "artifacts" / "report-benchmark" / "raw_result.json"
    output.parent.mkdir(parents=True)
    (folder / "request.json").write_text(
        json.dumps(
            {
                "owner": "alice",
                "pipeline_id": pipeline_id,
                "datasets": ["report-benchmark"],
                "max_cases": 1,
                "paid_execution_authorized": True,
                "created_at": "2026-11-21T00:00:00Z",
            }
        ),
        encoding="utf-8",
    )
    (folder / "status.json").write_text(
        json.dumps(
            {
                "pipeline_id": pipeline_id,
                "state": "completed",
                "steps": [],
                "updated_at": "2026-11-21T00:00:00Z",
            }
        ),
        encoding="utf-8",
    )
    output.write_text(
        json.dumps({"results": [{"id": BENCH_CASE, "planner_evidence": evidence}]}),
        encoding="utf-8",
    )

    saved = planner_evidence_for_bad_case(store, "alice", tmp_path, bad_id)
    assert saved["status"] == "recorded"
    assert saved["tasks"] == evidence["tasks"]
    assert saved["prompt_sha256"] == hashlib.sha256(PLANNER_PROMPT_V2.encode("utf-8")).hexdigest()
    assert saved["trace_id"] == "trace-01"
    assert saved["evidence_reference"].startswith(f"auto:{run_id}:")

    # 写入后归档被改动：取证必须降级为不可用，不能把改动后的任务当 Planner 原文。
    tampered = json.loads(output.read_text(encoding="utf-8"))
    tampered["results"][0]["planner_evidence"]["tasks"][0]["query"] = "伪造的检索任务"
    output.write_text(json.dumps(tampered), encoding="utf-8")
    assert (
        planner_evidence_for_bad_case(store, "alice", tmp_path, bad_id)["status"] == "unavailable"
    )


def _register_reviewable_real_run(
    store: EvaluationClosure,
    root: Path,
    *,
    case_id: str,
    owner: str = "alice",
    status: str = "passed",
) -> tuple[str, Path]:
    """模拟 Worker 已完成的真实报告案例；不触发模型调用。"""
    from app.services.planner_evolution_center import planner_evidence_for_case

    pipeline_id = "pipe-" + ("a" if case_id == BENCH_CASE else "b") * 24
    run_id = f"{pipeline_id}--report-benchmark"
    source_cases, _ = sources("report-benchmark", 0)
    question = source_cases[case_id]["case"]["query"]
    store.register_run(
        run_id=run_id,
        owner_id=owner,
        dataset_id="report-benchmark",
        execution_mode="real",
        sample_set_hash="v1",
        metrics_version="v1",
        model_route="configured-sha256:abc",
        judge_route="custom:judge",
        source_pipeline_id=pipeline_id,
    )
    store.ingest_case(
        run_id=run_id,
        owner_id=owner,
        case_id=case_id,
        status=status,
        input_sha256=input_hash(source_cases[case_id]),
        error_code="research_incomplete" if status == "failed" else None,
    )
    folder = root / "pipelines" / pipeline_id
    output = folder / "artifacts" / "report-benchmark" / "raw_result.json"
    output.parent.mkdir(parents=True)
    (folder / "request.json").write_text(
        json.dumps(
            {
                "owner": owner,
                "pipeline_id": pipeline_id,
                "datasets": ["report-benchmark"],
                "max_cases": 1,
                "paid_execution_authorized": True,
                "created_at": "2026-11-21T00:00:00Z",
            }
        ),
        encoding="utf-8",
    )
    (folder / "status.json").write_text(
        json.dumps(
            {
                "pipeline_id": pipeline_id,
                "state": "completed",
                "steps": [],
                "updated_at": "2026-11-21T00:00:00Z",
            }
        ),
        encoding="utf-8",
    )
    payload = {
        "schema_version": 1,
        "case_id": case_id,
        "query_sha256": hashlib.sha256(question.encode()).hexdigest(),
        "tasks": [{"id": "1", "query": "收集相关方法的技术路线和对照实验条件"}],
        "prompt_sha256": "b" * 64,
        "trace_id": "trace-review-case-01",
    }
    payload["evidence_sha256"] = hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    output.write_text(
        json.dumps({"results": [{"id": case_id, "planner_evidence": payload}]}), encoding="utf-8"
    )
    assert planner_evidence_for_case(store, owner, root, run_id, case_id)["status"] == "recorded"
    return run_id, output


def test_passed_evaluation_case_can_be_reviewed_without_becoming_bad_case(tmp_path: Path) -> None:
    from app.services.planner_evolution_center import (
        planner_evidence_for_case,
        reviewable_evaluation_cases,
    )

    store = make_store(tmp_path)
    run_id, artifact = _register_reviewable_real_run(store, tmp_path, case_id=BENCH_CASE)
    assert store.list_bad_cases("alice") == []
    assert reviewable_evaluation_cases(store, "alice")["cases"][0]["result_status"] == "passed"
    assert reviewable_evaluation_cases(store, "bob")["cases"] == []
    with pytest.raises(NotFound):
        planner_evidence_for_case(store, "bob", tmp_path, run_id, BENCH_CASE)
    evidence = planner_evidence_for_case(store, "alice", tmp_path, run_id, BENCH_CASE)
    aspects = load_benchmark_annotations()["aspects"][BENCH_CASE]["required_aspects"]
    record = backflow(
        store,
        "alice",
        tmp_path,
        case_id="review-complete-01",
        group="complete-agent-memory",
        aspects=aspects,
        review_note="六项维度逐一核对，已有可检索任务。",
        stage="planner_complete",
        evaluation_run_id=run_id,
        evaluation_case_id=BENCH_CASE,
        planner_tasks=[t["query"] for t in evidence["tasks"]],
        missing_aspects=[],
        evidence_reference=evidence["evidence_reference"],
        review_confirmed=True,
    )
    assert record["review"]["task_evidence_status"] == "trace_verified"
    merged = merged_dataset(tmp_path)
    assert merged["backflow_case_count"] == 0
    assert merged["review_record_count"] == 1
    with pytest.raises(BackflowError, match="已经人工复核"):
        backflow(
            store,
            "alice",
            tmp_path,
            case_id="review-complete-02",
            group="complete-agent-memory",
            aspects=aspects,
            review_note="重复审核",
            stage="planner_complete",
            evaluation_run_id=run_id,
            evaluation_case_id=BENCH_CASE,
            planner_tasks=[t["query"] for t in evidence["tasks"]],
            missing_aspects=[],
            evidence_reference=evidence["evidence_reference"],
            review_confirmed=True,
        )
    modified = json.loads(artifact.read_text(encoding="utf-8"))
    modified["results"][0]["planner_evidence"]["tasks"][0]["query"] = "伪造的规划任务"
    artifact.write_text(json.dumps(modified), encoding="utf-8")
    assert (
        planner_evidence_for_case(store, "alice", tmp_path, run_id, BENCH_CASE)["status"]
        == "unavailable"
    )


def test_confirmed_missing_from_passed_report_enters_train_only(tmp_path: Path) -> None:
    from app.services.planner_evolution_center import planner_evidence_for_case

    store = make_store(tmp_path)
    run_id, _ = _register_reviewable_real_run(store, tmp_path, case_id=BENCH_CASE)
    evidence = planner_evidence_for_case(store, "alice", tmp_path, run_id, BENCH_CASE)
    aspects = load_benchmark_annotations()["aspects"][BENCH_CASE]["required_aspects"]
    with pytest.raises(BackflowError, match="必须使用冻结的必答维度"):
        backflow(
            store,
            "alice",
            tmp_path,
            case_id="untrusted-label-01",
            group="agent-memory",
            aspects=["伪造维度", "另一个伪造维度"],
            review_note="不允许篡改已冻结标注",
            stage="planner",
            evaluation_run_id=run_id,
            evaluation_case_id=BENCH_CASE,
            planner_tasks=[evidence["tasks"][0]["query"]],
            missing_aspects=["伪造维度"],
            evidence_reference=evidence["evidence_reference"],
            review_confirmed=True,
        )
    record = backflow(
        store,
        "alice",
        tmp_path,
        case_id="review-missing-01",
        group="agent-memory",
        aspects=aspects,
        review_note="任务未覆盖所标注的检索维度。",
        stage="planner",
        evaluation_run_id=run_id,
        evaluation_case_id=BENCH_CASE,
        planner_tasks=[evidence["tasks"][0]["query"]],
        missing_aspects=[aspects[-1]],
        evidence_reference=evidence["evidence_reference"],
        review_confirmed=True,
    )
    assert record["review"]["source"]["type"] == "evaluation_case"
    assert record["review"]["task_evidence_status"] == "trace_verified"
    assert merged_dataset(tmp_path)["backflow_case_count"] == 1
    assert merged_dataset(tmp_path)["splits"]["train"] == 13


def test_reviewable_case_api_requires_owner_and_explicit_review_action(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, api_store: EvaluationClosure
) -> None:
    run_id, _ = _register_reviewable_real_run(api_store, tmp_path, case_id=BENCH_CASE)
    aspects = load_benchmark_annotations()["aspects"][BENCH_CASE]["required_aspects"]
    route = f"/api/v1/evaluation/evolution/reviewable-cases/{run_id}/{BENCH_CASE}/planner-evidence"
    body = {
        "case_id": "review-api-complete",
        "group": "agent-memory",
        "aspects": aspects,
        "review_note": "审阅完整规划并确认无漏项",
        "stage": "planner_complete",
        "evaluation_run_id": run_id,
        "evaluation_case_id": BENCH_CASE,
        "planner_tasks": ["收集相关方法的技术路线和对照实验条件"],
        "missing_aspects": [],
        "review_confirmed": True,
    }
    for client in api_client(build_api_app(monkeypatch, tmp_path)):
        listing = client.get("/api/v1/evaluation/evolution/reviewable-cases")
        assert listing.status_code == 200
        assert listing.json()["cases"][0]["result_status"] == "passed"
        ev = client.get(route)
        assert ev.status_code == 200
        body["evidence_reference"] = ev.json()["evidence_reference"]
        denied = client.post("/api/v1/evaluation/evolution/backflow", json=body)
        assert denied.status_code == 403
        accepted = client.post(
            "/api/v1/evaluation/evolution/backflow",
            json=body,
            headers={"X-ResearchFlow-Eval-Action": "planner-backflow-v1"},
        )
        assert accepted.status_code == 200, accepted.text
        assert accepted.json()["review"]["stage"] == "planner_complete"
    for client in api_client(build_api_app(monkeypatch, tmp_path, owner="bob")):
        assert client.get(route).status_code == 404
        assert client.get("/api/v1/evaluation/evolution/reviewable-cases").json()["cases"] == []


def test_planner_judge_suggestion_requires_explicit_cost_consent_and_owned_evidence(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, api_store: EvaluationClosure
) -> None:
    """不调用真实模型；验证 POST 准入、跨账号隔离和仅建议不回流。"""
    from app.services.planner_evolution_center import planner_evidence_for_case

    run_id, artifact = _register_reviewable_real_run(api_store, tmp_path, case_id=BENCH_CASE)
    evidence = planner_evidence_for_case(api_store, "alice", tmp_path, run_id, BENCH_CASE)
    endpoint = (
        f"/api/v1/evaluation/evolution/reviewable-cases/{run_id}/{BENCH_CASE}/judge-suggestion"
    )
    called = []

    async def fake_judge(*, evidence, question, aspects, timeout_seconds=90):
        called.append((evidence, question, aspects))
        return {"status": "suggested", "automatic_label": False, "aspects": []}

    import app.services.planner_review_judge as service

    monkeypatch.setattr(service, "suggest_planner_coverage", fake_judge)
    body = {"evidence_reference": evidence["evidence_reference"], "acknowledge_cost": True}
    for client in api_client(build_api_app(monkeypatch, tmp_path)):
        assert client.post(endpoint, json=body).status_code == 403
        assert (
            client.post(
                endpoint,
                headers={"X-ResearchFlow-Eval-Action": "planner-judge-v1"},
                json={**body, "acknowledge_cost": False},
            ).status_code
            == 422
        )
        assert called == []
        assert (
            client.post(
                endpoint,
                headers={"X-ResearchFlow-Eval-Action": "planner-judge-v1"},
                json={**body, "evidence_reference": "auto:stale:" + "x" * 64},
            ).status_code
            == 409
        )
        assert called == []
        answer = client.post(
            endpoint,
            headers={"X-ResearchFlow-Eval-Action": "planner-judge-v1"},
            json=body,
        )
        assert answer.status_code == 200, answer.text
        assert answer.json()["automatic_label"] is False
        assert len(called) == 1
        assert api_store.list_bad_cases("alice") == []
    for client in api_client(build_api_app(monkeypatch, tmp_path, owner="bob")):
        assert (
            client.post(
                endpoint,
                headers={"X-ResearchFlow-Eval-Action": "planner-judge-v1"},
                json=body,
            ).status_code
            == 404
        )
    assert len(called) == 1

    # 原始文件被修改后，指纹检查在模型调用之前阻止请求。
    doc = json.loads(artifact.read_text(encoding="utf-8"))
    doc["results"][0]["planner_evidence"]["tasks"][0]["query"] = "被篡改的检索内容"
    artifact.write_text(json.dumps(doc), encoding="utf-8")
    for client in api_client(build_api_app(monkeypatch, tmp_path)):
        assert (
            client.post(
                endpoint,
                headers={"X-ResearchFlow-Eval-Action": "planner-judge-v1"},
                json=body,
            ).status_code
            == 409
        )
    assert len(called) == 1
