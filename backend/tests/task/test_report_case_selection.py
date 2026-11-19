"""Single-case selection, real CLI plumbing and API/Worker wiring without paid calls."""

import asyncio
import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.v1 import evaluation
from app.services import evaluation_pipeline_v13 as pipeline
from evals import harness, query_selection
from evals.agent_eval_ext import closure_observer
from evals.agent_eval_ext import evaluation_pipeline_suite_v13 as suite
from evals.agent_eval_ext import evaluation_pipeline_worker_v13 as worker

CASE = "quantum-error-correction"
BACKEND = Path(__file__).resolve().parents[2]


def test_selects_original_quantum_input_without_reordering_or_mutation():
    original = query_selection.load_report_queries()
    frozen = json.dumps(original)
    selected = query_selection.select_queries(original, case_id=CASE)
    assert len(selected) == 1
    assert selected[0] is next(q for q in original if q["id"] == CASE)
    assert json.dumps(original) == frozen
    assert query_selection.select_queries(original, limit=2) == original[:2]
    assert query_selection.select_queries(original) == original
    assert query_selection.select_queries(original, limit=0) == original


@pytest.mark.parametrize(
    "case", ["unknown-case", "Quantum-error-correction", "", " quantum-error-correction"]
)
def test_unknown_or_invalid_id_never_returns_full_set(case):
    with pytest.raises(ValueError):
        query_selection.load_report_queries(case)


def test_duplicate_source_ids_fail_even_when_target_itself_is_unique():
    rows = [{"id": CASE}, {"id": "other"}, {"id": "other"}]
    with pytest.raises(ValueError, match="duplicate"):
        query_selection.select_queries(rows, case_id=CASE)
    with pytest.raises(ValueError, match="duplicate"):
        query_selection.select_queries([{"id": CASE}, {"id": CASE}], case_id=CASE)


@pytest.mark.parametrize("limit", [0, 1, 2, -1])
def test_selector_rejects_any_explicit_limit_with_case(limit):
    with pytest.raises(ValueError, match="conflicts"):
        query_selection.select_queries([{"id": CASE}], case_id=CASE, limit=limit)


@pytest.mark.parametrize(
    "args,duplicate",
    [
        (["--case-id", "absent"], False),
        (["--case-id", CASE], True),
        (["--case-id", CASE, "--limit", "1"], False),
        (["--case-id", CASE, "--limit", "0"], False),
        (["--mode", "memory", "--case-id", CASE], False),
    ],
)
def test_cli_rejects_before_real_judge_configuration_output_or_graph(
    tmp_path, monkeypatch, args, duplicate
):
    if duplicate:
        (tmp_path / "queries.json").write_text(
            json.dumps({"queries": [{"id": CASE}, {"id": CASE}]})
        )
        monkeypatch.setattr(harness, "EVALS_DIR", tmp_path)
    monkeypatch.setattr(harness, "RUN_CONFIG", replace(harness.RUN_CONFIG, llm_mode="real"))

    def forbidden(*_args, **_kwargs):
        pytest.fail("selection failure reached model configuration or graph/output")

    monkeypatch.setattr(harness, "configure_judge", forbidden)
    monkeypatch.setattr(harness, "run_one", forbidden)
    monkeypatch.setattr(harness, "_check_output_path", forbidden)
    monkeypatch.setattr(sys, "argv", ["harness", *args, "--out", str(tmp_path / "result.json")])
    with pytest.raises(SystemExit) as error:
        asyncio.run(harness.main())
    assert error.value.code == 2
    assert not (tmp_path / "result.json").exists()


@pytest.mark.parametrize(
    "selection,ids",
    [(["--case-id", CASE], [CASE]), (["--limit", "2"], ["llm-memory", "postgres-vs-mysql"])],
)
def test_real_cli_uses_compiled_graph_only_for_selected_ids_in_fake_mode(tmp_path, selection, ids):
    output = tmp_path / "result.json"
    env = {**os.environ, "LLM_MODE": "fake", "HF_HUB_OFFLINE": "1"}
    result = subprocess.run(
        [sys.executable, "-m", "evals.harness", *selection, "--out", str(output)],
        cwd=BACKEND,
        env=env,
        capture_output=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr.decode()
    document = json.loads(output.read_text())
    assert [r["id"] for r in document["results"]] == ids
    assert document["aggregate"]["queries"] == len(ids)
    assert document["llm_mode"] == "fake"
    if "--case-id" in selection:
        assert document["method"]["case_id"] == CASE
    else:
        assert "case_id" not in document["method"]


@pytest.fixture
def online(monkeypatch):
    monkeypatch.setenv("LLM_MODE", "real")
    monkeypatch.setenv("RF_EVAL_ONLINE_ENABLED", "1")
    monkeypatch.setenv("RF_EVAL_JUDGE_ROUTE", "custom:fixture-judge")


@pytest.mark.parametrize(
    "datasets,limit,case",
    [
        (["report-benchmark"], 1, "absent"),
        (["report-benchmark"], 0, CASE),
        (["report-benchmark"], 2, CASE),
        (["report-benchmark", "grounding-v1"], 1, CASE),
        (["grounding-v1"], 1, CASE),
    ],
)
def test_pipeline_refuses_invalid_selection_before_enqueue(tmp_path, online, datasets, limit, case):
    with pytest.raises(pipeline.PipelineError):
        pipeline.create_pipeline(
            tmp_path, "alice", datasets, str(uuid4()), limit, pipeline.PAID_CONFIRM, case_id=case
        )
    assert not (tmp_path / "pipelines").exists()


def test_pipeline_idempotency_includes_case_id_and_legacy_jobs_omit_it(tmp_path, online):
    request = str(uuid4())
    first = pipeline.create_pipeline(
        tmp_path, "alice", ["report-benchmark"], request, 1, pipeline.PAID_CONFIRM, case_id=CASE
    )
    assert first["case_id"] == CASE
    assert (
        pipeline.create_pipeline(
            tmp_path, "alice", ["report-benchmark"], request, 1, pipeline.PAID_CONFIRM, case_id=CASE
        )
        == first
    )
    with pytest.raises(pipeline.PipelineError, match="idempotency_conflict"):
        pipeline.create_pipeline(
            tmp_path,
            "alice",
            ["report-benchmark"],
            request,
            1,
            pipeline.PAID_CONFIRM,
            case_id="llm-memory",
        )
    with pytest.raises(pipeline.PipelineError, match="idempotency_conflict"):
        pipeline.create_pipeline(
            tmp_path, "alice", ["report-benchmark"], request, 1, pipeline.PAID_CONFIRM
        )
    old_root = tmp_path / "old"
    old_root.mkdir()
    legacy = pipeline.create_pipeline(
        old_root, "alice", ["report-benchmark"], str(uuid4()), 2, pipeline.PAID_CONFIRM
    )
    assert "case_id" not in legacy
    assert "case_id" not in json.loads(
        next((old_root / "pipelines").glob("*/request.json")).read_text()
    )


def test_api_to_online_worker_and_closure_select_exact_case_with_synthetic_result(
    tmp_path, monkeypatch, online
):
    monkeypatch.setenv("RF_EVAL_JOB_ROOT", str(tmp_path))
    app = FastAPI()
    app.include_router(evaluation.router, prefix="/api/v1")
    app.dependency_overrides[evaluation._evaluation_root] = lambda: tmp_path
    app.dependency_overrides[evaluation.get_current_user] = lambda: SimpleNamespace(id="alice")
    observed = []

    class Process:
        pid = 0

        def wait(self, timeout):
            return 0

    def launch(cmd, **kwargs):
        observed.append(cmd)
        assert cmd[-2:] == ["--case-id", CASE]
        assert kwargs["shell"] is False
        target = Path(cmd[cmd.index("--output") + 1])
        # Transport fixture only, not a claimed real model result.
        target.write_text(
            json.dumps({"results": [{"id": CASE, "completed": False}], "aggregate": {"queries": 1}})
        )
        return Process()

    monkeypatch.setattr(worker.subprocess, "Popen", launch)
    with TestClient(app) as client:
        response = client.post(
            "/api/v1/evaluation/pipelines",
            headers={"X-ResearchFlow-Eval-Action": "run-pipeline-v1.3"},
            json={
                "datasets": ["report-benchmark"],
                "request_id": str(uuid4()),
                "max_cases": 1,
                "case_id": CASE,
                "consent": pipeline.PAID_CONFIRM,
            },
        )
        assert response.status_code == 202, response.text
        pid = response.json()["pipeline_id"]
        assert worker.run_once(tmp_path, "online")
        state = client.get(f"/api/v1/evaluation/pipelines/{pid}").json()
        assert state["case_id"] == CASE and state["steps"][0]["case_count"] == 1
        assert state["steps"][0]["closure_status"] == "recorded"
        for invalid in [
            {"case_id": "missing"},
            {"max_cases": 0},
            {"max_cases": 2},
            {"datasets": ["report-benchmark", "grounding-v1"]},
        ]:
            rejected = client.post(
                "/api/v1/evaluation/pipelines",
                headers={"X-ResearchFlow-Eval-Action": "run-pipeline-v1.3"},
                json={
                    "datasets": ["report-benchmark"],
                    "request_id": str(uuid4()),
                    "max_cases": 1,
                    "case_id": CASE,
                    "consent": pipeline.PAID_CONFIRM,
                    **invalid,
                },
            )
            assert rejected.status_code == 422
    assert len(observed) == 1
    captured = closure_observer.store_at(tmp_path).runs("alice")[0]
    assert captured["expected_json"] == [CASE]
    for body in [{"case_id": 1}, {"case_id": True}, {"case_id": ""}]:
        with pytest.raises(ValueError):
            evaluation.PipelineRequestV13(
                datasets=["report-benchmark"], request_id=str(uuid4()), **body
            )


@pytest.mark.parametrize(
    "case_id,expected", [(CASE, ["--case-id", CASE]), (None, ["--limit", "2"])]
)
def test_suite_passes_case_to_original_harness_without_limit(
    tmp_path, monkeypatch, online, case_id, expected
):
    output = tmp_path / "result.json"
    calls = []

    def run(cmd, **kwargs):
        calls.append(cmd)
        output.write_text(json.dumps({"results": [{"id": case_id or "llm-memory"}]}))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(suite.subprocess, "run", run)
    suite.execute(
        "report-benchmark",
        output,
        1 if case_id else 2,
        judge_route="custom:fixture-judge",
        case_id=case_id,
    )
    assert calls[0][1:3] == ["-m", "evals.harness"]
    assert all(value in calls[0] for value in expected)
    assert ("--limit" in calls[0]) is (case_id is None)


def test_suite_invalid_case_and_wrong_result_identity_are_rejected(tmp_path, monkeypatch, online):
    called = []

    def run(cmd, **kwargs):
        called.append(cmd)
        Path(cmd[-1]).write_text(json.dumps({"results": [{"id": "llm-memory"}, {"id": CASE}]}))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(suite.subprocess, "run", run)
    with pytest.raises(ValueError):
        suite.execute(
            "report-benchmark",
            tmp_path / "unknown.json",
            1,
            judge_route="custom:fixture-judge",
            case_id="missing",
        )
    assert not called
    with pytest.raises(ValueError, match="identity_mismatch"):
        suite.execute(
            "report-benchmark",
            tmp_path / "wrong.json",
            1,
            judge_route="custom:fixture-judge",
            case_id=CASE,
        )


def test_closure_selects_original_id_and_source_fingerprint():
    all_rows, all_files = closure_observer.sources("report-benchmark", 0)
    one, files = closure_observer.sources("report-benchmark", 1, case_id=CASE)
    assert list(one) == [CASE] and one[CASE] == all_rows[CASE] and files == all_files


@pytest.mark.parametrize("entry", ["pipeline", "suite", "worker"])
def test_duplicate_dataset_rejected_by_every_launch_boundary(tmp_path, monkeypatch, online, entry):
    monkeypatch.setattr(query_selection, "__file__", str(tmp_path / "query_selection.py"))
    (tmp_path / "queries.json").write_text(json.dumps({"queries": [{"id": CASE}, {"id": CASE}]}))

    def forbidden(*args, **kwargs):
        pytest.fail("duplicate ID reached process launch")

    monkeypatch.setattr(worker.subprocess, "Popen", forbidden)
    with pytest.raises(ValueError, match="duplicate"):
        if entry == "pipeline":
            pipeline.create_pipeline(
                tmp_path,
                "alice",
                ["report-benchmark"],
                str(uuid4()),
                1,
                pipeline.PAID_CONFIRM,
                case_id=CASE,
            )
        elif entry == "suite":
            suite.execute(
                "report-benchmark",
                tmp_path / "result.json",
                1,
                judge_route="custom:fixture-judge",
                case_id=CASE,
            )
        else:
            worker._run_fixed("report-benchmark", tmp_path / "result.json", 1, case_id=CASE)
