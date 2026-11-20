"""验收 GEPA 留出隔离、真实节点调用、坏例归属及发布门禁。"""

from __future__ import annotations

import copy
import json
import sys
from dataclasses import replace

import pytest
from langchain_core.messages import AIMessage

from app.services.evaluation_closure import input_hash
from app.services.evaluation_closure.storage import store_at
from evals import harness
from evals.agent_eval_ext import long_run
from evals.agent_eval_ext.closure_observer import read, sources
from evals.agent_eval_ext.planner_evolution import experiment, runtime
from evals.agent_eval_ext.planner_evolution.__main__ import DEMO_DATASET, REPORT_DATASET, main
from evals.agent_eval_ext.planner_evolution.release import release_candidate, report_gate
from research_engine.runconfig import RunConfig, get_run_config


@pytest.fixture
def dataset():
    return experiment.Dataset.model_validate(read(DEMO_DATASET))


def test_gepa_uses_real_library_and_keeps_holdout_out_of_reflection(dataset):
    pytest.importorskip("gepa")
    calls = []

    def rollout(case, prompt):
        calls.append(case.split)
        return experiment.demo_rollout(case, prompt)

    def reflect(prompt):
        assert "holdout" not in calls
        assert any(
            label in prompt
            for case in dataset.cases
            if case.split == "train"
            for label in case.required_aspects
        )
        assert all(case.query not in prompt for case in dataset.cases if case.split == "holdout")
        assert all(
            label not in prompt
            for case in dataset.cases
            if case.split == "holdout"
            for label in case.required_aspects
        )
        return experiment.demo_reflection(prompt)

    result = experiment.optimize_planner(
        dataset, None, rollout, reflect, experiment.SearchLimits(40, 1, 42, 0.05)
    )
    assert result["holdout"]["gate"]["passed"] is True
    assert result["holdout"]["gate"]["baseline_coverage"] == pytest.approx(0.6875)
    assert result["holdout"]["gate"]["candidate_coverage"] == 1
    assert calls[-8:] == ["holdout"] * 8
    assert len(calls) <= 40
    assert result["baseline"]["sha256"] != result["candidate"]["sha256"]


def test_group_cannot_cross_splits(dataset):
    raw = dataset.model_dump()
    raw["cases"][-1]["group"] = raw["cases"][0]["group"]
    with pytest.raises(ValueError, match="multiple splits"):
        experiment.Dataset.model_validate(raw)


def test_missing_baseline_scores_stop_before_reflection(dataset):
    pytest.importorskip("gepa")

    def rollout(case, _):
        assert case.split == "validation"
        return experiment.Observation(
            case_id=case.case_id,
            split=case.split,
            required_count=len(case.required_aspects),
            error="judge unavailable",
        )

    def reflect(_):
        pytest.fail("unmeasured validation cannot produce reflective candidates")

    result = experiment.optimize_planner(
        dataset, None, rollout, reflect, experiment.SearchLimits(40, 1, 42, 0.05)
    )
    assert result["candidate"] is None
    assert result["holdout"]["gate"]["reasons"] == ["validation_unmeasured"]


def test_holdout_gate_detects_lost_aspect_even_if_mean_improves(dataset):
    cases = [case for case in dataset.cases if case.split == "holdout"]
    before = [experiment.demo_rollout(case, None) for case in cases]
    candidate = [experiment.demo_rollout(case, "REVIEW EVERY REQUIRED ASPECT") for case in cases]
    candidate[0] = candidate[0].model_copy(update={"covered_indices": [2, 3, 4]})
    gate = experiment.make_gate(cases, before, candidate, 0.05)
    assert gate["delta"] > 0.05
    assert gate["passed"] is False
    assert gate["losses"] == [{"case_id": cases[0].case_id, "lost_aspects": [1]}]
    candidate[0] = candidate[0].model_copy(update={"error": "judge timed out"})
    gate = experiment.make_gate(cases, before, candidate, 0.05)
    assert gate["delta"] is None
    assert gate["reasons"] == ["unmeasured_cases"]


@pytest.mark.parametrize("task_ids, measured", [([1], True), ([99], False)])
async def test_live_adapter_runs_planner_with_frozen_prompt_and_separate_judge(
    monkeypatch, dataset, task_ids, measured
):
    calls = []
    candidate = "Generate specific research tasks with independently searchable queries."
    config = RunConfig(llm_mode="real", models={**RunConfig().models, "planner": "custom:planner"})
    case = dataset.cases[0]

    class Model:
        def __init__(self, role):
            self.role = role

        def with_structured_output(self, schema, **kwargs):
            assert kwargs["include_raw"] is True
            self.schema = schema
            return self

        async def ainvoke(self, messages):
            calls.append((self.role, get_run_config().model_for(self.role)))
            if self.role == "planner":
                assert messages[0].content == candidate
                parsed = self.schema.model_validate({"tasks": [{"query": "independent query one"}]})
                served = "planner"
            else:
                assert messages[0].content == runtime.PLANNER_JUDGE_PROMPT
                assert candidate not in messages[-1].content
                fields = [
                    {"aspect_index": i + 1, "task_ids": task_ids}
                    for i in range(len(case.required_aspects))
                ]
                parsed = self.schema.model_validate({"matches": fields})
                served = "independent-coverage-judge"
            return {
                "parsed": parsed,
                "raw": AIMessage(content="", response_metadata={"model_name": served}),
                "parsing_error": None,
            }

    monkeypatch.setattr(runtime.graph, "get_llm", Model)
    monkeypatch.setattr(runtime.llm_factory, "get_llm", Model)
    original_config = get_run_config()
    adapter = runtime.LiveRuntime(
        config, "custom:independent-coverage-judge", "custom:planner", 10, ()
    )
    result = await adapter.observe(case, candidate)
    assert calls == [("planner", "custom:planner"), ("critic", "custom:independent-coverage-judge")]
    assert result.score == (1.0 if measured else None)
    assert get_run_config() is original_config


def test_seed_case_reuses_verified_input_and_checks_owner(tmp_path, dataset):
    original_cases, _ = sources("report-benchmark", 0)
    old_id = next(iter(original_cases))
    store = store_at(tmp_path)
    store.register_run(
        run_id="report-run",
        owner_id="owner",
        dataset_id="report-benchmark",
        execution_mode="real",
        sample_set_hash="sample",
        metrics_version="fixture",
        model_route="custom:planner",
    )
    store.ingest_case(
        run_id="report-run",
        owner_id="owner",
        case_id=old_id,
        status="failed",
        input_sha256=input_hash(original_cases[old_id]),
        error_code="coverage_missing",
    )
    bad_id = store.list_bad_cases("owner")[0]["bad_case_id"]
    out = tmp_path / "seed.json"
    command = [
        "seed-case",
        "--job-root",
        str(tmp_path),
        "--bad-case-id",
        bad_id,
        "--case-id",
        "my-fail-1",
        "--group",
        "memory",
        "--aspect",
        "latency trade-off",
        "--aspect",
        "scalability",
        "--out",
        str(out),
        "--owner",
    ]
    assert main([*command, "someone-else"]) == 2
    assert not out.exists()
    assert main([*command, "owner"]) == 0
    row = experiment.Case.model_validate(read(out))
    assert row.source_bad_case_id == bad_id
    assert row.split == "train"
    assert row.query == original_cases[old_id]["case"]["query"]
    assert dataset.provenance == "demo"


def _measured_experiment(dataset):
    # 控制装置只用于检查发布逻辑契约。已标为 live 的历史报告仍要经过报告门禁验证。
    reviewed = dataset.model_copy(update={"provenance": "reviewed"})
    candidate = experiment.demo_reflection("").removeprefix("```\n").removesuffix("\n```")
    cases = [case for case in dataset.cases if case.split == "holdout"]

    def measured_rows(prompt):
        rows = [experiment.demo_rollout(case, prompt) for case in cases]
        return [
            row.model_copy(
                update={
                    "task_matches": [
                        experiment.TaskMatch(
                            aspect_index=i,
                            task_ids=[i] if i in row.covered_indices else [],
                        )
                        for i in range(1, row.required_count + 1)
                    ],
                    "served_judge": "scripted-judge",
                }
            ).model_dump()
            for row in rows
        ]

    return {
        "schema_version": 1,
        "evidence_kind": "live_planner",
        "metric_version": experiment.METRIC_VERSION,
        "dataset": reviewed.model_dump(),
        "dataset_sha256": input_hash(reviewed.model_dump()),
        "limits": {"minimum_gain": 0.05},
        "baseline": {
            "override": "Approved custom baseline",
            "sha256": long_run.sha256_text("Approved custom baseline"),
        },
        "candidate": {"override": candidate, "sha256": long_run.sha256_text(candidate)},
        "holdout": {
            "baseline": measured_rows(None),
            "candidate": measured_rows(candidate),
        },
        "routes": {
            "pipeline": dict(harness.RUN_CONFIG.models),
            "judge": "custom:independent-judge",
            "reflection": harness.RUN_CONFIG.models["planner"],
        },
        "report_dataset_sha256": input_hash(read(REPORT_DATASET)["queries"]),
    }


async def test_release_uses_actual_harness_ab_and_rejects_partial_judgments(
    monkeypatch, tmp_path, dataset
):
    import test_long_run_reliability as fixtures

    measured = _measured_experiment(dataset)
    baseline, candidate = release_candidate(measured)
    before, after = tmp_path / "baseline.json", tmp_path / "candidate.json"
    out, progress = tmp_path / "report.json", tmp_path / "progress"
    before.write_text(json.dumps({"planner": baseline}), encoding="utf-8")
    after.write_text(json.dumps({"planner": candidate}), encoding="utf-8")
    monkeypatch.setattr(harness, "RUN_CONFIG", replace(harness.RUN_CONFIG, llm_mode="real"))
    monkeypatch.setattr(harness, "JUDGE_ROUTE", None)
    monkeypatch.setattr(harness, "build_graph", lambda *_: fixtures.FakeGraph())
    judge = fixtures.ScriptedJudge()
    monkeypatch.setattr(harness, "_judge_llm", lambda: judge)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "harness",
            "--judge",
            measured["routes"]["judge"],
            "--baseline",
            str(before),
            "--candidate",
            str(after),
            "--progress-dir",
            str(progress),
            "--out",
            str(out),
        ],
    )
    await harness.main()

    report = read(out)
    queries = read(REPORT_DATASET)["queries"]
    passed = report_gate(measured, report, queries)
    assert passed["passed"] is True
    assert passed["cases"] == len(queries)
    stored = report["reliability"]["cases"]
    case_id = queries[0]["id"]
    assert case_id in stored
    assert case_id + ":candidate-" + input_hash({"planner": candidate}) in stored

    partial = copy.deepcopy(report)
    del partial["candidate"]["results"][0]["claim_verdicts"][-1]
    with pytest.raises(ValueError, match="missing or uncommitted"):
        report_gate(measured, partial, queries)
    mismatch = copy.deepcopy(report)
    mismatch["method"]["baseline_prompt_sha256"]["planner"] = "different"
    with pytest.raises(ValueError, match="same models, prompt versions"):
        report_gate(measured, mismatch, queries)
    demo = {**measured, "evidence_kind": "scripted_demo"}
    with pytest.raises(ValueError, match="requires a real Planner experiment"):
        release_candidate(demo)

    exp_file = tmp_path / "experiment.json"
    exp_file.write_text(json.dumps(measured), encoding="utf-8")
    dest = tmp_path / "release"
    assert (
        main(
            [
                "release",
                "--experiment",
                str(exp_file),
                "--report",
                str(out),
                "--out-dir",
                str(dest),
                "--approved-by",
                "reviewer",
            ]
        )
        == 0
    )
    assert read(dest / "apply.json")["preferences"]["prompt_overrides"] == {"planner": candidate}
    assert read(dest / "rollback.json")["preferences"]["prompt_overrides"] == {"planner": baseline}
