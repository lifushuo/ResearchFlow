"""发布门禁与保存故障的反例回归；不调用真实模型。"""

from __future__ import annotations

import copy
import json
import sys
from dataclasses import replace

import pytest
from test_planner_evolution import _measured_experiment

from evals import harness
from evals.agent_eval_ext.closure_observer import read
from evals.agent_eval_ext.planner_evolution import __main__ as cli
from evals.agent_eval_ext.planner_evolution import experiment, runtime
from evals.agent_eval_ext.planner_evolution.release import release_candidate, report_gate


@pytest.fixture
def measured():
    dataset = experiment.Dataset.model_validate(read(cli.DEMO_DATASET))
    return _measured_experiment(dataset)


@pytest.fixture
async def report(monkeypatch, tmp_path, measured):
    import test_long_run_reliability as fixtures

    before, after = tmp_path / "baseline.json", tmp_path / "candidate.json"
    before.write_text(json.dumps({"planner": measured["baseline"]["override"]}), encoding="utf-8")
    after.write_text(json.dumps({"planner": measured["candidate"]["override"]}), encoding="utf-8")
    out = tmp_path / "report.json"
    monkeypatch.setattr(harness, "RUN_CONFIG", replace(harness.RUN_CONFIG, llm_mode="real"))
    monkeypatch.setattr(harness, "JUDGE_ROUTE", None)
    monkeypatch.setattr(harness, "build_graph", lambda *_: fixtures.FakeGraph())
    monkeypatch.setattr(harness, "_judge_llm", lambda: fixtures.ScriptedJudge())
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
            str(tmp_path / "progress"),
            "--out",
            str(out),
        ],
    )
    await harness.main()
    return read(out)


def gate(measured, report):
    return report_gate(measured, report, read(cli.REPORT_DATASET)["queries"])


@pytest.mark.parametrize("completed", [1, 0, "true", "false", None, [], {}])
async def test_malformed_completion_cannot_skip_judgment_checks(measured, report, completed):
    row = report["candidate"]["results"][0]
    row.update(completed=completed, report="", judge_status="judge_failed")
    with pytest.raises(ValueError, match="strict boolean"):
        gate(measured, report)


@pytest.mark.parametrize(
    "field,value",
    [
        ("citation_support_rate", True),
        ("citation_support_rate", float("nan")),
        ("citation_support_rate", "1.0"),
        ("report", None),
        ("claim_verdicts", None),
        ("claim_verdicts", [None]),
    ],
)
async def test_completed_report_requires_strict_evidence(measured, report, field, value):
    report["candidate"]["results"][0][field] = value
    with pytest.raises(ValueError):
        gate(measured, report)


@pytest.mark.parametrize("field,value", [("judged", 1), ("supported", 1), ("supported", "YES")])
async def test_verdict_flags_are_real_booleans(measured, report, field, value):
    report["candidate"]["results"][0]["claim_verdicts"][0][field] = value
    with pytest.raises(ValueError, match="Claim judgments"):
        gate(measured, report)


async def test_gateway_fallback_to_pipeline_model_blocks_release(monkeypatch, measured, report):
    from langchain_core.messages import AIMessage
    from test_long_run_reliability import REPORT, SOURCES

    # 使用真实 Judge 解析入口；不同配置路由不能掩盖响应披露的研究模型身份。
    model = measured["routes"]["pipeline"]["planner"].partition(":")[2]

    class FallbackJudge:
        async def ainvoke(self, messages):
            return AIMessage(content="Claim 1: YES", response_metadata={"model_name": model})

    monkeypatch.setattr(harness, "_judge_llm", FallbackJudge)
    _, verdicts = await harness.judge_citation_support(REPORT, SOURCES)
    assert verdicts[0]["judged_by"] == model
    for arm in (report["results"], report["candidate"]["results"]):
        for row in arm:
            for verdict in row["claim_verdicts"]:
                verdict["judged_by"] = model
    for arm in ("baseline", "candidate"):
        report["judge"][arm]["served_models"] = [model]
    with pytest.raises(ValueError, match="cannot also be its judge"):
        gate(measured, report)


@pytest.mark.parametrize("served", [None, "", "   "])
async def test_unknown_actual_judge_never_clears_release(measured, report, served):
    report["candidate"]["results"][0]["claim_verdicts"][0]["judged_by"] = served
    with pytest.raises(ValueError, match="unverified"):
        gate(measured, report)


async def test_judge_summary_is_rebuilt_from_verdicts(measured, report):
    report["judge"]["candidate"]["served_models"] = ["some-other-independent-model"]
    with pytest.raises(ValueError, match="individual Claim"):
        gate(measured, report)


@pytest.mark.parametrize("field", ["task_matches", "served_judge"])
def test_live_holdout_cannot_omit_evidence(measured, field):
    measured["holdout"]["candidate"][0][field] = None
    with pytest.raises(ValueError, match="missing_task_evidence|judge_identity_unverified"):
        release_candidate(measured)


def test_live_holdout_rejects_wrong_mapping(measured):
    measured["holdout"]["candidate"][0]["task_matches"][0]["task_ids"] = []
    with pytest.raises(ValueError, match="task evidence"):
        release_candidate(measured)


def test_live_holdout_rejects_self_judge(measured):
    for arm in ("baseline", "candidate"):
        for row in measured["holdout"][arm]:
            row["served_judge"] = measured["routes"]["pipeline"]["planner"].partition(":")[2]
    with pytest.raises(ValueError, match="cannot also be its judge"):
        release_candidate(measured)


def test_unwritable_output_stops_before_any_rollout(monkeypatch, tmp_path):
    monkeypatch.setattr(cli, "version", lambda _: "0.1.4")

    def cannot_write(*_):
        raise OSError("simulated file quota")

    def forbidden(*_, **__):
        pytest.fail("optimizer must not run before a successful file write")

    monkeypatch.setattr(cli, "atomic_create", cannot_write)
    monkeypatch.setattr(cli, "optimize_planner", forbidden)
    out = tmp_path / "experiment"
    assert cli.main(["optimize", "--demo", "--out-dir", str(out)]) == 2
    assert out.is_dir() and not list(out.iterdir())


def test_reflection_failure_preserves_completed_observations(monkeypatch, tmp_path):
    pytest.importorskip("gepa")
    dataset = read(cli.DEMO_DATASET)
    dataset["provenance"] = "reviewed"
    source = tmp_path / "dataset.json"
    source.write_text(json.dumps(dataset), encoding="utf-8")
    calls = []
    reflections = []

    class FailingRuntime:
        def __init__(self, *_, **__):
            pass

        def checked_text(self, text):
            return text

        async def observe(self, case, prompt):
            calls.append(case.case_id)
            return experiment.demo_rollout(case, prompt)

        async def reflect(self, _):
            reflections.append(1)
            raise ValueError("simulated reflection failure")

    routes = {
        "pipeline": dict(harness.RUN_CONFIG.models),
        "judge": "custom:independent",
        "reflection": "custom:planner",
    }
    monkeypatch.setattr(cli, "_live_routes", lambda *_: (harness.RUN_CONFIG, routes))
    monkeypatch.setattr(runtime, "LiveRuntime", FailingRuntime)
    out = tmp_path / "experiment"
    assert (
        cli.main(
            [
                "optimize",
                "--live",
                "--dataset",
                str(source),
                "--out-dir",
                str(out),
                "--proposals",
                "1",
            ]
        )
        == 2
    )
    events = [read(path) for path in sorted((out / "journal").glob("*.json"))]
    observations = [event for event in events if event["kind"] == "observation"]
    assert len(observations) == len(calls) > 0
    assert len(reflections) == 1
    assert events[-1]["kind"] == "interrupted"
    assert events[-1]["payload"] == {
        "error_type": "ValueError",
        "role": None,
        "slot_id": None,
        "cause_type": None,
    }
    assert not (out / "experiment.json").exists()


def test_journal_write_failure_prevents_next_model_call():
    pytest.importorskip("gepa")
    dataset = experiment.Dataset.model_validate(read(cli.DEMO_DATASET))
    calls = []

    def rollout(case, prompt):
        calls.append(case.case_id)
        return experiment.demo_rollout(case, prompt)

    def fail_record(*_):
        raise OSError("simulated journal write failure")

    with pytest.raises(OSError, match="journal write"):
        experiment.optimize_planner(
            dataset,
            None,
            rollout,
            experiment.demo_reflection,
            experiment.SearchLimits(40, 1, 42, 0.05),
            record=fail_record,
        )
    assert len(calls) == 1


def test_demo_runs_real_gepa_and_keeps_release_blocked(tmp_path):
    pytest.importorskip("gepa")
    out = tmp_path / "demo"
    assert cli.main(["optimize", "--demo", "--out-dir", str(out), "--proposals", "1"]) == 0
    result = read(out / "experiment.json")
    assert result["evidence_kind"] == "scripted_demo"
    assert result["holdout"]["gate"]["passed"] is True
    assert read(out / "candidate.json")["planner"] == result["candidate"]["override"]
    events = [read(path) for path in sorted((out / "journal").glob("*.json"))]
    assert events[-1]["kind"] == "completed"
    assert any(event["kind"] == "candidate_frozen" for event in events)
    with pytest.raises(ValueError, match="real Planner"):
        release_candidate(copy.deepcopy(result))
