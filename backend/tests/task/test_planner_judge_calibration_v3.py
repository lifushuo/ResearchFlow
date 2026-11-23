"""Planner Judge v3 校准协议的离线回归：聚合数学、fail-closed 与调用计划。

不发起任何模型调用；真实调用路径（--execute）需要 consent，
本文件只验证计划打印、刺激校验与 summarize 聚合。
"""

from __future__ import annotations

import hashlib
import json

import pytest

from evals.scripts.run_planner_judge_calibration_v3 import (
    DEFAULT_STIMULI,
    call_plan,
    load_stimuli,
    summarize,
)

STIMULI_PATH = DEFAULT_STIMULI


_TASKS = [
    "统计两种方案在固定基准上的召回率与误差分布",
    "测量不同配置下的 token 消耗与延迟中位数",
    "核对评审记录中的哈希与审计事件完整性",
]
_VAGUE_TASKS = ["了解相关领域的背景资料", "研究一般性的现状信息", "查找相关主题的介绍材料"]


def _row(case_id: str, variant: str, n: int, spec: int, plan: int) -> dict:
    """构造内部一致的评分行，形状与真实 Judge 投影一致。

    plan_quality 按必答维度逐条（未覆盖维度记 0）；完整变体的任务文本互不相似，
    避免冗余度惩罚人为压低完整计划的质量分。
    """
    task_count = n - 1 if variant == "missing" else n
    queries = (_VAGUE_TASKS if variant == "vague" else _TASKS)[:task_count]
    covered = task_count if variant != "missing" else n - 1
    plan_by_aspect = [plan] * task_count + ([0] if variant == "missing" else [])
    return {
        "case_id": case_id,
        "split": "train",
        "required_count": n,
        "task_queries": queries,
        "covered_indices": list(range(1, covered + 1)),
        "task_specificity": [spec] * task_count,
        "plan_quality": plan_by_aspect,
        "served_judge": None,
        "variant": variant,
        "repeat": 1 if variant == "complete" else None,
        "error": None,
    }


def test_suggestion_projection_feeds_case_score_v3():
    """suggest_planner_coverage_v2 的返回形状必须能投影为三轴可测的评分输入。

    缺漏维度在投影中得到 plan_quality=0（与实验契约一致），case_score_v3
    据此给出三轴与 quality——这是"Judge 实际输出可计算"的离线证明。
    """
    from evals.scripts.run_planner_judge_calibration_v3 import _observation_fields

    case = load_stimuli(STIMULI_PATH)["cases"][1]  # 3 维度题
    suggestion = {
        "status": "suggested",
        "coverage": 2 / 3,
        "aspects": [
            {
                "aspect_index": 1,
                "aspect": case["required_aspects"][0],
                "covered": True,
                "task_ids": [1],
                "plan_quality": 2,
            },
            {
                "aspect_index": 2,
                "aspect": case["required_aspects"][1],
                "covered": True,
                "task_ids": [2],
                "plan_quality": 2,
            },
            {
                "aspect_index": 3,
                "aspect": case["required_aspects"][2],
                "covered": False,
                "task_ids": [],
                "plan_quality": 0,
            },
        ],
        "task_specificity": [
            {
                "task_id": 1,
                "task_query": case["variants"]["missing"]["task_queries"][0],
                "specificity": 2,
            },
            {
                "task_id": 2,
                "task_query": case["variants"]["missing"]["task_queries"][1],
                "specificity": 1,
            },
        ],
        "automatic_label": False,
    }
    fields = _observation_fields(
        case=case,
        variant="missing",
        repeat=None,
        task_queries=case["variants"]["missing"]["task_queries"],
        suggestion=suggestion,
    )
    from evals.agent_eval_ext.planner_evolution.experiment import Observation
    from evals.agent_eval_ext.planner_evolution.metrics_v3 import case_score_v3

    scoring_input = {k: v for k, v in fields.items() if k not in ("variant", "repeat")}
    score = case_score_v3(Observation.model_validate(scoring_input))
    assert score is not None
    assert score["coverage"] == round(2 / 3, 4)
    assert score["specificity"] == 0.75
    assert score["plan_quality"] == round((2 + 2 + 0) / 3 / 2, 4)
    assert 0.0 < score["quality"] < 1.0


def test_shipped_stimuli_fixture_loads_and_has_three_variants():
    stimuli = load_stimuli(STIMULI_PATH)
    assert len(stimuli["cases"]) == 3
    for case in stimuli["cases"]:
        assert set(case["variants"]) == {"complete", "missing", "vague"}
        # 缺漏变体的预期未覆盖维度必须真的比完整变体少一条任务。
        complete = case["variants"]["complete"]["task_queries"]
        missing = case["variants"]["missing"]["task_queries"]
        assert len(missing) == len(complete) - 1
        assert case["variants"]["missing"]["expected"]["uncovered_aspects"]


def test_load_stimuli_rejects_wrong_purpose(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"purpose": "something_else", "cases": []}), encoding="utf-8")
    with pytest.raises(ValueError, match="wrong_stimuli_purpose"):
        load_stimuli(path)


def test_call_plan_counts_repeats_and_single_shot_variants():
    stimuli = load_stimuli(STIMULI_PATH)
    plan = call_plan(stimuli, repeats=3)
    assert len(plan) == len(stimuli["cases"]) * 5
    complete = [p for p in plan if p["variant"] == "complete"]
    assert len(complete) == len(stimuli["cases"]) * 3
    assert all(p["repeat"] is None for p in plan if p["variant"] != "complete")


def test_summarize_detects_discrimination_on_ideal_judge():
    stimuli = load_stimuli(STIMULI_PATH)
    observations = []
    for case in stimuli["cases"]:
        n = len(case["required_aspects"])
        # 两次完全一致的完整计划评分 → 重复性 MAE = 0。
        observations.extend([_row(case["case_id"], "complete", n, 2, 2)] * 2)
        observations.append(_row(case["case_id"], "missing", n, 2, 2))
        observations.append(_row(case["case_id"], "vague", n, 0, 1))
    report = summarize(stimuli, observations)
    assert report["discrimination"]["coverage"]["directional_win_rate"] == 1.0
    assert report["discrimination"]["specificity"]["directional_win_rate"] == 1.0
    assert (
        report["discrimination"]["coverage"]["missing_mean"]
        < report["discrimination"]["coverage"]["complete_mean"]
    )
    # 构造预期全部符合：完整=满分、缺漏=指定维度未覆盖、含混=低具体性。
    for variant in ("complete", "missing", "vague"):
        assert report["expected_outcome_conformance"][variant]["passed"] == 3
        assert report["expected_outcome_conformance"][variant]["unmeasured"] == 0
    # 完全一致的重复评分 → 质量轴 MAE = 0 → 噪声下界落在 0.01。
    assert report["repeatability"]["per_axis_mae"]["quality"] == 0.0
    assert report["gate_reference_min_gain"]["noise_lower_bound"] == 0.01
    # 质量分排序：完整 > 缺漏 与 完整 > 含混 的方向胜率。
    assert report["discrimination"]["quality"]["complete_gt_missing_win_rate"] == 1.0
    assert report["discrimination"]["quality"]["complete_gt_vague_win_rate"] == 1.0
    assert report["unmeasured"]["case_comparisons"] == 0
    assert report["unmeasured"]["observations"] == 0


def test_summarize_fails_closed_on_unmeasured_judge_calls():
    stimuli = load_stimuli(STIMULI_PATH)
    observations = []
    for case in stimuli["cases"]:
        n = len(case["required_aspects"])
        observations.append(_row(case["case_id"], "complete", n, 2, 2))
        observations.append(
            {
                "case_id": case["case_id"],
                "split": "train",
                "required_count": n,
                "task_queries": ["任一任务描述"],
                "covered_indices": [],
                "task_specificity": None,
                "plan_quality": None,
                "served_judge": None,
                "variant": "missing",
                "repeat": None,
                "error": "judge: TimeoutError",
            }
        )
        observations.append(_row(case["case_id"], "vague", n, 0, 1))
    report = summarize(stimuli, observations)
    # 缺漏臂全部未测量：覆盖率对照不可测量，不按 0 分计。
    assert report["discrimination"]["coverage"]["directional_win_rate"] is None
    assert report["discrimination"]["coverage"]["comparisons_measured"] == 0
    assert report["discrimination"]["specificity"]["directional_win_rate"] == 1.0
    # 每题缺漏臂未测量 → 覆盖率与质量分两组对照各 1 个不可测比较，共 6 个。
    assert report["unmeasured"]["case_comparisons"] == 6
    assert report["unmeasured"]["observations"] == 3
    assert report["discrimination"]["quality"]["complete_gt_missing_win_rate"] is None
    assert report["discrimination"]["quality"]["complete_gt_vague_win_rate"] == 1.0
    # 未测量在符合率中单独披露，绝不混入「不满足」。
    missing_conf = report["expected_outcome_conformance"]["missing"]
    assert missing_conf == {"checked": 0, "passed": 0, "unmeasured": 3}


def test_summarize_flags_vague_judge_that_accepts_vague_tasks():
    """含混任务被 Judge 评高分 = 判别失败证据：符合率必须下降而不是掩盖。"""
    stimuli = load_stimuli(STIMULI_PATH)
    observations = []
    for case in stimuli["cases"]:
        n = len(case["required_aspects"])
        observations.append(_row(case["case_id"], "complete", n, 2, 2))
        observations.append(_row(case["case_id"], "missing", n, 2, 2))
        # 含混任务仍被评 2 分——判别失败（已测量但不满足预期）。
        observations.append(_row(case["case_id"], "vague", n, 2, 2))
    report = summarize(stimuli, observations)
    vague_conf = report["expected_outcome_conformance"]["vague"]
    assert vague_conf["checked"] == 3
    assert vague_conf["passed"] == 0
    assert vague_conf["unmeasured"] == 0
    assert report["discrimination"]["specificity"]["directional_win_rate"] == 0.0
    # 含混计划与完整计划质量分打平（都满分）——排序胜率同样归零，不掩盖判别失败。
    assert report["discrimination"]["quality"]["complete_gt_vague_win_rate"] == 0.0


def test_plan_mode_prints_calls_without_consent_or_network(capsys):
    from evals.scripts.run_planner_judge_calibration_v3 import main

    exit_code = main([])
    assert exit_code == 0
    output = capsys.readouterr().out
    # 15 次 = 9 个基础刺激 + 6 次完整计划重复评分；每类刺激的人工预期逐条披露。
    assert "15 次 Judge 调用 = 9 个基础刺激 + 6 次完整计划重复评分" in output
    assert "预期: coverage=1.0" in output
    assert "维度 [2] 未被覆盖" in output
    assert "维度 [3] 未被覆盖" in output
    assert "specificity均值≤0.5" in output
    assert "未测量" in output
    assert "I_CONFIRM_JUDGE_API_COST" in output


def test_execute_requires_consent(tmp_path, capsys):
    from evals.scripts.run_planner_judge_calibration_v3 import main

    exit_code = main(["--execute", "--out", str(tmp_path / "evidence.json")])
    assert exit_code == 2
    assert "I_CONFIRM_JUDGE_API_COST" in capsys.readouterr().err
    assert not (tmp_path / "evidence.json").exists()


# ---------- Fake Judge 故障分类（零网络；错误归类正确、失败始终未测量） ----------
def _stub_suggestion(**kwargs):
    return {
        "status": "suggested",
        "coverage": 1.0,
        "aspects": [
            {
                "aspect_index": 1,
                "aspect": "维度一",
                "covered": True,
                "task_ids": [1],
                "plan_quality": 2,
            },
            {
                "aspect_index": 2,
                "aspect": "维度二",
                "covered": True,
                "task_ids": [2],
                "plan_quality": 2,
            },
        ],
        "task_specificity": [
            {"task_id": 1, "task_query": "任务一", "specificity": 2},
            {"task_id": 2, "task_query": "任务二", "specificity": 2},
        ],
        "judge_tokens": (120, 60),
        "served_judge": "qwen4-max",
        "automatic_label": False,
    }


def _run_execute(monkeypatch, behaviors: dict[str, Exception], variants: list[str]):
    """以桩替 suggest_planner_coverage_v2 跑 _execute，按调用序分派行为。"""
    from app.services import planner_review_judge as service
    from evals.scripts.run_planner_judge_calibration_v3 import _execute, load_stimuli

    stimuli = load_stimuli(STIMULI_PATH)
    case = stimuli["cases"][0]
    plan = [
        {"case_id": case["case_id"], "variant": v, "repeat": 1 if v == "complete" else None}
        for v in variants
    ]

    calls = {"n": 0}
    ordered = list(behaviors.values())

    async def fake_suggest_ordered(**kwargs):
        index = min(calls["n"], len(ordered) - 1)
        calls["n"] += 1
        behavior = ordered[index]
        if behavior is None:
            return _stub_suggestion()
        raise behavior

    monkeypatch.setattr(service, "suggest_planner_coverage_v2", fake_suggest_ordered)
    return _execute(stimuli, plan, timeout=90)


def test_execute_records_latency_and_tokens_for_measured_calls(monkeypatch):
    rows = _run_execute(monkeypatch, {"complete": None}, ["complete"])
    assert len(rows) == 1
    row = rows[0]
    assert row["error"] is None
    assert row["judge_tokens"] == (120, 60)
    assert row["served_judge"] == "qwen4-max"
    assert row["latency_seconds"] >= 0
    assert row["run_source"].startswith("planner-judge-v3-calibration-")
    assert row["covered_indices"] == [1, 2]


def test_execute_classifies_timeout_rate_limit_and_runtime_errors(monkeypatch):
    from evals.agent_eval_ext.planner_evolution.experiment import Observation
    from evals.scripts.run_planner_judge_calibration_v3 import _OBSERVATION_FIELDS

    class FakeRateLimitError(Exception):
        status_code = 429
        request_id = "req-fake-429"

    behaviors = {
        "complete": TimeoutError("request timed out"),
        "missing": RuntimeError("Event loop is closed"),
        "vague": FakeRateLimitError("Requests rate limit exceeded"),
    }
    rows = _run_execute(monkeypatch, behaviors, ["complete", "missing", "vague"])
    assert [r["error_stage"] for r in rows] == ["http_timeout", "runtime_or_unknown", "http_status"]
    for row in rows:
        # 失败保持未测量：子分全空，绝不补造评分。
        assert row["error"] is not None
        assert row["task_specificity"] is None and row["plan_quality"] is None
        assert row["covered_indices"] == []
        assert row["judge_tokens"] is None
        assert row["latency_seconds"] >= 0
        # 归类字段可安全并入 Observation（fail-closed 语义不变）。
        Observation.model_validate({k: v for k, v in row.items() if k in _OBSERVATION_FIELDS})
    # RuntimeError 的 cause 链被记录（脱敏后）。
    runtime_row = rows[1]
    assert "RuntimeError" in runtime_row["error"]
    assert runtime_row["error_status"] is None
    rate_row = rows[2]
    assert rate_row["error_status"] == 429
    assert rate_row["error_request_id"] == "req-fake-429"


def test_execute_records_error_cause_chain(monkeypatch):
    from evals.scripts.run_planner_judge_calibration_v3 import _execute, load_stimuli

    stimuli = load_stimuli(STIMULI_PATH)
    plan = [{"case_id": stimuli["cases"][0]["case_id"], "variant": "complete", "repeat": 1}]

    async def failing_suggest(**kwargs):
        try:
            raise ValueError("inner transport closed")
        except ValueError as inner:
            raise RuntimeError("Event loop is closed") from inner

    from app.services import planner_review_judge as service

    monkeypatch.setattr(service, "suggest_planner_coverage_v2", failing_suggest)
    rows = _execute(stimuli, plan, timeout=90)
    row = rows[0]
    assert row["error_stage"] == "runtime_or_unknown"
    assert row["error_causes"] and "ValueError" in row["error_causes"][0]
    assert row["task_specificity"] is None


def test_resume_from_patches_only_unmeasured_calls(tmp_path, monkeypatch, capsys):
    """首轮 8/15 → 补测 7 个未测量调用；合并文件逐条标注来源，首轮文件不动。"""
    import json as jsonlib

    from evals.scripts.run_planner_judge_calibration_v3 import load_stimuli

    stimuli = load_stimuli(STIMULI_PATH)
    stimuli_sha = hashlib.sha256(
        jsonlib.dumps(stimuli, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()
    # 构造首轮证据：8 测量 + 7 失败（与真实 r1 同构）。
    prior_rows = []
    measured_variants = {
        ("cal-rag-vs-lc", "complete", 1),
        ("cal-rag-vs-lc", "complete", 2),
        ("cal-rag-vs-lc", "vague", None),
        ("cal-planner-evolution", "complete", 1),
        ("cal-planner-evolution", "missing", None),
        ("cal-benchmark-report", "complete", 1),
        ("cal-benchmark-report", "complete", 2),
        ("cal-benchmark-report", "vague", None),
    }
    from evals.scripts.run_planner_judge_calibration_v3 import call_plan

    for item in call_plan(stimuli, repeats=3):
        key = (item["case_id"], item["variant"], item["repeat"])
        if key in measured_variants:
            prior_rows.append(
                {
                    "case_id": item["case_id"],
                    "split": "train",
                    "required_count": 2,
                    "task_queries": ["任务一的具体描述", "任务二的具体描述"],
                    "covered_indices": [1, 2],
                    "task_specificity": [2, 2],
                    "plan_quality": [2, 2],
                    "served_judge": "qwen4-max",
                    "variant": item["variant"],
                    "repeat": item["repeat"],
                    "error": None,
                }
            )
        else:
            prior_rows.append(
                {
                    "case_id": item["case_id"],
                    "split": "train",
                    "required_count": 2,
                    "task_queries": ["任务"],
                    "covered_indices": [],
                    "task_specificity": None,
                    "plan_quality": None,
                    "served_judge": None,
                    "variant": item["variant"],
                    "repeat": item["repeat"],
                    "error": "judge: RuntimeError",
                }
            )
    prior_path = tmp_path / "r1.json"
    prior_path.write_text(
        jsonlib.dumps(
            {
                "schema_version": 1,
                "purpose": "planner_judge_v3_calibration_evidence",
                "created_at": "2026-11-22T16:04:01+00:00",
                "stimuli_sha256": stimuli_sha,
                "observations": prior_rows,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    prior_bytes = prior_path.read_bytes()

    from app.services import planner_review_judge as service

    call_count = {"n": 0}

    async def patch_suggest(**kwargs):
        # 桩按真实变体的任务数/维度数返回自洽建议（机制测试不关心判别数值）。
        call_count["n"] += 1
        tasks = kwargs["evidence"]["tasks"]
        return {
            "status": "suggested",
            "coverage": 1.0,
            "aspects": [
                {
                    "aspect_index": index,
                    "aspect": aspect,
                    "covered": True,
                    "task_ids": [1],
                    "plan_quality": 2,
                }
                for index, aspect in enumerate(kwargs["aspects"], 1)
            ],
            "task_specificity": [
                {"task_id": index, "task_query": row["query"], "specificity": 2}
                for index, row in enumerate(tasks, 1)
            ],
            "judge_tokens": (120, 60),
            "served_judge": "qwen4-max",
            "automatic_label": False,
        }

    monkeypatch.setattr(service, "suggest_planner_coverage_v2", patch_suggest)
    monkeypatch.setattr(service, "configured_judge_route", lambda: "custom:qwen4-max")
    out = tmp_path / "r2.json"
    from evals.scripts.run_planner_judge_calibration_v3 import main

    exit_code = main(
        [
            "--execute",
            "--consent",
            "I_CONFIRM_JUDGE_API_COST",
            "--resume-from",
            str(prior_path),
            "--out",
            str(out),
        ]
    )
    assert exit_code == 0, capsys.readouterr().out
    # 只补测 7 个未测量逻辑调用，不重复 8 个已测量判定。
    assert call_count["n"] == 7
    merged = jsonlib.loads(out.read_text(encoding="utf-8"))
    assert len(merged["observations"]) == 7
    assert len(merged["prior_observations"]) == 15
    assert all(row["run_source"].startswith("prior:") for row in merged["prior_observations"])
    assert all(
        row["run_source"].startswith("planner-judge-v3-calibration-")
        for row in merged["observations"]
    )
    assert merged["merged_totals"] == {
        "prior_measured": 8,
        "new_executed": 7,
        "new_measured": 7,
        "combined_measured": 15,
    }
    # 合并摘要描述最终证据状态：同一逻辑调用取最新判定 → 全部 15 个逻辑调用可测量。
    assert merged["merged_final_state_rows"] == 15
    assert merged["summary"]["unmeasured"]["observations"] == 0
    # 合并后的质量排序对照在 3 道题上全部可测（真实判别数值来自真实证据文件）。
    assert merged["summary"]["discrimination"]["quality"]["comparisons_measured"]["vs_missing"] == 3
    assert (
        merged["summary"]["discrimination"]["quality"]["complete_gt_missing_win_rate"] is not None
    )
    assert "PATCHES a prior run" in merged["warning"]
    # 首轮证据文件字节级未动。
    assert prior_path.read_bytes() == prior_bytes


def test_resume_from_refuses_foreign_stimuli(tmp_path):
    import json as jsonlib

    import pytest as pytest_module

    from evals.scripts.run_planner_judge_calibration_v3 import main

    prior_path = tmp_path / "foreign.json"
    prior_path.write_text(
        jsonlib.dumps(
            {
                "schema_version": 1,
                "purpose": "planner_judge_v3_calibration_evidence",
                "created_at": "2026-11-22T16:04:01+00:00",
                "stimuli_sha256": "0" * 64,
                "observations": [],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    out = tmp_path / "r2.json"
    # 刺激集不同源时 main 按 CLI 契约抛 ValueError（__main__ 统一转退出码 2）。
    with pytest_module.raises(ValueError, match="prior_evidence_stimuli_mismatch"):
        main(
            [
                "--execute",
                "--consent",
                "I_CONFIRM_JUDGE_API_COST",
                "--resume-from",
                str(prior_path),
                "--out",
                str(out),
            ]
        )
    assert not out.exists()
