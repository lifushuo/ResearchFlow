"""GEPA v3 质量指标：覆盖率硬约束 + 去覆盖率质量分的离线回归。

v3 与 v2 的分界必须由测试钉住：quality 恒等于 v2 综合分去覆盖率后的归一化，
覆盖率不进入优化目标；门禁的覆盖约束独立于质量增益生效。
"""

from __future__ import annotations

import pytest

from evals.agent_eval_ext.planner_evolution.experiment import (
    Case,
    Dataset,
    Observation,
    SearchLimits,
    optimize_planner,
)
from evals.agent_eval_ext.planner_evolution.metrics_v2 import METRIC_VERSION_V2, case_score_v2
from evals.agent_eval_ext.planner_evolution.metrics_v3 import (
    METRIC_VERSION_V3,
    QUALITY_WEIGHTS,
    case_score_v3,
    gate_v3,
    rules_document,
    rules_hash,
)


def _row(
    case_id: str = "c1",
    required: int = 2,
    covered: list[int] | None = None,
    tasks: list[str] | None = None,
    spec: list[int] | None = None,
    plan: list[int] | None = None,
    error: str | None = None,
) -> Observation:
    return Observation(
        case_id=case_id,
        split="train",
        required_count=required,
        task_queries=tasks if tasks is not None else ["task one alpha", "task two beta"],
        covered_indices=covered if covered is not None else [1, 2],
        task_specificity=spec if spec is not None else [2, 1],
        plan_quality=plan if plan is not None else [2, 1],
        error=error,
    )


def _cases() -> list[Case]:
    return [
        Case(
            case_id=f"c{index}",
            group="g",
            split="train",
            query=f"研究两种候选提示词在第 {index} 组固定验证题上的质量差异。",
            depth="balanced",
            required_aspects=["aa", "bb"],
        )
        for index in (1, 2)
    ]


def test_quality_weights_are_v2_renormalized_not_newly_chosen():
    # 0.4/0.4/0.2 必须是 v2 质量权重 (0.2, 0.2, 0.1) 除以剩余权重 0.5；
    # 这是去覆盖率公式的恒等变形，不是一组新设计的权重。
    assert QUALITY_WEIGHTS == {
        "specificity": 0.2 / 0.5,
        "plan_quality": 0.2 / 0.5,
        "redundancy": 0.1 / 0.5,
    }


def test_case_score_v3_quality_excludes_coverage():
    score = case_score_v3(_row())
    assert score is not None
    assert score["coverage"] == 1.0
    assert score["specificity"] == 0.75
    assert score["plan_quality"] == 0.75
    assert score["redundancy"] == 1.0
    expected = round(
        0.4 * 0.75 + 0.4 * 0.75 + 0.2 * 1.0,
        4,
    )
    assert score["quality"] == expected


def test_v3_quality_is_renormalized_v2_composite_without_coverage():
    # 与用户修正后的公式逐值对账：(S_v2 − 0.5·coverage) / 0.5。
    # coverage 取 5/6 这类无限小数，覆盖 double-rounding 的边缘。
    for covered in ([1, 2], [1], [2], []):
        for spec, plan in (([2, 1], [2, 0]), ([1, 0], [1, 1]), ([2, 2], [2, 2])):
            row = _row(covered=covered, spec=spec, plan=plan)
            v2 = case_score_v2(row)
            v3 = case_score_v3(row)
            assert v2 is not None and v3 is not None
            identity = (v2["composite"] - 0.5 * v2["coverage"]) / 0.5
            assert v3["quality"] == pytest.approx(identity, abs=1e-4), (
                f"covered={covered} spec={spec}: {v3['quality']} != {identity}"
            )


def test_same_quality_different_coverage_scores_identically_on_quality():
    # 覆盖率不同的两个观测，质量子分相同时 quality 必须相同——
    # 覆盖率是约束不是目标，搜索不会因为多覆盖一个维度而获得质量分。
    full = _row(covered=[1, 2])
    partial = _row(covered=[1])
    full_score, partial_score = case_score_v3(full), case_score_v3(partial)
    assert full_score and partial_score
    assert full_score["coverage"] != partial_score["coverage"]
    assert full_score["quality"] == partial_score["quality"]


def test_unmeasured_rows_stay_unmeasured():
    assert case_score_v3(_row(error="planner: boom")) is None
    no_fields = _row().model_copy(update={"task_specificity": None, "plan_quality": None})
    assert case_score_v3(no_fields) is None
    bad_length = _row().model_copy(update={"task_specificity": [2]})
    assert case_score_v3(bad_length) is None


def test_gate_v3_passes_non_regressing_candidate_with_quality_gain():
    baseline = [_row("c1"), _row("c2", spec=[1, 1], plan=[1, 1])]
    candidate = [_row("c1", spec=[2, 2], plan=[2, 2]), _row("c2", spec=[2, 2], plan=[2, 2])]
    gate = gate_v3(_cases(), baseline, candidate, min_gain=0.05)
    assert gate["passed"] is True
    assert gate["coverage_delta"] == 0.0
    assert gate["quality_delta"] > 0.05
    assert gate["metric_version"] == METRIC_VERSION_V3


def test_gate_v3_rejects_coverage_regression_despite_large_quality_gain():
    # 覆盖率是硬约束：质量增益再大也不能换回覆盖退化。
    baseline = [_row("c1"), _row("c2")]
    regressed = [
        _row("c1", covered=[1], spec=[2, 2], plan=[2, 2]),
        _row("c2", covered=[1, 2], spec=[2, 2], plan=[2, 2]),
    ]
    gate = gate_v3(_cases(), baseline, regressed, min_gain=0.05)
    assert gate["passed"] is False
    assert "coverage_regression" in gate["reasons"]


def test_gate_v3_rejects_gain_below_threshold_with_full_coverage():
    baseline = [_row("c1"), _row("c2")]
    gate = gate_v3(_cases(), baseline, [_row("c1"), _row("c2")], min_gain=0.05)
    assert gate["passed"] is False
    assert "quality_gain_below_threshold" in gate["reasons"]


def test_gate_v3_fails_closed_on_unmeasured_quality():
    baseline = [_row("c1"), _row("c2")]
    broken = [_row("c1", error="planner: boom"), _row("c2")]
    gate = gate_v3(_cases(), baseline, broken, min_gain=0.05)
    assert gate["passed"] is False
    assert "quality_unmeasured" in gate["reasons"]
    # 未测量时不得给出任何质量分均值（unmeasured ≠ 0）。
    assert gate["quality_baseline"] is None
    assert gate["quality_candidate"] is None
    assert gate["quality_delta"] is None


def test_metric_versions_are_isolated():
    assert METRIC_VERSION_V3 != METRIC_VERSION_V2
    document = rules_document()
    assert document["metric_version"] == METRIC_VERSION_V3
    assert document["coverage_role"] == "hard_constraint"
    # Judge 轴与 v2 同一提示词：v3 改聚合不改评分对象。
    from evals.agent_eval_ext.planner_evolution.judge_prompt import PLANNER_JUDGE_PROMPT_V2

    assert document["judge_prompt"] == PLANNER_JUDGE_PROMPT_V2
    assert rules_hash() == rules_hash()
    assert len(rules_hash()) == 64


def _v3_dataset() -> Dataset:
    return Dataset(
        schema_version=1,
        dataset_id="test",
        provenance="reviewed",
        cases=[
            Case(
                case_id=f"t{i}",
                group="tg",
                split="train",
                query=f"训练题 {i} 的独立描述。",
                depth="balanced",
                required_aspects=["aa", "bb"],
            )
            for i in range(1, 4)
        ]
        + [
            Case(
                case_id=f"v{i}",
                group="vg",
                split="validation",
                query=f"验证题 {i} 的独立描述。",
                depth="balanced",
                required_aspects=["aa", "bb"],
            )
            for i in range(1, 3)
        ]
        + [
            Case(
                case_id="h1",
                group="hg",
                split="holdout",
                query="留出题：比较两种方案在未见场景下的表现。",
                depth="balanced",
                required_aspects=["aa", "bb"],
            ),
        ],
    )


def test_optimize_planner_v3_end_to_end_uses_quality_not_composite() -> None:
    """v3 全链路：GEPA 奖励与候选选择只用 quality，留出集走 gate_v3。

    候选 Prompt 让 rollout 产出更具体的任务与更高的计划质量，覆盖率两臂都满——
    若 v3 仍沿用 v2 综合分，覆盖率满分的两臂增益会被稀释为 0（这正是 v2 实测的
    门禁数学不可达形态），因此该测试同时验证了 v3 修复的就是那个形态。
    """

    def rollout(case: Case, prompt: str | None) -> Observation:
        improved = bool(prompt and "CONCRETE TASKS" in prompt)
        tasks = (
            ["统计两种提示词在固定验证集上的覆盖率与误差率"]
            if improved
            else ["研究该主题的相关背景资料"]
        )
        return Observation(
            case_id=case.case_id,
            split=case.split,
            required_count=len(case.required_aspects),
            task_queries=tasks * 2 if not improved else tasks,
            covered_indices=list(range(1, len(case.required_aspects) + 1)),
            task_specificity=[2] if improved else [0] * len(case.required_aspects),
            plan_quality=[2] if improved else [1] * len(case.required_aspects),
        )

    candidate_prompt = "Improve the planner. CONCRETE TASKS required. " + "x" * 40
    result = optimize_planner(
        _v3_dataset(),
        None,
        rollout,
        lambda _text: f"REVIEW EVERY REQUIRED ASPECT\n{candidate_prompt}",
        SearchLimits(evaluations=64, proposals=2, seed=42, minimum_gain=0.05),
        metric_version="v3",
    )
    gate = result["holdout"]["gate"]
    assert result["candidate"] is not None
    assert gate["passed"] is True
    assert gate["quality_delta"] is not None and gate["quality_delta"] > 0.05
    assert gate["coverage_delta"] == 0.0
    assert gate["metric_version"] == METRIC_VERSION_V3


def test_optimize_planner_v3_preexperiment_never_touches_holdout() -> None:
    """allow_holdout=False：留出集全程零调用，门禁显式记不适用，候选照常产出。

    预实验的数据集文件含留出拆分（哈希与正式实验同源钉死版本），
    由参数而非改数据来排除留出集——留出题在任何阶段不可见。
    """

    split_calls: dict[str, int] = {}

    def rollout(case: Case, prompt: str | None) -> Observation:
        split_calls[case.split] = split_calls.get(case.split, 0) + 1
        improved = bool(prompt and "CONCRETE TASKS" in prompt)
        if improved:
            return Observation(
                case_id=case.case_id,
                split=case.split,
                required_count=len(case.required_aspects),
                task_queries=["统计两种提示词在固定验证集上的覆盖率与误差率"],
                covered_indices=list(range(1, len(case.required_aspects) + 1)),
                task_specificity=[2],
                plan_quality=[2],
            )
        # 基线刻意低质量且任务文本重复（red=0），保证基线分 < 1 触发搜索；
        # 任务数与子分长度必须自洽，否则 fail-closed 跳过搜索。
        return Observation(
            case_id=case.case_id,
            split=case.split,
            required_count=len(case.required_aspects),
            task_queries=["研究该主题的相关背景资料", "研究该主题的相关背景资料"],
            covered_indices=list(range(1, len(case.required_aspects) + 1)),
            task_specificity=[1] * len(case.required_aspects),
            plan_quality=[1] * len(case.required_aspects),
        )

    candidate_prompt = "Improve the planner. CONCRETE TASKS required. " + "x" * 40
    dataset = _v3_dataset()
    holdout_ids = [case.case_id for case in dataset.cases if case.split == "holdout"]
    assert holdout_ids, "fixture must contain a holdout split for this test"
    result = optimize_planner(
        dataset,
        None,
        rollout,
        lambda _text: f"REVIEW EVERY REQUIRED ASPECT\n{candidate_prompt}",
        SearchLimits(evaluations=64, proposals=2, seed=42, minimum_gain=0.05),
        metric_version="v3",
        allow_holdout=False,
    )
    gate = result["holdout"]["gate"]
    assert gate["passed"] is None
    assert gate["reasons"] == ["preexperiment_no_holdout"]
    assert "holdout" not in split_calls, "预实验绝不能调用留出集"
    assert split_calls.get("validation", 0) > 0
    # 候选照常产出为诊断工件；验证选择增益已记录。
    assert result["candidate"] is not None
    assert gate["validation_selected_gain"] is not None


def test_optimize_planner_v3_rejects_candidate_that_trades_coverage_for_quality() -> None:
    """高质量低覆盖的候选在 Validation 接受阶段就被拒绝，绝不进留出集。

    v3 的 GEPA 奖励只含质量分——若覆盖约束只在留出集门禁生效，搜索会先接受
    这种候选并浪费一次性留出集暴露。本测试用 fake rollout 验证：候选在
    accepted_for_holdout 前就被覆盖非退化约束拦下，留出集零调用、候选不产出。
    """

    holdout_calls: list[str] = []

    def rollout(case: Case, prompt: str | None) -> Observation:
        if case.split == "holdout":
            holdout_calls.append(case.case_id)
        improved = bool(prompt and "CONCRETE TASKS" in prompt)
        if improved:
            return Observation(
                case_id=case.case_id,
                split=case.split,
                required_count=len(case.required_aspects),
                task_queries=["统计两种提示词在固定验证集上的覆盖率与误差率"],
                covered_indices=[1],
                task_specificity=[2],
                plan_quality=[2],
            )
        return Observation(
            case_id=case.case_id,
            split=case.split,
            required_count=len(case.required_aspects),
            task_queries=["研究该主题的相关背景资料", "研究该主题的相关背景资料"],
            covered_indices=list(range(1, len(case.required_aspects) + 1)),
            task_specificity=[0] * len(case.required_aspects),
            plan_quality=[1] * len(case.required_aspects),
        )

    candidate_prompt = "Improve the planner. CONCRETE TASKS required. " + "x" * 40
    result = optimize_planner(
        _v3_dataset(),
        None,
        rollout,
        lambda _text: f"REVIEW EVERY REQUIRED ASPECT\n{candidate_prompt}",
        SearchLimits(evaluations=64, proposals=2, seed=42, minimum_gain=0.05),
        metric_version="v3",
    )
    gate = result["holdout"]["gate"]
    assert gate["passed"] is False
    assert "validation_coverage_regression" in gate["reasons"]
    assert gate["reasons"] == ["validation_coverage_regression"]
    # 候选未被接受：不产出候选，留出集零调用（一次性暴露未被消耗）。
    assert result["candidate"] is None
    assert result["holdout"]["baseline"] == []
    assert result["holdout"]["candidate"] == []
    assert holdout_calls == []


def test_optimize_planner_v2_acceptance_also_requires_validation_coverage() -> None:
    """v2 的综合分含覆盖率但可被质量增益抵消——接受阶段同样复核覆盖非退化。"""

    def rollout(case: Case, prompt: str | None) -> Observation:
        improved = bool(prompt and "CONCRETE TASKS" in prompt)
        if improved:
            return Observation(
                case_id=case.case_id,
                split=case.split,
                required_count=len(case.required_aspects),
                task_queries=["统计两种提示词在固定验证集上的覆盖率与误差率"],
                covered_indices=[1],
                task_specificity=[2],
                plan_quality=[2],
            )
        return Observation(
            case_id=case.case_id,
            split=case.split,
            required_count=len(case.required_aspects),
            task_queries=["研究该主题的相关背景资料", "研究该主题的相关背景资料"],
            covered_indices=list(range(1, len(case.required_aspects) + 1)),
            task_specificity=[0] * len(case.required_aspects),
            plan_quality=[1] * len(case.required_aspects),
        )

    candidate_prompt = "Improve the planner. CONCRETE TASKS required. " + "x" * 40
    result = optimize_planner(
        _v3_dataset(),
        None,
        rollout,
        lambda _text: f"REVIEW EVERY REQUIRED ASPECT\n{candidate_prompt}",
        SearchLimits(evaluations=64, proposals=2, seed=42, minimum_gain=0.05),
        metric_version="v2",
    )
    gate = result["holdout"]["gate"]
    assert gate["passed"] is False
    assert "validation_coverage_regression" in gate["reasons"]
    assert result["candidate"] is None
    assert result["holdout"]["baseline"] == []
