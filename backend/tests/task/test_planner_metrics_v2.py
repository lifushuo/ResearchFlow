"""GEPA v2 综合质量指标：评分规则、门禁与版本隔离的离线回归。"""

from __future__ import annotations

import pytest

from evals.agent_eval_ext.planner_evolution.experiment import (
    METRIC_VERSION,
    Case,
    Dataset,
    Observation,
    SearchLimits,
    optimize_planner,
)
from evals.agent_eval_ext.planner_evolution.metrics_v2 import (
    METRIC_VERSION_V2,
    case_score_v2,
    gate_v2,
    redundancy_score,
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
    served: str | None = None,
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
        served_judge=served,
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


def test_case_score_v2_composite_follows_frozen_weights():
    row = _row()
    score = case_score_v2(row)
    assert score is not None
    assert score["coverage"] == 1.0
    assert score["specificity"] == 0.75
    assert score["plan_quality"] == 0.75
    assert score["redundancy"] == 1.0
    assert score["composite"] == round(0.5 * 1.0 + 0.2 * 0.75 + 0.2 * 0.75 + 0.1 * 1.0, 4)


def test_unmeasured_rows_stay_unmeasured():
    assert case_score_v2(_row(error="planner: boom")) is None
    no_fields = _row().model_copy(update={"task_specificity": None, "plan_quality": None})
    assert case_score_v2(no_fields) is None
    bad_length = _row().model_copy(update={"task_specificity": [2]})
    assert case_score_v2(bad_length) is None


def test_redundancy_is_deterministic():
    assert redundancy_score(["完全相同的任务描述", "完全相同的任务描述"]) == 0.0
    assert redundancy_score(["检索分布延迟", "核对幂等语义"]) == 1.0
    assert redundancy_score(["只有一条任务"]) == 1.0


def test_gate_v2_passes_non_regressing_candidate_with_quality_gain():
    baseline = [_row("c1"), _row("c2", spec=[1, 1], plan=[1, 1])]
    candidate = [_row("c1", spec=[2, 2], plan=[2, 2]), _row("c2", spec=[2, 2], plan=[2, 2])]
    gate = gate_v2(_cases(), baseline, candidate, min_gain=0.05)
    assert gate["passed"] is True
    assert gate["coverage_delta"] == 0.0
    assert gate["composite_delta"] > 0.05


def test_gate_v2_rejects_coverage_regression_and_missing_gain():
    baseline = [_row("c1"), _row("c2")]
    regressed = [
        _row("c1", covered=[1], spec=[2, 2], plan=[2, 2]),
        _row("c2", covered=[1, 2], spec=[2, 2], plan=[2, 2]),
    ]
    gate = gate_v2(_cases(), baseline, regressed, min_gain=0.05)
    assert gate["passed"] is False
    assert "coverage_regression" in gate["reasons"]

    no_gain = [_row("c1"), _row("c2")]
    gate = gate_v2(_cases(), baseline, no_gain, min_gain=0.05)
    assert gate["passed"] is False
    assert "composite_gain_below_threshold" in gate["reasons"]


def test_gate_v2_fails_closed_on_unmeasured_quality():
    baseline = [_row("c1"), _row("c2")]
    broken = [_row("c1", error="planner: boom"), _row("c2")]
    gate = gate_v2(_cases(), baseline, broken, min_gain=0.05)
    assert gate["passed"] is False
    assert "quality_unmeasured" in gate["reasons"]
    # 未测量时不得给出任何综合分均值（unmeasured ≠ 0）。
    assert gate["composite_baseline"] is None
    assert gate["composite_candidate"] is None
    assert gate["composite_delta"] is None


def test_metric_versions_are_isolated():
    assert METRIC_VERSION_V2 != METRIC_VERSION
    document = rules_document()
    assert document["metric_version"] == METRIC_VERSION_V2
    assert rules_hash() == rules_hash()
    assert len(rules_hash()) == 64


def test_optimize_planner_rejects_unknown_metric_version():
    extra = [
        Case(
            case_id=case_id,
            group=group,
            split=split,
            query=query,
            depth="balanced",
            required_aspects=["aa", "bb"],
        )
        for case_id, split, group, query in (
            ("c3", "train", "g", "第三道训练题的独立研究问题描述。"),
            ("c4", "validation", "gv", "第一道验证题的独立研究问题描述。"),
            ("c5", "holdout", "gh", "第一道留出题的独立研究问题描述。"),
        )
    ]
    dataset = Dataset(
        schema_version=1,
        dataset_id="test",
        provenance="reviewed",
        cases=_cases() + extra,
    )
    with pytest.raises(ValueError, match="unknown metric_version"):
        optimize_planner(
            dataset,
            None,
            rollout=None,
            reflection=None,
            limits=SearchLimits(64, 4, 42, 0.05),
            metric_version="v4",
        )


def _build_validation_retry_dataset() -> Dataset:
    """3 train + 2 validation + 1 holdout，足以触发定向恢复路径。"""
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


def test_single_case_flaky_failure_recovered_by_targeted_retry() -> None:
    """8+1 场景：1 题 Judge 首次失败后定向重试恢复 → GEPA 搜索不被跳过。"""

    dataset = _build_validation_retry_dataset()
    validation_cases = [c for c in dataset.cases if c.split == "validation"]
    flaky_case_id = validation_cases[0].case_id
    call_counts: dict[str, int] = {}

    def flaky_rollout(case, override):
        call_counts[case.case_id] = call_counts.get(case.case_id, 0) + 1
        if case.case_id == flaky_case_id and call_counts[case.case_id] == 1:
            return Observation(
                case_id=case.case_id,
                split=case.split,
                required_count=len(case.required_aspects),
                task_queries=["some query here"],
                error="judge: ValueError: Judge returned no valid structured coverage verdict",
            )
        return Observation(
            case_id=case.case_id,
            split=case.split,
            required_count=len(case.required_aspects),
            task_queries=["valid task query"],
            covered_indices=list(range(1, len(case.required_aspects) + 1)),
            task_specificity=[2] * len(case.required_aspects),
            plan_quality=[2] * len(case.required_aspects),
        )

    def noop_reflection(text):
        return "reflection response"

    journal_events = []
    result = optimize_planner(
        dataset,
        None,
        flaky_rollout,
        noop_reflection,
        SearchLimits(evaluations=64, proposals=2, seed=42, minimum_gain=0.05),
        record=lambda kind, payload: journal_events.append({"kind": kind, **payload}),
        metric_version="v2",
    )
    # 定向恢复成功：validation_unmeasured 不在门禁原因中。
    gate = result["holdout"]["gate"]
    assert "validation_unmeasured" not in gate.get("reasons", [])
    # Journal 记录了恢复尝试。
    retry_events = [e for e in journal_events if "retry" in e.get("kind", "")]
    assert len(retry_events) >= 1
    # flaky 题被调用了 2 次（首次失败 + 重试成功），其他题只调用 1 次。
    assert call_counts[flaky_case_id] == 2
    for case in dataset.cases:
        if case.case_id != flaky_case_id:
            assert call_counts.get(case.case_id, 0) <= 1, f"{case.case_id} called more than once"


def test_persistent_failure_still_blocks_search_after_bounded_retry() -> None:
    """持续失败的题目在 2 轮重试后仍失败 → 保持未测量并阻止不可信搜索。"""
    dataset = _build_validation_retry_dataset()
    validation_cases = [c for c in dataset.cases if c.split == "validation"]
    bad_case_id = validation_cases[0].case_id
    call_counts: dict[str, int] = {}

    def persistent_fail_rollout(case, override):
        call_counts[case.case_id] = call_counts.get(case.case_id, 0) + 1
        if case.case_id == bad_case_id:
            return Observation(
                case_id=case.case_id,
                split=case.split,
                required_count=len(case.required_aspects),
                task_queries=[],
                error="judge: persistent structured output failure",
            )
        return Observation(
            case_id=case.case_id,
            split=case.split,
            required_count=len(case.required_aspects),
            task_queries=["valid query"],
            covered_indices=list(range(1, len(case.required_aspects) + 1)),
            task_specificity=[2] * len(case.required_aspects),
            plan_quality=[2] * len(case.required_aspects),
        )

    def noop_reflection(text):
        return "reflection response"

    result = optimize_planner(
        dataset,
        None,
        persistent_fail_rollout,
        noop_reflection,
        SearchLimits(evaluations=64, proposals=2, seed=42, minimum_gain=0.05),
        metric_version="v2",
    )
    gate = result["holdout"]["gate"]
    assert "validation_unmeasured" in gate.get("reasons", [])
    # 持续失败的题目被调用 3 次（1 次初始 + 2 轮重试），不会无限重试。
    assert call_counts.get(bad_case_id, 0) == 3
