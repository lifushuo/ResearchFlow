"""GEPA v3 质量指标：覆盖率退为硬约束，优化目标只含质量子分。

v2 的实测教训（gepa-00a9f8c5，64 次观测 coverage 全为 1.0）：覆盖率是饱和指标，
0.5 的权重锁定后 composite 的可达上界只剩 0.0441，任何 ≥0.05 的 min_gain 在数学上
不可达，且饱和分量持续稀释质量信号。v3 的应对（doc/治理原则：规则先冻结后优化）：

- coverage：仍是 covered/required，但退出优化目标，只作硬约束——
  候选均值守门（candidate ≥ baseline，非退化），复现 v2 的 coverage_regression 语义
- quality：v2 综合分去掉覆盖率后的归一化重标定
  quality = (v2_composite − 0.5·coverage) / 0.5
          = 0.4·specificity + 0.4·plan_quality + 0.2·redundancy
  权重是 v2 质量权重 (0.2, 0.2, 0.1) 除以剩余权重 0.5，不是新设定的权重
- Judge 轴不变：specificity / plan_quality / coverage 三个轴与 v2 同一提示词、
  同一结构化契约（CoverageV2）。v3 改变的只是聚合方式，因此校准要求是
  验证 Judge 在完整/缺漏/含混计划上的判别行为，而不是新契约
- 门禁 gate_v3：结构检查（用例一致性、Judge 身份、逐题损失）+ 覆盖率非退化 +
  quality 增益 ≥ min_gain。任一子分缺失即 quality_unmeasured，不补造、不按 0 分计

评价对象仍是 Planner 的研究任务。不得利用留出集结果调整本规则；
min_gain 由开发/校准集上的 Judge 判别与重复性测量决定，不从留出集反推。
"""

from __future__ import annotations

import json
from typing import Any

from evals.agent_eval_ext.planner_evolution.experiment import Observation
from evals.agent_eval_ext.planner_evolution.judge_prompt import PLANNER_JUDGE_PROMPT_V2
from evals.agent_eval_ext.planner_evolution.metrics_v2 import (
    REDUNDANCY_THRESHOLD,
    redundancy_score,
)

METRIC_VERSION_V3 = "planner-coverage-quality/v3"
# v2 质量权重 (0.2, 0.2, 0.1) / 剩余权重 0.5 的归一化，与去覆盖率公式恒等。
QUALITY_WEIGHTS: dict[str, float] = {
    "specificity": 0.4,
    "plan_quality": 0.4,
    "redundancy": 0.2,
}
V2_COVERAGE_WEIGHT = 0.5
_MIN_GAIN = 1e-9


def case_score_v3(row: Observation) -> dict[str, Any] | None:
    """单题 v3 分；error 行或缺失 v2 子分返回 None（未测量，绝不按 0 分计）。

    quality 不含 coverage——两个 coverage 不同但质量子分相同的观测，
    quality 相同而 coverage 不同，这正是"覆盖率是约束不是目标"的度量形态。
    """
    if row.error is not None:
        return None
    if row.task_specificity is None or row.plan_quality is None:
        return None
    if len(row.task_specificity) != len(row.task_queries):
        return None
    coverage = len(row.covered_indices) / row.required_count
    specificity = (
        sum(row.task_specificity) / len(row.task_specificity) / 2 if row.task_specificity else 0.0
    )
    plan_quality = sum(row.plan_quality) / len(row.plan_quality) / 2 if row.plan_quality else 0.0
    # 冗余度是确定性计算，不经 Judge——与 v2 完全同一实现，两版规则不得分叉。
    redundancy = redundancy_score(row.task_queries)
    quality = round(
        QUALITY_WEIGHTS["specificity"] * specificity
        + QUALITY_WEIGHTS["plan_quality"] * plan_quality
        + QUALITY_WEIGHTS["redundancy"] * redundancy,
        4,
    )
    return {
        "coverage": round(coverage, 4),
        "specificity": round(specificity, 4),
        "plan_quality": round(plan_quality, 4),
        "redundancy": redundancy,
        "quality": quality,
    }


def gate_v3(
    cases: list,
    baseline_rows: list[Observation],
    candidate_rows: list[Observation],
    min_gain: float,
    *,
    require_live_evidence: bool = False,
) -> dict[str, Any]:
    """v3 门禁：覆盖率非退化（硬约束）+ quality 增益 ≥ min_gain；结构检查复用 make_gate。"""
    from evals.agent_eval_ext.planner_evolution.experiment import make_gate

    # 与 gate_v2 相同的结构复用：min_gain=0 换取用例集一致性、Judge 身份与逐题损失检查，
    # 其 gain_below_threshold 由本函数自己的覆盖约束与质量增益判定替代。
    coverage_gate = make_gate(
        cases,
        baseline_rows,
        candidate_rows,
        min_gain=0.0,
        require_live_evidence=require_live_evidence,
    )
    structural = [r for r in coverage_gate["reasons"] if r != "gain_below_threshold"]

    base_scores = [case_score_v3(row) for row in baseline_rows]
    cand_scores = [case_score_v3(row) for row in candidate_rows]
    reasons = list(structural)
    if any(score is None for score in base_scores + cand_scores):
        reasons.append("quality_unmeasured")

    def _mean(values: list[float]) -> float | None:
        return round(sum(values) / len(values), 4) if values else None

    base_coverage = _mean([row.score for row in baseline_rows if row.score is not None])
    cand_coverage = _mean([row.score for row in candidate_rows if row.score is not None])
    if base_coverage is None or cand_coverage is None or cand_coverage < base_coverage - _MIN_GAIN:
        if (
            "coverage_regression" not in reasons
            and base_coverage is not None
            and cand_coverage is not None
        ):
            reasons.append("coverage_regression")

    quality_baseline = None
    quality_candidate = None
    quality_delta = None
    if not any(s is None for s in base_scores + cand_scores):
        quality_baseline = _mean([s["quality"] for s in base_scores])
        quality_candidate = _mean([s["quality"] for s in cand_scores])
        quality_delta = round(quality_candidate - quality_baseline, 4)
        if quality_delta <= 0 or quality_delta + _MIN_GAIN < min_gain:
            reasons.append("quality_gain_below_threshold")

    return {
        "passed": not reasons,
        "reasons": reasons,
        "case_count": len(cases),
        "coverage_baseline": base_coverage,
        "coverage_candidate": cand_coverage,
        "coverage_delta": (
            round(cand_coverage - base_coverage, 4)
            if base_coverage is not None and cand_coverage is not None
            else None
        ),
        "quality_baseline": quality_baseline,
        "quality_candidate": quality_candidate,
        "quality_delta": quality_delta,
        "losses": coverage_gate.get("losses"),
        "metric_version": METRIC_VERSION_V3,
    }


def rules_document() -> dict[str, Any]:
    """冻结的评分规则文档；其哈希随实验记录落盘，规则改动必然改变哈希。"""
    return {
        "metric_version": METRIC_VERSION_V3,
        "quality_weights": QUALITY_WEIGHTS,
        "renormalization": "quality = (v2_composite − 0.5·coverage) / 0.5",
        "coverage_role": "hard_constraint",
        "coverage_constraint": "candidate 均值 ≥ baseline 均值（非退化）；coverage 不参与优化目标",
        "redundancy_threshold": REDUNDANCY_THRESHOLD,
        "judge_prompt": PLANNER_JUDGE_PROMPT_V2,
        "specificity_rubric": "0 泛化主题串；1 指名具体方法/来源/数据集类；2 指名具体产物且带对比或测量目标",
        "plan_quality_rubric": "0 仅主题相关；1 可检索到部分证据；2 可检索到可测量/可对比的证据且范围明确",
        "redundancy_rule": "归一化相似度 ≥0.8 的任务对视为重复；得分 = 1 − 重复对数/全部任务对数",
        "gate_rule": "覆盖率非退化（硬约束）且 quality 增益 ≥ min_gain；min_gain 由校准集测量决定",
    }


def rules_hash() -> str:
    from evals.agent_eval_ext.long_run import sha256_text

    return sha256_text(_canonical(rules_document()))


def _canonical(value: Any) -> str:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
