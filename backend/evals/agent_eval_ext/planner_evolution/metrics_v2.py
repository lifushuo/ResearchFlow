"""GEPA v2 综合质量指标：v1 覆盖率 + 任务具体性 + 证据计划质量 + 冗余度。

规则冻结（先于任何真实优化）：
- coverage：covered/required（v1 契约原样保留，独立展示并作为质量非退化约束）
- specificity：Judge 对每条任务评 0–2 —— 0 泛化主题串；1 指名具体方法/来源/数据集类；
  2 指名具体产物且带对比或测量目标。归一化 = 均值/2
- plan_quality：Judge 对每个必答维度评 0–2 —— 0 仅主题相关；1 可检索到部分证据；
  2 可检索到可测量/可对比的证据且范围明确。未覆盖维度记 0。归一化 = 均值/2
- redundancy：确定性计算，不经 Judge —— 归一化相似度 ≥0.8 的任务对视为重复，
  得分 = 1 − 重复对数 / 全部任务对数
- composite = 0.5·coverage + 0.2·specificity + 0.2·plan_quality + 0.1·redundancy

门禁 gate_v2：覆盖率非退化（candidate 均值 ≥ baseline 均值）且 composite 增益 ≥ min_gain；
任一子分缺失即 quality_unmeasured，不补造、不按 0 分计。
评价对象是 Planner 的研究任务，不是最终报告。不得利用留出集结果调整本规则。
"""

from __future__ import annotations

import json
from difflib import SequenceMatcher
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from evals.agent_eval_ext.planner_evolution.experiment import Observation, TaskMatch
from evals.agent_eval_ext.planner_evolution.judge_prompt import PLANNER_JUDGE_PROMPT_V2

METRIC_VERSION_V2 = "planner-coverage-quality/v2"
WEIGHTS: dict[str, float] = {
    "coverage": 0.5,
    "specificity": 0.2,
    "plan_quality": 0.2,
    "redundancy": 0.1,
}
REDUNDANCY_THRESHOLD = 0.8
_MIN_GAIN = 1e-9


class TaskMatchV2(TaskMatch):
    """v1 覆盖匹配 + 该维度下任务集的证据获取计划质量（0–2）。"""

    plan_quality: int = Field(ge=0, le=2, strict=True)


class TaskSpecificity(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_id: int = Field(ge=1, strict=True)
    specificity: int = Field(ge=0, le=2, strict=True)


class CoverageV2(BaseModel):
    model_config = ConfigDict(extra="forbid")

    matches: list[TaskMatchV2]
    task_specificity: list[TaskSpecificity]


def redundancy_score(task_queries: list[str]) -> float:
    """确定性冗余得分：1 − 重复任务对占比；不足两条任务时无重复可言，记 1.0。"""
    if len(task_queries) < 2:
        return 1.0
    pairs = 0
    duplicates = 0
    for i, a in enumerate(task_queries):
        for b in task_queries[i + 1 :]:
            pairs += 1
            if SequenceMatcher(None, a.strip(), b.strip()).ratio() >= REDUNDANCY_THRESHOLD:
                duplicates += 1
    return round(1 - duplicates / pairs, 4) if pairs else 1.0


def case_score_v2(row: Observation) -> dict[str, Any] | None:
    """单题 v2 综合分；error 行或缺失 v2 子分返回 None（未测量，绝不按 0 分计）。"""
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
    redundancy = redundancy_score(row.task_queries)
    composite = round(
        WEIGHTS["coverage"] * coverage
        + WEIGHTS["specificity"] * specificity
        + WEIGHTS["plan_quality"] * plan_quality
        + WEIGHTS["redundancy"] * redundancy,
        4,
    )
    return {
        "coverage": round(coverage, 4),
        "specificity": round(specificity, 4),
        "plan_quality": round(plan_quality, 4),
        "redundancy": redundancy,
        "composite": composite,
    }


def gate_v2(
    cases: list,
    baseline_rows: list[Observation],
    candidate_rows: list[Observation],
    min_gain: float,
    *,
    require_live_evidence: bool = False,
) -> dict[str, Any]:
    """v2 门禁：覆盖率非退化 + composite 增益 ≥ min_gain；结构/身份检查复用 make_gate。"""
    from evals.agent_eval_ext.planner_evolution.experiment import make_gate

    # make_gate 以覆盖 delta > min_gain 判通过；v2 的覆盖率约束是"非退化"（delta ≥ 0），
    # 因此这里以 min_gain=0 复用它的用例集一致性、Judge 身份与逐题损失检查，
    # 并把它的 gain_below_threshold 替换为本函数自己的非退化判定。
    coverage_gate = make_gate(
        cases,
        baseline_rows,
        candidate_rows,
        min_gain=0.0,
        require_live_evidence=require_live_evidence,
    )
    structural = [r for r in coverage_gate["reasons"] if r != "gain_below_threshold"]

    base_scores = [case_score_v2(row) for row in baseline_rows]
    cand_scores = [case_score_v2(row) for row in candidate_rows]
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

    composite_baseline = None
    composite_candidate = None
    composite_delta = None
    if not any(s is None for s in base_scores + cand_scores):
        composite_baseline = _mean([s["composite"] for s in base_scores])
        composite_candidate = _mean([s["composite"] for s in cand_scores])
        composite_delta = round(composite_candidate - composite_baseline, 4)
        if composite_delta <= 0 or composite_delta + _MIN_GAIN < min_gain:
            reasons.append("composite_gain_below_threshold")

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
        "composite_baseline": composite_baseline,
        "composite_candidate": composite_candidate,
        "composite_delta": composite_delta,
        "losses": coverage_gate.get("losses"),
        "metric_version": METRIC_VERSION_V2,
    }


def rules_document() -> dict[str, Any]:
    """冻结的评分规则文档；其哈希随实验记录落盘，规则改动必然改变哈希。"""
    return {
        "metric_version": METRIC_VERSION_V2,
        "weights": WEIGHTS,
        "redundancy_threshold": REDUNDANCY_THRESHOLD,
        "judge_prompt": PLANNER_JUDGE_PROMPT_V2,
        "specificity_rubric": "0 泛化主题串；1 指名具体方法/来源/数据集类；2 指名具体产物且带对比或测量目标",
        "plan_quality_rubric": "0 仅主题相关；1 可检索到部分证据；2 可检索到可测量/可对比的证据且范围明确",
        "redundancy_rule": "归一化相似度 ≥0.8 的任务对视为重复；得分 = 1 − 重复对数/全部任务对数",
        "gate_rule": "覆盖率非退化（candidate 均值 ≥ baseline 均值）且 composite 增益 ≥ min_gain",
    }


def rules_hash() -> str:
    from evals.agent_eval_ext.long_run import sha256_text

    return sha256_text(_canonical(rules_document()))


def _canonical(value: Any) -> str:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
