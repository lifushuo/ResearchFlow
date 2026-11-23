"""按需分析已归档 Planner 任务的维度覆盖；只给人工复核建议，不写入训练集。

与 GEPA 使用相同的 PLANNER_JUDGE_PROMPT、Coverage 和 covered_indices 契约。
不重新运行 Planner。调用者必须先从评测闭环验证 Run 的归属与原始证据指纹。
"""

from __future__ import annotations

import asyncio
import json
import os
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, ConfigDict, Field

from evals.agent_eval_ext.planner_evolution.experiment import Coverage, TaskMatch, covered_indices
from evals.agent_eval_ext.planner_evolution.judge_prompt import PLANNER_JUDGE_PROMPT
from evals.judge_config import judge_client_config
from research_engine import llm_factory
from research_engine.runconfig import get_run_config


class ReviewTaskMatch(TaskMatch):
    """任务编号仍由原版 Coverage 校验；理由仅作为可审核的模型建议。"""

    reason: str = Field(min_length=2, max_length=500)


class ReviewCoverage(BaseModel):
    model_config = ConfigDict(extra="forbid")
    matches: list[ReviewTaskMatch]


def build_review_advice(
    *,
    evidence: dict[str, Any],
    question: str,
    aspects: list[str],
    verdict: ReviewCoverage,
    judge_route: str,
    served_model: str | None,
    tokens: tuple[int, int] | None,
) -> dict[str, Any]:
    """纯函数；结构不正确则拒绝，而不是把没有测量误报为 Planner 漏项。"""
    tasks = evidence["tasks"]
    original = Coverage(
        matches=[
            TaskMatch(aspect_index=m.aspect_index, task_ids=m.task_ids) for m in verdict.matches
        ]
    )
    covered = covered_indices(original, len(aspects), len(tasks))
    # 去掉重复 task ID，并使用证据的顺序展示；不采信模型输出的任务文本。
    matches = {match.aspect_index: match for match in verdict.matches}
    return {
        "status": "suggested",
        "run_id": evidence["run_id"],
        "case_id": evidence["case_id"],
        "evidence_reference": evidence["evidence_reference"],
        "question": question,
        "judge_route": judge_route,
        "served_judge": served_model,
        "judge_tokens": {"input": tokens[0], "output": tokens[1]} if tokens else None,
        "coverage": len(covered) / len(aspects),
        "aspects": [
            {
                "aspect": aspect,
                "aspect_index": i,
                "covered": i in covered,
                "task_ids": sorted(set(matches[i].task_ids)),
                "reason": matches[i].reason,
            }
            for i, aspect in enumerate(aspects, start=1)
        ],
        "automatic_label": False,
    }


def configured_judge_route() -> str:
    """优先与统一评测采用相同的 Judge；不默认回退到研究 Critic。"""
    route = (
        os.environ.get("RF_EVAL_JUDGE_ROUTE") or os.environ.get("EVAL_JUDGE_MODEL") or ""
    ).strip()
    provider, _, model = route.partition(":")
    if (
        not provider
        or not model
        or model.lower() == "auto"
        or model.lower().startswith("auto/")
        or model.lower().endswith("/auto")
    ):
        raise ValueError("请先配置固定的独立 EVAL_JUDGE_MODEL（provider:model）")
    config = get_run_config()
    if config.llm_mode != "real":
        raise ValueError("当前 API 是 fake 模式，不能生成真实 Judge 评分")
    # 与当前服务端研究路由作显式独立性检查；若历史路由已变更，须人工复核实际评测元数据。
    for research in config.models.values():
        used = research.partition(":")[2]
        if (
            used.lower() == "auto"
            or used.lower().startswith("auto/")
            or used.lower().endswith("/auto")
        ):
            raise ValueError("研究模型使用自动路由，无法证明 Judge 独立")
        if research == route or used.rsplit("/", 1)[-1].lower() == model.rsplit("/", 1)[-1].lower():
            raise ValueError("Judge 与当前研究模型重合，请配置独立模型")
    return route


async def suggest_planner_coverage(
    *,
    evidence: dict[str, Any],
    question: str,
    aspects: list[str],
    timeout_seconds: int = 90,
) -> dict[str, Any]:
    """一次明确授权的真实 Judge 调用，既不触发 Planner 也不持久化任何标注。"""
    route = configured_judge_route()
    tasks = evidence["tasks"]
    payload = {
        "question": question,
        "required_aspects": dict(enumerate(aspects, 1)),
        "tasks": dict(enumerate((row["query"] for row in tasks), 1)),
    }
    with judge_client_config(route):
        llm = llm_factory.get_llm("critic")
        method = (
            {"method": "function_calling"}
            if route.startswith(("custom:", "ollama:", "openrouter:"))
            else {}
        )
        judge = llm.with_structured_output(ReviewCoverage, include_raw=True, **method)
        response = await asyncio.wait_for(
            judge.ainvoke(
                [
                    SystemMessage(
                        content=(
                            PLANNER_JUDGE_PROMPT
                            + " For every aspect, include a short reason explaining which tasks"
                            + " cover it, or why none does."
                        )
                    ),
                    HumanMessage(content=json.dumps(payload, ensure_ascii=False)),
                ]
            ),
            timeout=timeout_seconds,
        )
    if response.get("parsing_error") or response.get("parsed") is None:
        raise ValueError("Judge 未返回可解析的逐维度判断；本次不生成任何漏项标签")
    verdict = ReviewCoverage.model_validate(response["parsed"])
    raw = response.get("raw")
    return build_review_advice(
        evidence=evidence,
        question=question,
        aspects=aspects,
        verdict=verdict,
        judge_route=route,
        served_model=llm_factory.served_model_id(raw),
        tokens=llm_factory.token_counts(raw),
    )


async def suggest_planner_coverage_v2(
    *,
    evidence: dict[str, Any],
    question: str,
    aspects: list[str],
    timeout_seconds: int = 90,
) -> dict[str, Any]:
    """一次 v2 Judge 调用：按冻结的 v2 Rubric 输出覆盖率、计划质量与任务具体性建议。

    复用 v1 的独立性预检与 Judge 配置机制；只更换 schema 与提示词。
    任务必须逐条评分、维度必须逐个返回，否则 fail-closed（不生成任何标签）。
    """
    from evals.agent_eval_ext.planner_evolution.judge_prompt import PLANNER_JUDGE_PROMPT_V2
    from evals.agent_eval_ext.planner_evolution.metrics_v2 import (
        METRIC_VERSION_V2,
        CoverageV2,
    )

    route = configured_judge_route()
    tasks = evidence["tasks"]
    payload = {
        "question": question,
        "required_aspects": dict(enumerate(aspects, 1)),
        "tasks": dict(enumerate((row["query"] for row in tasks), 1)),
    }
    with judge_client_config(route):
        llm = llm_factory.get_llm("critic")
        method = (
            {"method": "function_calling"}
            if route.startswith(("custom:", "ollama:", "openrouter:"))
            else {}
        )
        scorer = llm.with_structured_output(CoverageV2, include_raw=True, **method)
        response = await asyncio.wait_for(
            scorer.ainvoke(
                [
                    SystemMessage(content=PLANNER_JUDGE_PROMPT_V2),
                    HumanMessage(content=json.dumps(payload, ensure_ascii=False)),
                ]
            ),
            timeout=timeout_seconds,
        )
    if response.get("parsing_error") or response.get("parsed") is None:
        raise ValueError("Judge 未返回可解析的 v2 结构化评分；本次不生成任何标签")
    verdict = CoverageV2.model_validate(response["parsed"])
    # fail-closed 契约：每个任务恰好评一次、每个维度恰好返回一次。
    spec_by_id = {t.task_id: t.specificity for t in verdict.task_specificity}
    if set(spec_by_id) != set(range(1, len(tasks) + 1)):
        raise ValueError("judge must return a specificity score for every task exactly once")
    aspects_returned = sorted(m.aspect_index for m in verdict.matches)
    if aspects_returned != list(range(1, len(aspects) + 1)):
        raise ValueError("judge must return each aspect exactly once")
    plan_by_index = {m.aspect_index: m.plan_quality for m in verdict.matches}
    raw_response = response.get("raw")
    return {
        "status": "suggested",
        "metric_version": METRIC_VERSION_V2,
        "run_id": evidence.get("run_id"),
        "case_id": evidence.get("case_id"),
        "evidence_reference": evidence.get("evidence_reference"),
        "judge_route": route,
        "served_judge": llm_factory.served_model_id(raw_response),
        "judge_tokens": llm_factory.token_counts(raw_response),
        "coverage": len([m for m in verdict.matches if m.task_ids]) / len(aspects),
        "aspects": [
            {
                "aspect_index": index,
                "aspect": aspect,
                "covered": index in {m.aspect_index for m in verdict.matches if m.task_ids},
                "task_ids": sorted(
                    next(m.task_ids for m in verdict.matches if m.aspect_index == index)
                ),
                "plan_quality": plan_by_index.get(index, 0),
            }
            for index, aspect in enumerate(aspects, 1)
        ],
        "task_specificity": [
            {
                "task_id": index,
                "task_query": tasks[index - 1]["query"],
                "specificity": spec_by_id[index],
            }
            for index in range(1, len(tasks) + 1)
        ],
        "automatic_label": False,
    }
