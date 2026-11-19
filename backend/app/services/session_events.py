"""两个主机共用 RunOutcome 到生命周期事件的转换。

计划关卡、报告关卡、完成、失败不仅名称一致，负载中的任务数、来源、耗时、费用与原因也必须一致。
仅依赖标准库与研究引擎，禁止为桌面导入 app.config。"""

from __future__ import annotations

from research_engine.events import make_event
from research_engine.runner import RunOutcome

# 从 RunOutcome.status 映射到客户端识别的事件名，两端停止列表都用这些名称。
EVENT_TYPES: dict[str, str] = {
    "awaiting_plan": "PLAN_READY",
    "awaiting_approval": "HITL_READY",
    "completed": "COMPLETED",
    "failed": "FAILED",
}


def lifecycle_event(outcome: RunOutcome) -> dict:
    """结果提交后应发布的事件。

    所有数字取实际观测，不用零凑结构；计划关卡也披露实际费用，恢复执行到此时可能已经有真实消耗。"""
    event_type = EVENT_TYPES[outcome.status]

    if outcome.status == "awaiting_plan":
        data = {
            "task_count": len(outcome.plan_tasks),
            "outline_section_count": len(outcome.plan_outline),
            "cost_usd": round(outcome.cost_usd, 4),
        }
    elif outcome.status == "awaiting_approval":
        data = {
            "word_count": len((outcome.draft_report or "").split()),
            "source_count": len(outcome.sources),
            "cost_usd": round(outcome.cost_usd, 4),
        }
    elif outcome.status == "failed":
        # 失败原因放入 data.reason，不放 message，保证两端客户端读取一致。
        data = {"reason": outcome.error}
    else:
        data = {
            "elapsed_s": float(outcome.elapsed_seconds or 0),
            "cost_usd": round(outcome.cost_usd, 4),
        }

    return make_event(event_type, data=data)
