"""LangGraph 流水线状态，见 docs/architecture/04-agent-architecture.md §2。

使用普通 TypedDict，由 LangGraph 管理并逐步保存到检查点；只能包含可 JSON 序列化的数据。"""

from __future__ import annotations

from typing import Any, TypedDict


class AgentState(TypedDict, total=False):
    # 运行身份。
    session_id: str
    user_id: str

    # 输入参数。
    original_query: str
    research_depth: str

    # 规划阶段。
    tasks: list[dict[str, Any]]
    # 规划器提议的报告结构，可在计划关卡编辑；未提议或跳过时为空。
    proposed_outline: list[dict[str, Any]]
    # 通过或跳过计划关卡后设为真，使恢复运行能区分尚未到达和已经通过。
    plan_approved: bool | None

    # 执行与评审轮次；并发任务按各自 ID 追踪进度，不能靠单个移动索引。
    #
    # 每轮按任务定义顺序重建 evidence，不按完成顺序，避免并发速度改变引用编号。
    # 每项携带 task_id，因此也可用于逐任务追踪。
    #
    # verdicts、retries 以字符串任务 ID 作键，因为检查点 JSON 对象键只能是字符串。
    evidence: list[dict[str, Any]]
    verdicts: dict[str, dict[str, Any]]
    retries: dict[str, int]
    research_round: int
    # 整个运行实际取得文本的独立来源数。与 evidence 分开，
    # 用来区分没有读到来源和读到很多来源却没提取出证据，两种故障处理方法不同。
    sources_seen: int

    # 综合与人工审核。
    draft_report: str | None
    sources: list[dict[str, Any]]
    # 检测出的冲突观点对，展示到报告和关卡中，不自动裁决，见 docs/12 M11。
    contradictions: list[dict[str, Any]]
    human_feedback: str | None
    rework_count: int
    approved: bool | None
    # 指定文件研究的逐 Claim 核验记录：计数、原因、状态和被核验正文哈希。
    # 综合路由与人工关卡都据此判断；其他运行保持 None。
    grounding_report: dict[str, Any] | None

    # 最终输出。
    final_report: str | None

    # 用量与耗时统计。
    tokens_input: int
    tokens_output: int
    cost_usd: float
    started_at: float

    # 终止状态。
    error: str | None
