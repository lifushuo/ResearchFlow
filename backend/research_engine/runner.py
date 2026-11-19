"""不依赖宿主的图运行与恢复，见 docs/13 §4–§5、docs/12 M6。

只接收准备好的检查点与端口，安装上下文、驱动图并返回 RunOutcome。
数据库会话、队列、锁、saver 构造与建表均由宿主负责。
结果持久化和生命周期事件也归宿主：先提交状态，再通知客户端，避免终态事件后仍读到旧状态。"""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Literal

from langgraph.types import Command

from research_engine import events, llm_factory, offline
from research_engine.cache import reset_cache, set_cache
from research_engine.corpus import reset_corpus, set_corpus
from research_engine.graph import build_graph
from research_engine.ports import Cache, Corpus, EventSink
from research_engine.runconfig import (
    RunConfig,
    get_run_config,
    reset_run_config,
    set_run_config,
)

#: awaiting_plan 是计划审核暂停，awaiting_approval 是草稿审核暂停。
#: 两者恢复参数不同，混用会在尚无草稿时错误批准报告。
RunStatus = Literal["awaiting_plan", "awaiting_approval", "completed", "failed"]

#: 两个人工关卡 interrupt 载荷的 type，必须与图节点生产端一致。
_PLAN_INTERRUPT = "PLAN_READY"

# 与服务端错误消息列的截断长度一致。
_MAX_ERROR_CHARS = 500


@dataclass(frozen=True)
class RunOutcome:
    """一次图调用的结果，只有普通数据，不含 ORM 或宿主类型。"""

    status: RunStatus
    draft_report: str | None = None
    final_report: str | None = None
    sources: list[dict] = field(default_factory=list)
    cost_usd: float = 0.0
    tokens_input: int = 0
    tokens_output: int = 0
    rework_count: int = 0
    elapsed_seconds: float | None = None
    error: str | None = None
    # 仅在 awaiting_plan 携带规划提议，供宿主持久化及展示。
    # 其他状态保持空：未到关卡无提议，过关后则已有决定而非仍待审提议。
    plan_tasks: list[dict] = field(default_factory=list)
    plan_outline: list[dict] = field(default_factory=list)

    @property
    def report(self) -> str | None:
        """返回当前最佳报告文本：有定稿则取定稿，否则取草稿。"""
        return self.final_report or self.draft_report


@contextmanager
def _installed(
    run_config: RunConfig | None,
    provider_keys: dict[str, str] | None,
    event_sink: EventSink | None,
    cache: Cache | None,
    corpus: Corpus | None,
):
    """安装运行上下文，并按相反顺序恢复。

    各项使用 ContextVar，隔离同一进程的并发路由、预算与用户密钥。"""
    undo: list[tuple] = []
    try:
        if run_config is not None:
            undo.append((reset_run_config, set_run_config(run_config)))
        if provider_keys is not None:
            undo.append((llm_factory.reset_user_keys, llm_factory.set_user_keys(provider_keys)))
        if event_sink is not None:
            undo.append((events.reset_emitter, events.set_emitter(event_sink)))
        if cache is not None:
            undo.append((reset_cache, set_cache(cache)))
        if corpus is not None:
            undo.append((reset_corpus, set_corpus(corpus)))
        yield
    finally:
        for reset, token in reversed(undo):
            reset(token)


def initial_state(*, session_id: str, user_id: str, query: str, depth: str = "balanced") -> dict:
    """构造新研究的图初始状态。"""
    return {
        "session_id": session_id,
        "user_id": user_id,
        "original_query": query,
        "research_depth": depth,
        "evidence": [],
        "verdicts": {},
        "retries": {},
        "research_round": 0,
        "rework_count": 0,
        "cost_usd": 0.0,
        "tokens_input": 0,
        "tokens_output": 0,
        "started_at": time.time(),
    }


def _outcome(result: dict, state: dict) -> RunOutcome:
    """将图状态映射为 RunOutcome。

    先检查中断再检查 error，因为等待人工不是失败。
    从中断载荷读取关卡类型，不能默认草稿已完成；计划关卡时可能尚未搜索。"""
    totals = {
        "sources": state.get("sources") or [],
        "cost_usd": round(state.get("cost_usd", 0.0), 6),
        "tokens_input": state.get("tokens_input", 0),
        "tokens_output": state.get("tokens_output", 0),
        "rework_count": state.get("rework_count", 0),
    }

    if "__interrupt__" in result:
        payload = result["__interrupt__"][0].value or {}
        if payload.get("type") == _PLAN_INTERRUPT:
            return RunOutcome(
                status="awaiting_plan",
                # 从中断载荷读取用户实际看到的提议，不能取可能已被修改的状态值。
                plan_tasks=list(payload.get("tasks") or []),
                plan_outline=list(payload.get("proposed_outline") or []),
                **totals,
            )
        return RunOutcome(
            status="awaiting_approval", draft_report=state.get("draft_report"), **totals
        )

    if state.get("error"):
        return RunOutcome(status="failed", error=str(state["error"])[:_MAX_ERROR_CHARS], **totals)

    return RunOutcome(
        status="completed",
        draft_report=state.get("draft_report"),
        final_report=state.get("final_report") or state.get("draft_report"),
        elapsed_seconds=round(time.time() - state.get("started_at", time.time()), 2),
        **totals,
    )


async def _drive(
    *,
    checkpointer,
    session_id: str,
    payload,
    run_config: RunConfig | None,
    provider_keys: dict[str, str] | None,
    event_sink: EventSink | None,
    cache: Cache | None,
    corpus: Corpus | None,
) -> RunOutcome:
    config = {"configurable": {"thread_id": session_id}}
    with _installed(run_config, provider_keys, event_sink, cache, corpus):
        # 每次启动和恢复都先验证 Offline，保证不符合本地承诺时没有任何模型调用。
        refused = offline.refusal(
            get_run_config(), custom_base_url=llm_factory.api_key_for("custom_base_url")
        )
        if refused:
            return RunOutcome(status="failed", error=refused)
        graph = build_graph(checkpointer)
        result = await graph.ainvoke(payload, config)
        # 异步检查点必须使用异步状态读取器；运行中的事件循环不能调用同步 get_state。
        state = (await graph.aget_state(config)).values
    return _outcome(result, state)


async def run(
    *,
    checkpointer,
    session_id: str,
    user_id: str,
    query: str,
    depth: str = "balanced",
    run_config: RunConfig | None = None,
    provider_keys: dict[str, str] | None = None,
    event_sink: EventSink | None = None,
    cache: Cache | None = None,
    corpus: Corpus | None = None,
) -> RunOutcome:
    """启动新研究，运行到首次暂停或失败。"""
    return await _drive(
        checkpointer=checkpointer,
        session_id=session_id,
        payload=initial_state(session_id=session_id, user_id=user_id, query=query, depth=depth),
        run_config=run_config,
        provider_keys=provider_keys,
        event_sink=event_sink,
        cache=cache,
        corpus=corpus,
    )


async def resume(
    *,
    checkpointer,
    session_id: str,
    approved: bool | None = None,
    feedback: str | None = None,
    plan: dict | None = None,
    run_config: RunConfig | None = None,
    provider_keys: dict[str, str] | None = None,
    event_sink: EventSink | None = None,
    cache: Cache | None = None,
    corpus: Corpus | None = None,
) -> RunOutcome:
    """从人工关卡的检查点恢复，不重新规划。

    approved 配可选 feedback 用于草稿审批；plan 用于编辑计划，tasks 与 outline 均可省略。
    plan={} 表示按原提议批准，tasks=[] 表示排除全部任务，两者不同。
    两种决定必须且只能给一种，不设置默认批准，以免把错误恢复请求变成对不存在草稿的批准。"""
    if (approved is None) == (plan is None):
        raise ValueError(
            "resume() needs exactly one of `approved=` (draft review gate) or "
            "`plan=` (research design gate); got "
            f"approved={approved!r}, plan={plan!r}."
        )
    decision = (
        {"approved": approved, "feedback": feedback}
        if plan is None
        else {"tasks": plan.get("tasks"), "outline": plan.get("outline")}
    )
    return await _drive(
        checkpointer=checkpointer,
        session_id=session_id,
        payload=Command(resume=decision),
        run_config=run_config,
        provider_keys=provider_keys,
        event_sink=event_sink,
        cache=cache,
        corpus=corpus,
    )
