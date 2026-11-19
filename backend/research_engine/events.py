"""流水线事件发送的间接层，见 docs/02 §5、docs/05 §4。

节点调用 emit，实际发送器由 ContextVar 提供。Worker 同时落库用于 SSE 回放并发布到 Redis。
测试默认收集或忽略事件，因此引擎可以不依赖数据库与 Redis。"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from contextvars import ContextVar

import structlog

logger = structlog.get_logger()

Emitter = Callable[[str, dict], Awaitable[None]]


async def _noop(session_id: str, event: dict) -> None:  # pragma: no cover — 默认空实现
    return None


_emitter: ContextVar[Emitter] = ContextVar("pipeline_emitter", default=_noop)


def set_emitter(fn: Emitter):
    """为当前上下文安装事件发送器，返回可恢复原设置的 ContextVar 令牌。"""
    return _emitter.set(fn)


def reset_emitter(token) -> None:
    _emitter.reset(token)


def make_event(
    event_type: str,
    *,
    agent: str | None = None,
    message: str | None = None,
    detail: dict | None = None,
    data: dict | None = None,
) -> dict:
    """只构造事件，不发送。

    宿主可在会话事务提交后再发 HITL_READY、COMPLETED、FAILED，
    避免客户端收到完成通知后立即查询却仍看到 RUNNING。图内事件仍通过 emit 发送。"""
    return {
        "type": event_type,
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "agent": agent,
        "message": message,
        "detail": detail,
        "data": data,
    }


async def emit(
    session_id: str,
    event_type: str,
    *,
    agent: str | None = None,
    message: str | None = None,
    detail: dict | None = None,
    data: dict | None = None,
) -> None:
    event = make_event(event_type, agent=agent, message=message, detail=detail, data=data)
    await _write_through_cancellation(_emitter.get()(session_id, event))


async def _write_through_cancellation(write: Awaitable[None]) -> None:
    """取消生效前等待已经开始的事件写入完成。

    并发兄弟任务失败会触发取消；若在提交中途取消，可能泄漏或丢弃数据库连接。
    因此先完成当前写入，再传播取消，不再启动新写入。"""
    task = asyncio.ensure_future(write)
    try:
        await asyncio.shield(task)
    except asyncio.CancelledError:
        # 等待期间再次收到取消，也先保证当前写入结束。
        while not task.done():
            try:
                await asyncio.wait([task])
            except asyncio.CancelledError:
                pass
        if not task.cancelled() and task.exception() is not None:
            logger.warning(
                "event_write_failed_during_cancellation",
                error=f"{type(task.exception()).__name__}: {task.exception()}",
            )
        raise
