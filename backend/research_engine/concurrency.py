"""让图节点的并发子任务不会在节点退出后继续运行。

asyncio.gather 遇到首个异常会返回，但其他任务仍可能调用模型或写事件。
这既造成失败后的额外费用，也可能与宿主关闭数据库会话冲突，掩盖原始模型异常。"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Iterable

import structlog

logger = structlog.get_logger()


async def gather_or_cancel[T](*aws: Awaitable[T]) -> list[T]:
    """并发执行；任一任务失败时取消并等待其余任务清理。

    成功结果按参数顺序返回；失败则抛出最早发生的原始异常。
    不额外包装异常，避免宿主记录到包装文字而非真正原因。"""
    tasks = [asyncio.ensure_future(aw) for aw in aws]
    if not tasks:
        return []
    # 按完成顺序追踪最早故障。一个子任务可能破坏共享状态，继而使其他任务失败；
    # 按参数顺序挑异常会把后果误报成原因。
    finished: list[asyncio.Task] = []
    for task in tasks:
        task.add_done_callback(finished.append)
    try:
        await asyncio.wait(tasks, return_when=asyncio.FIRST_EXCEPTION)
    except asyncio.CancelledError:
        await _cancel_and_wait(tasks)
        raise
    # 任务可能已结束但回调尚未运行；这种任务排在已有完成记录之后。
    ordered = finished + [t for t in tasks if t.done() and t not in finished]
    failed = [t for t in ordered if not t.cancelled() and t.exception() is not None]
    if not failed:
        return [t.result() for t in tasks]
    await _cancel_and_wait(tasks)
    first = failed[0]
    for other in tasks:
        # 取出并记录其他异常，既避免未读取异常警告，也不静默丢失同轮的其他故障。
        if other is not first and not other.cancelled() and other.exception() is not None:
            logger.warning(
                "fan_out_sibling_also_failed",
                error=f"{type(other.exception()).__name__}: {other.exception()}",
            )
    raise first.exception()


async def _cancel_and_wait(tasks: Iterable[asyncio.Task]) -> None:
    pending = [t for t in tasks if not t.done()]
    for task in pending:
        task.cancel()
    # 即使当前等待被取消，也先等子任务清理结束，再传播取消。
    # 使用 wait 避免重新抛出子异常或再次取消任务，防止清理过程无人等待。
    interrupted = False
    while pending:
        try:
            await asyncio.wait(pending)
        except asyncio.CancelledError:
            interrupted = True
        pending = [t for t in pending if not t.done()]
    if interrupted:
        raise asyncio.CancelledError
