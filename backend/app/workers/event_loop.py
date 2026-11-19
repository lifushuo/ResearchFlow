"""每个 Celery Worker 进程只维护一个事件循环，所有任务共用。

模型 SDK 会在进程内缓存异步 HTTP 客户端，连接属于创建它的循环。若每个任务另建并关闭循环，下个任务复用连接时就可能报 Event loop is closed；某些 SDK 还会用重试掩盖问题，造成重复请求。
循环与进程同寿命可保护这些缓存。每个任务结束前仍取消并等待其遗留协程，且为每个任务复制独立上下文，避免日志身份或配置串用。
API、桌面本来就运行在各自的单一循环上；CLI 每个进程只运行一次，无需此适配。"""

from __future__ import annotations

import asyncio
import contextvars
import os
import threading
from collections.abc import Coroutine
from typing import Any

_runner: asyncio.Runner | None = None
_runner_pid: int | None = None
_runner_thread: int | None = None


def _process_runner() -> asyncio.Runner:
    global _runner, _runner_pid, _runner_thread
    # fork 后不能沿用父进程的 selector，子进程必须创建自己的事件循环。
    if _runner is None or _runner_pid != os.getpid():
        # 通过 loop_factory 创建循环，但不把它设置为线程默认循环；避免干扰测试进程自己的循环管理。
        _runner = asyncio.Runner(loop_factory=asyncio.new_event_loop)
        _runner_pid = os.getpid()
        _runner_thread = threading.get_ident()
    elif _runner_thread != threading.get_ident():
        # 模型客户端缓存属于进程，不能使用线程池驱动这套循环。
        # 每线程一个循环会共享同一个缓存客户端；多线程共用并驱动同一个循环也不安全。
        raise RuntimeError(
            "app.workers.event_loop.run was called from a second thread; one event loop per "
            "worker process requires Celery's prefork (the default) or solo pool"
        )
    return _runner


def run[T](coro: Coroutine[Any, Any, T]) -> T:
    """在当前进程的固定事件循环上，将一个 Celery 任务的协程运行到结束。"""
    try:
        runner = _process_runner()
    except RuntimeError:
        coro.close()  # 协程尚未启动；关闭它，避免拒绝操作之外再产生未等待警告。
        raise
    root: list[asyncio.Task] = []

    async def as_root() -> T:
        root.append(asyncio.current_task())
        return await coro

    try:
        # 每个任务复制一份新上下文；asyncio.Runner 默认会复用上下文，否则本次绑定的日志身份可能无法生效。
        return runner.run(as_root(), context=contextvars.copy_context())
    finally:
        loop = runner.get_loop()
        if root and not root[0].done():
            _unwind(loop, root[0])
        _cancel_leftovers(loop)


def _unwind(loop: asyncio.AbstractEventLoop, root: asyncio.Task) -> None:
    """任务被中断时，先取消它的主协程，让清理沿正常调用链完成。

    Celery 软时限会从信号处理器跳出循环。先取消主协程，才能让并发子任务有序取消、受保护的事件写入提交完成，最后释放锁和会话。若一次取消所有协程，会直接打断受保护写入，使事务只完成一半。"""
    root.cancel()
    loop.run_until_complete(asyncio.wait([root]))
    if not root.cancelled() and root.exception() is not None:
        loop.call_exception_handler(
            {
                "message": "a Celery task's coroutine raised while unwinding after an interruption",
                "exception": root.exception(),
                "task": root,
            }
        )


def _cancel_leftovers(loop: asyncio.AbstractEventLoop) -> None:
    """取消并等待该任务留下的协程，保留原来每任务关闭循环时的清理效果。"""
    leftovers = [task for task in asyncio.all_tasks(loop) if not task.done()]
    if not leftovers:
        return
    for task in leftovers:
        task.cancel()
    loop.run_until_complete(asyncio.gather(*leftovers, return_exceptions=True))
    for task in leftovers:
        if not task.cancelled() and task.exception() is not None:
            loop.call_exception_handler(
                {
                    "message": "a task left running by a finished Celery task raised",
                    "exception": task.exception(),
                    "task": task,
                }
            )


def shutdown() -> None:
    """关闭当前进程的事件循环；下次 run 会重新创建。

    由 Celery 的 worker_process_shutdown 触发，确保子进程退出时显式清理，而不是留给解释器兜底。"""
    global _runner, _runner_pid, _runner_thread
    if _runner is not None and _runner_pid == os.getpid():
        _runner.close()
    _runner = None
    _runner_pid = None
    _runner_thread = None
