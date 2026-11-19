"""两种主机、四个入口共用的 SSE 帧生成器。

统一执行连接通知、历史回放、当前状态判断和实时监听；历史数据库查询与实时订阅由各主机传入。
terminal_stop 用于实时流，包含人工关卡；replay_stop 用于历史，只含真正终态。混用会使重连在旧计划关卡停止，漏掉后续执行。
仅依赖标准库，桌面不能因此导入 app.config。"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Iterable, Sequence


def _frame(event_id: int | None, payload: dict) -> str:
    """生成单条 SSE 消息。

    没有持久化 ID 时不输出 id 行：仍可把临时事件交给在线用户，但不能让客户端把不存在的数据库行当作恢复游标。"""
    body = f"data: {json.dumps(payload)}\n\n"
    return body if event_id is None else f"id: {event_id}\n{body}"


async def sse_frames(
    *,
    connected: dict,
    backlog: Iterable[tuple[int | None, dict]],
    live: AsyncIterator[tuple[int | None, dict]],
    replay_stop: Sequence[str],
    terminal_stop: Sequence[str],
    already_done: bool,
    seen_from: int = 0,
) -> AsyncIterator[str]:
    """先回放历史，再按独立停止列表监听实时消息。

    seen_from 是客户端 Last-Event-ID，实时订阅可能包含快照中已见事件，需去重。already_done 由调用方根据当前状态决定；任务终结或停在关卡时回放完即可关闭，不再空等。"""
    yield _frame(None, connected)

    seen_max = seen_from
    for event_id, payload in backlog:
        if event_id is not None:
            seen_max = max(seen_max, event_id)
        yield _frame(event_id, payload)
        if payload.get("type") in replay_stop:
            return

    if already_done:
        return

    async for event_id, payload in live:
        # 先订阅再读取历史快照，因此两者有重叠，同一事件可能收到两次。
        if event_id is not None and event_id <= seen_max:
            continue
        yield _frame(event_id, payload)
        if event_id is not None:
            seen_max = event_id
        if payload.get("type") in terminal_stop:
            return
