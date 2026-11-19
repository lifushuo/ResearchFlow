"""模型读取最近的聊天窗口，但按实际发生顺序提供。

先倒序取最新 limit 条，再反转成从旧到新；直接升序 LIMIT 只会永久取最早消息，长对话连当前问题都看不到。
报告聊天、项目线程和桌面统一复用。"""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.chat_message import ChatMessage

# 统一限制模型可见消息数，维持连续性同时约束上下文和费用，两个聊天入口共用。
HISTORY_LIMIT = 20


async def recent_turns(db: AsyncSession, where, *, limit: int = HISTORY_LIMIT) -> list[ChatMessage]:
    """选取 where 范围内最新 limit 条消息，再按旧到新返回。

    报告通过 session_id 限定，项目线程通过 thread_id 限定，其余逻辑相同。"""
    newest_first = (
        (
            await db.execute(
                select(ChatMessage)
                .where(where)
                .order_by(ChatMessage.created_at.desc())
                .limit(limit)
            )
        )
        .scalars()
        .all()
    )
    return list(reversed(newest_first))
