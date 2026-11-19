"""两个主机共用的会话启动状态更新，必须让用户取消优先。

会话没有运行表那样的取消约束，所有写入都需遵守 cancelled_at。创建提交到 Worker 接手之间可能发生取消，不能再直接写 RUNNING。
本模块不依赖 app.config、Redis、Celery，供两种主机共用。"""

from __future__ import annotations

import uuid

from sqlalchemy import update
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.session import Session, SessionStatus


async def mark_running(db: AsyncSession, session_id: uuid.UUID) -> bool:
    """仅在用户尚未取消时改为 RUNNING，成功更新返回 True。

    取消条件放在 UPDATE 内，防止读取与写入间发生竞争；不更新旧内存对象，需要最新状态的调用方重新查行。"""
    result = await db.execute(
        update(Session)
        .where(Session.id == session_id, Session.cancelled_at.is_(None))
        .values(status=SessionStatus.RUNNING)
        .execution_options(synchronize_session=False)
    )
    return result.rowcount == 1
