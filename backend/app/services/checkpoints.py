"""清理 LangGraph 检查点。

检查点保存在独立表中，以 thread_id（即会话 ID）为键，没有指向 sessions 的外键。
删除会话行不会清除包含完整智能体历史和已抓取网页内容的图状态。
因此 API 必须在成功删除会话后调用此模块，确保删除操作不会遗留研究内容。"""

from __future__ import annotations

import structlog

from app.config import settings

logger = structlog.get_logger()


def _dsn() -> str:
    # LangGraph 的 Postgres 检查点存储使用 psycopg，而非 asyncpg。
    return settings.database_url.replace("postgresql+asyncpg://", "postgresql://")


async def delete_thread(thread_id: str) -> None:
    """删除一个会话的全部检查点状态。失败时抛出异常，由调用方记录日志。"""
    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

    async with AsyncPostgresSaver.from_conn_string(_dsn()) as saver:
        await saver.adelete_thread(thread_id)
    logger.info("checkpoints_deleted", thread_id=thread_id)


async def get_thread_state(thread_id: str) -> dict:
    """从检查点存储读取最新的流水线状态。"""
    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

    from research_engine.graph import build_graph

    async with AsyncPostgresSaver.from_conn_string(_dsn()) as saver:
        graph = build_graph(saver)
        # aget_state 返回 StateSnapshot；此处只读取其 values。
        snapshot = await graph.aget_state({"configurable": {"thread_id": thread_id}})
        return snapshot.values if snapshot else {}
