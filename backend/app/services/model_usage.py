"""复用已有运行事件表读取用量，两种主机共用，不新增数据库结构。"""

from sqlalchemy import select

from app.models.agent_log import AgentLog
from research_engine.token_usage import summarize_usage


async def load_model_usage(db, run_id):
    payloads = (
        (
            await db.execute(
                select(AgentLog.payload)
                .where(AgentLog.session_id == run_id, AgentLog.event_type == "llm_usage")
                .order_by(AgentLog.id)
            )
        )
        .scalars()
        .all()
    )
    return summarize_usage([row.get("detail") for row in payloads if isinstance(row, dict)])
