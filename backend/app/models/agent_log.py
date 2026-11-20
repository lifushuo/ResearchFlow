"""持久化代理事件流（docs/05 §1，docs/02 §5）。

每个事件先入库再发布到 Redis，SSE 重连从这些记录回放；递增 ID 同时作为 Last-Event-ID 游标。"""

import uuid
from datetime import datetime

from sqlalchemy import DateTime, String, func
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base
from app.models.types import BigIntAutoType, JsonType, UuidType


class AgentLog(Base):
    __tablename__ = "agent_logs"

    id: Mapped[int] = mapped_column(BigIntAutoType, primary_key=True, autoincrement=True)
    # 事件所属对象可能是 sessions.id，也可能是 research_runs.id。
    #
    # 该多态字段不设外键，因为单个外键不能同时指向两张运行表。
    # 删除时由各入口显式清理，与检查点清理原则相同；否则原生运行的日志会因错误的会话外键而无法保存。
    session_id: Mapped[uuid.UUID] = mapped_column(UuidType, nullable=False, index=True)
    event_type: Mapped[str] = mapped_column(
        String(40), nullable=False
    )  # 事件类型包含 agent_log、llm_usage、人工关卡及终态；用量只含安全元数据。
    agent_name: Mapped[str | None] = mapped_column(String(50), nullable=True)
    payload: Mapped[dict] = mapped_column(JsonType, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), index=True
    )

    def __repr__(self) -> str:
        return f"<AgentLog id={self.id} type={self.event_type} session={self.session_id}>"
