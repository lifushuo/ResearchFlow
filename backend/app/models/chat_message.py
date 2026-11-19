import uuid
from datetime import datetime

from sqlalchemy import CheckConstraint, DateTime, ForeignKey, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base
from app.models.types import JsonType, UuidType


class ChatMessage(Base):
    """聊天消息只能归属于报告会话或项目线程之一。

    0008 迁移的 CHECK 要求恰好一个父对象，避免两者皆空时出现不在任何历史列表中的消息。"""

    __tablename__ = "chat_messages"
    __table_args__ = (
        CheckConstraint(
            "(session_id IS NOT NULL) <> (thread_id IS NOT NULL)",
            name="one_parent",  # 命名约定会将此约束生成为 ck_chat_messages_one_parent。
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(UuidType, primary_key=True, default=uuid.uuid4)
    session_id: Mapped[uuid.UUID | None] = mapped_column(
        UuidType,
        ForeignKey("sessions.id", ondelete="CASCADE"),
        nullable=True,
        index=True,
    )
    thread_id: Mapped[uuid.UUID | None] = mapped_column(
        UuidType,
        ForeignKey("chat_threads.id", ondelete="CASCADE"),
        nullable=True,
        index=True,
    )
    role: Mapped[str] = mapped_column(
        String(50), nullable=False
    )  # 角色为 user（用户）或 assistant（助手）。
    content: Mapped[str] = mapped_column(Text, nullable=False)
    # 项目线程消息保存回答实际引用的报告信息，写入时解析完成。
    # 即使原始检索早已结束，引用标记仍能定位报告（docs/14 §5）。
    citations: Mapped[list | None] = mapped_column(JsonType, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    # ORM 关系
    session: Mapped[Session | None] = relationship("Session", back_populates="chat_messages")  # noqa: F821
    thread: Mapped[ChatThread | None] = relationship("ChatThread", back_populates="messages")  # noqa: F821

    def __repr__(self) -> str:
        return f"<ChatMessage id={self.id} role={self.role}>"
