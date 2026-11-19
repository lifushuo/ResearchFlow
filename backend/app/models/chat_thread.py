"""项目范围的聊天线程（docs/14 §3）。

报告聊天只围绕一份报告，项目线程围绕项目全部获批研究。多个线程相互独立，可分别讨论不同研究问题。"""

import uuid
from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, String, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base
from app.models.types import UuidType

# 从首条消息生成标题，不强迫用户预先命名，也不为纯展示额外调用模型。
TITLE_MAX_CHARS = 60


class ChatThread(Base):
    __tablename__ = "chat_threads"

    id: Mapped[uuid.UUID] = mapped_column(UuidType, primary_key=True, default=uuid.uuid4)
    project_id: Mapped[uuid.UUID] = mapped_column(
        UuidType,
        ForeignKey("projects.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    title: Mapped[str] = mapped_column(String(200), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    # 线程列表按最后使用时间排序，而不是创建时间。
    last_message_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    project: Mapped[Project] = relationship("Project", back_populates="chat_threads")  # noqa: F821
    messages: Mapped[list[ChatMessage]] = relationship(  # noqa: F821
        "ChatMessage",
        back_populates="thread",
        cascade="all, delete-orphan",
        passive_deletes=True,
        lazy="select",
    )

    def __repr__(self) -> str:
        return f"<ChatThread id={self.id} title={self.title!r}>"


def derive_title(first_message: str) -> str:
    """用开场消息生成标题，按单词边界截断。"""
    cleaned = " ".join((first_message or "").split()) or "New chat"
    if len(cleaned) <= TITLE_MAX_CHARS:
        return cleaned
    cut = cleaned.rfind(" ", 0, TITLE_MAX_CHARS)
    return cleaned[: cut if cut > TITLE_MAX_CHARS // 2 else TITLE_MAX_CHARS].rstrip() + "…"
