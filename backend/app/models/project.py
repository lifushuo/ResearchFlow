"""项目是用户研究的容器（docs/14 §3）。

删除会同时删除所属研究，因此属于破坏性操作，不能当成简单整理列表。"""

import uuid
from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, String, Text, UniqueConstraint, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base
from app.models.types import UuidType


class Project(Base):
    __tablename__ = "projects"
    # 同一用户的项目名不区分大小写。迁移用函数索引强制保证，避免界面出现仅大小写不同的重复项目。
    __table_args__ = (UniqueConstraint("user_id", "name", name="uq_projects_user_name"),)

    id: Mapped[uuid.UUID] = mapped_column(UuidType, primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(
        UuidType,
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    name: Mapped[str] = mapped_column(String(120), nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    # 归档只隐藏、可恢复；删除真正移除数据。
    archived_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    sessions: Mapped[list[Session]] = relationship(  # noqa: F821
        "Session",
        back_populates="project",
        cascade="all, delete-orphan",
        passive_deletes=True,
        lazy="select",
    )
    # 项目记忆随项目一起删除，ORM 和数据库级联双重保证，不留下孤立向量。
    memory_chunks: Mapped[list[MemoryChunk]] = relationship(  # noqa: F821
        "MemoryChunk",
        back_populates="project",
        cascade="all, delete-orphan",
        passive_deletes=True,
        lazy="select",
    )
    chat_threads: Mapped[list[ChatThread]] = relationship(  # noqa: F821
        "ChatThread",
        back_populates="project",
        cascade="all, delete-orphan",
        passive_deletes=True,
        lazy="select",
    )

    def __repr__(self) -> str:
        return f"<Project id={self.id} name={self.name!r}>"
