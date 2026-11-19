"""项目记忆中的一段获批报告向量。

只在批准操作后写入，草稿和失败报告根本不进入存储，从数据来源保证检索可信。"""

import uuid
from datetime import datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import DateTime, ForeignKey, Integer, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base
from app.models.types import UuidType

# 固定向量维度，与 0007 迁移及 app.adapters 的供应商输出配置一致。
EMBEDDING_DIMENSIONS = 768


class MemoryChunk(Base):
    __tablename__ = "memory_chunks"

    id: Mapped[uuid.UUID] = mapped_column(UuidType, primary_key=True, default=uuid.uuid4)
    project_id: Mapped[uuid.UUID] = mapped_column(
        UuidType,
        ForeignKey("projects.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    # 记录文本所属报告及来源链，使检索引用可回到报告自身的证据来源。
    #
    # 来源 ID 同时支持研究运行和会话，因此不设单表外键；由 ORM 删除逻辑清理。
    # project_id 仍在数据库级联删除，保证删项目时不残留记忆。
    source_report_id: Mapped[uuid.UUID] = mapped_column(UuidType, nullable=False, index=True)
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    embedding: Mapped[list[float]] = mapped_column(Vector(EMBEDDING_DIMENSIONS), nullable=False)
    # 记录生成向量的模型，例如 ollama:nomic-embed-text。同维度不代表可比较，检索必须按模型筛选。
    embedding_model: Mapped[str] = mapped_column(String(120), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    project: Mapped[Project] = relationship("Project", back_populates="memory_chunks")  # noqa: F821

    def __repr__(self) -> str:
        return f"<MemoryChunk id={self.id} project={self.project_id} index={self.chunk_index}>"
