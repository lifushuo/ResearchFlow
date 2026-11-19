"""项目记忆及其指向获批产物的来源关系。

来源表只允许引用 ResearchArtifact，而产物必须有报告批准，因此“记忆只来自获批研究”由结构保证，不能指向草稿、运行或聊天消息。
使用关联表而非单个 artifact_id，允许一条知识将来关联多个产物，不把当前切块方式误当业务约束。
向量列使用 pgvector，仅 Postgres 提供；桌面通过 POSTGRES_ONLY_TABLES 同时排除记忆和来源表。"""

from __future__ import annotations

import uuid
from datetime import datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    CheckConstraint,
    DateTime,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base
from app.models.memory_chunk import EMBEDDING_DIMENSIONS
from app.models.types import UuidType


class ProjectMemoryItem(Base):
    """可检索的一段获批知识，完全派生自最终产物。"""

    __tablename__ = "project_memory_items"

    id: Mapped[uuid.UUID] = mapped_column(UuidType, primary_key=True, default=uuid.uuid4)
    project_id: Mapped[uuid.UUID] = mapped_column(
        UuidType, ForeignKey("projects.id", ondelete="CASCADE"), nullable=False, index=True
    )
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    embedding: Mapped[list[float]] = mapped_column(Vector(EMBEDDING_DIMENSIONS), nullable=False)
    # 不同模型向量即使同维度也不能混用，检索按生成模型过滤。
    embedding_model: Mapped[str] = mapped_column(String(128), nullable=False)
    # 对有序的来源产物哈希集合计算 SHA-256，保证重复摄取可识别；也可从来源表重新计算。
    provenance_digest: Mapped[str] = mapped_column(String(64), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    __table_args__ = (
        UniqueConstraint(
            "project_id",
            "embedding_model",
            "provenance_digest",
            "chunk_index",
            name="uq_memory_chunk",
        ),
        CheckConstraint("chunk_index >= 0", name="ck_memory_chunk"),
        CheckConstraint("length(provenance_digest) = 64", name="ck_memory_digest"),
    )


class ProjectMemoryProvenance(Base):
    """记录一条记忆派生自哪些获批产物。

    每条持久记忆必须至少关联一个获批 ResearchArtifact；数据库不能直接要求子行存在，这部分由应用层保证。"""

    __tablename__ = "project_memory_provenance"

    memory_item_id: Mapped[uuid.UUID] = mapped_column(
        UuidType,
        ForeignKey("project_memory_items.id", ondelete="CASCADE"),
        primary_key=True,
    )
    artifact_id: Mapped[uuid.UUID] = mapped_column(
        UuidType,
        ForeignKey("research_artifacts.id", ondelete="CASCADE"),
        primary_key=True,
    )
