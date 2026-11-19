"""报告版本和结论关系图。

revisions 不包含可修改字段，状态由审核、运行状态和版本位置推导，避免另存状态后不一致。
claim_evidence_links 是权威结论关联，正文 [n] 是它的展示，不再只把文字标记当全部证据数据。"""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import (
    BigInteger,
    CheckConstraint,
    DateTime,
    ForeignKey,
    ForeignKeyConstraint,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base
from app.models.research import _in
from app.models.types import JsonType, UuidType

CLAIM_EXTRACTION_METHODS = ("DERIVED_FROM_REPORT", "MODEL_STRUCTURED", "HUMAN_EDITED")
CLAIM_VERIFICATION_STATES = (
    "SUPPORTED",
    "UNSUPPORTED",
    "INSUFFICIENT_EVIDENCE",
    "UNCHECKED",
)
CLAIM_VERIFICATION_METHODS = ("NUMERIC_GROUNDING", "MODEL_JUDGE", "NOT_RUN")
LINK_STANCES = ("SUPPORTS", "CONTRADICTS", "CONTEXT")
LINK_ORIGINS = ("CITATION_MARKER", "MODEL_ASSERTED", "HUMAN_ASSERTED")


class Revision(Base):
    """一版报告，不是整个研究状态快照。

    证据和来源属于运行，返工只回到综合器，继续使用已有证据。"""

    __tablename__ = "revisions"

    id: Mapped[uuid.UUID] = mapped_column(UuidType, primary_key=True, default=uuid.uuid4)
    run_id: Mapped[uuid.UUID] = mapped_column(
        UuidType, ForeignKey("research_runs.id", ondelete="CASCADE"), nullable=False
    )
    version: Mapped[int] = mapped_column(Integer, nullable=False)
    # 保存人工批准的原文字节且不可变，使审核哈希永久对应报告，而不是下次返工前碰巧一致。
    report_markdown: Mapped[str] = mapped_column(Text, nullable=False)
    report_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    # 结构化 ReportDocument 从 report_markdown 派生，不反过来修改正文。
    # 早期版本为空，不能补造；权威哈希仍对 Markdown 原文计算。
    report_document: Mapped[dict | None] = mapped_column(JsonType, nullable=True)
    # 综合时可见的最大 evidence.sequence，是阈值而非条数，允许序号空缺。
    # 不作为外键，也不赋予某条证据特殊身份；无证据时 0 合法。
    evidence_watermark: Mapped[int] = mapped_column(BigInteger, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    __table_args__ = (
        UniqueConstraint("run_id", "version", name="uq_revision_version"),
        UniqueConstraint("id", "run_id", name="uq_revision_run"),
        CheckConstraint("version >= 1", name="ck_revision_version"),
        CheckConstraint("length(report_hash) = 64", name="ck_revision_hash"),
        CheckConstraint("evidence_watermark >= 0", name="ck_revision_wm"),
        # 按运行读取最新报告修订，是详情页常用查询。
        Index("ix_revision_run_version", "run_id", "version"),
    )


class Claim(Base):
    """属于一个报告修订的持久结论。

    目前从正文提取，lineage_id 全部为空。无法仅凭相似文字证明新版与旧版是同一断言；只能在综合器真正输出结构化谱系后填写，不能模糊匹配回填。"""

    __tablename__ = "claims"

    id: Mapped[uuid.UUID] = mapped_column(UuidType, primary_key=True, default=uuid.uuid4)
    revision_id: Mapped[uuid.UUID] = mapped_column(UuidType, nullable=False)
    run_id: Mapped[uuid.UUID] = mapped_column(UuidType, nullable=False)
    position: Mapped[int] = mapped_column(Integer, nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    extraction_method: Mapped[str] = mapped_column(String(24), nullable=False)
    verification_state: Mapped[str] = mapped_column(String(24), nullable=False)
    verification_method: Mapped[str] = mapped_column(String(20), nullable=False)
    lineage_id: Mapped[uuid.UUID | None] = mapped_column(UuidType, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    __table_args__ = (
        ForeignKeyConstraint(
            ["revision_id", "run_id"],
            ["revisions.id", "revisions.run_id"],
            ondelete="CASCADE",
            name="fk_claim_revision",
        ),
        UniqueConstraint("revision_id", "position", name="uq_claim_position"),
        UniqueConstraint("id", "run_id", name="uq_claim_run"),
        CheckConstraint("position >= 0", name="ck_claim_position"),
        CheckConstraint(
            _in("extraction_method", CLAIM_EXTRACTION_METHODS), name="ck_claim_extract"
        ),
        CheckConstraint(
            _in("verification_state", CLAIM_VERIFICATION_STATES), name="ck_claim_state"
        ),
        CheckConstraint(
            _in("verification_method", CLAIM_VERIFICATION_METHODS), name="ck_claim_method"
        ),
        # 未检查的结论不能同时声称已执行某种核验，测量状态与方法记录必须一致。
        CheckConstraint(
            "(verification_state = 'UNCHECKED') = (verification_method = 'NOT_RUN')",
            name="ck_claim_unchecked",
        ),
        # lineage_id 当前全部为空，暂不建索引以免徒增写入成本；有真实非空写入后再建。
    )


class ClaimEvidenceLink(Base):
    """结论与证据的权威关联，正文 [n] 只是展示。

    两个复合外键共用同一个 run_id，因此两端必须属于同次运行，从结构上禁止跨运行串证据。"""

    __tablename__ = "claim_evidence_links"

    id: Mapped[uuid.UUID] = mapped_column(UuidType, primary_key=True, default=uuid.uuid4)
    run_id: Mapped[uuid.UUID] = mapped_column(UuidType, nullable=False)
    claim_id: Mapped[uuid.UUID] = mapped_column(UuidType, nullable=False)
    evidence_id: Mapped[uuid.UUID] = mapped_column(UuidType, nullable=False)
    stance: Mapped[str] = mapped_column(String(12), nullable=False)
    # CITATION_MARKER 明确表示关联来自文字引用标记，不是模型或人工已经判断内容一致。
    origin: Mapped[str] = mapped_column(String(20), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    __table_args__ = (
        ForeignKeyConstraint(
            ["claim_id", "run_id"],
            ["claims.id", "claims.run_id"],
            ondelete="CASCADE",
            name="fk_link_claim",
        ),
        ForeignKeyConstraint(
            ["evidence_id", "run_id"],
            ["evidence.id", "evidence.run_id"],
            ondelete="CASCADE",
            name="fk_link_evidence",
        ),
        UniqueConstraint("claim_id", "evidence_id", name="uq_link"),
        CheckConstraint(_in("stance", LINK_STANCES), name="ck_link_stance"),
        CheckConstraint(_in("origin", LINK_ORIGINS), name="ck_link_origin"),
        Index("ix_link_claim", "claim_id"),
        # 反向索引支持查询哪些结论引用了这条证据，供证据侧栏使用。
        Index("ix_link_evidence", "evidence_id"),
    )
