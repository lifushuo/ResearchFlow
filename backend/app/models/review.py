"""人工审核、最终产物与审计模型。

reviews 是唯一批准来源；结论批注没有 decision，不能当成另一套审批。最终产物通过冗余的批准状态与复合外键，只能引用真正的报告批准。
审计事件不外键绑定被操作对象，避免删对象时同时抹掉审计。"""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import (
    Boolean,
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
from app.models.types import BigIntAutoType, JsonType, UuidType

REVIEW_GATES = ("PLAN", "REPORT")
REVIEW_DECISIONS = ("APPROVED", "REWORK_REQUESTED", "REJECTED")
ANNOTATION_KINDS = ("FLAG_UNSUPPORTED", "REQUEST_EVIDENCE", "COMMENT")


class Review(Base):
    """人工对某个版本对象作出的决定，是系统信任边界。

    RESTRICT 外键阻止普通级联删除悄悄擦掉审核记录；需要删除时由明确的应用流程处理，而不是让数据库自动连带清空。"""

    __tablename__ = "reviews"

    id: Mapped[uuid.UUID] = mapped_column(UuidType, primary_key=True, default=uuid.uuid4)
    # 审核还必须关联整个 RUN，而不只是被审核版本。
    # 计划挂在 research_plans、报告挂在 revisions；只查一个父表会漏掉另一关卡的决定。
    run_id: Mapped[uuid.UUID] = mapped_column(
        UuidType, ForeignKey("research_runs.id", ondelete="RESTRICT"), nullable=False
    )
    # sequence 给同次运行的决定建立完整顺序。时间可能相同，uuid5 也没有插入顺序含义，不能拿它们排序代替。
    sequence: Mapped[int] = mapped_column(Integer, nullable=False)
    # 计划审核发生在草稿生成前，因此 revision_id 为空是正常情况。
    revision_id: Mapped[uuid.UUID | None] = mapped_column(
        UuidType, ForeignKey("revisions.id", ondelete="RESTRICT"), nullable=True
    )
    reviewer_id: Mapped[uuid.UUID] = mapped_column(
        UuidType, ForeignKey("users.id", ondelete="RESTRICT"), nullable=False
    )
    plan_version_id: Mapped[uuid.UUID | None] = mapped_column(
        UuidType, ForeignKey("research_plans.id", ondelete="RESTRICT"), nullable=True
    )
    gate: Mapped[str] = mapped_column(String(8), nullable=False)
    decision: Mapped[str] = mapped_column(String(20), nullable=False)
    feedback: Mapped[str | None] = mapped_column(Text, nullable=True)
    # 哈希含义随 gate 区分：报告关卡绑定 Revision.report_markdown，计划关卡绑定标准化计划。
    # 不能把两种含义混在一个未校验的 draft_hash 中。
    reviewed_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    __table_args__ = (
        # 产物复合外键同时引用审核的 gate 和 decision，只限制批准状态不足以排除计划批准。
        UniqueConstraint("id", "decision", "gate", name="uq_review_decision"),
        UniqueConstraint("run_id", "sequence", name="uq_review_sequence"),
        CheckConstraint(_in("gate", REVIEW_GATES), name="ck_review_gate"),
        CheckConstraint(_in("decision", REVIEW_DECISIONS), name="ck_review_decision"),
        CheckConstraint("length(reviewed_hash) = 64", name="ck_review_hash"),
        CheckConstraint("(gate = 'PLAN') = (plan_version_id IS NOT NULL)", name="ck_review_plan"),
        # 报告审核必须有关联修订，计划审核必须没有，与 ck_review_plan 对应。
        CheckConstraint("(gate = 'REPORT') = (revision_id IS NOT NULL)", name="ck_review_report"),
        CheckConstraint("sequence >= 1", name="ck_review_sequence"),
        Index("ix_review_revision", "revision_id", "created_at"),
        Index("ix_review_run", "run_id", "sequence"),
    )


# 每个修订最多一个报告批准，但允许多次返工审核。
# 索引在类定义后配置，并同时指定两种数据库条件，避免误变为全表唯一而破坏返工。
Index(
    "uq_review_approval",
    Review.__table__.c.revision_id,
    unique=True,
    postgresql_where=(Review.__table__.c.decision == "APPROVED")
    & (Review.__table__.c.gate == "REPORT"),
    sqlite_where=(Review.__table__.c.decision == "APPROVED")
    & (Review.__table__.c.gate == "REPORT"),
)


class ClaimAnnotation(Base):
    """审核者对单条结论的建议，不具有批准权。

    不新增 Review.gate 类型，也没有 decision 字段。批注随结论级联删除，不自动跨修订继承，避免编造结论谱系。"""

    __tablename__ = "claim_annotations"

    id: Mapped[uuid.UUID] = mapped_column(UuidType, primary_key=True, default=uuid.uuid4)
    claim_id: Mapped[uuid.UUID] = mapped_column(
        UuidType, ForeignKey("claims.id", ondelete="CASCADE"), nullable=False, index=True
    )
    author_id: Mapped[uuid.UUID] = mapped_column(
        UuidType, ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    kind: Mapped[str] = mapped_column(String(24), nullable=False)
    note: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    __table_args__ = (CheckConstraint(_in("kind", ANNOTATION_KINDS), name="ck_annotation_kind"),)


class ResearchArtifact(Base):
    """不可变、独立且可通过哈希验证的获批记录。

    payload 是冻结快照，不重新连接实时表读取；项目重命名或删除不能改变其内容。
    owner_id 是持续保留的归属锚点；其他对象删除后 ID 可置空，产物仍不能脱离用户隔离。"""

    __tablename__ = "research_artifacts"

    id: Mapped[uuid.UUID] = mapped_column(UuidType, primary_key=True, default=uuid.uuid4)
    owner_id: Mapped[uuid.UUID] = mapped_column(
        UuidType, ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    run_id: Mapped[uuid.UUID | None] = mapped_column(
        UuidType, ForeignKey("research_runs.id", ondelete="SET NULL"), nullable=True
    )
    project_id: Mapped[uuid.UUID | None] = mapped_column(
        UuidType, ForeignKey("projects.id", ondelete="SET NULL"), nullable=True
    )
    revision_id: Mapped[uuid.UUID | None] = mapped_column(
        UuidType, ForeignKey("revisions.id", ondelete="SET NULL"), nullable=True
    )
    review_id: Mapped[uuid.UUID | None] = mapped_column(UuidType, nullable=True)
    review_decision: Mapped[str] = mapped_column(String(20), nullable=False)
    # 冗余保存报告关卡并约束它，让计划批准无法授权产物，而不只是靠调用方自觉检查。
    review_gate: Mapped[str] = mapped_column(String(8), nullable=False, default="REPORT")

    # 沿用 research_engine.bundle 格式，共用同一结构和校验器。
    format_version: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    payload: Mapped[dict] = mapped_column(JsonType, nullable=False)
    artifact_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    demo: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    __table_args__ = (
        # 关键数据库约束：批准必须是实际存在的审核事实。
        ForeignKeyConstraint(
            ["review_id", "review_decision", "review_gate"],
            ["reviews.id", "reviews.decision", "reviews.gate"],
            ondelete="SET NULL",
            name="fk_artifact_review",
        ),
        CheckConstraint("review_decision = 'APPROVED'", name="ck_artifact_approved"),
        # 复合外键要求关卡与决定匹配真实审核，且关卡固定为 REPORT，计划批准无法通过。
        CheckConstraint("review_gate = 'REPORT'", name="ck_artifact_gate"),
        UniqueConstraint("artifact_hash", name="uq_artifact_hash"),
        CheckConstraint("format_version >= 1", name="ck_artifact_format"),
        CheckConstraint("length(artifact_hash) = 64", name="ck_artifact_hashlen"),
        # 产物在项目删除后仍可保留，因此不能靠项目关联作为唯一访问入口。
        Index("ix_artifact_owner", "owner_id", "created_at"),
    )


# 运行存在时最多一个产物；采用部分索引，让运行删除、run_id 变空后允许保留多个独立产物。
Index(
    "uq_artifact_run",
    ResearchArtifact.__table__.c.run_id,
    unique=True,
    postgresql_where=ResearchArtifact.__table__.c.run_id.isnot(None),
    sqlite_where=ResearchArtifact.__table__.c.run_id.isnot(None),
)


class AuditEvent(Base):
    """只追加的责任审计记录；审核决定本身保存在 Review。

    subject_type、subject_id 是多态且不设外键，使被操作对象删除后仍能保留审计，不受其他对象级联影响。"""

    __tablename__ = "audit_events"

    id: Mapped[int] = mapped_column(BigIntAutoType, primary_key=True, autoincrement=True)
    actor_id: Mapped[uuid.UUID | None] = mapped_column(
        UuidType, ForeignKey("users.id", ondelete="SET NULL"), nullable=True
    )
    action: Mapped[str] = mapped_column(String(48), nullable=False)
    subject_type: Mapped[str] = mapped_column(String(32), nullable=False)
    subject_id: Mapped[uuid.UUID | None] = mapped_column(UuidType, nullable=True)
    metadata_json: Mapped[dict | None] = mapped_column("metadata", JsonType, nullable=True)
    occurred_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    __table_args__ = (
        Index("ix_audit_subject", "subject_type", "subject_id", "id"),
        Index("ix_audit_actor", "actor_id", "occurred_at"),
    )
