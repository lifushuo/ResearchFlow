"""研究运行及其收集的数据模型。

使用复合外键保证来源、证据等属于同一次运行：子行的 run_id 必须同时满足自身范围和父对象归属。为此设置的 UNIQUE(id, run_id) 并非多余。
约束兼容 Postgres 与 SQLite；不可变性由应用和独立证据包校验器保证，避免依赖某种数据库专有触发器或列权限。"""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import (
    BigInteger,
    Boolean,
    CheckConstraint,
    DateTime,
    ForeignKey,
    ForeignKeyConstraint,
    Index,
    Integer,
    Numeric,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base
from app.models.types import JsonType, UuidType
from app.services.run_config import PROMPT_OVERRIDE_STATUSES

# 领域状态词汇表
#
# 状态定义为模块常量，数据库 CHECK 和业务代码共用，避免数据库接受应用不认识的值。

RUN_STATUSES = (
    "PENDING",
    "RUNNING",
    "AWAITING_PLAN",
    "AWAITING_REVIEW",
    "COMPLETED",
    "FAILED",
    "CANCELLED",
)
RESEARCH_DEPTHS = ("fast", "balanced", "comprehensive")
PLAN_ORIGINS = ("MODEL_PROPOSED", "HUMAN_EDITED", "TEMPLATE", "UNKNOWN")
SOURCE_KINDS = ("WEB", "CORPUS")
# corpus:// 来源应允许 CORPUS_DOCUMENT。
# 常量必须包含此值，否则真实语料结果写入会被 CHECK 约束拒绝，不能因为过去未触达该路径就假定合法。
RETRIEVAL_STATUSES = ("FETCHED", "SEARCH_RESULT_ONLY", "FAILED", "UNKNOWN", "CORPUS_DOCUMENT")
PROVENANCE_STATES = ("ATTESTED", "UNATTESTED", "UNCHECKED")
ATTESTATION_GRADES = ("FETCHED_BODY", "SEARCH_SNIPPET", "CORPUS_DOCUMENT")
CONTRADICTION_DETECTION = ("DETECTED", "NOT_RUN", "DETECTOR_UNAVAILABLE")
CONTRADICTION_DIMENSIONS = (
    "TIMEFRAME",
    "METHODOLOGY",
    "POPULATION",
    "WORKLOAD",
    "SOURCE_QUALITY",
    "UNCLASSIFIED",
)
CONTRADICTION_REVIEW_STATES = ("UNREVIEWED", "ACKNOWLEDGED", "DISMISSED")

# 记录是否真正从图检查点读取过本运行的证据。
#:
# READ 表示读取成功，evidence 表中的零条或多条都是实测结果。
# 读取失败意味着证据数量未知，空表不能当实测为零；run_bundle 必须拒绝据此生成虚假的引用证明。
#:
# NOT_READ 表示还未产生可读取证据，例如停在计划关卡、取消或图产出状态前就失败。
EVIDENCE_OUTCOMES = ("NOT_READ", "READ", "CHECKPOINT_MISSING", "CHECKPOINT_UNREADABLE")


def _in(column: str, values: tuple[str, ...]) -> str:
    """生成跨数据库兼容的 col IN 条件片段。"""
    return f"{column} IN (" + ", ".join(f"'{v}'" for v in values) + ")"


class ResearchRun(Base):
    """记录一次研究执行本身，不等同于最终报告结果。"""

    __tablename__ = "research_runs"

    id: Mapped[uuid.UUID] = mapped_column(UuidType, primary_key=True, default=uuid.uuid4)
    project_id: Mapped[uuid.UUID] = mapped_column(
        UuidType, ForeignKey("projects.id", ondelete="CASCADE"), nullable=False, index=True
    )
    # 将项目 owner_id 冗余保存在运行表，让每次权限校验只查一张表，减少漏掉隔离条件的风险。
    owner_id: Mapped[uuid.UUID] = mapped_column(
        UuidType, ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )

    question: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(String(20), nullable=False)
    depth: Mapped[str] = mapped_column(String(16), nullable=False)

    corpus_mode: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    demo: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    skip_plan_gate: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    topic_seeds: Mapped[list | None] = mapped_column(JsonType, nullable=True)
    outline_template: Mapped[str | None] = mapped_column(String(64), nullable=True)
    model_routing: Mapped[dict | None] = mapped_column(JsonType, nullable=True)

    # 提示词覆盖配置在首次开始时冻结，此后不重新读取。
    # 恢复必须重建原运行，而不是套用最新用户偏好。旧运行字段为 NULL 表示当时使用项目默认提示词，不能事后回填猜测值。
    effective_prompt_overrides: Mapped[dict | None] = mapped_column(JsonType, nullable=True)

    # 按用途保存运行中实际组成的系统提示词，包含用途、角色、策略、是否覆盖和有效提示词。
    #:
    # 来源必须在执行当时记录，不能导出时重新组成。
    # 运行到导出之间可能升级代码，重新组成只会反映今天的默认值，不能证明历史模型真正收到过什么。
    #:
    # 只保存实际执行过的提示词用途，条件分支未执行就没有记录。
    # 旧运行来源缺失无法补造，继续使用证据包 v1；不能为了生成 v2 而伪造历史。
    effective_prompt_provenance: Mapped[dict | None] = mapped_column(JsonType, nullable=True)

    # 保存 PROMPT_OVERRIDE_STATUSES 中的配置状态；旧运行可为空。
    # 不能仅根据空快照推断，因为未配置和配置损坏都可能回退默认提示词，用户需要区分。
    prompt_overrides_status: Mapped[str | None] = mapped_column(String(16), nullable=True)

    cost_usd: Mapped[float] = mapped_column(Numeric(12, 6), nullable=False, default=0)
    tokens_input: Mapped[int] = mapped_column(BigInteger, nullable=False, default=0)
    tokens_output: Mapped[int] = mapped_column(BigInteger, nullable=False, default=0)
    elapsed_seconds: Mapped[float | None] = mapped_column(Numeric(12, 3), nullable=True)
    # NULL 表示未测量，不能用 0 代替。
    citation_resolution_rate: Mapped[float | None] = mapped_column(Numeric(5, 4), nullable=True)

    # 图结果保存时一次性写入证据读取状态，定义见 EVIDENCE_OUTCOMES。
    evidence_outcome: Mapped[str] = mapped_column(
        String(24), nullable=False, default="NOT_READ", server_default="NOT_READ"
    )

    # 记录实际读取的语料库及打开时状态；网页模式和早期运行可为空，不补造零值。
    corpus_id: Mapped[str | None] = mapped_column(Text, nullable=True)
    corpus_version: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # 语料运行在创建时冻结请求选择及解析到的具体文档版本，见 corpus_selection.py。
    # 每次开始和恢复读取同一快照，不重新解析；网页和旧运行为空，旧版仍按全库语义处理。
    corpus_selection: Mapped[dict | None] = mapped_column(JsonType, nullable=True)

    # 记录用户是否要求离线保证；启用时引擎只接受本地模型（RFC V3.1 §6.2）。
    # 所选文件运行另保存从图状态复制的逐条证据核验记录；审批必须确认记录核验过当前修订正文，其他运行保持空值。
    grounding_report: Mapped[dict | None] = mapped_column(JsonType, nullable=True)

    offline: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False, server_default="false"
    )

    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    # 取消状态持久化在数据库行中，不依赖会过期的缓存键。
    cancelled_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    cancel_requested_by: Mapped[uuid.UUID | None] = mapped_column(
        UuidType, ForeignKey("users.id", ondelete="SET NULL"), nullable=True
    )
    archived_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )

    __table_args__ = (
        CheckConstraint(_in("status", RUN_STATUSES), name="ck_run_status"),
        CheckConstraint(_in("evidence_outcome", EVIDENCE_OUTCOMES), name="ck_run_evidence_outcome"),
        # NULL 在这里是合法业务值，不等于尚未初始化，含义见列说明。
        CheckConstraint(
            "prompt_overrides_status IS NULL OR "
            + _in("prompt_overrides_status", PROMPT_OVERRIDE_STATUSES),
            name="ck_run_prompt_overrides_status",
        ),
        CheckConstraint(_in("depth", RESEARCH_DEPTHS), name="ck_run_depth"),
        # 取消状态和取消时间必须同时存在或同时不存在。
        # Postgres 布尔与 SQLite 的 0/1 通过相等比较实现同一约束。
        CheckConstraint(
            "(status = 'CANCELLED') = (cancelled_at IS NOT NULL)", name="ck_run_cancelled"
        ),
        CheckConstraint(
            "cost_usd >= 0 AND tokens_input >= 0 AND tokens_output >= 0", name="ck_run_metrics"
        ),
        CheckConstraint(
            "citation_resolution_rate IS NULL OR "
            "(citation_resolution_rate >= 0 AND citation_resolution_rate <= 1)",
            name="ck_run_resolution",
        ),
        # 历史列表常用查询：同项目、未归档、按创建时间倒序。
        Index("ix_run_project_recent", "project_id", "archived_at", "created_at"),
        Index("ix_run_owner_status", "owner_id", "status"),
    )


class ResearchPlan(Base):
    """带版本且不可变的模型计划或人工编辑。

    只插入新版本，不覆盖 plan_json，以便同时保留模型提案和人工修改。"""

    __tablename__ = "research_plans"

    id: Mapped[uuid.UUID] = mapped_column(UuidType, primary_key=True, default=uuid.uuid4)
    run_id: Mapped[uuid.UUID] = mapped_column(
        UuidType, ForeignKey("research_runs.id", ondelete="CASCADE"), nullable=False
    )
    version: Mapped[int] = mapped_column(Integer, nullable=False)
    tasks: Mapped[list] = mapped_column(JsonType, nullable=False)
    outline_sections: Mapped[list] = mapped_column(JsonType, nullable=False)
    origin: Mapped[str] = mapped_column(String(16), nullable=False)
    approved_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    __table_args__ = (
        UniqueConstraint("run_id", "version", name="uq_plan_version"),
        CheckConstraint("version >= 1", name="ck_plan_version"),
        # UNKNOWN 仅兼容过去未记录计划来源的数据，新代码不得写入。
        CheckConstraint(_in("origin", PLAN_ORIGINS), name="ck_plan_origin"),
    )


# 部分唯一索引保证每次运行最多一个获批计划。
#
# 索引条件需要映射完成后的列对象，因此在类定义后声明。
# Postgres 与 SQLite 的条件都要填写，漏一项会把部分索引变成全表唯一，错误阻止第二个未批准版本。
Index(
    "uq_plan_approved",
    ResearchPlan.__table__.c.run_id,
    unique=True,
    postgresql_where=ResearchPlan.__table__.c.approved_at.isnot(None),
    sqlite_where=ResearchPlan.__table__.c.approved_at.isnot(None),
)


class Source(Base):
    """一次检索得到的文档，也是引用最终解析到的来源。"""

    __tablename__ = "sources"

    id: Mapped[uuid.UUID] = mapped_column(UuidType, primary_key=True, default=uuid.uuid4)
    run_id: Mapped[uuid.UUID] = mapped_column(
        UuidType, ForeignKey("research_runs.id", ondelete="CASCADE"), nullable=False
    )
    url: Mapped[str] = mapped_column(Text, nullable=False)
    normalized_url: Mapped[str] = mapped_column(Text, nullable=False)
    title: Mapped[str | None] = mapped_column(Text, nullable=True)
    kind: Mapped[str] = mapped_column(String(8), nullable=False)
    # 必须区分网页正文与搜索摘要，证据核验等级依赖此差别。
    retrieval_status: Mapped[str] = mapped_column(String(20), nullable=False)
    # NULL 表示检索到但尚未被综合器赋予引用编号，例如综合前失败或未引用该页。
    # 不能自行补号，否则是在伪造执行历史。
    # .
    citation_index: Mapped[int | None] = mapped_column(Integer, nullable=True)
    # 不设文档外键：删除上传文件不能使历史来源记录失效。
    corpus_document_id: Mapped[str | None] = mapped_column(Text, nullable=True)
    retrieved_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    __table_args__ = (
        UniqueConstraint("run_id", "normalized_url", name="uq_source_url"),
        # 作为证据复合外键的目标，证明父子来源属于同一次运行，并非重复主键。
        UniqueConstraint("id", "run_id", name="uq_source_run"),
        CheckConstraint(_in("kind", SOURCE_KINDS), name="ck_source_kind"),
        CheckConstraint(_in("retrieval_status", RETRIEVAL_STATUSES), name="ck_source_ret"),
        CheckConstraint("citation_index IS NULL OR citation_index >= 1", name="ck_source_cidx"),
        CheckConstraint(
            "(kind = 'CORPUS') = (corpus_document_id IS NOT NULL)", name="ck_source_corpus"
        ),
    )


# 只对非空引用编号建立运行内唯一约束；未引用来源允许多条。
# 必须同时配置两种数据库的部分索引条件，避免某个主机错误使用全表唯一。
Index(
    "uq_source_index",
    Source.__table__.c.run_id,
    Source.__table__.c.citation_index,
    unique=True,
    postgresql_where=Source.__table__.c.citation_index.isnot(None),
    sqlite_where=Source.__table__.c.citation_index.isnot(None),
)


class Evidence(Base):
    """不可变的证据片段及其来源核验状态。

    引擎可能清空虚构片段；这里保存原文字并标为 UNATTESTED，以便审计，不能销毁错误证据。"""

    __tablename__ = "evidence"

    id: Mapped[uuid.UUID] = mapped_column(UuidType, primary_key=True, default=uuid.uuid4)
    run_id: Mapped[uuid.UUID] = mapped_column(
        UuidType, ForeignKey("research_runs.id", ondelete="CASCADE"), nullable=False
    )
    source_id: Mapped[uuid.UUID] = mapped_column(UuidType, nullable=False)
    # sequence 只保证顺序，不保证连续。并发收集无需为凑连续编号额外加锁；按 sequence、id 排序处理同序号情况。
    sequence: Mapped[int] = mapped_column(BigInteger, nullable=False)
    task_id: Mapped[str | None] = mapped_column(Text, nullable=True)
    snippet: Mapped[str] = mapped_column(Text, nullable=False)
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    key_fact: Mapped[str | None] = mapped_column(Text, nullable=True)

    provenance_state: Mapped[str] = mapped_column(String(12), nullable=False)
    attested_against: Mapped[str | None] = mapped_column(String(20), nullable=True)
    attestation_run_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    __table_args__ = (
        ForeignKeyConstraint(
            ["source_id", "run_id"],
            ["sources.id", "sources.run_id"],
            ondelete="CASCADE",
            name="fk_evidence_source",
        ),
        UniqueConstraint("run_id", "sequence", name="uq_evidence_seq"),
        UniqueConstraint("id", "run_id", name="uq_evidence_run"),
        # 用字段和约束表达来源核验状态。
        CheckConstraint(_in("provenance_state", PROVENANCE_STATES), name="ck_ev_state"),
        CheckConstraint(
            "(provenance_state = 'UNCHECKED') = (attestation_run_at IS NULL)",
            name="ck_ev_unchecked",
        ),
        CheckConstraint(
            "(attested_against IS NOT NULL) = (provenance_state = 'ATTESTED')",
            name="ck_ev_grade",
        ),
        CheckConstraint(
            "attested_against IS NULL OR " + _in("attested_against", ATTESTATION_GRADES),
            name="ck_ev_grade_vocab",
        ),
        CheckConstraint("sequence >= 1", name="ck_ev_seq"),
        Index("ix_evidence_source", "source_id"),
    )


class Contradiction(Base):
    """检测到的两段带来源引文之间的冲突，保存原结果，不在此裁决。

    检测器只看到来源 URL 与片段分组，不知道具体证据行；因此来源锚点是确定信息，证据锚点只在引文恰好匹配一条证据时补充，不能伪装成更高定位精度。"""

    __tablename__ = "contradictions"

    id: Mapped[uuid.UUID] = mapped_column(UuidType, primary_key=True, default=uuid.uuid4)
    run_id: Mapped[uuid.UUID] = mapped_column(
        UuidType, ForeignKey("research_runs.id", ondelete="CASCADE"), nullable=False, index=True
    )
    # 来源锚点是权威定位；validate_pairs 已排除输入中不存在的 URL。
    source_a_id: Mapped[uuid.UUID | None] = mapped_column(UuidType, nullable=True)
    source_b_id: Mapped[uuid.UUID | None] = mapped_column(UuidType, nullable=True)
    # 证据锚点是进一步定位；零条或多条匹配时保持 NULL。
    evidence_a_id: Mapped[uuid.UUID | None] = mapped_column(UuidType, nullable=True)
    evidence_b_id: Mapped[uuid.UUID | None] = mapped_column(UuidType, nullable=True)
    # 保存检测器引用的原文及两侧概括，忠实记录它说过什么，不把模型概括当世界事实。
    # 这里使用 summary，Claim 专指报告中的结论句。
    quote_a: Mapped[str | None] = mapped_column(Text, nullable=True)
    quote_b: Mapped[str | None] = mapped_column(Text, nullable=True)
    summary_a: Mapped[str | None] = mapped_column(Text, nullable=True)
    summary_b: Mapped[str | None] = mapped_column(Text, nullable=True)
    # 保存两侧不能同时成立的解释，并用于报告冲突区展示。
    nature: Mapped[str | None] = mapped_column(Text, nullable=True)
    dimension: Mapped[str | None] = mapped_column(String(20), nullable=True)
    # 检测器未运行与运行后未发现冲突是不同结果。
    detection_state: Mapped[str] = mapped_column(String(24), nullable=False)
    review_state: Mapped[str] = mapped_column(String(16), nullable=False, default="UNREVIEWED")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    __table_args__ = (
        ForeignKeyConstraint(
            ["source_a_id", "run_id"],
            ["sources.id", "sources.run_id"],
            ondelete="CASCADE",
            name="fk_contra_src_a",
        ),
        ForeignKeyConstraint(
            ["source_b_id", "run_id"],
            ["sources.id", "sources.run_id"],
            ondelete="CASCADE",
            name="fk_contra_src_b",
        ),
        ForeignKeyConstraint(
            ["evidence_a_id", "run_id"],
            ["evidence.id", "evidence.run_id"],
            ondelete="CASCADE",
            name="fk_contra_a",
        ),
        ForeignKeyConstraint(
            ["evidence_b_id", "run_id"],
            ["evidence.id", "evidence.run_id"],
            ondelete="CASCADE",
            name="fk_contra_b",
        ),
        CheckConstraint(_in("detection_state", CONTRADICTION_DETECTION), name="ck_contra_state"),
        CheckConstraint(
            "dimension IS NULL OR " + _in("dimension", CONTRADICTION_DIMENSIONS),
            name="ck_contra_dim",
        ),
        CheckConstraint(_in("review_state", CONTRADICTION_REVIEW_STATES), name="ck_contra_review"),
        # DETECTED 表示检测器在它实际处理的来源粒度上发现了冲突对。
        CheckConstraint(
            "(detection_state = 'DETECTED') = "
            "(source_a_id IS NOT NULL AND source_b_id IS NOT NULL)",
            name="ck_contra_pair",
        ),
        # 进一步证据定位必须成对成功，不能只关联一侧。
        CheckConstraint(
            "(evidence_a_id IS NULL) = (evidence_b_id IS NULL)", name="ck_contra_refine"
        ),
        CheckConstraint(
            "source_a_id IS NULL OR source_a_id <> source_b_id", name="ck_contra_src_distinct"
        ),
        CheckConstraint(
            "evidence_a_id IS NULL OR evidence_a_id <> evidence_b_id", name="ck_contra_distinct"
        ),
    )
