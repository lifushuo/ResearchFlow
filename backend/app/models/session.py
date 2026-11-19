import enum
import uuid
from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, Integer, Numeric, String, Text, func
from sqlalchemy import Enum as SAEnum
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base
from app.models.types import JsonType, UuidType


class SessionStatus(enum.StrEnum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    # 规划完成后停在研究设计关卡，尚未开始搜索，等待审核任务与大纲。
    # 与报告 AWAITING_APPROVAL 分开，因为恢复负载不同，不能给不存在的草稿展示批准按钮。
    AWAITING_PLAN = "AWAITING_PLAN"
    AWAITING_APPROVAL = "AWAITING_APPROVAL"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class Session(Base):
    __tablename__ = "sessions"

    id: Mapped[uuid.UUID] = mapped_column(UuidType, primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(
        UuidType,
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    # 每个会话必须归属项目；0005 迁移已将旧数据放入用户默认 General 项目，所以该字段非空。
    project_id: Mapped[uuid.UUID] = mapped_column(
        UuidType,
        ForeignKey("projects.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    prompt: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[SessionStatus] = mapped_column(
        SAEnum(SessionStatus, name="session_status"),
        nullable=False,
        default=SessionStatus.PENDING,
        index=True,
    )
    research_depth: Mapped[str] = mapped_column(String(20), nullable=False, default="balanced")
    draft_report: Mapped[str | None] = mapped_column(Text, nullable=True)
    final_report: Mapped[str | None] = mapped_column(Text, nullable=True)
    # 前端展示的编号来源表，包含 index、url、title、snippet（docs/05 §1）。
    sources: Mapped[list | None] = mapped_column(JsonType, nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    # 金额使用精确 Numeric，不使用浮点数（docs/05 §5）。
    total_cost_usd: Mapped[float] = mapped_column(Numeric(10, 6), nullable=False, default=0)
    total_tokens_input: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    total_tokens_output: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    elapsed_seconds: Mapped[float | None] = mapped_column(Numeric(10, 2), nullable=True)
    rework_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    # 归档保留记录且可恢复；删除是独立操作。
    # 用时间戳而非布尔值，便于知道何时离开活动列表。
    archived_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    # 用户取消时间；从未取消则为 NULL。
    #:
    # 取消必须持久化，旧的短时缓存标记不能跨 Worker 重启保护状态。
    # 结果写入器会先检查 is_cancelled，避免在途结果返回后把已停止会话改回其他状态。
    #:
    # 与运行侧使用相同取消时间字段；既能追溯停止时间，也能区分用户取消与执行失败。
    cancelled_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    # 保存实际生成本报告的模型快照，而不是查询用户当前偏好（docs/12 M8）。
    # 设置事后可变化，但报告来源必须一直可追溯。
    model_routing: Mapped[dict | None] = mapped_column(JsonType, nullable=True)

    # 报告引用可解析比例允许为空。
    # NULL 表示没测量或没有可测引用；0.0 表示已测且所有标记都无法解析，两者不能混用。
    citation_resolution_rate: Mapped[float | None] = mapped_column(Numeric(5, 4), nullable=True)

    # 隔离的语料库模式（internal/12 M10）。
    corpus_mode: Mapped[bool] = mapped_column(nullable=False, server_default="false")

    # 记录实际用了脚本模型和受控检索，而非真实供应商（docs/17 §6.2）。
    # 必须保存此事实，不能导出时再看进程环境猜测，确保演示报告始终带正确标记。
    demo: Mapped[bool] = mapped_column(nullable=False, server_default="false")

    # 记录用户配置过可用的提示词覆盖，但旧会话流程未应用。
    # 会话按设计不支持运行侧的覆盖契约，必须如实披露，不能让用户误以为配置已生效。
    prompt_overrides_not_applied: Mapped[bool] = mapped_column(
        nullable=False, server_default="false", default=False
    )

    # 计划审核关卡（internal/07 Phase 4）。
    # plan_json、outline_json 保存人工编辑后实际决定的内容，而非模型最初提案，与模型路由快照原则一致。
    plan_json: Mapped[dict | None] = mapped_column(JsonType, nullable=True)
    outline_json: Mapped[dict | None] = mapped_column(JsonType, nullable=True)
    plan_approved_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    # 创建入口会明确写入是否跳过计划关卡。
    # 数据库默认 false 仅用于绕过正常入口创建的记录，保留迁移已发布的更安全默认值。
    #
    # 请求和 RunConfig 默认 true、即跳过关卡，是为了兼容无法处理暂停的旧脚本、CLI 和评测。
    # 产品表单显式传 false，不能为了表面统一而把三个默认值改成一样。
    skip_plan_gate: Mapped[bool] = mapped_column(nullable=False, server_default="false")
    # 保存研究开始时用户提出的主题和大纲要求，区别于关卡处的最终决定。
    # 恢复时 RunConfig 从行记录重建，若只存在于初始请求中，第二次执行就会丢失。
    topic_seeds: Mapped[list | None] = mapped_column(JsonType, nullable=True)
    outline_template: Mapped[str | None] = mapped_column(String(64), nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    @property
    def is_cancelled(self) -> bool:
        """两个主机共用的用户取消判定。

        结果写入规则在 pipeline_runner._persist_outcome 与 sidecar._apply_outcome，判定本身不能再各写一份。"""
        return self.cancelled_at is not None

    # 普通关系让数据库 ON DELETE CASCADE 清理，避免 ORM 为删除而加载所有子行（docs/05 §5）。
    user: Mapped[User] = relationship("User", back_populates="sessions")  # noqa: F821
    project: Mapped[Project] = relationship("Project", back_populates="sessions")  # noqa: F821
    # agent_logs.session_id 同时引用会话和运行，无单表外键。
    # 显式给出关联条件，并由 ORM 删除孤儿日志，确保删除会话时移除其轨迹。
    agent_logs: Mapped[list[AgentLog]] = relationship(  # noqa: F821
        "AgentLog",
        primaryjoin="Session.id == foreign(AgentLog.session_id)",
        cascade="all, delete-orphan",
        lazy="select",
    )
    # memory_chunks.source_report_id 也是多态引用，没有数据库外键级联；删除会话后必须清理索引文本，避免聊天继续引用无法打开的报告。
    #
    # 此关系只读，删除由两种主机的 memory.purge_report 显式完成。
    # 不能配置无条件 ORM 级联，否则桌面删除会话时会查询并不存在的 pgvector 表而报错。
    memory_chunks: Mapped[list[MemoryChunk]] = relationship(  # noqa: F821
        "MemoryChunk",
        primaryjoin="Session.id == foreign(MemoryChunk.source_report_id)",
        viewonly=True,
        lazy="select",
        overlaps="project,memory_chunks",
    )
    chat_messages: Mapped[list[ChatMessage]] = relationship(  # noqa: F821
        "ChatMessage",
        back_populates="session",
        cascade="all, delete-orphan",
        passive_deletes=True,
        lazy="select",
        order_by="ChatMessage.created_at",
    )

    def __repr__(self) -> str:
        return f"<Session id={self.id} status={self.status}>"
