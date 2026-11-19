"""ORM 模型注册入口。

这里导入的模型会加入 Base.metadata，供服务端 Alembic 自动生成迁移以及桌面 create_all 创建表。"""

from app.models.agent_log import AgentLog
from app.models.audit_log import AuditLog
from app.models.base import Base
from app.models.chat_message import ChatMessage
from app.models.chat_thread import ChatThread
from app.models.memory_chunk import MemoryChunk
from app.models.project import Project
from app.models.project_memory import ProjectMemoryItem, ProjectMemoryProvenance
from app.models.refresh_token import RefreshToken
from app.models.research import (
    Contradiction,
    Evidence,
    ResearchPlan,
    ResearchRun,
    Source,
)
from app.models.review import AuditEvent, ClaimAnnotation, ResearchArtifact, Review
from app.models.revision import Claim, ClaimEvidenceLink, Revision
from app.models.session import Session, SessionStatus
from app.models.user import User

# 统一定义 SQLite 不支持的表。
#:
# 这些表自身带 pgvector 列或引用此类表，桌面 create_all 必须统一跳过，不能在 sidecar 再维护一份名单。
#:
# 桌面按设计不提供项目记忆；本地语料有独立 SQLite 与向量实现（internal/12 M10）。
POSTGRES_ONLY_TABLES = frozenset(
    {
        MemoryChunk.__tablename__,
        ProjectMemoryItem.__tablename__,
        # 来源关系指向的表不存在时，该关系表也必须一起排除。
        ProjectMemoryProvenance.__tablename__,
    }
)

__all__ = [
    "Base",
    "POSTGRES_ONLY_TABLES",
    "User",
    "Project",
    "Session",
    "SessionStatus",
    "AgentLog",
    "ChatMessage",
    "ChatThread",
    "MemoryChunk",
    "AuditLog",
    "RefreshToken",
    # 注册研究领域模型。
    "ResearchRun",
    "ResearchPlan",
    "Source",
    "Evidence",
    "Contradiction",
    "Revision",
    "Claim",
    "ClaimEvidenceLink",
    "Review",
    "ClaimAnnotation",
    "ResearchArtifact",
    "AuditEvent",
    "ProjectMemoryItem",
    "ProjectMemoryProvenance",
]
