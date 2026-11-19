"""项目聊天线程及记忆状态结构（docs/14 §8）。

与 research.py 分开，因为这里讨论整个项目，不是单次研究。"""

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field

from app.services.chat_scope import ChatScope


class ThreadCreateRequest(BaseModel):
    model_config = {"str_strip_whitespace": True}

    # 标题可选；未提供时从首条消息生成，不强迫用户开聊前先命名。
    title: str | None = Field(default=None, max_length=200)


class ThreadResponse(BaseModel):
    id: UUID
    project_id: UUID
    title: str
    message_count: int = 0
    created_at: datetime
    last_message_at: datetime

    model_config = {"from_attributes": True}


class ThreadListResponse(BaseModel):
    threads: list[ThreadResponse]
    total: int


class Citation(BaseModel):
    """将 R 编号定位到获批报告。

    excerpt 保存检索原片段，用户可以核对结论依据，而不只是看到装饰性引用（docs/27）。"""

    marker: str
    session_id: UUID
    title: str
    created_at: datetime
    excerpt: str


class ThreadMessageSchema(BaseModel):
    id: UUID
    thread_id: UUID | None = None
    role: str
    content: str
    citations: list[Citation] | None = None
    created_at: datetime

    model_config = {"from_attributes": True}


class ThreadMessageRequest(BaseModel):
    model_config = {"str_strip_whitespace": True}

    message: str = Field(..., min_length=1, max_length=4000)
    # 定义本轮允许读取的范围，和报告聊天使用同一套 scope 值及 chat_scope.py 解释。
    # 此处 report 指项目内获批研究。
    scope: ChatScope = "report"


class MemoryModelBreakdown(BaseModel):
    embedding_model: str
    chunks: int
    reports: int


class MemoryStatusResponse(BaseModel):
    """让记忆状态可见（docs/14 §8）。

    pending_reports 披露摄取失败，stale_models 披露旧模型向量尚未重建，避免只能从聊天找不到内容才发现缺口。"""

    available: bool
    chunk_count: int
    indexed_reports: int
    approved_reports: int
    pending_reports: int
    current_model: str
    models: list[MemoryModelBreakdown]
    stale_models: list[str]
    last_ingest_at: datetime | None = None
