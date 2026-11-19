"""项目 API 数据契约（docs/14 §8）。"""

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field, field_validator


class ProjectCreateRequest(BaseModel):
    model_config = {"str_strip_whitespace": True}

    name: str = Field(..., min_length=1, max_length=120)
    description: str | None = Field(default=None, max_length=2000)

    @field_validator("name")
    @classmethod
    def _not_blank(cls, v: str) -> str:
        # 先去掉首尾空白，再校验最小长度，避免纯空格名称漏过检查。
        if not v.strip():
            raise ValueError("Project name cannot be blank.")
        return v


class ProjectUpdateRequest(BaseModel):
    model_config = {"str_strip_whitespace": True}

    name: str | None = Field(default=None, min_length=1, max_length=120)
    description: str | None = Field(default=None, max_length=2000)
    archived: bool | None = None


class ProjectResponse(BaseModel):
    id: UUID
    name: str
    description: str | None = None
    archived_at: datetime | None = None
    created_at: datetime
    # 列表中附带项目统计，避免客户端为每个项目额外请求一次计数。
    session_count: int = 0

    model_config = {"from_attributes": True}


class ProjectListResponse(BaseModel):
    projects: list[ProjectResponse]
    total: int
