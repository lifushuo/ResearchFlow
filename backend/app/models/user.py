import uuid
from datetime import datetime

from sqlalchemy import Boolean, DateTime, Integer, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base
from app.models.types import JsonType, UuidType


class User(Base):
    __tablename__ = "users"

    id: Mapped[uuid.UUID] = mapped_column(UuidType, primary_key=True, default=uuid.uuid4)
    email: Mapped[str] = mapped_column(String(255), unique=True, nullable=False, index=True)
    hashed_pw: Mapped[str] = mapped_column(String(255), nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    # 公开用户资料；上面的 id 是界面展示的唯一用户标识。
    display_name: Mapped[str | None] = mapped_column(String(80), nullable=True)
    # 头像可选；为空时前端使用姓名首字母，不影响完整展示。
    avatar_url: Mapped[str | None] = mapped_column(Text, nullable=True)

    # 用户自带密钥以 Fernet 密文存储，接口绝不返回明文。
    # hint 只保留用于识别的末尾显示片段（docs/06 §1）。
    api_key_encrypted: Mapped[str | None] = mapped_column(Text, nullable=True)
    api_key_provider: Mapped[str | None] = mapped_column(String(20), nullable=True)
    api_key_base_url: Mapped[str | None] = mapped_column(Text, nullable=True)
    api_key_hint: Mapped[str | None] = mapped_column(String(16), nullable=True)
    api_key_set_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    # 用户给当前连接起的名称，区别于供应商固定目录标签。
    # NULL 表示显示目录默认名称；删除密钥时一起清空，因为连接已不存在。
    api_key_label: Mapped[str | None] = mapped_column(String(60), nullable=True)

    # 滚动月度总 Token 上限；自托管默认 0 表示不限。
    monthly_token_limit: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    # 按角色保存 provider:model 路由（docs/12 M8）。
    # NULL 表示跟随部署 MODEL_* 默认值；写入前按目录校验，拒绝不可路由或不满足计价要求的选择。
    model_routing: Mapped[dict | None] = mapped_column(JsonType, nullable=True)

    # 研究和界面偏好使用 JSON，由 UserPreferences 校验结构（internal/07 Phase 3）。
    # 避免每增加一个选项就迁移表；NULL 表示全部未设置，读取时采用默认值。
    preferences: Mapped[dict | None] = mapped_column(JsonType, nullable=True)

    sessions: Mapped[list[Session]] = relationship(  # noqa: F821
        "Session",
        back_populates="user",
        cascade="all, delete-orphan",
        passive_deletes=True,
        lazy="select",
    )

    def __repr__(self) -> str:
        return f"<User id={self.id} email={self.email}>"
