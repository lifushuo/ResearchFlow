from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import AnyHttpUrl, BaseModel, EmailStr, Field, field_validator

from app.services.passwords import MIN_LENGTH
from research_engine.prompt_composition import ROLES


class ConnectionVerdict(BaseModel):
    """供应商连接检查的 API 结构（internal/07 Phase 2a）。

    保留三态：ok 正常；degraded 表示有响应但密钥、配额或服务有问题；failed 表示完全未响应。不能只用布尔值混淆处理方式。"""

    state: Literal["ok", "degraded", "failed"]
    reason: str
    checked_at: str
    model_count: int | None = None


class RegisterRequest(BaseModel):
    email: EmailStr
    # 此处校验长度下限；泄露密码名单和字节上限由服务层检查。
    password: str = Field(min_length=MIN_LENGTH, max_length=200)


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


# 每个角色可保存的替换提示词长度上限（范围冻结 §7，D-2）。
#:
# 上限 2500，允许用户写出与最长默认提示词 2431 字符相当的内容。
# 限制主要控制证据包体积，而非上下文窗口，因此逐角色限制，不另设五个角色总长度。
MAX_PROMPT_OVERRIDE_CHARS = 2500


class UserPreferences(BaseModel):
    """用户可调整的研究与界面设置（internal/07 Phase 3）。

    全部可选，None 表示使用默认值。尚未被界面读取的字段也先校验并保存，保证设置页写入的是真实配置。"""

    model_config = {"extra": "forbid"}

    retrieval_k: int | None = Field(default=None, ge=1, le=20)
    min_sources_per_task: int | None = Field(default=None, ge=0, le=20)
    snippet_max_chars: int | None = Field(default=None, ge=100, le=500)
    density: Literal["comfortable", "compact"] | None = None
    tavily_api_key: str | None = Field(default=None, max_length=200)
    brave_api_key: str | None = Field(default=None, max_length=200)

    # 按角色保存替换系统提示词；缺省使用项目默认值。
    #:
    # 键仅允许五个公开角色，不允许填写内部用途。
    # 具体哪个用途可被覆盖由 prompt_composition.is_overridable 决定，critic 的受保护提示词不会因此开放。
    prompt_overrides: dict[str, str | None] | None = None

    @field_validator("prompt_overrides", mode="before")
    @classmethod
    def _validate_prompt_overrides(cls, value: object) -> object:
        """校验角色名与提示词内容，明确告诉用户错误原因及允许值。

        两种主机共用此验证器，避免一端给通用错误、另一端给专门提示而形成不同契约。"""
        if value is None:
            return None
        if not isinstance(value, dict):
            raise ValueError("prompt_overrides must be an object keyed by agent role")
        for role, text in value.items():
            if role not in ROLES:
                raise ValueError(f"unknown agent role {role!r}; valid roles are {', '.join(ROLES)}")
            if text is None:
                continue  # 重置该角色，恢复项目默认提示词。
            if not isinstance(text, str):
                raise ValueError(f"prompt override for {role!r} must be a string or null")
            if not text.strip():
                raise ValueError(
                    f"prompt override for {role!r} is empty; send null to reset it to the "
                    "shipped prompt"
                )
            if len(text) > MAX_PROMPT_OVERRIDE_CHARS:
                raise ValueError(
                    f"prompt override for {role!r} is {len(text)} characters; the maximum "
                    f"is {MAX_PROMPT_OVERRIDE_CHARS}"
                )
        return value


class UserResponse(BaseModel):
    id: UUID
    email: str
    is_active: bool
    created_at: datetime

    # 公开资料
    display_name: str | None = None
    avatar_url: str | None = None
    monthly_token_limit: int = 0

    # 用户自带密钥的状态，不包含密钥原文。
    api_key_provider: str | None = None
    api_key_base_url: str | None = None
    api_key_hint: str | None = None
    api_key_label: str | None = None
    api_key_set_at: datetime | None = None
    # 只有保存密钥接口填写本次探测结果，避免显示未经验证的连接成功。
    # 其他返回此结构的接口填 None（internal/07 Phase 2a）。
    connection_verdict: ConnectionVerdict | None = None

    preferences: UserPreferences = Field(default_factory=UserPreferences)

    model_config = {"from_attributes": True}

    @field_validator("preferences", mode="before")
    @classmethod
    def preferences_default_when_unset(cls, v: dict | None) -> dict:
        """未修改设置的旧账号在数据库中为 NULL，响应转换成完整默认偏好对象，而不是报结构校验错误。"""
        return v or {}


class ProfileUpdate(BaseModel):
    """可编辑资料；请求未提供的字段保持不变。"""

    model_config = {"str_strip_whitespace": True}

    display_name: str | None = Field(default=None, max_length=80)
    avatar_url: str | None = Field(default=None, max_length=2000)
    monthly_token_limit: int | None = Field(default=None, ge=0)
    # 与已保存偏好合并，不能整份替换，避免研究设置覆盖外观设置。
    preferences: UserPreferences | None = None

    @field_validator("avatar_url")
    @classmethod
    def avatar_must_be_http(cls, v: str | None) -> str | None:
        """头像用于 img，只允许 http(s)，拒绝 javascript: 和 data: 地址，避免危险内容被存入后回显。"""
        if v is None or v == "":
            return None
        if not v.startswith(("http://", "https://")):
            raise ValueError("Avatar URL must start with http:// or https://")
        return v


class PasswordChangeRequest(BaseModel):
    """修改密码需证明知道旧密码，避免被盗会话 Cookie 升级成永久账号接管。"""

    current_password: str = Field(min_length=1, max_length=200)
    new_password: str = Field(min_length=MIN_LENGTH, max_length=200)


class ApiKeyRequest(BaseModel):
    """用户自带供应商密钥，加密存储，响应绝不返回。"""

    model_config = {"str_strip_whitespace": True}

    provider: Literal["google", "anthropic", "openai", "openrouter", "custom"]
    api_key: str = Field(min_length=8, max_length=500)
    api_base_url: AnyHttpUrl | None = Field(default=None)


class ApiKeyLabelRequest(BaseModel):
    """只重命名连接，不重新验证密钥，也不再次调用供应商探测。与 ApiKeyRequest 明确分开。"""

    model_config = {"str_strip_whitespace": True}

    # 空名称表示恢复目录默认标签，与清空头像使用相同的空值约定。
    label: str | None = Field(default=None, max_length=60)

    @field_validator("label")
    @classmethod
    def blank_clears(cls, v: str | None) -> str | None:
        return v or None


class UsageWindow(BaseModel):
    tokens_input: int = 0
    tokens_output: int = 0
    tokens_total: int = 0
    cost_usd: float = 0.0
    sessions: int = 0


class UsageResponse(BaseModel):
    """个人资料页展示的 Token 与费用用量（docs/07）。"""

    month: UsageWindow  # 当前自然月，和月度限额使用同一窗口。
    week: UsageWindow  # 滚动最近 7 天。
    last_session: UsageWindow  # 只统计最近一次会话。
    monthly_token_limit: int = 0  # 0 表示不限。
    limit_remaining: int | None = None  # 不限额时为 None。
    limit_reached: bool = False
