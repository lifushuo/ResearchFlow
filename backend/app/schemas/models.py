"""模型路由接口共用的请求和响应结构。

两种主机必须返回一致字段；这些模型只依赖 Pydantic，不能依赖服务端 app.config 或 app.db，否则缺少数据库和 JWT 设置的桌面无法导入。路由模块继续重新导出以兼容旧位置。"""

from __future__ import annotations

from typing import Literal

from pydantic import AnyHttpUrl, BaseModel, Field


class ModelInfo(BaseModel):
    route: str
    provider: str
    model_id: str
    display_name: str
    input_per_mtok: float | None
    output_per_mtok: float | None
    context_window: int | None
    max_output_tokens: int | None
    supports_tools: bool
    supports_structured_output: bool
    notes: str
    available: bool = Field(description="Whether this user currently has a usable key.")


class CatalogResponse(BaseModel):
    roles: list[str]
    models: list[ModelInfo]
    presets: dict[str, dict[str, dict[str, str]]]
    preset_names: list[str]
    available_providers: list[str]
    # 当前运行实际使用的路由：用户保存值优先，否则部署默认值。
    effective_routing: dict[str, str]
    # 用户没有独立偏好、跟随部署默认时为 None。
    user_routing: dict[str, str] | None
    deployment_routing: dict[str, str]


class LocalModelInfo(BaseModel):
    name: str
    size_bytes: int | None
    route: str | None
    in_catalog: bool
    likely_underpowered: bool
    is_embedding: bool
    params_b: float | None


class LocalLLMStatusResponse(BaseModel):
    configured_base_url: str
    reachable: bool
    usable: bool
    models: list[LocalModelInfo]
    error: str | None
    hint: str | None
    # 区分未安装和未启动，两者需要不同操作，不能都叫未检测到（internal/07 Phase 2b）。
    install_state: Literal["running", "installed_not_running", "not_installed"]


class CustomEndpointStatusResponse(BaseModel):
    """自定义兼容端点实际报告的模型 ID 列表。

    不套用目录 ModelInfo，因为端点没有可验证的价格、上下文或能力数据；不能补造零价格冒充免费。"""

    configured_base_url: str
    reachable: bool
    models: list[str]
    error: str | None
    hint: str | None


class ReadinessResponse(BaseModel):
    """当前用户是否真正具备启动研究的条件（docs/17 §8a）。"""

    ready: bool
    # 分别报告哪些条件满足，便于界面给出具体下一步，不能只说未配置。
    has_cloud_key: bool
    local_reachable: bool
    local_chat_models: int


class ProviderTestRequest(BaseModel):
    """保存前待探测的密钥（internal/07 Phase 2a）。"""

    model_config = {"str_strip_whitespace": True}

    provider: Literal["google", "anthropic", "openai", "openrouter", "custom"]
    api_key: str = Field(min_length=8, max_length=500)
    api_base_url: AnyHttpUrl | None = Field(default=None)


class RoutingRequest(BaseModel):
    routing: dict[str, str]


class RoutingResponse(BaseModel):
    routing: dict[str, str] | None
    effective_routing: dict[str, str]


class PromptDefault(BaseModel):
    """单个角色可编辑的默认提示词（范围冻结 §12，D-1）。"""

    role: str
    # 移除不可信内容说明后的默认文本，也就是用户实际替换的部分。
    default_prompt: str
    # 若系统会在替换内容前后补上安全说明，则为 True。
    untrusted_content_framed: bool


class PromptDefaultsResponse(BaseModel):
    """提示词编辑器的初始默认内容。

    由构建决定，对所有用户和主机相同，不包含用户状态、受保护用途或哈希；用户覆盖另由 GET /auth/me 获取。"""

    max_chars: int
    roles: list[PromptDefault]
