"""用户与会话模型路由（docs/12 M8）。

优先级为会话开始快照、用户偏好、部署 MODEL_*。服务层统一实现，确保接口和 Worker 解析一致。"""

from __future__ import annotations

from app.config import settings
from app.models.user import User

# 规则实现在引擎层，桌面复用同一实现。
# 此处重新导出兼容既有服务端调用位置，避免桌面复制后漏掉自定义端点和带标签的 Ollama 路由。
from research_engine.routing_rules import (  # noqa: F401 — 重新导出共用实现
    ENDPOINT_DEFINED_PROVIDERS,
    InvalidRouting,
    validate,
)


def available_providers(user: User) -> list[str]:
    """列出用户有密钥、部署有密钥或无需密钥即可使用的供应商。

    界面据此标记不可选，而非隐藏模型，添加密钥后可在同一列表中启用。"""
    usable: set[str] = {"ollama"}  # 本地推理不需要密钥。

    if user.api_key_provider:
        usable.add(user.api_key_provider)

    for provider, server_key in (
        ("google", settings.google_api_key),
        ("anthropic", settings.anthropic_api_key),
        ("openai", settings.openai_api_key),
        ("openrouter", settings.openrouter_api_key),
    ):
        if server_key:
            usable.add(provider)

    # 自定义兼容端点可能不需要密钥，是否配置 URL 才是其可用配置依据。
    #
    # custom 和 openrouter 的模型列表属于端点，不在固定目录中，也必须列入可用供应商。
    # 否则全自定义路由部署的选择器反而看不到它实际使用的模型。
    if settings.custom_base_url:
        usable.add("custom")

    return sorted(usable)


def deployment_default() -> dict[str, str]:
    """运行与用户均未指定时采用的部署路由。"""
    return {
        "planner": settings.model_planner,
        "executor": settings.model_executor,
        "critic": settings.model_critic,
        "synthesizer": settings.model_synthesizer,
        "chat": settings.model_chat,
    }


def resolve(*, session_routing: dict | None, user_routing: dict | None) -> dict[str, str]:
    """按会话快照、用户偏好、部署默认值顺序解析。"""
    return session_routing or user_routing or deployment_default()
