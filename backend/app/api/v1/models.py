"""模型目录与用户路由配置接口（docs/12 M8）。

模型选择器读取价格、上下文长度、能力、预设及用户可访问的供应商。没有密钥的模型仍返回，但标明不可用，便于用户知道添加密钥后能使用什么。"""

from __future__ import annotations

import json

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.dependencies import get_current_user, get_db
from app.models.user import User
from app.schemas.auth import MAX_PROMPT_OVERRIDE_CHARS, ConnectionVerdict
from app.services import crypto, custom_endpoint, local_llm, model_routing, provider_health
from research_engine import catalog
from research_engine.prompt_composition import editable_defaults
from research_engine.runconfig import ROLES

router = APIRouter(prefix="/models", tags=["Models"])


# 请求响应结构共用 app/schemas/models.py，让桌面声明相同 response_model 时不引入服务端配置链；此处只重新导出兼容旧导入。
from app.schemas.models import (  # noqa: E402 — 重新导出共用结构
    CatalogResponse,
    CustomEndpointStatusResponse,
    LocalLLMStatusResponse,
    LocalModelInfo,
    ModelInfo,
    PromptDefault,
    PromptDefaultsResponse,
    ProviderTestRequest,
    ReadinessResponse,
    RoutingRequest,
    RoutingResponse,
)

__all__ = [
    "CatalogResponse",
    "CustomEndpointStatusResponse",
    "LocalLLMStatusResponse",
    "LocalModelInfo",
    "ModelInfo",
    "PromptDefault",
    "PromptDefaultsResponse",
    "ProviderTestRequest",
    "ReadinessResponse",
    "RoutingRequest",
    "RoutingResponse",
    "router",
]


async def _ollama_presets_from_installed() -> dict[str, dict[str, str]] | None:
    """按本机实际安装的 Ollama 模型生成预设，无可用模型返回 None。

    跳过向量模型；均衡和最佳优先足以处理结构化证据的模型，快速模式可用最小聊天模型。不能让静态预设指向未安装标签，直到研究执行才出现 404。"""
    try:
        status_ = await local_llm.probe()
    except Exception:  # noqa: BLE001 — 探测服务故障不能使模型目录查询失败
        return None
    chat_models = [
        m for m in status_.models if not m.is_embedding and getattr(m, "supports_tools", True)
    ]
    if not chat_models:
        return None

    def size(m) -> float:
        return m.params_b if m.params_b is not None else (m.size_bytes or 0) / 1e9

    smallest = min(chat_models, key=size)
    research_ready = [m for m in chat_models if not m.likely_underpowered]
    strongest = max(research_ready or chat_models, key=size)

    # 必须路由到精确安装标签，不能替换成目录家族名称。
    # 家族名只有存在 latest 时才可用，否则会 404，或运行与用户选择不同的模型。
    def route_of(m) -> str:
        return f"ollama:{m.name}"

    return {
        "fast": dict.fromkeys(ROLES, route_of(smallest)),
        "balanced": dict.fromkeys(ROLES, route_of(strongest)),
        "best": dict.fromkeys(ROLES, route_of(strongest)),
    }


@router.get("/readiness", response_model=ReadinessResponse)
async def get_readiness(current_user: User = Depends(get_current_user)):
    """判断当前用户现在能否启动研究（docs/17 §8a）。

    每次请求重新计算，不保存容易过期的入门完成标记。密钥被撤销或本地服务启动后，结果应立即变化。
    available_providers 总会包含不需要密钥的 Ollama，不能单凭列表非空判断就绪，还必须确认服务可达且存在聊天模型。"""
    has_cloud_key = bool(
        current_user.api_key_provider
        or settings.google_api_key
        or settings.anthropic_api_key
        or settings.openai_api_key
        or settings.openrouter_api_key
        or settings.custom_api_key
    )

    local_reachable = False
    chat_models = 0
    try:
        status_ = await local_llm.probe()
        local_reachable = status_.reachable
        chat_models = sum(1 for m in status_.models if not m.is_embedding)
    except Exception:  # noqa: BLE001 — 探测失败视为没有可用本地模型，不作为接口错误
        pass

    return ReadinessResponse(
        # 只有向量模型的 Ollama 无法承担研究角色；必须统计聊天模型，不能只检查 reachable。
        ready=has_cloud_key or chat_models > 0,
        has_cloud_key=has_cloud_key,
        local_reachable=local_reachable,
        local_chat_models=chat_models,
    )


@router.get("", response_model=CatalogResponse)
async def get_catalog(current_user: User = Depends(get_current_user)):
    usable = model_routing.available_providers(current_user)
    deployment = model_routing.deployment_default()
    user_routing = current_user.model_routing or None

    models = [
        ModelInfo(
            route=spec.route,
            provider=spec.provider,
            model_id=spec.model_id,
            display_name=spec.display_name,
            input_per_mtok=spec.input_per_mtok,
            output_per_mtok=spec.output_per_mtok,
            context_window=spec.context_window,
            max_output_tokens=spec.max_output_tokens,
            supports_tools=spec.supports_tools,
            supports_structured_output=spec.supports_structured_output,
            notes=spec.notes,
            available=spec.provider in usable,
        )
        # 按费用从低到高排列，本地免费模型优先展示，便于控制成本。
        for spec in sorted(
            catalog.CATALOG.values(),
            key=lambda s: (s.provider, s.output_per_mtok if s.priced else float("inf")),
        )
    ]

    # 优先返回本机实际可运行的预设；未安装模型或探测失败时回退静态表。
    presets = dict(catalog.PRESETS)
    installed = await _ollama_presets_from_installed()
    if installed:
        presets["ollama"] = installed

    return CatalogResponse(
        roles=list(ROLES),
        models=models,
        presets=presets,
        preset_names=list(catalog.PRESET_NAMES),
        available_providers=usable,
        effective_routing=model_routing.resolve(session_routing=None, user_routing=user_routing),
        user_routing=user_routing,
        deployment_routing=deployment,
    )


@router.get("/local/status", response_model=LocalLLMStatusResponse)
async def local_llm_status(_current_user: User = Depends(get_current_user)):
    """探测已配置的本地模型服务（docs/12 M15）。

    与模型目录分开，因为实时网络探测可能超时，而目录查询应保持快速且始终可展示。"""
    status_ = await local_llm.probe()
    return LocalLLMStatusResponse(
        configured_base_url=status_.configured_base_url,
        reachable=status_.reachable,
        usable=status_.usable,
        models=[
            LocalModelInfo(
                name=m.name,
                size_bytes=m.size_bytes,
                route=m.route,
                in_catalog=m.in_catalog,
                likely_underpowered=m.likely_underpowered,
                is_embedding=m.is_embedding,
                params_b=m.params_b,
            )
            for m in status_.models
        ],
        error=status_.error,
        hint=status_.hint,
        install_state=status_.install_state,
    )


@router.get("/custom/status", response_model=CustomEndpointStatusResponse)
async def custom_endpoint_status(_current_user: User = Depends(get_current_user)):
    """询问自定义端点实际提供的模型，规则见 services/custom_endpoint。

    与目录接口分开，避免远程服务故障拖慢模型列表。自定义和本地模型列表由端点决定，不能在项目目录里猜测。"""
    status_ = await custom_endpoint.probe()
    return CustomEndpointStatusResponse(
        configured_base_url=status_.configured_base_url,
        reachable=status_.reachable,
        models=status_.models,
        error=status_.error,
        hint=status_.hint,
    )


@router.get("/prompt-defaults", response_model=PromptDefaultsResponse)
async def get_prompt_defaults(_current_user: User = Depends(get_current_user)):
    """返回各角色提示词编辑器的项目默认内容（范围冻结 §12）。

    单独设接口，避免运行表单读取模型目录时携带无用的长提示词。所有用户和主机得到相同的构建默认值，不包含用户覆盖。仅通过 editable_defaults 选择允许公开的用途，受保护提示词不能进入响应。"""
    return PromptDefaultsResponse(
        max_chars=MAX_PROMPT_OVERRIDE_CHARS,
        roles=[
            PromptDefault(role=d.role, default_prompt=d.text, untrusted_content_framed=d.framed)
            for d in editable_defaults()
        ],
    )


@router.post("/providers/test", response_model=ConnectionVerdict)
async def test_provider(
    payload: ProviderTestRequest, _current_user: User = Depends(get_current_user)
):
    """在保存之前探测提交的密钥，对应“测试连接”（internal/07 Phase 2a）。

    它可能较慢或超时，因此独立于必须快速返回的模型目录。"""
    verdict = await provider_health.probe(
        payload.provider,
        payload.api_key,
        str(payload.api_base_url) if payload.api_base_url else None,
    )
    return ConnectionVerdict(
        state=verdict.state,
        reason=verdict.reason,
        checked_at=verdict.checked_at,
        model_count=verdict.model_count,
    )


@router.get("/providers/health", response_model=ConnectionVerdict)
async def provider_health_check(current_user: User = Depends(get_current_user)):
    """按用户要求重新检查其已保存的自带密钥。

    没有保存密钥时返回 404；此时没有用户密钥可验证，不必再返回一份重复的降级状态。"""
    if not current_user.api_key_provider or not current_user.api_key_encrypted:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="No key stored to check.")

    key = crypto.decrypt(current_user.api_key_encrypted)
    verdict = (
        provider_health.failed_verdict("The stored key could not be decrypted — save it again.")
        if not key
        else await provider_health.probe(
            current_user.api_key_provider, key, current_user.api_key_base_url
        )
    )
    return ConnectionVerdict(
        state=verdict.state,
        reason=verdict.reason,
        checked_at=verdict.checked_at,
        model_count=verdict.model_count,
    )


@router.post("/local/pull")
async def pull_local_model(request: Request, _current_user: User = Depends(get_current_user)):
    """流式返回推荐模型下载进度（internal/07 Phase 2b）。

    通过 HTTP 调用已运行的 Ollama，不需要启动本地进程，因此服务端和桌面都可使用。每行 JSON 对应一条 PullProgress。"""
    body = await request.json()
    model = (body.get("model") or "").strip()
    if not model:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, detail="No model given.")

    async def gen():
        async for progress in local_llm.pull(model):
            yield (
                json.dumps(
                    {
                        "status": progress.status,
                        "completed": progress.completed,
                        "total": progress.total,
                        "error": progress.error,
                    }
                )
                + "\n"
            )

    return StreamingResponse(gen(), media_type="application/x-ndjson")


@router.get("/routing", response_model=RoutingResponse)
async def get_routing(current_user: User = Depends(get_current_user)):
    """读取用户保存的路由及本次会真正使用的路由。

    routing 为空表示用户未选偏好，effective_routing 始终有实际默认值；两者不能合并，否则无法区分跟随部署配置和用户明确选择。
    只读数据库，不调用真实供应商，保持与模型目录一样轻量。"""
    return RoutingResponse(
        routing=current_user.model_routing,
        effective_routing=model_routing.resolve(
            session_routing=None, user_routing=current_user.model_routing
        ),
    )


@router.put("/routing", response_model=RoutingResponse)
async def set_routing(
    payload: RoutingRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """保存用户按角色选择的模型路由；先按目录校验，避免保存后直到研究执行才发现模型不能使用。"""
    try:
        cleaned = model_routing.validate(payload.routing)
    except model_routing.InvalidRouting as e:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(e)) from e

    current_user.model_routing = cleaned
    await db.commit()
    await db.refresh(current_user)

    return RoutingResponse(
        routing=current_user.model_routing,
        effective_routing=model_routing.resolve(
            session_routing=None, user_routing=current_user.model_routing
        ),
    )


@router.delete("/routing", response_model=RoutingResponse)
async def clear_routing(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """清除用户偏好，回退部署的 MODEL_* 路由。"""
    current_user.model_routing = None
    await db.commit()
    await db.refresh(current_user)

    return RoutingResponse(
        routing=None,
        effective_routing=model_routing.resolve(session_routing=None, user_routing=None),
    )
