"""发现自定义 OpenAI 兼容端点提供的模型。

模型列表由网关实时决定，不属于静态目录；主动询问端点才能让选择器展示实际可调用模型。所有故障转换为不可达状态和提示，不向外抛出。
从引擎 RunConfig 读取端点和密钥，不导入要求数据库与 JWT 配置的 app.config，保证桌面可用。"""

from __future__ import annotations

from dataclasses import dataclass, field

import httpx
import structlog

from research_engine.llm_factory import api_key_for, map_local_host
from research_engine.runconfig import get_run_config

logger = structlog.get_logger()

# 设置页探测时限要短；五秒内无法列出模型就报告不可达，不让页面一直等待。
_PROBE_TIMEOUT_SECONDS = 5.0


@dataclass
class CustomEndpointStatus:
    """记录配置端点及它实际报告的模型能力。"""

    configured_base_url: str
    reachable: bool
    # 模型 ID 按端点原样保留，不清理大小写或前缀，以免改变实际调用时的匹配身份。
    models: list[str] = field(default_factory=list)
    error: str | None = None
    hint: str | None = None


def configured_base_url() -> str:
    """返回 custom: 路由的部署端点；未配置时为空。"""
    return api_key_for("custom_base_url")


async def probe(base_url: str | None = None) -> CustomEndpointStatus:
    """通过兼容 /models 接口读取模型列表。

    configured 是用户配置和界面显示的地址，dial 是容器内实际连接地址；统一用 map_local_host 映射，保证探测与研究调用同一服务。"""
    configured = base_url or configured_base_url()
    if not configured:
        return CustomEndpointStatus(
            configured_base_url="",
            reachable=False,
            hint=(
                "No custom endpoint is configured. Set CUSTOM_BASE_URL to an "
                "OpenAI-compatible base URL to route runs through it."
            ),
        )

    dial = map_local_host(configured)
    if get_run_config().enforce_ssrf_guards:
        # 探测也属于服务端请求用户提供的 URL，必须沿用 get_llm 的防护。
        # 在设置页面就拒绝危险地址，不等到研究执行才失败。
        from research_engine.net_guard import SSRFBlocked, validate_url

        try:
            validate_url(dial)
        except SSRFBlocked as exc:
            return CustomEndpointStatus(
                configured_base_url=configured,
                reachable=False,
                error=str(exc),
                hint="This address is blocked by the deployment's SSRF guard.",
            )

    url = f"{dial.rstrip('/')}/models"
    key = api_key_for("custom")
    headers = {"Authorization": f"Bearer {key}"} if key else {}

    try:
        async with httpx.AsyncClient(timeout=_PROBE_TIMEOUT_SECONDS) as client:
            response = await client.get(url, headers=headers)
            response.raise_for_status()
            payload = response.json()
    except Exception as exc:  # noqa: BLE001 — 任何失败都转换为不可达状态
        logger.info("custom_endpoint_probe_failed", configured=configured, error=str(exc))
        return CustomEndpointStatus(
            configured_base_url=configured,
            reachable=False,
            error=str(exc),
            hint=(
                "No OpenAI-compatible endpoint answered at this address. Check that the "
                "gateway is running and that CUSTOM_BASE_URL includes its API prefix."
            ),
        )

    # 接口应返回 data 内含 id 的列表。
    # 200 但结构不符表示服务可达却无法读取模型，不能显示空列表让用户误以为没有模型。
    entries = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(entries, list):
        return CustomEndpointStatus(
            configured_base_url=configured,
            reachable=True,
            error="The endpoint answered without an OpenAI-compatible model list.",
            hint="Enter the model id by hand — this gateway does not advertise one.",
        )

    models = sorted(
        {
            str(e["id"])
            for e in entries
            if isinstance(e, dict)
            and e.get("id")
            and "deepseek-r1" not in str(e.get("id", "")).lower()
            and (
                e.get("capabilities", {}).get("tool_calling", True)
                if isinstance(e.get("capabilities"), dict)
                else True
            )
        },
    )
    return CustomEndpointStatus(
        configured_base_url=configured,
        reachable=True,
        models=models,
        hint=(None if models else "The endpoint listed no models. Enter the model id by hand."),
    )
