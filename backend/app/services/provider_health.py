"""供应商密钥连接探测（internal/07 Phase 2a）。

始终返回可操作状态，不向外抛异常。degraded 表示服务有响应但拒绝密钥等问题，failed 表示完全未响应，两者处理方式不同。
使用最便宜的鉴权端点：可鉴权的模型列表，或 OpenRouter 的密钥信息接口；不调用模型，不消耗 Token。"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Literal

import httpx
import structlog

from research_engine.llm_factory import map_local_host

logger = structlog.get_logger()

# 探测应快速结束，不能拖慢设置页。
_PROBE_TIMEOUT_SECONDS = 6.0

VerdictState = Literal["ok", "degraded", "failed"]


@dataclass
class Verdict:
    state: VerdictState
    reason: str
    checked_at: str  # ISO-8601 时间，方便界面显示多久前检查过。
    model_count: int | None = None


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _ok(reason: str, model_count: int | None = None) -> Verdict:
    return Verdict(state="ok", reason=reason, checked_at=_now_iso(), model_count=model_count)


def _degraded(reason: str) -> Verdict:
    return Verdict(state="degraded", reason=reason, checked_at=_now_iso())


def _failed(reason: str) -> Verdict:
    return Verdict(state="failed", reason=reason, checked_at=_now_iso())


def failed_verdict(reason: str) -> Verdict:
    """供调用方构造统一 failed 结果，例如未发请求前就发现已存密钥不能解密，避免各处手工拼结构。"""
    return _failed(reason)


@dataclass
class _ProbeRequest:
    url: str
    headers: dict[str, str]
    # 供应商端点确实提供模型列表时，从成功响应中提取数量。
    count_of: Callable[[dict], int] | None = None


def _google_request(key: str) -> _ProbeRequest:
    # Generative Language API 通过查询参数鉴权，而非请求头。
    return _ProbeRequest(
        url=f"https://generativelanguage.googleapis.com/v1beta/models?key={key}",
        headers={},
        count_of=lambda body: len(body.get("models", [])),
    )


def _anthropic_request(key: str) -> _ProbeRequest:
    return _ProbeRequest(
        url="https://api.anthropic.com/v1/models",
        headers={"x-api-key": key, "anthropic-version": "2023-06-01"},
        count_of=lambda body: len(body.get("data", [])),
    )


def _openai_request(key: str) -> _ProbeRequest:
    return _ProbeRequest(
        url="https://api.openai.com/v1/models",
        headers={"Authorization": f"Bearer {key}"},
        count_of=lambda body: len(body.get("data", [])),
    )


def _openrouter_request(key: str) -> _ProbeRequest:
    # OpenRouter /models 是公开目录，错误密钥也可能返回 200；要用 /auth/key 验证调用者自己的密钥。
    return _ProbeRequest(
        url="https://openrouter.ai/api/v1/auth/key",
        headers={"Authorization": f"Bearer {key}"},
        count_of=None,
    )


def _custom_request(key: str, base_url: str) -> _ProbeRequest:
    # 自定义端点使用 OpenAI 兼容协议，因此尝试标准 /models 路径。
    dial = map_local_host(base_url)
    return _ProbeRequest(
        url=f"{dial.rstrip('/')}/models",
        headers={"Authorization": f"Bearer {key}"},
        count_of=lambda body: len(body.get("data", [])),
    )


_BUILDERS: dict[str, Callable[[str, str | None], _ProbeRequest]] = {
    "google": lambda key, base_url: _google_request(key),
    "anthropic": lambda key, base_url: _anthropic_request(key),
    "openai": lambda key, base_url: _openai_request(key),
    "openrouter": lambda key, base_url: _openrouter_request(key),
    "custom": lambda key, base_url: _custom_request(key, base_url or ""),
}


async def probe(provider: str, key: str, base_url: str | None = None) -> Verdict:
    """检查供应商是否实际接受 key，自定义供应商还使用 base_url。

    所有故障转成 Verdict 状态，不向外抛出，与本地探测契约一致。"""
    if not key:
        return _degraded("No key to test.")

    builder = _BUILDERS.get(provider)
    if builder is None:
        return _failed(f"Unknown provider '{provider}'.")
    if provider == "custom" and not base_url:
        return _degraded("A base URL is required to test a custom endpoint.")

    req = builder(key, base_url)

    try:
        async with httpx.AsyncClient(timeout=_PROBE_TIMEOUT_SECONDS) as client:
            resp = await client.get(req.url, headers=req.headers)
    except Exception as exc:  # noqa: BLE001 — 传输故障表示没有响应
        logger.info("provider_health_probe_failed", provider=provider, error=str(exc))
        return _failed(f"No response from {provider}: {exc}")

    if resp.status_code in (401, 403):
        return _degraded(f"{provider} answered, but rejected this key (HTTP {resp.status_code}).")
    if resp.status_code == 429:
        return _degraded(f"{provider} answered, but this key is rate-limited or out of quota.")
    if 500 <= resp.status_code < 600:
        return _degraded(
            f"{provider} answered with a server error (HTTP {resp.status_code}) — try again shortly."
        )
    if resp.status_code != 200:
        return _degraded(f"{provider} answered with HTTP {resp.status_code}.")

    if req.count_of is None:
        return _ok("Connected.")
    try:
        count = req.count_of(resp.json())
    except Exception:  # noqa: BLE001 — 正文无法解析仍不能否认服务已经响应
        return _ok("Connected.")
    return _ok(f"Connected · {count} models", model_count=count)
