"""检索缓存的间接访问层，见 docs/13 §4、docs/12 M6。

检索器通过 ContextVar 中的 Cache 端口访问缓存，不直接导入服务端 Redis。
默认空缓存，让测试、桌面端和独立 CLI 即使没有缓存服务也能运行。"""

from __future__ import annotations

from contextvars import ContextVar

from research_engine.ports import Cache


class NullCache:
    """不缓存任何内容，作为无需宿主配置的默认实现。"""

    async def get(self, key: str) -> str | None:  # noqa: ARG002 - 保留协议要求的参数形式
        return None

    async def set(self, key: str, value: str, ttl: int) -> None:  # noqa: ARG002
        return None


_NULL: Cache = NullCache()

_cache: ContextVar[Cache] = ContextVar("engine_cache", default=_NULL)


def set_cache(cache: Cache):
    """为当前上下文安装缓存，返回可供 reset_cache 恢复的令牌。"""
    return _cache.set(cache)


def reset_cache(token) -> None:
    _cache.reset(token)


def get_cache() -> Cache:
    return _cache.get()
