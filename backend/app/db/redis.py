from __future__ import annotations

from typing import TYPE_CHECKING

import structlog

from app.config import settings

if TYPE_CHECKING:
    import redis.asyncio as aioredis

logger = structlog.get_logger()

_redis_pool: aioredis.Redis | None = None


async def init_redis_pool() -> None:
    """初始化共用 Redis 连接池；必须在函数内导入 redis。

    桌面端会通过共享运行路由间接导入本模块，但不使用 Redis，打包时也不包含它。若在模块顶部导入，桌面端所有运行路由都会因缺少 redis 包而失败；放在这里则只有真正启用 Redis 的服务端需要它。"""
    global _redis_pool

    import redis.asyncio as aioredis

    _redis_pool = await aioredis.from_url(
        settings.redis_url,
        encoding="utf-8",
        decode_responses=True,
        max_connections=50,
    )
    # 发送一次请求，确认 Redis 能连接
    await _redis_pool.ping()
    logger.info("redis_init", message="Redis connection pool initialized")


async def close_redis_pool() -> None:
    global _redis_pool
    if _redis_pool:
        pool = _redis_pool
        _redis_pool = None
        await pool.aclose()
        logger.info("redis_close", message="Redis connection pool closed")


def get_redis() -> aioredis.Redis:
    """供 FastAPI 请求使用，返回已经初始化的共用 Redis 连接池。"""
    if _redis_pool is None:
        raise RuntimeError("Redis pool not initialized. Call init_redis_pool() first.")
    return _redis_pool


async def publish_event(session_id: str, event: dict) -> None:
    """把 JSON 事件发布到该会话的 SSE 消息通道。"""
    import json

    redis = get_redis()
    channel = f"session:{session_id}:events"
    await redis.publish(channel, json.dumps(event))


# 先核对令牌再删除锁，避免旧 worker 误释放别人后来取得的锁（docs/02 §6）。
_RELEASE_LUA = """
if redis.call('GET', KEYS[1]) == ARGV[1] then
  return redis.call('DEL', KEYS[1])
end
return 0
"""


async def acquire_session_lock(session_id: str, token: str, ttl: int) -> bool:
    """用唯一令牌申请会话锁；锁的有效期必须长于任务超时时间。"""
    redis = get_redis()
    return bool(await redis.set(f"lock:session:{session_id}", token, ex=ttl, nx=True))


async def release_session_lock(session_id: str, token: str) -> None:
    """只有令牌仍匹配、锁仍属于自己时，才释放它。"""
    redis = get_redis()
    await redis.eval(_RELEASE_LUA, 1, f"lock:session:{session_id}", token)
