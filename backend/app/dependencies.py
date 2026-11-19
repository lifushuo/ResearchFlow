"""身份认证依赖（docs/architecture/06-security.md §1）。

浏览器使用 httpOnly access_token Cookie，经同源代理自动携带，兼容 SSE。非浏览器客户端也可使用 Bearer 头。密码哈希和令牌逻辑分别位于 passwords、tokens 服务。"""

from __future__ import annotations

import jwt
import structlog
from fastapi import Depends, HTTPException, Request, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.db.base import get_db
from app.db.redis import get_redis
from app.models.user import User
from app.services import rate_limit, tokens
from app.services.passwords import (
    hash_password,
    verify_password,
)  # 重新导出共用依赖，兼容现有导入位置。

logger = structlog.get_logger()

__all__ = ["hash_password", "verify_password", "get_current_user", "get_client_ip"]


def get_client_ip(request: Request) -> str:
    fwd = request.headers.get("x-forwarded-for")
    if fwd:
        return fwd.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def _extract_token(request: Request) -> str | None:
    cookie = request.cookies.get(tokens.ACCESS_COOKIE)
    if cookie:
        return cookie
    auth = request.headers.get("authorization", "")
    if auth.lower().startswith("bearer "):
        return auth[7:]
    return None


async def get_current_user(
    request: Request,
    db: AsyncSession = Depends(get_db),
) -> User:
    token = _extract_token(request)
    if not token:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Not authenticated")
    try:
        user_id = tokens.decode_access_token(token)
    except (jwt.ExpiredSignatureError, jwt.InvalidTokenError, ValueError, KeyError) as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or expired token"
        ) from e

    user = (
        await db.execute(select(User).where(User.id == user_id, User.is_active.is_(True)))
    ).scalar_one_or_none()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found or deactivated"
        )
    return user


async def enforce_research_rate_limit(
    current_user: User = Depends(get_current_user), redis=Depends(get_redis)
) -> None:
    """用户研究请求限流，默认 0 关闭，规则见 Settings。

    关闭时不执行 Redis INCR，也不积累隐藏计数；重新启用时从新的时间窗口开始。"""
    limit = settings.research_rate_limit_per_hour
    if limit <= 0:
        return
    res = await rate_limit.check(
        redis, rate_limit.key_research(current_user.id), rate_limit.RateLimit(limit, 3600)
    )
    if not res.allowed:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Research limit reached ({limit}/hour). "
            f"Retry in {max(res.ttl, 0) // 60 + 1} minutes.",
        )


async def enforce_chat_rate_limit(
    current_user: User = Depends(get_current_user), redis=Depends(get_redis)
) -> None:
    """用户聊天请求限流，默认 0 关闭，规则见 Settings。"""
    limit = settings.chat_rate_limit_per_hour
    if limit <= 0:
        return
    res = await rate_limit.check(
        redis, rate_limit.key_chat(current_user.id), rate_limit.RateLimit(limit, 3600)
    )
    if not res.allowed:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Chat limit reached ({limit}/hour). "
            f"Retry in {max(res.ttl, 0) // 60 + 1} minutes.",
        )
