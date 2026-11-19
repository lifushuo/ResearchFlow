"""短期访问 JWT 与不透明刷新令牌服务（docs/06 §1，docs/05 §1）。

访问令牌为 15 分钟 HS256 JWT；刷新令牌是 256 位随机串，服务端只存 SHA-256 与 jti，每次使用轮换，旧令牌复用会吊销整族。Cookie 为 httpOnly、SameSite=Lax，生产启用 Secure。"""

from __future__ import annotations

import hashlib
import secrets
from datetime import UTC, datetime, timedelta
from uuid import UUID

import jwt

from app.config import settings

ACCESS_COOKIE = "access_token"
REFRESH_COOKIE = "refresh_token"


def create_access_token(user_id: UUID) -> str:
    now = datetime.now(UTC)
    payload = {
        "sub": str(user_id),
        "type": "access",
        "iat": now,
        "exp": now + timedelta(minutes=settings.jwt_access_token_expire_minutes),
    }
    return jwt.encode(payload, settings.jwt_secret_key, algorithm=settings.jwt_algorithm)


def decode_access_token(token: str) -> UUID:
    """返回用户 ID；令牌无效时抛 JWT 异常或 ValueError。"""
    payload = jwt.decode(token, settings.jwt_secret_key, algorithms=[settings.jwt_algorithm])
    if payload.get("type") != "access":
        raise ValueError("not an access token")
    return UUID(payload["sub"])


def generate_refresh_token() -> str:
    return secrets.token_urlsafe(32)


def hash_refresh_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def refresh_expiry() -> datetime:
    return datetime.now(UTC) + timedelta(days=settings.jwt_refresh_token_expire_days)


def cookie_kwargs(max_age: int, *, path: str) -> dict:
    """认证 Cookie 共用的 Set-Cookie 属性。"""
    return {
        "httponly": True,
        "secure": settings.is_production,
        "samesite": "lax",
        "max_age": max_age,
        "path": path,
    }


def access_cookie_kwargs() -> dict:
    return cookie_kwargs(settings.jwt_access_token_expire_minutes * 60, path="/")


def refresh_cookie_kwargs() -> dict:
    return cookie_kwargs(settings.jwt_refresh_token_expire_days * 86400, path="/api/v1/auth")
