"""刷新令牌的签发、轮换、吊销和重复使用检测（docs/06 §1）。

只保存不透明令牌的 SHA-256。轮换后旧令牌失效，再次提交已吊销令牌视为泄露，吊销该用户整族令牌。"""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.refresh_token import RefreshToken
from app.services import tokens


async def issue_refresh_token(db: AsyncSession, user_id: UUID) -> str:
    raw = tokens.generate_refresh_token()
    db.add(
        RefreshToken(
            user_id=user_id,
            token_hash=tokens.hash_refresh_token(raw),
            expires_at=tokens.refresh_expiry(),
        )
    )
    await db.flush()
    return raw


async def rotate_refresh_token(db: AsyncSession, raw: str) -> tuple[UUID, str] | None:
    """校验并轮换令牌，成功返回用户 ID 和新原始令牌，无效返回 None；发现旧令牌复用则吊销该用户整族令牌。"""
    token_hash = tokens.hash_refresh_token(raw)
    row = (
        await db.execute(select(RefreshToken).where(RefreshToken.token_hash == token_hash))
    ).scalar_one_or_none()
    if row is None:
        return None

    now = datetime.now(UTC)
    if row.revoked_at is not None:
        # 重复使用已轮换或已吊销令牌视为泄露，吊销整个令牌家族。
        await _revoke_all_for_user(db, row.user_id)
        return None
    if row.expires_at <= now:
        return None

    row.revoked_at = now
    new_raw = await issue_refresh_token(db, row.user_id)
    return row.user_id, new_raw


async def revoke_refresh_token(db: AsyncSession, raw: str) -> None:
    token_hash = tokens.hash_refresh_token(raw)
    await db.execute(
        update(RefreshToken)
        .where(RefreshToken.token_hash == token_hash, RefreshToken.revoked_at.is_(None))
        .values(revoked_at=datetime.now(UTC))
    )


async def revoke_all_for_user(db: AsyncSession, user_id: UUID) -> None:
    """吊销用户全部有效刷新令牌，让所有设备退出。

    供改密码等主动全账号退出使用；重复使用检测也复用同一逻辑。"""
    await _revoke_all_for_user(db, user_id)


async def _revoke_all_for_user(db: AsyncSession, user_id: UUID) -> None:
    await db.execute(
        update(RefreshToken)
        .where(RefreshToken.user_id == user_id, RefreshToken.revoked_at.is_(None))
        .values(revoked_at=datetime.now(UTC))
    )
