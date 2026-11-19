"""登录、注册与会话认证接口（docs/05 §3，docs/06 §1—§2）。

登录设置 httpOnly 访问和刷新 Cookie；刷新时轮换令牌，退出时吊销刷新令牌。注册使用中性提示防止枚举账号；登录同时限制 IP 与账号失败次数。"""

from __future__ import annotations

from datetime import UTC, datetime

import structlog
from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.db.base import get_db
from app.db.redis import get_redis
from app.dependencies import get_client_ip, get_current_user
from app.models.user import User
from app.schemas.auth import (
    ApiKeyLabelRequest,
    ApiKeyRequest,
    ConnectionVerdict,
    LoginRequest,
    PasswordChangeRequest,
    ProfileUpdate,
    RegisterRequest,
    UsageResponse,
    UserResponse,
)
from app.services import auth_service, crypto, provider_health, rate_limit, tokens, usage
from app.services.passwords import WeakPassword, hash_password, verify_password
from app.services.preferences import merge_preferences
from research_engine.net_guard import SSRFBlocked, validate_url

logger = structlog.get_logger()
router = APIRouter(prefix="/auth", tags=["Authentication"])


def _set_auth_cookies(response: Response, access: str, refresh: str) -> None:
    response.set_cookie(tokens.ACCESS_COOKIE, access, **tokens.access_cookie_kwargs())
    response.set_cookie(tokens.REFRESH_COOKIE, refresh, **tokens.refresh_cookie_kwargs())


async def _issue_session(response: Response, db: AsyncSession, user: User) -> None:
    access = tokens.create_access_token(user.id)
    refresh = await auth_service.issue_refresh_token(db, user.id)
    _set_auth_cookies(response, access, refresh)


@router.post("/register", status_code=201)
async def register(
    payload: RegisterRequest,
    request: Request,
    db: AsyncSession = Depends(get_db),
    redis=Depends(get_redis),
):
    ip = get_client_ip(request)
    limited = await rate_limit.check(redis, rate_limit.key_register_ip(ip), rate_limit.REGISTER_IP)
    if not limited.allowed:
        raise HTTPException(
            status.HTTP_429_TOO_MANY_REQUESTS, detail="Too many registrations. Try later."
        )

    try:
        hashed = hash_password(payload.password)
    except WeakPassword as e:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(e)) from e

    existing = (
        await db.execute(select(User).where(User.email == payload.email))
    ).scalar_one_or_none()
    if existing is None:
        db.add(
            User(
                email=payload.email,
                hashed_pw=hashed,
                # 公开部署可限制月度用量，避免单个账号耗尽共用密钥额度；默认 0 表示不限。
                monthly_token_limit=settings.default_monthly_token_limit,
            )
        )
        await db.commit()
        logger.info("user_registered", email=payload.email)
    # 无论账号是否存在都返回相同提示，防止枚举（docs/06 §1）。
    return {"message": "If this email is available, the account has been created. Please log in."}


@router.post("/login")
async def login(
    payload: LoginRequest,
    request: Request,
    response: Response,
    db: AsyncSession = Depends(get_db),
    redis=Depends(get_redis),
):
    ip = get_client_ip(request)
    ip_rl = await rate_limit.check(redis, rate_limit.key_login_ip(ip), rate_limit.LOGIN_IP)
    # 账号限流只累计失败次数，因此先读取限额，再在下面记录失败。
    # 不能把成功登录也计为失败，否则频繁正常登录的用户也会被锁定（docs/06 §4）。
    email_key = rate_limit.key_login_email(payload.email)
    email_rl = await rate_limit.peek(redis, email_key, rate_limit.LOGIN_EMAIL)
    if not ip_rl.allowed or not email_rl.allowed:
        raise HTTPException(
            status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Too many attempts. Please wait before trying again.",
        )

    user = (await db.execute(select(User).where(User.email == payload.email))).scalar_one_or_none()
    if not user or not verify_password(payload.password, user.hashed_pw):
        # 不存在的邮箱与密码错误同样计数，避免通过锁定行为判断账号是否存在。
        await rate_limit.check(redis, email_key, rate_limit.LOGIN_EMAIL)
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, detail="Invalid email or password.")
    if not user.is_active:
        raise HTTPException(status.HTTP_403_FORBIDDEN, detail="Account is deactivated.")

    await _issue_session(response, db, user)
    await db.commit()
    logger.info("user_login", user_id=str(user.id))
    return {"message": "Logged in."}


@router.post("/refresh")
async def refresh(
    request: Request,
    response: Response,
    db: AsyncSession = Depends(get_db),
):
    raw = request.cookies.get(tokens.REFRESH_COOKIE)
    if not raw:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, detail="No refresh token.")
    rotated = await auth_service.rotate_refresh_token(db, raw)
    if rotated is None:
        await db.commit()  # 提交同一刷新令牌家族的吊销记录。
        response.delete_cookie(tokens.ACCESS_COOKIE, path="/")
        response.delete_cookie(tokens.REFRESH_COOKIE, path="/api/v1/auth")
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, detail="Invalid refresh token.")
    user_id, new_raw = rotated
    access = tokens.create_access_token(user_id)
    _set_auth_cookies(response, access, new_raw)
    await db.commit()
    return {"message": "Refreshed."}


@router.post("/logout")
async def logout(
    request: Request,
    response: Response,
    db: AsyncSession = Depends(get_db),
):
    raw = request.cookies.get(tokens.REFRESH_COOKIE)
    if raw:
        await auth_service.revoke_refresh_token(db, raw)
        await db.commit()
    response.delete_cookie(tokens.ACCESS_COOKIE, path="/")
    response.delete_cookie(tokens.REFRESH_COOKIE, path="/api/v1/auth")
    return {"message": "Logged out."}


@router.get("/me", response_model=UserResponse)
async def get_me(current_user: User = Depends(get_current_user)):
    return UserResponse.model_validate(current_user)


@router.patch("/me", response_model=UserResponse)
async def update_me(
    payload: ProfileUpdate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """更新允许编辑的用户资料；只修改请求明确提供的字段。"""
    fields = payload.model_dump(exclude_unset=True)
    for field in ("display_name", "avatar_url", "monthly_token_limit"):
        if field in fields:
            value = fields[field]
            # 空字符串表示清空可选文本字段。
            if field in ("display_name", "avatar_url") and value == "":
                value = None
            setattr(current_user, field, value)
    if payload.preferences is not None:
        # 按字段合并偏好，不能整份替换，否则一个设置分区会清空另一个分区的配置。
        # prompt_overrides 还需再合并一层；与桌面共用函数，避免重复规则（internal/07 Phase 3）。
        current_user.preferences = merge_preferences(
            current_user.preferences, payload.preferences.model_dump(exclude_unset=True)
        )
    await db.commit()
    await db.refresh(current_user)
    logger.info("profile_updated", user_id=str(current_user.id), fields=sorted(fields))
    return UserResponse.model_validate(current_user)


@router.post("/me/password", status_code=200)
async def change_password(
    payload: PasswordChangeRequest,
    request: Request,
    response: Response,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
    redis=Depends(get_redis),
):
    """修改账号密码。

    必须验证旧密码，防止仅持有被盗 Cookie 就能改密码；新密码遵守注册时的强度规则。吊销全部刷新令牌使其他设备退出，同时向当前设备签发新令牌对。"""
    # 此接口也验证密码，属于暴力尝试目标，因此复用登录限流。
    ip = get_client_ip(request)
    limited = await rate_limit.check(redis, rate_limit.key_login_ip(ip), rate_limit.LOGIN_IP)
    if not limited.allowed:
        raise HTTPException(
            status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Too many attempts. Please wait before trying again.",
        )

    if not verify_password(payload.current_password, current_user.hashed_pw):
        raise HTTPException(status.HTTP_403_FORBIDDEN, detail="Current password is incorrect.")
    if payload.new_password == payload.current_password:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="New password must be different from the current one.",
        )

    try:
        current_user.hashed_pw = hash_password(payload.new_password)
    except WeakPassword as e:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(e)) from e

    # 先退出所有设备，再为当前设备签发新令牌。
    await auth_service.revoke_all_for_user(db, current_user.id)
    await _issue_session(response, db, current_user)
    await db.commit()
    logger.info("password_changed", user_id=str(current_user.id))
    return {"message": "Password updated. Other devices have been signed out."}


@router.put("/me/api-key", response_model=UserResponse)
async def set_api_key(
    payload: ApiKeyRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """保存用户自带供应商密钥。

    密钥加密存储，接口不返回原值，只返回供应商和显示提示（docs/06 §1）。"""
    if payload.provider == "custom" and payload.api_base_url and settings.is_production:
        try:
            validate_url(str(payload.api_base_url))
        except SSRFBlocked as e:
            raise HTTPException(status.HTTP_400_BAD_REQUEST, detail=str(e)) from e

    current_user.api_key_encrypted = crypto.encrypt(payload.api_key)
    current_user.api_key_provider = payload.provider
    current_user.api_key_base_url = str(payload.api_base_url) if payload.api_base_url else None
    current_user.api_key_hint = crypto.hint(payload.api_key)
    current_user.api_key_set_at = datetime.now(UTC)
    await db.commit()
    await db.refresh(current_user)
    # 只记录操作事件，不记录密钥及其显示提示。
    logger.info("api_key_set", user_id=str(current_user.id), provider=payload.provider)

    # 提交保存后再探测连接，使检查对应实际已存储密钥；探测故障不影响保存本身（internal/07 Phase 2a）。
    verdict = await provider_health.probe(
        payload.provider,
        payload.api_key,
        str(payload.api_base_url) if payload.api_base_url else None,
    )
    response = UserResponse.model_validate(current_user)
    response.connection_verdict = ConnectionVerdict(
        state=verdict.state,
        reason=verdict.reason,
        checked_at=verdict.checked_at,
        model_count=verdict.model_count,
    )
    return response


@router.delete("/me/api-key", response_model=UserResponse)
async def delete_api_key(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """删除用户自带密钥，后续回退部署密钥。"""
    current_user.api_key_encrypted = None
    current_user.api_key_provider = None
    current_user.api_key_base_url = None
    current_user.api_key_hint = None
    current_user.api_key_label = None
    current_user.api_key_set_at = None
    await db.commit()
    await db.refresh(current_user)
    logger.info("api_key_removed", user_id=str(current_user.id))
    return UserResponse.model_validate(current_user)


@router.patch("/me/api-key/label", response_model=UserResponse)
async def rename_api_key(
    payload: ApiKeyLabelRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """给当前自定义连接设置显示名称。

    名称描述网关身份，与密钥轮换无关；重命名只更新标签，不要求重填密钥或再次探测供应商。"""
    if not current_user.api_key_provider:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="No connection to rename.")
    current_user.api_key_label = payload.label
    await db.commit()
    await db.refresh(current_user)
    logger.info("api_key_renamed", user_id=str(current_user.id))
    return UserResponse.model_validate(current_user)


@router.get("/me/usage", response_model=UsageResponse)
async def get_usage(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """查询本月、近 7 天及最近一次会话的 Token 与费用用量。"""
    return await usage.summary(db, current_user.id, current_user.monthly_token_limit)
