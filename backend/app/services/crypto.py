"""加密存储用户自带模型密钥（docs/06 §1）。

使用 Fernet；仅在用户自己的执行上下文需要时解密，不返回接口、不记录日志，浏览器只看到不可还原的尾部提示。
优先从 ENCRYPTION_KEY，否则 JWT_SECRET_KEY 经 HKDF-SHA256 派生，并用独立用途标签隔离 JWT 签名。源密钥轮换后旧密文不能解密，应提示重填，而不是使应用崩溃。"""

from __future__ import annotations

import base64

from cryptography.fernet import Fernet, InvalidToken
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

from app.config import settings

_HKDF_INFO = b"mara.user-api-key.v1"


def _fernet() -> Fernet:
    secret = (getattr(settings, "encryption_key", "") or settings.jwt_secret_key).encode("utf-8")
    derived = HKDF(
        algorithm=hashes.SHA256(),
        length=32,
        salt=None,  # 启动时已保证源密钥至少 32 字符，具备足够熵。
        info=_HKDF_INFO,
    ).derive(secret)
    return Fernet(base64.urlsafe_b64encode(derived))


def encrypt(plaintext: str) -> str:
    """加密待存储密钥，返回可安全放入 URL 的 base64 密文。"""
    return _fernet().encrypt(plaintext.encode("utf-8")).decode("ascii")


def decrypt(ciphertext: str) -> str | None:
    """解密已存密钥；无法解密时返回 None，让调用方降级，例如源密钥已轮换。"""
    try:
        return _fernet().decrypt(ciphertext.encode("ascii")).decode("utf-8")
    except InvalidToken, ValueError, TypeError:
        return None


def user_provider_keys(user) -> dict[str, str]:
    """统一生成用户自带密钥的供应商映射，无可用密钥时为空。

    必须同时有供应商和可解密密文；自定义端点还按 llm_factory.api_key_for 约定附带供应商 base_url，避免多处手工拼装不一致。"""
    if not (user.api_key_encrypted and user.api_key_provider):
        return {}
    plaintext = decrypt(user.api_key_encrypted)
    if not plaintext:
        return {}
    keys = {user.api_key_provider: plaintext}
    if user.api_key_base_url:
        keys[f"{user.api_key_provider}_base_url"] = user.api_key_base_url
    return keys


def hint(plaintext: str) -> str:
    """只展示最后 4 个字符作为识别提示，不能还原完整密钥。"""
    tail = plaintext.strip()[-4:]
    return f"…{tail}" if tail else "…"
