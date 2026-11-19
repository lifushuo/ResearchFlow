"""密码策略与哈希（docs/architecture/06-security.md §1）。

直接使用 bcrypt，超过 72 字节明确拒绝，不能任其静默截断。内置常见泄露密码集合，生产可通过 BREACHED_PATH 提供更大名单。"""

from __future__ import annotations

import bcrypt

MIN_LENGTH = 12
MAX_BYTES = 72

# 默认仅包含少量常见泄露密码；真实部署建议加载更大的名单，不能把最小样本宣传为完整防护。
_COMMON = {
    "password",
    "password123",
    "123456789012",
    "qwertyuiop12",
    "letmein12345",
    "changeme1234",
    "administrator",
    "welcome12345",
}


class WeakPassword(ValueError):
    """密码不满足策略时抛出。"""


def validate_password(password: str) -> None:
    if len(password) < MIN_LENGTH:
        raise WeakPassword(f"Password must be at least {MIN_LENGTH} characters.")
    if len(password.encode("utf-8")) > MAX_BYTES:
        raise WeakPassword(f"Password must not exceed {MAX_BYTES} bytes.")
    if password.lower() in _COMMON:
        raise WeakPassword("This password is too common. Choose something less guessable.")


def hash_password(password: str) -> str:
    validate_password(password)
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt(rounds=12)).decode("utf-8")


def verify_password(plain: str, hashed: str) -> bool:
    try:
        return bcrypt.checkpw(plain.encode("utf-8"), hashed.encode("utf-8"))
    except ValueError:
        return False
