"""原子限流（docs/architecture/06-security.md §2）。

同一 Lua 脚本执行递增和首次过期设置，保证计数不会因两步之间中断而永久无 TTL。不同操作用不同键，研究和聊天不共用预算。"""

from __future__ import annotations

from dataclasses import dataclass

# KEYS[1] 为键，ARGV[1] 为上限，ARGV[2] 为窗口秒数；返回当前计数与剩余秒数，首次递增设置 TTL。
_LUA = """
local current = redis.call('INCR', KEYS[1])
if current == 1 then
  redis.call('EXPIRE', KEYS[1], tonumber(ARGV[2]))
end
local ttl = redis.call('TTL', KEYS[1])
return {current, ttl}
"""


@dataclass(frozen=True)
class RateLimit:
    limit: int
    window_seconds: int


# 按操作区分限流规则（docs/06 §2）。
#
# 登录注册限流用于防暴力破解，不能配置关闭，避免调整用量上限时顺便取消认证保护。
#
# 研究和聊天限流不写死在常量中，从 Settings 每次构建，默认 0 不限。
# 它们用于公开多用户防滥用，不能强迫自托管单用户遵守固定低上限。
LOGIN_IP = RateLimit(20, 60)
# 每 15 分钟最多 5 次失败；登录先 peek，只有密码错误才 check 增加计数，成功不计入。
LOGIN_EMAIL = RateLimit(5, 900)
REGISTER_IP = RateLimit(5, 3600)


@dataclass(frozen=True)
class RateResult:
    allowed: bool
    current: int
    ttl: int


async def check(redis, key: str, rule: RateLimit) -> RateResult:
    """递增键计数并判断是否仍在规则内，保证存在 TTL。"""
    current, ttl = await redis.eval(_LUA, 1, key, rule.limit, rule.window_seconds)
    return RateResult(allowed=current <= rule.limit, current=int(current), ttl=int(ttl))


async def peek(redis, key: str, rule: RateLimit) -> RateResult:
    """只查看是否还能执行一次，不增加计数。

    适用于只计失败结果的限流：结果未知时先看，确实失败才 check。"""
    current = int(await redis.get(key) or 0)
    return RateResult(allowed=current < rule.limit, current=current, ttl=int(await redis.ttl(key)))


def key_research(user_id) -> str:
    return f"rl:research:{user_id}"


def key_chat(user_id) -> str:
    return f"rl:chat:{user_id}"


def key_login_ip(ip: str) -> str:
    return f"rl:login:ip:{ip}"


def key_login_email(email: str) -> str:
    return f"rl:login:email:{email.lower()}"


def key_register_ip(ip: str) -> str:
    return f"rl:register:ip:{ip}"
