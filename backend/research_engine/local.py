"""本地宿主适配器，见 docs/13 §4、docs/12 M6。

SQLite 搜索缓存和进程内事件发送器，配合 AsyncSqliteSaver，
让完整研究及审批无需 Docker、Postgres、Redis 或登录即可运行。
作为 CLI 与桌面共用的参考实现放在引擎包中。
缓存使用线程内标准库 sqlite3；检查点另需 pyproject.toml 的 local 扩展依赖。"""

from __future__ import annotations

import asyncio
import os
import sqlite3
import time
from collections.abc import Callable
from pathlib import Path

from research_engine.runconfig import DEFAULT_MODELS, ROLES, RunConfig

_PROVIDER_ENV = {
    "google": "GOOGLE_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
    "custom": "CUSTOM_API_KEY",
}

# 本地 Ollama 和自托管 custom 可不带供应商密钥。
# 与服务端 app/config.py 的 _KEYLESS_PROVIDERS 保持同一含义。
_KEYLESS_PROVIDERS = ("ollama", "custom")


def providers_missing_a_key(models: dict[str, str], keys: dict[str, str]) -> list[str]:
    """返回因缺密钥而不能运行的供应商列表；全部可用时为空。

    只检查确实需要密钥的供应商；CLI 和桌面共用，避免拒绝完整的无密钥本地配置。"""
    routed = sorted({m.split(":", 1)[0] for m in models.values()})
    needs_key = [p for p in routed if p not in _KEYLESS_PROVIDERS]
    # custom_base_url 是地址而非凭据，不能算作密钥。
    have_a_key = any(name != "custom_base_url" for name in keys)
    return needs_key if not have_a_key else []


def usable_providers(keys: dict[str, str]) -> list[str]:
    """返回当前密钥及地址支持的供应商，供自行管理密钥的宿主使用。

    Ollama 始终可用；custom 配有地址即可，不强制密钥，与服务端规则一致。"""
    usable = {"ollama"} | {p for p in keys if p != "custom_base_url"}
    if keys.get("custom_base_url"):
        usable.add("custom")
    return sorted(usable)


def load_env_file(path: str | Path = "../.env") -> bool:
    """宿主显式请求时加载 .env，返回是否成功加载。

    已有环境变量优先，不能被旧文件覆盖；引擎不自行读取环境。
    python-dotenv 延迟导入，依赖缺失时跳过，避免成为引擎必需依赖。"""
    env_path = Path(path)
    if not env_path.is_file():
        return False
    try:
        from dotenv import load_dotenv
    except ImportError:  # pragma: no cover — 可选的环境加载便利功能
        return False
    return bool(load_dotenv(env_path, override=False))


def provider_keys_from_env() -> dict[str, str]:
    """统一读取供应商密钥及 custom_base_url，供 CLI 与桌面共用。

    自定义路由必须同时传递地址；只传密钥会造成客户端没有正确端点。"""
    keys = {p: os.environ[env] for p, env in _PROVIDER_ENV.items() if os.environ.get(env)}
    if os.environ.get("CUSTOM_BASE_URL"):
        keys["custom_base_url"] = os.environ["CUSTOM_BASE_URL"]
    return keys


def run_config_from_env(*, fake: bool, model_routes: dict[str, str] | None = None) -> RunConfig:
    """从进程环境构造引擎配置，供本地 CLI 和评测宿主共用。

    对应服务端 app/runtime.py，但无需数据库或 JWT 配置。
    这里故意不设置 skip_plan_gate，保留 RunConfig 默认 True；
    CLI 与无人值守评测无法展示计划审批，打开关卡会让任务停住无人处理。
    真正的服务端和桌面运行则从会话明确读取该选项。"""
    if fake:
        return RunConfig(llm_mode="fake")

    # 辅助单用途调用可指定各自路由，同时复用研究运行的供应商、本地地址及环境策略。
    models = (
        model_routes
        if model_routes is not None
        else {role: os.environ.get(f"MODEL_{role.upper()}", DEFAULT_MODELS[role]) for role in ROLES}
    )
    if model_routes is not None and (
        set(models) != set(ROLES)
        or any(not isinstance(route, str) or ":" not in route for route in models.values())
    ):
        raise ValueError("model_routes must specify a provider:model route for every role")
    keys = provider_keys_from_env()

    # 先拦截完全忘配密钥的常见情况；具体供应商缺失仍由工厂给出明确提示。
    if needs_key := providers_missing_a_key(models, keys):
        wanted = ", ".join(_PROVIDER_ENV.get(p, f"{p.upper()}_API_KEY") for p in needs_key)
        raise SystemExit(
            f"No provider key found. Routing needs one of: {wanted}.\n"
            f"Export it, or run with --fake for a keyless demo."
        )

    return RunConfig(
        llm_mode="real",
        models=models,
        provider_keys=keys,
        tavily_api_key=os.environ.get("TAVILY_API_KEY", ""),
        brave_api_key=os.environ.get("BRAVE_API_KEY", ""),
        # 与服务端一致：生产启用严格 SSRF，开发放宽。否则笔记本上的 Ollama、
        # 本地代理端口会被仅允许常用 Web 端口的生产策略拦截。
        enforce_ssrf_guards=os.environ.get("ENVIRONMENT", "development") == "production",
        # 本机用 localhost，Docker 常用 host.docker.internal；读取环境变量使同一套配置可覆盖两种场景。
        ollama_base_url=os.environ.get("OLLAMA_BASE_URL")
        or RunConfig.__dataclass_fields__["ollama_base_url"].default,
        **run_limits_from_env(),
    )


def run_limits_from_env() -> dict:
    """统一从环境读取各运行限制，供 CLI 与桌面共用。

    0 表示无限制，默认与 app/config.py 一致，见 docs/04 §6。
    不能在桌面私设时间或费用上限，也不能漏读 MAX_INPUT_TOKENS；需要上限时由宿主显式设置。"""

    def _int_env(name: str, default: int) -> int:
        raw = os.environ.get(name)
        try:
            return int(raw) if raw else default
        except ValueError:
            return default

    def _float_env(name: str, default: float) -> float:
        raw = os.environ.get(name)
        try:
            return float(raw) if raw else default
        except ValueError:
            return default

    return {
        "max_critic_loops": _int_env("MAX_CRITIC_LOOPS", 2),
        "max_cost_per_session_usd": _float_env("MAX_COST_PER_SESSION_USD", 0.0),
        "max_wallclock_seconds": _int_env("MAX_WALLCLOCK_SECONDS", 0),
        "max_input_tokens": _int_env("MAX_INPUT_TOKENS", 0),
        "max_parallel_tasks": _int_env("MAX_PARALLEL_TASKS", 4),
    }


_SCHEMA = """
CREATE TABLE IF NOT EXISTS search_cache (
    key        TEXT PRIMARY KEY,
    value      TEXT NOT NULL,
    expires_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_search_cache_expires ON search_cache (expires_at);
"""


def default_data_dir() -> Path:
    """返回本地运行数据库目录，调用方可以覆盖。"""
    return Path.home() / ".research-engine"


class SqliteCache:
    """SQLite 搜索结果缓存，对应服务端 RedisCache。

    每操作在线程里使用短连接，不跨线程共享连接。读取时检查过期，顺便删除过期项。
    缓存只是性能优化，锁定或损坏时退化成无缓存，不让搜索失败。"""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=5.0)
        # WAL 允许读写并存，供桌面研究与界面同时访问。
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(_SCHEMA)

    def _get_sync(self, key: str) -> str | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT value, expires_at FROM search_cache WHERE key = ?", (key,)
            ).fetchone()
            if row is None:
                return None
            value, expires_at = row
            if expires_at <= time.time():
                conn.execute("DELETE FROM search_cache WHERE key = ?", (key,))
                return None
            return value

    def _set_sync(self, key: str, value: str, ttl: int) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO search_cache (key, value, expires_at) VALUES (?, ?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value, "
                "expires_at = excluded.expires_at",
                (key, value, time.time() + ttl),
            )

    async def get(self, key: str) -> str | None:
        return await asyncio.to_thread(self._get_sync, key)

    async def set(self, key: str, value: str, ttl: int) -> None:
        await asyncio.to_thread(self._set_sync, key, value, ttl)


class InProcessEventSink:
    """按顺序在内存收集事件，可同步转发到实时展示。

    服务端读者跨进程，因此需要落库与 Redis；本地读者同进程，可直接输出 CLI 或 SSE。
    on_event 实时回调，events 保留完整顺序历史。"""

    def __init__(self, on_event: Callable[[dict], None] | None = None) -> None:
        self.events: list[dict] = []
        self._on_event = on_event

    async def __call__(self, session_id: str, event: dict) -> None:  # noqa: ARG002
        self.events.append(event)
        if self._on_event is not None:
            self._on_event(event)

    def by_agent(self, agent: str) -> list[dict]:
        return [e for e in self.events if e.get("agent") == agent]
