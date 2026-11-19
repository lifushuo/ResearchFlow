"""可替换供应商的模型工厂，见 docs/architecture/04-agent-architecture.md §7、docs/13 §6。

按 provider:model 为各角色构造客户端；价格与能力统一来自 catalog。
需定价的路由缺少价格时在启动阶段拒绝。fake 模式返回无需密钥或网络的脚本模型。"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar

import structlog
from langchain_core.language_models.chat_models import BaseChatModel

from research_engine import catalog
from research_engine.runconfig import RunConfig, get_run_config

logger = structlog.get_logger()

# 角色对应的生成参数。
_ROLE_CONFIG = {
    "planner": {"temperature": 0.1, "max_tokens": 2000},
    "executor": {"temperature": 0.2, "max_tokens": 4000},
    "critic": {"temperature": 0.0, "max_tokens": 1000},
    "synthesizer": {"temperature": 0.2, "max_tokens": 6000},
    "chat": {"temperature": 0.3, "max_tokens": 2000},
}

# 仅在客户端构造期间覆盖生成参数，不修改角色默认值，也不影响并发研究。
_OUTPUT_LIMITS: ContextVar[dict[str, int] | None] = ContextVar("model_output_limits", default=None)


@contextmanager
def output_limit(role: str, max_tokens: int):
    """为当前上下文构造的指定角色客户端设置独立输出上限。"""
    if role not in _ROLE_CONFIG or type(max_tokens) is not int or max_tokens <= 0:
        raise ValueError("invalid role or output token limit")
    token = _OUTPUT_LIMITS.set({**(_OUTPUT_LIMITS.get() or {}), role: max_tokens})
    try:
        yield
    finally:
        _OUTPUT_LIMITS.reset(token)


# ── 每用户 BYOK 密钥 ──
# Worker 仅在本轮运行上下文安装用户解密密钥，避免并发运行串用。
# 不写文件或日志；未配置用户密钥时回退部署密钥。
_user_keys: ContextVar[dict[str, str] | None] = ContextVar("user_provider_keys", default=None)


def set_user_keys(keys: dict[str, str]):
    """为当前上下文安装用户密钥，返回用于恢复的令牌。"""
    return _user_keys.set(keys or {})


def reset_user_keys(token) -> None:
    _user_keys.reset(token)


def api_key_for(provider: str) -> str:
    """优先取用户 BYOK 密钥，缺失时取部署密钥。"""
    user_key = (_user_keys.get() or {}).get(provider, "")
    if user_key:
        return user_key
    return get_run_config().provider_keys.get(provider, "")


def model_name_for(role: str) -> str:
    """返回角色实际配置的裸模型 ID，去掉供应商前缀，用于定价。"""
    return get_run_config().model_for(role).split(":", 1)[-1]


def validate_pricing(cfg: RunConfig | None = None) -> None:
    """启动时检查需要定价的路由，缺价就明确失败。

    custom、OpenRouter 的模型由端点定义，无法靠本地目录完整定价，因此例外放行。
    这些路由 estimate_cost 为 0.0，费用上限实际上不起作用；界面零费用不能当作免费证明。
    启动时提示部署者改在供应商侧限制费用。"""
    cfg = cfg or get_run_config()
    if cfg.llm_mode == "fake":
        return

    # Ollama 本地推理没有远程 API 账单；以下两种路由可能收费，因此需要未知费用提示。
    uncapped = sorted(
        {
            cfg.model_for(role)
            for role in cfg.models
            if cfg.model_for(role).partition(":")[0] in ("custom", "openrouter")
        }
    )
    if uncapped:
        logger.warning(
            "spend_cap_not_enforced",
            models=uncapped,
            reason=(
                "these providers serve model ids this catalog cannot price, so "
                "max_cost_per_session_usd does not bind and reported cost reads $0.00 — "
                "set a spending limit at the provider"
            ),
        )

    missing = set()
    for role in cfg.models:
        provider, _, name = cfg.model_for(role).partition(":")
        # 端点自定义路由不要求目录条目；Ollama 也可使用任意已拉取的精确 tag。
        # 不能让这里与 model_routing.validate 对同一路由给出矛盾判断。
        if provider in ("custom", "openrouter", "ollama"):
            continue
        spec = catalog.get(name)
        if spec is None or not spec.priced:
            missing.add(name)
    missing = sorted(missing)
    if missing:
        raise ValueError(
            f"No price for routed model(s): {missing}. Add a ModelSpec to "
            f"research_engine/catalog.py (or call catalog.register() at startup) with the "
            f"provider's published per-1M-token prices — never estimated (docs/13 §6)."
        )


def sampling_supported(model: str) -> bool:
    """判断模型是否接受 temperature、top_p、top_k。

    未知模型默认接受；若不接受会明确返回 400，而不是静默降低运行质量。"""
    spec = catalog.get(model)
    return True if spec is None else spec.sampling_params_supported


def map_local_host(base_url: str) -> str:
    """在容器中把 localhost 模型地址改为 host.docker.internal。

    容器内 localhost 指容器自己，容器外该特殊域名又未必可解析，因此按运行环境改写。
    公开此函数供模型调用和健康探针共用，避免相同地址在两条路径行为不同。"""
    for prefix in (
        "http://localhost:",
        "http://127.0.0.1:",
        "https://localhost:",
        "https://127.0.0.1:",
    ):
        if base_url.startswith(prefix):
            try:
                import socket

                socket.gethostbyname("host.docker.internal")
            except Exception:
                return base_url  # 不在容器中，localhost 已经正确。
            return base_url.replace("localhost", "host.docker.internal", 1).replace(
                "127.0.0.1", "host.docker.internal", 1
            )
    return base_url


def _build(provider: str, model: str, role: str) -> BaseChatModel:
    """为 provider:model 路由构造 LangChain 聊天模型。

    各供应商参数、密钥、本地地址及 SSRF 规则确有差异，按供应商显式分支处理。"""
    cfg = {**_ROLE_CONFIG[role]}
    limits = _OUTPUT_LIMITS.get() or {}
    if role in limits:
        cfg["max_tokens"] = limits[role]
    from research_engine.response_diagnostics import models_enabled

    # 仅评测宿主显式启用诊断；两种客户端仍使用固定版本 SDK 的传输。
    if models_enabled() and provider in {"openai", "openrouter", "ollama", "custom"}:
        from research_engine.observed_openai import ObservedChatOpenAI as ChatOpenAI
    elif provider in {"openai", "openrouter", "ollama", "custom"}:
        from langchain_openai import ChatOpenAI
    key = api_key_for(provider)
    # Ollama 和本地自定义端点允许不配置密钥。
    if not key and provider not in ("ollama", "custom"):
        # 密钥缺失时给出可操作提示，不让用户只看到深层供应商 401 堆栈。
        raise ValueError(
            f"No API key available for provider '{provider}'. Add your own key in "
            f"Settings, or configure {provider.upper()}_API_KEY on the server."
        )

    if provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=model,
            google_api_key=key,
            temperature=cfg["temperature"],
            max_output_tokens=cfg["max_tokens"],
        )

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        # 由目录声明哪些模型拒绝采样参数，并在请求中省略；不能靠模型名前缀猜测。
        # 输出上限 max_tokens 保持传递，避免影响结构化输出与工具调用。
        kwargs: dict = {"model": model, "api_key": key, "max_tokens": cfg["max_tokens"]}
        if sampling_supported(model):
            kwargs["temperature"] = cfg["temperature"]
        return ChatAnthropic(**kwargs)

    if provider == "openai":
        kwargs = {"model": model, "api_key": key, "max_tokens": cfg["max_tokens"]}
        if sampling_supported(model):
            kwargs["temperature"] = cfg["temperature"]
        return ChatOpenAI(**kwargs)

    if provider == "openrouter":
        # OpenRouter 兼容 OpenAI 协议，通过地址覆盖即可使用同一密钥访问其模型，
        # 模型 ID 带命名空间，例如 anthropic/claude-opus-5，见 docs/13 §6。
        return ChatOpenAI(
            model=model,
            api_key=key,
            base_url="https://openrouter.ai/api/v1",
            max_tokens=cfg["max_tokens"],
            temperature=cfg["temperature"],
        )

    if provider == "ollama":
        # 本地推理也使用 OpenAI 兼容协议，无需密钥；地址允许配置为另一台机器。
        # 只有实际指向本机时才能满足本地隐私承诺。
        return ChatOpenAI(
            model=model,
            api_key=key or "ollama",  # SDK 要求非空占位值，该值不用于认证。
            base_url=map_local_host(get_run_config().ollama_base_url),
            max_tokens=cfg["max_tokens"],
            temperature=cfg["temperature"],
        )

    if provider == "custom":
        base_url = api_key_for("custom_base_url")
        if not base_url:
            # OpenAI SDK 默认连接 api.openai.com；custom 未配地址时必须拒绝，
            # 不能把用户自定义密钥和提示词误发到未选择的供应商。
            raise ValueError(
                f"The {role} route '{provider}:{model}' has no endpoint address. Save the "
                "Custom Endpoint's URL in Settings, or set CUSTOM_BASE_URL."
            )
        if get_run_config().enforce_ssrf_guards:
            from research_engine.net_guard import SSRFBlocked, validate_url

            try:
                validate_url(base_url)
            except SSRFBlocked as e:
                raise ValueError(f"Blocked by SSRF guard: {e}") from e

        kwargs = {
            "model": model,
            "api_key": key or "custom",
            "max_tokens": cfg["max_tokens"],
            "base_url": map_local_host(base_url),
        }
        if sampling_supported(model):
            kwargs["temperature"] = cfg["temperature"]
        return ChatOpenAI(**kwargs)

    raise ValueError(
        f"Unknown LLM provider '{provider}' for role '{role}'. "
        f"Known providers: {', '.join(catalog.KNOWN_PROVIDERS)}."
    )


def get_llm(role: str) -> BaseChatModel:
    """为指定代理角色返回聊天模型。"""
    cfg = get_run_config()
    if cfg.offline and cfg.llm_mode != "fake":
        # 运行前检查之外的最后保护：Offline 运行无论经哪条路径，都不能构造远程模型客户端。
        from research_engine import offline

        refused = offline.refusal(cfg, custom_base_url=api_key_for("custom_base_url"))
        if refused:
            raise ValueError(refused)
    if cfg.llm_mode == "fake":
        # 共用同一脚本调用流程；demo 仅选择用户可见内容，默认仍用测试固定数据，见 docs/17 §6.1。
        if cfg.demo:
            from research_engine.demo_fixtures import demo_model

            return demo_model(role)
        from research_engine.fakes import fake_model

        return fake_model(role)
    provider, _, model = cfg.model_for(role).partition(":")
    return _build(provider, model, role)


def text_of(message_or_chunk) -> str:
    """从模型响应提取普通可见文本。

    LangChain content 可能是字符串，也可能是带类型的内容块列表。
    不能直接 str(list) 把 Python 表示混入报告，也不能忽略列表造成正文丢失。"""
    content = getattr(message_or_chunk, "content", message_or_chunk)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                # 只取用户可见正文，跳过 thinking/reasoning 内容块。
                if block.get("type") in (None, "text") and isinstance(block.get("text"), str):
                    parts.append(block["text"])
        return "".join(parts)
    return ""


def estimate_cost(response, role: str) -> float:
    """根据响应 usage_metadata 和角色模型价格估算本次费用。

    未知价格在此贡献 0；需定价路由依赖启动校验保护，custom/OpenRouter 的零值不代表免费。"""
    usage = getattr(response, "usage_metadata", None) or {}
    in_tok = usage.get("input_tokens", 0)
    out_tok = usage.get("output_tokens", 0)
    spec = catalog.get(model_name_for(role))
    if spec is None or not spec.priced:
        return 0.0
    return (in_tok / 1_000_000 * spec.input_per_mtok) + (out_tok / 1_000_000 * spec.output_per_mtok)


def token_counts(response) -> tuple[int, int]:
    usage = getattr(response, "usage_metadata", None) or {}
    return usage.get("input_tokens", 0), usage.get("output_tokens", 0)


def served_model_id(response) -> str | None:
    """提取供应商实际披露的响应模型 ID，与配置的 provider:model 路由区分。

    auto/* 别名每次可能落到不同模型，不能冒充固定模型。
    没有披露时返回 None，fake 响应通常没有元数据，不补造默认模型。"""
    meta = getattr(response, "response_metadata", None) or {}
    # 兼容不同 SDK 的 model_name 与 model 字段，让调用方无需判断具体供应商客户端。
    served = meta.get("model_name") or meta.get("model")
    return str(served) if served else None
