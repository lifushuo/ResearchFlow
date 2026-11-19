"""模型目录，见 docs/13 §6、docs/12 M8。

每项集中描述价格、采样参数支持、工具调用、结构化输出、显示名和上下文窗口，
供费用核算、请求构造和界面共用。新增模型通常只需增加目录项。

价格为 None 表示部署者尚需提供价格，不能理解为免费。
validate_pricing 会拒绝缺少必需价格的路由，防止费用预算静默失真。
Anthropic 价格来自其公开资料；Google 沿用原表；OpenAI/OpenRouter 由部署方补充。
Ollama 的远程 API 费用为零，推理运行在用户自己的硬件上。"""

from __future__ import annotations

from dataclasses import dataclass

# 集中导出角色名称，便于与下方预设共同使用。
from research_engine.runconfig import ROLES  # noqa: F401


@dataclass(frozen=True)
class ModelSpec:
    """一个可通过路由选择的模型。"""

    provider: str
    model_id: str
    display_name: str

    # 价格单位为美元/百万 Token；None 表示部署者尚未提供价格。
    input_per_mtok: float | None
    output_per_mtok: float | None

    # 无法核实供应商上限时保持 None，界面显示未知，不能填写猜测值。
    context_window: int | None = None
    max_output_tokens: int | None = None

    supports_tools: bool = True
    supports_structured_output: bool = True

    # 供应商不接受 temperature/top_p/top_k 时设为 False，工厂会省略这些参数以免请求被拒绝。
    sampling_params_supported: bool = True

    # 记录最近核对供应商模型及价格的 ISO 日期。条目不会自动失效，
    # stale_entries 只提示哪些内容需要重新核实，避免人工目录静默过期。
    verified_on: str | None = None

    notes: str = ""

    @property
    def route(self) -> str:
        """返回 MODEL_* 和 RunConfig 使用的 provider:model 路由字符串。"""
        return f"{self.provider}:{self.model_id}"

    @property
    def priced(self) -> bool:
        return self.input_per_mtok is not None and self.output_per_mtok is not None


def _spec(*args, **kwargs) -> ModelSpec:
    return ModelSpec(*args, **kwargs)


# ── Anthropic ─────────────────────────────────────────────────────────────────────
#
# 部分较新 Anthropic 模型拒绝采样参数；Sonnet 5 对非默认值有限制。
# 统一省略不支持的参数，避免维护逐参数默认值表。旧模型仍按目录声明处理。
#
# claude-opus-5 曾同时遗漏价格和采样能力声明，导致启动或请求失败，见 docs/12 M8。

_ANTHROPIC = [
    _spec(
        "anthropic",
        "claude-opus-5",
        "Claude Opus 5",
        input_per_mtok=5.00,
        output_per_mtok=25.00,
        context_window=1_000_000,
        max_output_tokens=128_000,
        sampling_params_supported=False,
        notes="Thinking is on by default; raw reasoning is never returned.",
    ),
    _spec(
        "anthropic",
        "claude-opus-4-8",
        "Claude Opus 4.8",
        input_per_mtok=5.00,
        output_per_mtok=25.00,
        context_window=1_000_000,
        max_output_tokens=128_000,
        sampling_params_supported=False,
    ),
    _spec(
        "anthropic",
        "claude-opus-4-7",
        "Claude Opus 4.7",
        input_per_mtok=5.00,
        output_per_mtok=25.00,
        context_window=1_000_000,
        max_output_tokens=128_000,
        sampling_params_supported=False,
    ),
    _spec(
        "anthropic",
        "claude-opus-4-6",
        "Claude Opus 4.6",
        input_per_mtok=5.00,
        output_per_mtok=25.00,
        context_window=1_000_000,
        max_output_tokens=128_000,
        sampling_params_supported=True,
    ),
    _spec(
        "anthropic",
        "claude-sonnet-5",
        "Claude Sonnet 5",
        # 采用标准价格而非截至 2026-08-31 的促销价，避免预算低估。
        input_per_mtok=3.00,
        output_per_mtok=15.00,
        context_window=1_000_000,
        max_output_tokens=128_000,
        sampling_params_supported=False,
    ),
    _spec(
        "anthropic",
        "claude-sonnet-4-6",
        "Claude Sonnet 4.6",
        input_per_mtok=3.00,
        output_per_mtok=15.00,
        context_window=1_000_000,
        max_output_tokens=128_000,
        sampling_params_supported=True,
    ),
    _spec(
        "anthropic",
        "claude-haiku-4-5",
        "Claude Haiku 4.5",
        input_per_mtok=1.00,
        output_per_mtok=5.00,
        context_window=200_000,
        max_output_tokens=64_000,
        sampling_params_supported=True,
    ),
    _spec(
        "anthropic",
        "claude-fable-5",
        "Claude Fable 5",
        input_per_mtok=10.00,
        output_per_mtok=50.00,
        context_window=1_000_000,
        max_output_tokens=128_000,
        sampling_params_supported=False,
        notes="Highest-capability tier; thinking always on. Requires 30-day data retention.",
    ),
]

# ── Google ────────────────────────────────────────────────────────────────────────

_GOOGLE = [
    _spec(
        "google",
        "gemini-2.5-pro",
        "Gemini 2.5 Pro",
        input_per_mtok=1.25,
        output_per_mtok=5.00,
    ),
    _spec(
        "google",
        "gemini-2.5-flash",
        "Gemini 2.5 Flash",
        input_per_mtok=0.075,
        output_per_mtok=0.30,
    ),
    # 保留旧模型价格，保证已有 MODEL_* 配置仍可启动。
    _spec(
        "google",
        "gemini-1.5-pro",
        "Gemini 1.5 Pro",
        input_per_mtok=1.25,
        output_per_mtok=5.00,
    ),
    _spec(
        "google",
        "gemini-1.5-flash",
        "Gemini 1.5 Flash",
        input_per_mtok=0.075,
        output_per_mtok=0.30,
    ),
]

# ── Ollama 本地模型 ──
#
# 不再列出固定本地聊天模型：原模型或超出普通笔记本显存，或达不到本项目质量门槛。
# 仍保留 Ollama 嵌入模型和 ollama:<tag> 动态路由，不把具体聊天模型冒称为受支持档位。

CATALOG: dict[str, ModelSpec] = {spec.model_id: spec for spec in (*_ANTHROPIC, *_GOOGLE)}

# 工厂支持的供应商。OpenAI/OpenRouter 可路由，但模型和价格由部署方显式 register。
# Ollama 虽无固定聊天条目，仍支持嵌入和动态路由。
KNOWN_PROVIDERS = ("anthropic", "google", "openai", "openrouter", "ollama", "custom")


def stale_entries(today: str, max_age_days: int = 120) -> list[tuple[str, str | None]]:
    """列出超过 max_age_days 未核实的目录项，返回路由和上次核实日期。

    日期缺失也会列出。这里只提示维护，不拒绝路由，也不让条目自动过期。
    today 由调用方传入，避免引擎依赖外部时钟并便于测试。"""
    from datetime import date

    def parse(value: str) -> date:
        y, m, d = (int(part) for part in value.split("-"))
        return date(y, m, d)

    now = parse(today)
    out: list[tuple[str, str | None]] = []
    for spec in CATALOG.values():
        if spec.verified_on is None:
            out.append((spec.route, None))
            continue
        try:
            age = (now - parse(spec.verified_on)).days
        except ValueError, TypeError:
            out.append((spec.route, spec.verified_on))  # 日期无法解析表示没有可靠核实日期。
            continue
        if age > max_age_days:
            out.append((spec.route, spec.verified_on))
    return sorted(out)


def register(spec: ModelSpec) -> None:
    """运行时新增或覆盖目录项，供新模型、微调模型或部署方自行定价的路由使用。"""
    CATALOG[spec.model_id] = spec


def get(model_id: str) -> ModelSpec | None:
    return CATALOG.get(model_id)


def by_provider(provider: str) -> list[ModelSpec]:
    return [s for s in CATALOG.values() if s.provider == provider]


def providers() -> list[str]:
    return sorted({s.provider for s in CATALOG.values()})


# ── 角色路由预设 ──
#
# 按角色选择模型也影响质量：执行器需要多轮工具调用的速度与覆盖面，
# 综合器负责最终正文，更重视写作能力。预设简化常用配置，仍允许逐角色调整。
#
# 没有通过质量门槛的固定本地聊天模型，因此不提供会误导用户的 Ollama 聊天预设。

PRESETS: dict[str, dict[str, dict[str, str]]] = {
    "anthropic": {
        "fast": {r: "anthropic:claude-haiku-4-5" for r in ROLES},
        "balanced": {
            "planner": "anthropic:claude-sonnet-5",
            "executor": "anthropic:claude-haiku-4-5",
            "critic": "anthropic:claude-haiku-4-5",
            "synthesizer": "anthropic:claude-sonnet-5",
            "chat": "anthropic:claude-haiku-4-5",
        },
        "best": {
            "planner": "anthropic:claude-opus-5",
            "executor": "anthropic:claude-sonnet-5",
            "critic": "anthropic:claude-sonnet-5",
            "synthesizer": "anthropic:claude-opus-5",
            "chat": "anthropic:claude-sonnet-5",
        },
    },
    "google": {
        "fast": {r: "google:gemini-2.5-flash" for r in ROLES},
        "balanced": {
            "planner": "google:gemini-2.5-pro",
            "executor": "google:gemini-2.5-flash",
            "critic": "google:gemini-2.5-flash",
            "synthesizer": "google:gemini-2.5-pro",
            "chat": "google:gemini-2.5-flash",
        },
        "best": {r: "google:gemini-2.5-pro" for r in ROLES},
    },
}

PRESET_NAMES = ("fast", "balanced", "best")


def preset(provider: str, name: str) -> dict[str, str] | None:
    """返回命名预设的角色到路由映射；未知组合返回 None。"""
    return (PRESETS.get(provider) or {}).get(name)
