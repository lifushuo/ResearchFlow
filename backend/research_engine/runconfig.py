"""运行范围的引擎配置，见 docs/13 §4、docs/12 M6。

引擎不读取 app.config 或进程环境，所需设置统一由宿主传入 RunConfig。
set_process_default 安装进程基线；set_run_config 用 ContextVar 设置单轮覆盖，
让同一进程中的并发运行可使用不同路由与预算。
工具内部无法获得图状态，因此使用上下文配置而不是把设置塞进 AgentState。
读取优先级为本轮覆盖、进程默认、模块默认；模块默认与服务端保持一致。"""

from __future__ import annotations

from collections.abc import Mapping
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Literal

# 需要模型路由的代理角色，与 llm_factory._ROLE_CONFIG 对应。
ROLES: tuple[str, ...] = ("planner", "executor", "critic", "synthesizer", "chat")

# 与 app/config.py 的 MODEL_* 默认值一致。
DEFAULT_MODELS: Mapping[str, str] = {
    "planner": "google:gemini-2.5-pro",
    "executor": "google:gemini-2.5-flash",
    "critic": "google:gemini-2.5-flash",
    "synthesizer": "google:gemini-2.5-pro",
    "chat": "google:gemini-2.5-flash",
}


@dataclass(frozen=True)
class RunConfig:
    """进程或单次运行使用的不可变引擎配置。

    默认值与 app/config.py 一致；此处携带部署密钥，用户 BYOK 密钥由 set_user_keys 逐运行优先覆盖。"""

    llm_mode: Literal["real", "fake"] = "real"
    # 仅选择脚本输出内容，不决定是否脚本运行。演示采用 llm_mode=fake、demo=True。
    # 不增加第三种 llm_mode，避免漏过现有 fake 网络保护而意外调用真实模型。
    demo: bool = False
    models: Mapping[str, str] = field(default_factory=lambda: dict(DEFAULT_MODELS))
    provider_keys: Mapping[str, str] = field(default_factory=dict)

    # 检索设置。
    tavily_api_key: str = ""
    brave_api_key: str = ""

    # 仅资料库模式：检索只走 Corpus，读取只接受 corpus://，嵌入必须本地。
    # 但代理模型仍按 models 路由，单有本地嵌入并不代表整轮离线。
    # 未安装资料库时明确失败，不能静默回退外网。
    corpus_mode: bool = False

    # 冻结的选中文档版本 ID，控制本轮可检索和读取范围。
    # None 表示没有指定选择；宿主据此安装 ScopedCorpus。这是控制数据，不是用户偏好。
    corpus_allowed_ids: frozenset[str] | None = None

    # Offline 要求所有模型调用留在本机；图启动前与模型工厂均检查，且必须配合 corpus_mode。
    offline: bool = False

    # SSRF 防护，见 docs/06；服务端严格，本地可放宽以允许 Ollama。
    enforce_ssrf_guards: bool = True

    # Ollama 服务地址，可覆盖为局域网机器地址，见 docs/13 §6。
    ollama_base_url: str = "http://localhost:11434/v1"

    # 费用、时间、Token 预算均以 0 表示无限制，默认不私设上限，见 docs/04 §6。
    # custom/OpenRouter 价格未知时费用限制不起作用，需要供应商侧另设限额。
    max_critic_loops: int = 2
    max_cost_per_session_usd: float = 0.0
    max_wallclock_seconds: int = 0
    # 整个会话累计输入 Token，包含评审循环与返工。
    # 曾硬编码百万上限并终止真实运行，现由部署者明确配置。
    max_input_tokens: int = 0

    # 最多同时研究的任务数；1 恢复顺序执行。并发可能造成已发出调用的预算超支，见 graph._BudgetGuard。
    max_parallel_tasks: int = 4

    # ── 可配置功能，见 internal/07 Phase 3 ──
    # 默认值保留已有行为，用户未修改偏好时不应受影响；与其他硬编码值对应时注明来源。

    # 沿用 tools.py 中 web_search 原来的默认结果数。
    retrieval_k: int = 5
    # 0 表示不设证据下限，与现有 0 不限额约定一致。
    min_sources_per_task: int = 0
    # 在 EvidenceChunk 校验前截断片段，因此只能收紧，不能放宽数据结构自身的最大长度。
    snippet_max_chars: int = 500

    # 计划关卡使用的主题与大纲配置；为空时保留原有无约束规划行为。
    outline_template: str | None = None
    topic_seeds: tuple[str, ...] = ()
    # 角色对应的替换提示词正文，由统一提示词组装层按用途权限决定是否应用。
    prompt_overrides: Mapping[str, str] = field(default_factory=dict)

    # ── 计划审核关卡，见 internal/07 Phase 4 ──
    # 这里只限制规划器默认提议数，用户仍可增加；默认 6 保留原 PlannerOutput 上限行为。
    max_planner_tasks: int = 6
    # 引擎默认 True，让现有测试、CLI 和无人值守评测跳过无法展示的计划审批。
    # 没有显式配置的宿主必须保持原有行为，不能突然停在第二个人工关卡。
    #
    # 产品是否启用计划关卡由宿主明确决定：pipeline_runner 与桌面会话驱动读取 Session.skip_plan_gate。
    # 数据库列默认 False 与裸引擎默认 True 服务不同场景，不能强行统一。
    skip_plan_gate: bool = True

    def model_for(self, role: str) -> str:
        """返回角色对应的 provider:model 路由。"""
        try:
            return self.models[role]
        except KeyError:
            raise ValueError(
                f"No model routed for role '{role}'. Known roles: {sorted(self.models)}"
            ) from None


_MODULE_DEFAULT = RunConfig()

# 进程基线，由宿主启动时替换。
_process_default: RunConfig = _MODULE_DEFAULT

# 本轮覆盖；None 表示使用进程默认值。
_override: ContextVar[RunConfig | None] = ContextVar("engine_run_config", default=None)


def set_process_default(cfg: RunConfig) -> None:
    """由宿主启动时安装进程级基线配置。"""
    global _process_default
    _process_default = cfg


def reset_process_default() -> None:
    """恢复模块默认值，供验证无宿主配置行为的测试使用。"""
    global _process_default
    _process_default = _MODULE_DEFAULT


def set_run_config(cfg: RunConfig):
    """为当前上下文安装配置，返回用于 reset_run_config 的令牌。"""
    return _override.set(cfg)


def reset_run_config(token) -> None:
    _override.reset(token)


def get_run_config() -> RunConfig:
    """返回生效配置：本轮覆盖优先，其次进程默认，最后模块默认。"""
    return _override.get() or _process_default
