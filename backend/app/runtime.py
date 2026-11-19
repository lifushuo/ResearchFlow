"""服务端配置适配器：把部署设置转换成引擎 RunConfig（docs/13 §4）。

引擎不导入 app.config。API 和 Celery 等服务端进程启动时安装基础配置；桌面端自行读取本地设置与系统密钥库。
这里只映射字段，不混入业务流程，让引擎不必区分主机类型。"""

from __future__ import annotations

from app.config import Settings, settings
from research_engine.runconfig import RunConfig, set_process_default


def run_config_from_settings(s: Settings | None = None) -> RunConfig:
    """将从环境读取的服务端设置映射到引擎配置对象。"""
    s = s or settings
    return RunConfig(
        llm_mode=s.llm_mode,
        models={
            "planner": s.model_planner,
            "executor": s.model_executor,
            "critic": s.model_critic,
            "synthesizer": s.model_synthesizer,
            "chat": s.model_chat,
        },
        provider_keys={
            "google": s.google_api_key,
            "anthropic": s.anthropic_api_key,
            "openai": s.openai_api_key,
            "openrouter": s.openrouter_api_key,
            "custom": s.custom_api_key,
            "custom_base_url": s.custom_base_url,
        },
        tavily_api_key=s.tavily_api_key,
        brave_api_key=s.brave_api_key,
        enforce_ssrf_guards=s.is_production,
        ollama_base_url=s.ollama_base_url,
        max_critic_loops=s.max_critic_loops,
        max_cost_per_session_usd=s.max_cost_per_session_usd,
        max_wallclock_seconds=s.max_wallclock_seconds,
        max_input_tokens=s.max_input_tokens,
        max_parallel_tasks=s.max_parallel_tasks,
    )


def install_process_default(s: Settings | None = None) -> RunConfig:
    """把配置安装成本进程的引擎默认值。"""
    cfg = run_config_from_settings(s)
    set_process_default(cfg)
    return cfg
