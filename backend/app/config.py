"""应用配置统一来自环境变量，开发时也可读取 ../.env。

启动时尽早拒绝危险配置（docs/09 §3，docs/06 §7）。"""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# 常见占位密钥，禁止带入实际运行。
_PLACEHOLDER_SECRETS = {
    "change-me-to-a-long-random-secret-string-in-production",
    "changeme",
    "secret",
    "dev-secret",
}

# 本地推理供应商不需要 API 密钥，生产环境校验也不能强制要求。
_KEYLESS_PROVIDERS = {"ollama", "custom"}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file="../.env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # 运行环境
    environment: Literal["development", "production", "test"] = "development"
    # fake 使用固定脚本模型和受控检索数据，供测试及 CI 使用。
    llm_mode: Literal["real", "fake"] = "real"

    # 模型供应商与用户自带密钥
    google_api_key: str = ""
    anthropic_api_key: str = ""
    openai_api_key: str = ""
    openrouter_api_key: str = ""
    custom_api_key: str = ""
    custom_base_url: str = ""

    # 模型路由格式为 provider:model（docs/04 §7）
    model_planner: str = "google:gemini-2.5-pro"
    model_executor: str = "google:gemini-2.5-flash"
    model_critic: str = "google:gemini-2.5-flash"
    model_synthesizer: str = "google:gemini-2.5-pro"
    model_chat: str = "google:gemini-2.5-flash"

    # 本地或局域网 Ollama 的 OpenAI 兼容端点。
    # Docker 内访问宿主机通常使用 http://host.docker.internal:11434/v1；容器中的 localhost 指容器自身。
    ollama_base_url: str = "http://localhost:11434/v1"

    # 项目记忆与向量化（docs/14 §4）。
    # 指定将获批报告转成向量的供应商。auto 优先本地 Ollama，再回退有密钥的远程服务。
    # none 表示禁用记忆摄取和项目聊天，不暗中换成其他证据来源。
    embeddings_provider: Literal["auto", "ollama", "google", "openai", "none"] = "auto"
    # 留空使用供应商默认向量模型。更换模型后旧片段需重建索引；即使维度相同也不能混用不同模型的向量。
    # 检索按模型过滤，项目 memory/status 会披露不匹配情况。
    embeddings_model: str = ""

    # 搜索与检索配置
    tavily_api_key: str = ""
    brave_api_key: str = ""

    # 数据存储配置
    database_url: str
    redis_url: str = "redis://localhost:6379/0"
    # 原始目录默认可为相对路径，但不可直接使用。
    # 不同进程工作目录不同会产生两个语料根目录；必须通过下面的 corpus_path 取得统一绝对路径。
    corpus_dir: str = "data/corpus"

    # 身份认证配置
    jwt_secret_key: str
    jwt_algorithm: str = "HS256"
    jwt_access_token_expire_minutes: int = 15
    jwt_refresh_token_expire_days: int = 14
    require_email_verification: bool = False
    # 加密用户自带密钥。默认通过 HKDF 从 jwt_secret_key 派生独立用途的密钥。
    # 可单独配置，以便轮换 JWT 密钥时不影响已存模型密钥解密。
    encryption_key: str = ""

    # 新用户的默认月度 Token 上限，0 表示不限。公开部署应设置限额，避免单个账号耗尽共用密钥额度。
    default_monthly_token_limit: int = 0

    # 代理预算：费用、时长和输入 Token 都采用 0 表示不限（docs/04 §6）。
    # openrouter/custom 的本地费用估算为 0，不能依赖它作为实际费用兜底；真实开支还需在供应商侧限制。
    # 本地配置路径 research_engine/local.py 使用相同规则，修改时两处同步。
    max_critic_loops: int = 2
    max_cost_per_session_usd: float = 0.0
    max_wallclock_seconds: int = 0
    max_input_tokens: int = 0
    # 同一轮内研究任务可并发。1 表示严格串行，只有串行才能避免并发任务同时消耗预算导致超支（docs/12 M7）。
    max_parallel_tasks: int = 4
    celery_task_timeout_seconds: int = 660

    # 用户用量限制默认 0 表示不限，用于多用户公开部署防滥用。
    # 自托管单用户自己承担费用，不默认限流；公开演示应为两个限制设置正值（docs/06 §2）。
    #
    # 不包含登录和注册，它们的限流用于防暴力破解，始终启用。
    research_rate_limit_per_hour: int = 0
    chat_rate_limit_per_hour: int = 0

    # 可选 LangSmith 追踪
    langchain_tracing_v2: bool = False
    langchain_api_key: str = ""
    langchain_project: str = "multi-agent-research-assistant"

    # 仅开发使用前端跨域来源；生产通过同源代理访问。
    frontend_url: str = "http://localhost:3031"

    @model_validator(mode="after")
    def _validate_secrets(self) -> Settings:
        secret = self.jwt_secret_key.strip()
        if len(secret) < 32 or secret.lower() in _PLACEHOLDER_SECRETS:
            raise ValueError(
                "JWT_SECRET_KEY must be >= 32 chars of real randomness "
                "(generate with: openssl rand -hex 32). "
                "Placeholder values are refused — see docs/architecture/06-security.md §1."
            )
        if self.llm_mode == "real" and self.environment == "production":
            routed_providers = {
                m.split(":", 1)[0]
                for m in (
                    self.model_planner,
                    self.model_executor,
                    self.model_critic,
                    self.model_synthesizer,
                    self.model_chat,
                )
            }
            provider_keys = {
                "google": self.google_api_key,
                "anthropic": self.anthropic_api_key,
                "openai": self.openai_api_key,
                "openrouter": self.openrouter_api_key,
                "custom": self.custom_api_key,
            }
            for provider in routed_providers:
                if provider in _KEYLESS_PROVIDERS:
                    continue  # 本地推理（如 Ollama）不需要密钥。
                if not provider_keys.get(provider, ""):
                    raise ValueError(
                        f"Model routing uses provider '{provider}' but no API key is "
                        f"configured for it. Set the key or change MODEL_* routing."
                    )
        return self

    @property
    def is_production(self) -> bool:
        return self.environment == "production"

    @property
    def corpus_path(self) -> Path:
        """返回项目语料库的统一绝对目录。

        相对 corpus_dir 以 backend 包根目录为基准，而不是进程启动位置；绝对路径原样使用。否则不同入口虽都能成功写入，却会在不同目录查找同一文档。"""
        configured = Path(self.corpus_dir)
        if configured.is_absolute():
            return configured
        # config.py 位于 backend/app，parents[1] 就是 backend 根目录。
        return Path(__file__).resolve().parents[1] / configured


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
