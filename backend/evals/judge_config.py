"""评测裁判的独立端点配置，避免与被测系统共用 custom 凭据。"""

import os
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import replace

from research_engine.llm_factory import reset_user_keys, set_user_keys
from research_engine.runconfig import get_run_config, reset_run_config, set_run_config


def judge_max_output_tokens() -> int:
    """独立 Judge 的输出预算包含推理 Token，不借用研究 Critic 的 1000 上限。"""
    raw = os.environ.get("EVAL_JUDGE_MAX_OUTPUT_TOKENS", "4096").strip()
    if not raw.isascii() or not raw.isdigit() or not 1 <= int(raw) <= 32768:
        raise ValueError("EVAL_JUDGE_MAX_OUTPUT_TOKENS must be an integer in 1..32768")
    return int(raw)


def judge_generation_fingerprint() -> dict:
    """1000 显式兼容旧指纹；其他上限必须隔离批次，不能静默复用旧评分。"""
    maximum = judge_max_output_tokens()
    return {} if maximum == 1000 else {"judge_generation": {"max_output_tokens": maximum}}


def judge_provider_keys(route: str) -> dict[str, str]:
    if route.partition(":")[0] != "custom":
        return {}
    endpoint = os.environ.get("EVAL_JUDGE_BASE_URL", "").strip()
    key = os.environ.get("EVAL_JUDGE_API_KEY", "").strip()
    if bool(endpoint) != bool(key):
        raise ValueError("EVAL_JUDGE_BASE_URL 与 EVAL_JUDGE_API_KEY 必须同时配置")
    return {"custom": key, "custom_base_url": endpoint} if endpoint else {}


def benchmark_judge_route(default: str) -> str:
    return os.environ.get("BENCHMARK_JUDGE_MODEL") or os.environ.get("EVAL_JUDGE_MODEL") or default


@contextmanager
def judge_client_config(route: str) -> Iterator[None]:
    from research_engine.llm_factory import output_limit

    maximum = judge_max_output_tokens()
    keys = judge_provider_keys(route)
    base = get_run_config()
    token = set_run_config(
        replace(
            base,
            models={**base.models, "critic": route},
            provider_keys={**base.provider_keys, **keys},
        )
    )
    # BYOK 的优先级高于 RunConfig；独立裁判不能继承研究用户的 custom 密钥。
    user_token = set_user_keys({}) if keys else None
    try:
        with output_limit("critic", maximum):
            yield
    finally:
        if user_token is not None:
            reset_user_keys(user_token)
        reset_run_config(token)
