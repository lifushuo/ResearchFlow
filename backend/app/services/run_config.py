"""两种主机共用的运行配置业务规则。

主机各自负责读取环境、设置或密钥库；这里只统一哪些偏好生效、是否演示、以及提示词快照的执行规则。
不得引入 FastAPI、app.config、app.db 等主机依赖。读取偏好使用结构化协议，不要求完整 User 模型。"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import replace
from typing import Any, Protocol

from research_engine.runconfig import (
    RunConfig,
    get_run_config,
    reset_run_config,
    set_run_config,
)


class HasPreferences(Protocol):
    """两种主机的用户结构，只需提供偏好映射，无需依赖具体数据库模型。"""

    preferences: Mapping[str, Any] | None


# 与 RunConfig 同名一一对应的偏好字段。
# 缺失或 None 表示使用部署默认，不参与 replace，避免把其他字段清空。
#:
# 四个构建入口共用这一个字段列表：两个服务端驱动和两个桌面驱动。
# 不能靠备注要求多份列表同步，否则很容易新增入口时漏掉全部偏好。
PREFERENCE_FIELDS: tuple[str, ...] = (
    "retrieval_k",
    "min_sources_per_task",
    "snippet_max_chars",
    "tavily_api_key",
    "brave_api_key",
)


def preference_overrides(user: HasPreferences | None) -> dict[str, Any]:
    """只生成用户明确设置的 replace 参数；未提供和 None 都不产生覆盖项。"""
    prefs = (user.preferences if user else None) or {}
    return {k: prefs[k] for k in PREFERENCE_FIELDS if prefs.get(k) is not None}


def is_scripted(*, row_demo: bool, host_is_scripted: bool) -> bool:
    """统一判断运行是否会访问真实供应商。

    桌面在构建配置前就需要此答案，以选择脚本模型或检查密钥，不能再新增一份私有判断。"""
    return row_demo or host_is_scripted


def apply_demo_rule(
    base: RunConfig, *, row_demo: bool, host_is_scripted: bool
) -> tuple[RunConfig, bool]:
    """返回实际演示配置及是否需要修正数据库演示标记。

    请求 demo 或部署 fake 都代表脚本运行，必须统一记录，不能在证据包里冒称真实模型。结果再次输入应稳定，保证恢复不切换研究内容。
    已记录为演示的运行，即使恢复到 real 部署仍保持演示。写入具体记录的事务由调用方负责。"""
    if not is_scripted(row_demo=row_demo, host_is_scripted=host_is_scripted):
        return base, False
    return replace(base, llm_mode="fake", demo=True), not row_demo


# 持久保存提示词覆盖的实际状态，区分未配置与配置不可用，不能从最终空结果猜。
OVERRIDES_NONE = "NONE"
OVERRIDES_APPLIED = "APPLIED"
OVERRIDES_UNUSABLE = "UNUSABLE"

# 状态枚举与解析规则集中定义，模型约束也导入这里，避免词汇和行为漂移。
PROMPT_OVERRIDE_STATUSES = (OVERRIDES_NONE, OVERRIDES_APPLIED, OVERRIDES_UNUSABLE)


def prompt_settings_payload(overrides: dict[str, str | None]) -> dict[str, Any]:
    """构造已校验的 Agent 设置更新；某个角色传 null 表示恢复默认提示词。"""
    from app.schemas.auth import UserPreferences

    preferences = UserPreferences(prompt_overrides=overrides)
    return {"preferences": preferences.model_dump(exclude_unset=True)}


def snapshot_overrides(user: HasPreferences | None) -> tuple[dict[str, str], str]:
    """首次开始时读取并冻结覆盖及状态，之后不重新解析。

    其他五项普通偏好沿用恢复时重读的原行为；提示词必须冻结，避免同一报告前后指令不一致。"""
    return usable_overrides(((user.preferences if user else None) or {}).get("prompt_overrides"))


def usable_overrides(raw: Any) -> tuple[dict[str, str], str]:
    """校验已存提示词快照，要么整体使用，要么整体回退。

    一项损坏就全部标 UNUSABLE，不能生成用户从未选过的混合配置。即使接口已经校验，数据库手工编辑或迁移也可能产生坏快照，执行边界必须复核。
    角色集合在这里校验，用途是否允许覆盖仍由 prompt_composition 统一授权。"""
    from research_engine.prompt_composition import ROLES

    if not raw:
        return {}, OVERRIDES_NONE
    if not isinstance(raw, dict):
        return {}, OVERRIDES_UNUSABLE
    out: dict[str, str] = {}
    for role, text in raw.items():
        if role not in ROLES or not isinstance(text, str) or not text.strip():
            return {}, OVERRIDES_UNUSABLE
        out[role] = text
    return (out, OVERRIDES_APPLIED) if out else ({}, OVERRIDES_NONE)


@contextmanager
def chat_prompt_context(user: HasPreferences | None) -> Iterator[None]:
    """仅在当前聊天轮次上下文安装用户提示词，两个主机共用。

    退出后恢复配置，避免同一事件循环后续任务沿用他人的指令。"""
    token = set_run_config(chat_run_config(get_run_config(), user))
    try:
        yield
    finally:
        reset_run_config(token)


def chat_run_config(base: RunConfig, user: HasPreferences | None) -> RunConfig:
    """为本轮聊天给 base 应用当前用户提示词偏好。

    聊天没有跨关卡运行快照，应实时读取；两种主机共用构建函数，避免一端忽略偏好。"""
    usable, status = usable_overrides(
        ((user.preferences if user else None) or {}).get("prompt_overrides")
    )
    if status != OVERRIDES_APPLIED:
        return base
    return replace(base, prompt_overrides=usable)
