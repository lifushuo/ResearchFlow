"""读取 LangGraph 检查点，严格区分三种结果。

READ：成功解码，证据可以为空；MISSING：线程没有快照；
UNREADABLE：快照存在但无法按检查点格式读取。只有 READ 才能判断内容是否为空。

直接返回 snapshot.values 或空字典会把缺失与空内容混淆。这里通过 saver 协议提供纯读取逻辑，
供迁移和运行结果持久化共用。还需检查解码后的结构：损坏数据可能解码成整数 0 而不抛异常，
不能仅凭没有异常就宣称读取成功。"""

from __future__ import annotations

import enum
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any


class CheckpointOutcome(enum.StrEnum):
    READ = "READ"
    MISSING = "MISSING"
    UNREADABLE = "UNREADABLE"


@dataclass(frozen=True)
class CheckpointRead:
    outcome: CheckpointOutcome
    values: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    @property
    def evidence(self) -> list[dict]:
        """只有 outcome 为 READ 时才有意义；缺失或不可读时为空，调用方必须先判断 outcome。"""
        return list(self.values.get("evidence") or [])

    @property
    def contradictions(self) -> list[dict]:
        return list(self.values.get("contradictions") or [])


async def read_checkpoint(saver, thread_id: str) -> CheckpointRead:
    """读取线程最新状态，保留读取成功、缺失、不可读三种结果。

    使用 aget_tuple，因为 graph.aget_state 会把不存在的快照正规化成空状态。"""
    try:
        tup = await saver.aget_tuple({"configurable": {"thread_id": thread_id}})
    except Exception as exc:  # noqa: BLE001 — 解码或 I/O 失败需要作为独立读取结果返回
        return CheckpointRead(
            CheckpointOutcome.UNREADABLE, error=f"{type(exc).__name__}: {exc}"[:400]
        )

    if tup is None:
        return CheckpointRead(CheckpointOutcome.MISSING)

    # 没有抛异常不代表解码结果有效，还必须核对结构。
    #
    # 实际损坏的 saver 数据曾解码成整数 0，旧代码又把它转成空字典并报告 READ。
    # 这样会把读取失败误报为没有证据，因此必须保留不可读状态。
    #
    # 检查结构及 channel_values 是否存在；未知格式不能当作空证据。
    raw = tup.checkpoint
    if not isinstance(raw, Mapping):
        return CheckpointRead(
            CheckpointOutcome.UNREADABLE,
            error=f"checkpoint decoded to {type(raw).__name__}, not a mapping",
        )
    channels = raw.get("channel_values")
    if channels is None:
        return CheckpointRead(
            CheckpointOutcome.UNREADABLE, error="checkpoint carries no channel_values"
        )
    if not isinstance(channels, Mapping):
        return CheckpointRead(
            CheckpointOutcome.UNREADABLE,
            error=f"channel_values is {type(channels).__name__}, not a mapping",
        )

    try:
        values = dict(channels)
    except Exception as exc:  # noqa: BLE001 — 快照存在，但无法解包
        return CheckpointRead(
            CheckpointOutcome.UNREADABLE, error=f"{type(exc).__name__}: {exc}"[:400]
        )

    return CheckpointRead(CheckpointOutcome.READ, values=values)
