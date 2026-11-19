"""重新打开本次运行或会话曾引用的特定文档版本（RFC V3.1 §7.3，AC-27）。

桌面旧文档迁移到历史项目后，原报告仍需能打开引用。允许读取的版本由调用方自己的来源记录决定，不能用客户端指定项目扩大权限；冻结过哈希的版本必须匹配。
主机决定查找原项目和合法历史项目，身份不匹配则跳过。只返回那一个版本，不提供其他项目列表或搜索。所有拒绝表现为文档不存在，两种主机共用。"""

from __future__ import annotations

import uuid
from collections.abc import Iterable

from app.services.corpus_selection import document_id_of


def no_other_projects() -> list[uuid.UUID]:
    """服务端没有旧共用语料项目，不额外搜索其他项目；桌面可传入其合法迁移历史项目。"""
    return []


def recorded_ids(urls: Iterable[str | None]) -> set[str]:
    """从本运行或会话记录的来源 URL 提取引用过的文档版本 ID。"""
    return {doc for url in urls if (doc := document_id_of(url or "")) is not None}


def recorded_hashes(selection: dict | None) -> dict[str, str]:
    """冻结选择中的文档版本 ID 到 SHA-256 映射；3.1 前为空。"""
    try:
        return {
            str(item["document_id"]): str(item["sha256"])
            for item in (selection or {}).get("resolved", [])
            if item.get("sha256")
        }
    except AttributeError, KeyError, TypeError:
        return {}


async def store_holding(
    locator,
    document_id: str,
    *,
    cited: set[str],
    hashes: dict[str, str],
    corpus_id: str | None,
    projects: list[uuid.UUID],
):
    """寻找持有目标文档且允许调用方访问的存储；所有拒绝均返回 None。"""
    if document_id not in cited:
        return None
    for project_id in dict.fromkeys(projects):
        store = await locator.for_project(project_id)
        if store is None:
            continue
        if corpus_id is not None and (await store.identity())[0] != corpus_id:
            continue
        stored = await store.sha256_of(document_id)
        if stored is None:
            continue
        if document_id in hashes and stored != hashes[document_id]:
            return None
        return store
    return None
