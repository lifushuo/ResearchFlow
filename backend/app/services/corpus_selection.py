"""创建运行时一次性冻结文件选择（RFC V3.1 §8.2）。

把请求解析为确切文档版本并存入运行。计划关卡、恢复、保存和导出都读同一快照；文件后来替换、删除或移动，不能扩大或改变原范围。两种主机共用，不依赖服务器设置或数据库。"""

from __future__ import annotations

from typing import Any
from urllib.parse import unquote, urlsplit


class SelectionError(ValueError):
    """无法冻结文件选择；code 为稳定的错误原因标识。"""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


async def freeze_selection(
    store: Any,
    *,
    folder_ids: list[str] | None,
    doc_keys: list[str] | None,
    whole_corpus: bool,
) -> dict:
    """把选择请求解析成冻结记录，失败抛 SelectionError。

    whole_corpus 同样冻结当时整个项目的版本集合；store 为 None 表示项目还没有语料。"""
    folder_ids = list(folder_ids or [])
    doc_keys = list(doc_keys or [])
    if not whole_corpus and not folder_ids and not doc_keys:
        raise SelectionError("selection_empty", "Select at least one file or folder.")

    folder_keys: set[str] = set()
    if folder_ids:
        # 文件夹只展开为创建当时其中及子目录的文档；后来加入的文件不属于此次运行。
        found = await store.folder_doc_keys(folder_ids) if store is not None else None
        if found is None:
            # 不区分 ID 存在于别的项目还是根本不存在，避免泄露。
            raise SelectionError(
                "selection_not_found", "Some selected files or folders are not in this project."
            )
        folder_keys = found

    requested_keys = sorted(set(doc_keys) | folder_keys)
    if store is None:
        resolved: list[dict] = []
    else:
        resolved = await store.current_versions(None if whole_corpus else requested_keys)

    if not whole_corpus:
        found = {item["doc_key"] for item in resolved}
        # 明确指定的每个文件都必须能解析；文件夹中没有当前版本的文档则不纳入。
        if set(doc_keys) - found:
            raise SelectionError(
                "selection_not_found",
                "Some selected files are not in this project, or have no current version.",
            )

    if not resolved:
        raise SelectionError(
            "selection_resolved_empty",
            "The selection contains no documents that can be read. Upload a document, or "
            "choose files that are not auto-saved reports.",
        )

    corpus_id = corpus_version = None
    if store is not None:
        corpus_id, corpus_version = await store.identity()
    return {
        "requested": {
            "folder_ids": sorted(set(folder_ids)),
            "doc_keys": requested_keys,
            "whole_corpus": whole_corpus,
        },
        "resolved": resolved,
        "corpus_id": corpus_id,
        "corpus_version": corpus_version,
    }


def allowed_ids(selection: dict | None) -> frozenset[str] | None:
    """返回冻结选择允许的文档版本集合；只有确实没配置选择才返回 None。

    记录损坏返回空集合，让 ScopedCorpus 明确拒绝；若返回 None 会被解释为不限范围，反而允许读取全部文件。"""
    if selection is None:
        return None
    try:
        return frozenset(str(item["document_id"]) for item in selection["resolved"])
    except KeyError, TypeError:
        return frozenset()


def document_id_of(url: str) -> str | None:
    """从 corpus:// 地址提取文档版本 ID，其他地址返回 None。

    这里只取身份，不取片段位置；不能把含完整片段的 Source.corpus_document_id 直接与选择集合比较。"""
    parts = urlsplit(url or "")
    if parts.scheme != "corpus" or not parts.netloc:
        return None
    return unquote(parts.netloc)


def violations(selection: dict | None, urls: list[str]) -> list[str]:
    """找出冻结选择不允许引用的 URL，作为 ScopedCorpus 后的第二道检查（RFC V3.1 §8.5）。

    只有指向所选版本的语料 URL 合法，网页 URL 也算越界。发现越界说明存在绕过端口的路径；未配置选择则无此限制。"""
    allowed = allowed_ids(selection)
    if allowed is None:
        return []
    bad: list[str] = []
    for url in urls:
        doc = document_id_of(url)
        if doc is None or doc not in allowed:
            bad.append(url)
    return sorted(set(bad))
