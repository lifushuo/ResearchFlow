"""两种主机统一的上传校验、错误与返回结构。

底层语料布局不同，但客户端契约必须相同。仅依赖标准库、Pydantic 和领域层，禁止 FastAPI、app.config、app.db，保证桌面可导入。
抛出领域错误，再由两端统一错误响应处理器转换状态，避免同一问题一端 400、另一端 422。"""

from __future__ import annotations

from pathlib import Path

from app import metrics
from app.errors import DependencyUnavailable, Invalid, PayloadTooLarge, Unprocessable
from app.schemas.corpus import CorpusStatusResponse, DocumentResponse
from research_engine.corpus import UnknownDocumentKey
from research_engine.documents import MAX_DOCUMENT_BYTES

__all__ = [
    "MAX_DOCUMENT_BYTES",
    "clean_upload",
    "document_response",
    "ingest_document",
    "ingested_response",
    "status_response",
]

# 文档不属于语料时统一返回同一提示，避免两端连标点都不一致。
DOCUMENT_NOT_FOUND = "Document not found."


def clean_upload(filename: str | None, data: bytes) -> str:
    """校验上传并返回安全保存名称。

    文件过大使用独立错误，区别于内容不支持或格式有误，便于客户端给出正确处理方式。"""
    name = Path(filename or "").name  # 仅用于元数据，绝不作为磁盘路径。
    if not name:
        raise Invalid("A filename is required.")
    if not data:
        raise Invalid("The document is empty.")
    if len(data) > MAX_DOCUMENT_BYTES:
        raise PayloadTooLarge(
            f"The document is {len(data)} bytes; the limit is {MAX_DOCUMENT_BYTES}."
        )
    return name


# 限制上传相对路径深度和长度，既容纳正常文件夹，又防止恶意路径无限扩充文件夹表（RFC V3.1 §13）。
MAX_PATH_DEPTH = 32
MAX_COMPONENT_CHARS = 255


def folder_components(relative_path: str | None) -> list[str]:
    """将上传相对路径转换为从外到内的文件夹名称列表，不接触磁盘。

    兼容两种分隔符，拒绝空、. 和 ..，Unicode 规范化为 NFC，避免越界和同目录多种表示。"""
    import unicodedata

    if not relative_path:
        return []
    parts = [unicodedata.normalize("NFC", p) for p in relative_path.replace("\\", "/").split("/")]
    if any(not p.strip() or p.strip() in (".", "..") or "\0" in p for p in parts):
        raise Invalid("The folder path contains an empty, '.' or '..' part.")
    folders = [p.strip() for p in parts[:-1]]
    if len(folders) > MAX_PATH_DEPTH:
        raise Invalid(f"Folders are limited to {MAX_PATH_DEPTH} levels deep.")
    if any(len(p) > MAX_COMPONENT_CHARS for p in folders):
        raise Invalid(f"Folder names are limited to {MAX_COMPONENT_CHARS} characters.")
    return folders


async def ingest_document(
    store,
    filename: str,
    data: bytes,
    doc_key: str | None = None,
    relative_path: str | None = None,
) -> DocumentResponse:
    """共用的校验、摄取和响应转换。

    指定相对目录时按需建文件夹；同字节文件上传到第二个文件夹，只新增成员关系，不复制文档（RFC V3.1 §7.1）。
    格式问题是用户输入错误，向量服务不可用是依赖故障，分别报告。所有出口在此统一计数，标签使用固定结果，不携带可能含文件名的错误文本。"""
    from research_engine.embeddings import EmbeddingsUnavailable

    try:
        name = clean_upload(filename, data)
        folders = folder_components(relative_path)
        result = await store.ingest(name, data, doc_key=doc_key)
    except UnknownDocumentKey as e:
        # 请求明确表示替换某文档，但本语料没有该文档时返回 422；不能偷偷创建新文档并返回 201。
        metrics.observe_ingest("rejected")
        raise Unprocessable(f"No document with key {e.args[0]} in this corpus.") from e
    except Invalid, PayloadTooLarge:  # clean_upload 已拒绝此输入。
        metrics.observe_ingest("rejected")
        raise
    except ValueError as e:  # 不支持的扩展名或无法读取的文档。
        metrics.observe_ingest("rejected")
        raise Invalid(str(e)) from e
    except EmbeddingsUnavailable as e:
        metrics.observe_ingest("unavailable")
        raise DependencyUnavailable(str(e)) from e
    if folders and result.doc_id and result.doc_id != "skip":
        folder_id = await store.ensure_folder_path(folders)
        key = result.doc_key or await store.doc_key_of(result.doc_id)
        if key and folder_id:
            await store.add_membership(key, folder_id, name)
    # 已存在而 skipped 是成功跳过，但没有索引新字节，不计入吞吐字节总数。
    metrics.observe_ingest("skipped" if result.skipped else "ingested", byte_count=len(data))
    return ingested_response(result)


def ingested_response(result) -> DocumentResponse:
    """将 Ingested 转为客户端结构。

    识别为重复文档时 doc_id 为 None，对外沿用字符串 skip，让客户端与真实 ID 区分。"""
    return DocumentResponse(
        id=result.doc_id or "skip",
        filename=result.filename,
        chunks=result.chunks_written,
        doc_key=result.doc_key,
        version=result.version,
    )


def document_response(row: dict) -> DocumentResponse:
    """将 CorpusStore.documents 行转换成统一 API 结构。

    存储内部字段 chunk_count、ingested_at 对应接口 chunks、created_at；列表必须是裸数组，不能加 documents 包装导致前端误判为空。"""
    return DocumentResponse(
        id=row["id"],
        filename=row["filename"],
        chunks=row["chunk_count"],
        created_at=row.get("ingested_at"),
        size_bytes=row.get("size_bytes"),
        downloadable=row.get("downloadable", False),
        origin=row.get("origin", "uploaded"),
        doc_key=row.get("doc_key"),
        version=row.get("version"),
        is_current=row.get("is_current", True),
    )


def status_response(stat: dict) -> CorpusStatusResponse:
    """将 CorpusStore.status 转为客户端结构。

    存储层没有总 chunks，在此统一求和，避免桌面漏算。"""
    return CorpusStatusResponse(**{**stat, "chunks": sum(stat["chunks_by_model"].values())})
