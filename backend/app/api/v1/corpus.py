"""项目语料库接口（docs/12 M10）。

每个项目使用独立本地 SQLite 数据库，隔离文档和向量。"""

from __future__ import annotations

import uuid

import structlog
from fastapi import APIRouter, Depends, Form, HTTPException, Response, UploadFile, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.v1.projects import resolve_project
from app.db.base import get_db
from app.dependencies import get_current_user
from app.models.user import User
from app.ports import CorpusLocator

# 重新导出共用结构，不复制定义；结构位于 app/schemas/corpus.py，校验和响应转换位于 corpus_ingest。
from app.schemas.corpus import (
    CorpusStatusResponse,
    CorpusTreeResponse,
    DocumentResponse,
    ExtractedTextResponse,
    FolderCreateRequest,
    FolderDeleteResponse,
    FolderResponse,
    FolderUpdateRequest,
)
from app.services import corpus_ingest

# 下载策略放在仅依赖标准库的模块，桌面可导入而不引入服务端配置链。
# 这里继续导出，兼容已有调用方和测试。
from app.services.document_headers import (  # noqa: F401 — 供其他模块导入
    _MEDIA_TYPES,
    download_headers,
    media_type_for,
)
from app.services.host_ports import get_corpus_locator
from research_engine.corpus import CorpusStore, FolderNotFound

logger = structlog.get_logger()
router = APIRouter(prefix="/projects/{project_id}/corpus", tags=["Corpus"])


async def _get_corpus_store(project_id: uuid.UUID, corpus: CorpusLocator) -> CorpusStore:
    """取得项目语料库，不存在时创建。

    路径统一由 CorpusLocator 解析，避免上传和研究各自拼路径。通过依赖注入取得定位器，让桌面能传入针对其单库布局的实现。"""
    return await corpus.ensure(project_id)


@router.post("/documents", response_model=DocumentResponse, status_code=status.HTTP_201_CREATED)
async def upload_document(
    project_id: uuid.UUID,
    file: UploadFile,
    # 文档版本标识是可选字段，旧客户端仍按原行为创建新文档。
    # 只有明确提供标识才作为已有文档的新修订；不能根据文件名猜，因为不同论文可能同名。
    doc_key: str | None = Form(default=None),
    # 记录文件在上传文件夹内的相对路径，并按需建目录；未提供则存放在根目录。
    relative_path: str | None = Form(default=None),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
    corpus: CorpusLocator = Depends(get_corpus_locator),
):
    """将新文档写入项目语料。

    校验、错误映射和返回结构共用 app.services.corpus_ingest，保证服务端与桌面拒绝相同输入时返回相同语义。"""
    # 检查当前用户是否拥有项目，保证项目隔离。
    await resolve_project(db, current_user.id, project_id)

    store = await _get_corpus_store(project_id, corpus)
    return await corpus_ingest.ingest_document(
        store, file.filename, await file.read(), doc_key, relative_path
    )


@router.get("/documents", response_model=list[DocumentResponse])
async def list_documents(
    project_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
    corpus: CorpusLocator = Depends(get_corpus_locator),
):
    """列出项目语料库的全部文档。"""
    await resolve_project(db, current_user.id, project_id)

    store = await _get_corpus_store(project_id, corpus)
    return [corpus_ingest.document_response(d) for d in await store.documents()]


@router.get("/documents/{doc_id}/download")
async def download_document(
    project_id: uuid.UUID,
    doc_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
    corpus: CorpusLocator = Depends(get_corpus_locator),
):
    """返回上传原文件，让用户可以用原生应用打开。

    必须先由 resolve_project 检查归属，不能只凭文档 ID 放行。download_headers 统一决定类型和下载行为，防止上传内容在本站来源执行；只有 PDF 预览采用有限例外。"""
    await resolve_project(db, current_user.id, project_id)

    store = await _get_corpus_store(project_id, corpus)
    found = await store.blob(doc_id)
    if found is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No stored file for this document. Documents added before file "
            "retention was enabled keep their searchable text but not the original.",
        )
    data, filename, kind = found
    return Response(
        content=data,
        media_type=media_type_for(kind),
        headers=download_headers(kind, filename),
    )


@router.delete("/documents/{doc_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_document(
    project_id: uuid.UUID,
    doc_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
    corpus: CorpusLocator = Depends(get_corpus_locator),
):
    """从项目语料库删除文档。"""
    await resolve_project(db, current_user.id, project_id)

    store = await _get_corpus_store(project_id, corpus)
    if not await store.delete(doc_id):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=corpus_ingest.DOCUMENT_NOT_FOUND
        )
    # 明确返回无内容响应；若 204 路由直接返回 None，FastAPI 可能附上 JSON 类型但正文为空，客户端会解析失败。
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.get("/status", response_model=CorpusStatusResponse)
async def get_status(
    project_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
    corpus: CorpusLocator = Depends(get_corpus_locator),
):
    """读取项目语料库状态。"""
    await resolve_project(db, current_user.id, project_id)

    store = await _get_corpus_store(project_id, corpus)
    return corpus_ingest.status_response(await store.status())


# 文件夹管理（RFC V3.1 §7.1）


@router.get("/tree", response_model=CorpusTreeResponse)
async def get_tree(
    project_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
    corpus: CorpusLocator = Depends(get_corpus_locator),
):
    """以平铺列表返回项目中的文件夹、文档及归属关系。"""
    await resolve_project(db, current_user.id, project_id)
    store = await _get_corpus_store(project_id, corpus)
    return CorpusTreeResponse(
        folders=await store.folders(),
        documents=[corpus_ingest.document_response(d) for d in await store.documents()],
        memberships=await store.memberships(),
    )


@router.post("/folders", response_model=FolderResponse, status_code=status.HTTP_201_CREATED)
async def create_folder(
    project_id: uuid.UUID,
    body: FolderCreateRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
    corpus: CorpusLocator = Depends(get_corpus_locator),
):
    await resolve_project(db, current_user.id, project_id)
    store = await _get_corpus_store(project_id, corpus)
    try:
        folder_id = await store.create_folder(body.name, body.parent_id)
    except FolderNotFound as e:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="Folder not found.") from e
    except ValueError as e:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, detail=str(e)) from e
    return FolderResponse(id=folder_id, parent_id=body.parent_id, name=body.name.strip())


@router.patch("/folders/{folder_id}", response_model=FolderResponse)
async def update_folder(
    project_id: uuid.UUID,
    folder_id: str,
    body: FolderUpdateRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
    corpus: CorpusLocator = Depends(get_corpus_locator),
):
    """重命名或移动：name 为新名称；move=true 时使用 parent_id，空值表示根目录。"""
    await resolve_project(db, current_user.id, project_id)
    store = await _get_corpus_store(project_id, corpus)
    try:
        if body.move:
            await store.update_folder(folder_id, name=body.name, parent_id=body.parent_id)
        else:
            await store.update_folder(folder_id, name=body.name)
    except FolderNotFound as e:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="Folder not found.") from e
    except ValueError as e:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, detail=str(e)) from e
    folder = next(f for f in await store.folders() if f["id"] == folder_id)
    return FolderResponse(**folder)


@router.delete("/folders/{folder_id}", response_model=FolderDeleteResponse)
async def delete_folder(
    project_id: uuid.UUID,
    folder_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
    corpus: CorpusLocator = Depends(get_corpus_locator),
):
    """删除文件夹及子文件夹；只属于这些目录的文档一并删除。"""
    await resolve_project(db, current_user.id, project_id)
    store = await _get_corpus_store(project_id, corpus)
    try:
        return FolderDeleteResponse(**await store.delete_folder(folder_id))
    except FolderNotFound as e:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="Folder not found.") from e


@router.get("/documents/{doc_id}/text", response_model=ExtractedTextResponse)
async def get_document_text(
    project_id: uuid.UUID,
    doc_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
    corpus: CorpusLocator = Depends(get_corpus_locator),
):
    """返回特定版本的标准提取正文，引用核验与预览高亮都以同一份文本为准（RFC V3.1 §12）。"""
    await resolve_project(db, current_user.id, project_id)
    store = await _get_corpus_store(project_id, corpus)
    found = await store.extracted(doc_id)
    if found is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail=corpus_ingest.DOCUMENT_NOT_FOUND)
    return ExtractedTextResponse(**found)
