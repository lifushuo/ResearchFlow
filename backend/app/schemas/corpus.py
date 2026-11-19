"""项目语料接口共用的响应结构。

服务端和桌面底层布局可以不同，客户端字段和列表形状必须一致。此前桌面返回 documents 包装对象，前端按数组读取，导致有内容却显示为空。
结构移到 schemas 后，桌面无需导入依赖 app.config 的服务端路由。禁止引入 app.config、app.db 或间接依赖它们的模块。"""

from __future__ import annotations

from pydantic import BaseModel


class DocumentResponse(BaseModel):
    id: str
    filename: str
    chunks: int
    created_at: str | None = None
    size_bytes: int | None = None
    # 早期未保留原文件的文档为 False；界面据此决定是否提供打开和下载，不能假设每条都有文件。
    downloadable: bool = False
    # 来源类型 uploaded 或 generated；生成报告必须单独标识，不能冒充上传的原始证据。检索不会将生成报告当作来源。
    origin: str = "uploaded"
    # doc_key 是存储层生成的逻辑文档 UUID，不从文件名推断，另记具体版本。
    # 客户端用它明确表达新上传替换哪个文档，避免同名论文被错误合并。
    doc_key: str | None = None
    version: int | None = None
    # 被替换版本保留供历史引用读取，但不再作为新检索的证据。
    is_current: bool = True


class CorpusStatusResponse(BaseModel):
    documents: int
    chunks: int
    chunks_by_model: dict[str, int]
    current_model: str
    # 用存于数据中的身份和状态识别语料库，不能用主机特有、可能移动的文件路径代替。
    corpus_id: str | None = None
    corpus_version: int | None = None


class ExtractedTextResponse(BaseModel):
    """引用核验所依据的文档正文，供预览高亮使用。"""

    filename: str
    kind: str
    text: str
    segments: list[list] = []
    extraction_mode: str = "verbatim"


class FolderResponse(BaseModel):
    id: str
    parent_id: str | None = None
    name: str


class MembershipResponse(BaseModel):
    doc_key: str
    folder_id: str
    display_name: str


class CorpusTreeResponse(BaseModel):
    """平铺返回文件夹、文档和成员关系，由客户端统一构树；无成员关系的文档位于根目录（RFC V3.1 §7.1）。"""

    folders: list[FolderResponse]
    documents: list[DocumentResponse]
    memberships: list[MembershipResponse]


class FolderCreateRequest(BaseModel):
    name: str
    parent_id: str | None = None


class FolderUpdateRequest(BaseModel):
    """name 修改名称，parent_id 移动位置，空值表示根目录；未提供的字段不变。"""

    name: str | None = None
    parent_id: str | None = None
    move: bool = False


class FolderDeleteResponse(BaseModel):
    folders_deleted: int
    documents_deleted: int
    versions_deleted: int
