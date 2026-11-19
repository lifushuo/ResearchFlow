"""本地资料库，可作为研究唯一证据源，见 internal/12 M10、docs/13 隐私分层。

通过 ContextVar 安装 Corpus 端口，供工具内部获取。corpus_mode 开启后，
检索只走此端口，read_webpage 只允许 corpus://，无需修改图内搜索与读取步骤。

CorpusStore 负责 PDF/MD/TXT 提取、精确偏移切块、嵌入和 SQLite 余弦检索。
桌面端没有 pgvector，本地规模使用 SQLite 与 numpy 即可。
每块保存原文 [start,end)，URL 携带文档、页码和字符区间，读取时精确还原位置。"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
import sqlite3
import uuid
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import unquote, urlsplit

import structlog

from research_engine.chunking import chunk_document
from research_engine.documents import extract
from research_engine.embeddings import EmbeddingsUnavailable
from research_engine.ports import Corpus, Embeddings
from research_engine.runconfig import get_run_config

logger = structlog.get_logger()

_CORPUS_SCHEME = "corpus"

# 逐文档分批嵌入，限制同时持有的向量数量，避免导入大量文档时内存峰值失控。
_EMBED_BATCH = 32

# 读取引用附近的适量上下文，兼顾句子完整与工具返回成本。
_READ_CONTEXT_CHARS = 600
_READ_MAX_CHARS = 8000


# ── 端口安装 ──


class NoCorpus:
    """未安装资料库的默认实现，明确拒绝调用。

    仅资料库模式下静默返回空结果会导致无来源报告；抛错才能让代理与日志看到配置问题。"""

    async def search(self, query: str, max_results: int) -> list[dict]:  # noqa: ARG002
        raise RuntimeError(
            "Corpus-only mode is active but no corpus is installed. Ingest documents "
            "first (POST /api/v1/corpus/documents), or disable corpus-only mode."
        )

    async def read(self, url: str) -> dict:
        return {"url": url, "title": "", "text": "", "error": "no corpus is installed"}


_NO_CORPUS: Corpus = NoCorpus()

_corpus: ContextVar[Corpus] = ContextVar("engine_corpus", default=_NO_CORPUS)


def set_corpus(corpus: Corpus):
    """为当前上下文安装资料库，返回可供 reset_corpus 恢复的令牌。"""
    return _corpus.set(corpus)


def reset_corpus(token) -> None:
    _corpus.reset(token)


def get_corpus() -> Corpus:
    return _corpus.get()


class ScopedCorpus:
    """仅允许访问本轮冻结的文档版本选择。

    限制在端口处强制执行，而非事后过滤或仅告知模型；因此检索、读取以及模型编造的
    corpus:// 地址都不能越过选择范围，见 RFC V3.1 §8.4。"""

    def __init__(self, inner: CorpusStore, allowed_ids: frozenset[str]):
        if not allowed_ids:
            # 空选择应在运行前被拒绝；若仍到这里说明存在缺陷，不能退回全库检索。
            raise ValueError("a scoped corpus needs at least one selected document")
        self._inner = inner
        self._allowed = frozenset(allowed_ids)

    @property
    def allowed_ids(self) -> frozenset[str]:
        return self._allowed

    async def search(self, query: str, max_results: int) -> list[dict]:
        # 每个选中文件最多取 max(1, ceil(k/n)) 条，避免单文件独占结果。
        # 检索策略变化需用 §17 评测验证，不能假定一定改善。
        per_document = max(1, -(-max_results // len(self._allowed)))
        return await self._inner.search(
            query, max_results, document_ids=self._allowed, per_document=per_document
        )

    async def read(self, url: str) -> dict:
        location = parse_corpus_url(url)
        if location is None:
            return {"url": url, "title": "", "text": "", "error": "not a valid corpus URL"}
        if location.doc_id not in self._allowed:
            # 拒绝读取时不提供可引用正文，也不透露文档是否存在。
            return {"url": url, "title": "", "text": "", "error": OUTSIDE_SELECTION}
        return await self._inner.read(url)


OUTSIDE_SELECTION = "outside_selection: this document is not one of the run's selected files"


# ── 资料库 URL 格式 ──
#
# corpus://<document-id>#chars=<start>-<end>[&page=<n>]
#
# URL 片段表示提取文本中的精确偏移和可选页码。
# 使用 corpus:// 而非 HTTP，避免资料库位置被当作网络地址抓取或被网页结果冒充。


def corpus_url(doc_id: str, *, start: int, end: int, page: int | None) -> str:
    fragment = f"chars={start}-{end}"
    if page is not None:
        fragment += f"&page={page}"
    return f"{_CORPUS_SCHEME}://{doc_id}#{fragment}"


@dataclass(frozen=True)
class CorpusLocation:
    """解析后的资料库位置：文档、字符区间及可选页码。"""

    doc_id: str
    start: int
    end: int
    page: int | None


def parse_corpus_url(url: str) -> CorpusLocation | None:
    """解析资料库 URL；格式不符时返回 None，不能放宽接受范围。"""
    parts = urlsplit(url)
    if parts.scheme != _CORPUS_SCHEME or not parts.netloc:
        return None
    match = re.fullmatch(r"chars=(\d+)-(\d+)(?:&page=(\d+))?", parts.fragment)
    if match is None:
        return None
    start, end = int(match.group(1)), int(match.group(2))
    if end <= start:
        return None
    page = int(match.group(3)) if match.group(3) else None
    return CorpusLocation(doc_id=unquote(parts.netloc), start=start, end=end, page=page)


def segment_label(segments: list, start: int, end: int) -> str:
    """把 [start,end) 转成人可读位置；跨区段时显示起止范围，没有区段时返回空文本。"""
    first = last = ""
    for seg_start, label in segments:
        if seg_start <= start:
            first = label
        if seg_start < end:
            last = label
    if not first:
        return ""
    if last and last != first:
        head, _, n1 = first.rpartition(" ")
        head2, _, n2 = last.rpartition(" ")
        if head == head2 and n1.isdigit() and n2.isdigit():
            return f"{head}s {n1}–{n2}" if not head.endswith("s") else f"{head} {n1}–{n2}"
        return f"{first} – {last}"
    return first


class FolderNotFound(LookupError):
    """请求的文件夹 ID 不属于本资料库。"""


def _folder_name(name: str) -> str:
    """规范化文件夹名称：去首尾空白、NFC、仅单层路径且不能为空。"""
    import unicodedata

    cleaned = unicodedata.normalize("NFC", (name or "").strip())
    if (
        not cleaned
        or cleaned in (".", "..")
        or "/" in cleaned
        or "\\" in cleaned
        or "\0" in cleaned
    ):
        raise ValueError(f"{name!r} is not a usable folder name.")
    if len(cleaned) > 255:
        raise ValueError("Folder names are limited to 255 characters.")
    return cleaned


# ── SQLite 存储 ──

_SCHEMA = """
CREATE TABLE IF NOT EXISTS corpus_documents (
    id          TEXT PRIMARY KEY,
    filename    TEXT NOT NULL,
    kind        TEXT NOT NULL,
    sha256      TEXT NOT NULL,
    text        TEXT NOT NULL,
    page_starts TEXT NOT NULL,
    chunk_count INTEGER NOT NULL DEFAULT 0,
    ingested_at TEXT NOT NULL,
    -- The original upload, verbatim. Extracted `text` is what the agents search; this is
    -- what a human opens. Without it the UI could only offer "view extracted text", which
    -- is not the same as opening your own PDF. Nullable on purpose: documents ingested
    -- before this column existed keep working and simply cannot be downloaded.
    blob        BLOB,
    -- 'uploaded' (a human added it) or 'generated' (this project's own approved report,
    -- auto-saved after the report-review gate — internal/12 M10 follow-up). Retrieval excludes
    -- 'generated' unconditionally (see `_search_sync`): without this, an approved report
    -- becomes citable evidence for the next one, and a `[n]` marker would resolve cleanly
    -- while pointing at the model's own earlier output rather than a real source — the
    -- unverifiable-citation class AGENTS.md calls P0, just laundered through a real file.
    origin      TEXT NOT NULL DEFAULT 'uploaded'
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_corpus_doc_dedupe
    ON corpus_documents (filename, sha256);
CREATE TABLE IF NOT EXISTS corpus_chunks (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    document_id     TEXT NOT NULL REFERENCES corpus_documents(id) ON DELETE CASCADE,
    chunk_index     INTEGER NOT NULL,
    start           INTEGER NOT NULL,
    "end"           INTEGER NOT NULL,
    page            INTEGER,
    text            TEXT NOT NULL,
    embedding       BLOB NOT NULL,
    embedding_model TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_corpus_chunks_doc ON corpus_chunks (document_id);
"""

#: 单行保存资料库身份及变更版本。
#:
#: 身份保存在数据中而非依赖文件路径，这样跨宿主或移动文件后仍能确认来源资料库。
#:
#: 任何影响检索内容的变更都递增 version，运行可据此记录其使用的资料库状态。
_META_SCHEMA = """
CREATE TABLE IF NOT EXISTS corpus_meta (
    corpus_id      TEXT PRIMARY KEY,
    schema_version INTEGER NOT NULL,
    version        INTEGER NOT NULL,
    created_at     TEXT NOT NULL
);
"""

#: 文件夹成员关系绑定逻辑文档 doc_key；新版本继承文件夹，同一文档可属于多个文件夹。
#: 没有成员关系的文档显示在根目录。
_FOLDER_SCHEMA = """
CREATE TABLE IF NOT EXISTS corpus_folders (
    id         TEXT PRIMARY KEY,
    parent_id  TEXT REFERENCES corpus_folders(id) ON DELETE CASCADE,
    name       TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_corpus_folder_name
    ON corpus_folders (parent_id, name) WHERE parent_id IS NOT NULL;
CREATE UNIQUE INDEX IF NOT EXISTS idx_corpus_folder_root_name
    ON corpus_folders (name) WHERE parent_id IS NULL;
CREATE TABLE IF NOT EXISTS corpus_memberships (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    doc_key      TEXT NOT NULL,
    folder_id    TEXT NOT NULL REFERENCES corpus_folders(id) ON DELETE CASCADE,
    display_name TEXT NOT NULL,
    added_at     TEXT NOT NULL,
    UNIQUE (folder_id, doc_key)
);
CREATE INDEX IF NOT EXISTS idx_corpus_membership_doc ON corpus_memberships (doc_key);
"""

#: 当前写入的资料库文件结构版本；_migrate 增加步骤时同步递增。
_SCHEMA_VERSION = 3


class UnknownDocumentKey(LookupError):
    """替换请求指向的逻辑文档不存在。

    必须明确报错，不能静默创建新文档，否则用户会误以为旧版本已被替代。"""


@dataclass(frozen=True)
class Ingested:
    """一次导入的实际结果；skipped 表示已存在而跳过，属于成功。"""

    doc_id: str | None = None
    filename: str = ""
    chunks_written: int = 0
    skipped: bool = False
    reason: str | None = None
    #: 返回逻辑文档 doc_key 与版本，供调用方以后明确替换；不要求调用方猜测不透明 ID。
    doc_key: str | None = None
    version: int | None = None


class CorpusStore:
    """Corpus 端口的 SQLite 向量存储实现。

    每次操作使用短连接，阻塞工作放在线程中；启用 WAL 和外键，使删除文档同时删除块。
    向量按小端 float32 保存，以余弦相似度扫描。按 embedding_model 过滤，
    因为不同模型的向量不可混排，否则会产生看似可信但无意义的结果。"""

    def __init__(self, path: str | Path, embedder: Embeddings) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._embedder = embedder
        with self._connect() as conn:
            conn.executescript(_SCHEMA)
            self._migrate(conn)

    @staticmethod
    def _migrate(conn: sqlite3.Connection) -> None:
        """把现有资料库文件迁移到当前结构。

        CREATE TABLE IF NOT EXISTS 不会给旧表加列；每项目独立 SQLite 文件没有 Alembic，
        因此在这里按需检查并增加字段，保持增量、幂等。"""
        have = {row[1] for row in conn.execute("PRAGMA table_info(corpus_documents)")}
        if "blob" not in have:
            conn.execute("ALTER TABLE corpus_documents ADD COLUMN blob BLOB")
        if "origin" not in have:
            conn.execute(
                "ALTER TABLE corpus_documents ADD COLUMN origin TEXT NOT NULL DEFAULT 'uploaded'"
            )

        # ── 结构版本 2：逻辑文档与版本 ──
        #
        # 依次建身份表、加列、回填、建唯一索引；旧数据回填前不能先加唯一约束。
        conn.executescript(_META_SCHEMA)
        if not conn.execute("SELECT 1 FROM corpus_meta").fetchone():
            conn.execute(
                "INSERT INTO corpus_meta (corpus_id, schema_version, version, created_at) "
                "VALUES (?, ?, 0, ?)",
                (str(uuid.uuid4()), _SCHEMA_VERSION, datetime.now(UTC).isoformat()),
            )
        if "doc_key" not in have:
            conn.execute("ALTER TABLE corpus_documents ADD COLUMN doc_key TEXT")
            conn.execute("ALTER TABLE corpus_documents ADD COLUMN version INTEGER")
            conn.execute("ALTER TABLE corpus_documents ADD COLUMN superseded_at TEXT")

        # 旧行各自成为独立逻辑文档的第 1 版。
        #
        # 不能按同名文件推断版本关系：同名可能是不同论文。
        # 保留原历史，不编造替代关系；只有用户明确 supersede 才建立版本联系。
        for (row_id,) in conn.execute(
            "SELECT id FROM corpus_documents WHERE doc_key IS NULL ORDER BY ingested_at, id"
        ).fetchall():
            conn.execute(
                "UPDATE corpus_documents SET doc_key = ?, version = 1 WHERE id = ?",
                (str(uuid.uuid4()), row_id),
            )
        conn.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_corpus_doc_version "
            "ON corpus_documents (doc_key, version)"
        )

        # ── 结构版本 3：记录工作表、行、幻灯片、单元格等区段及文本模式。
        # 新列可空，旧数据按无区段、原文模式读取；旧版本程序可以忽略。
        if "segments" not in have:
            conn.execute("ALTER TABLE corpus_documents ADD COLUMN segments TEXT")
        if "extraction_mode" not in have:
            conn.execute("ALTER TABLE corpus_documents ADD COLUMN extraction_mode TEXT")
        # 仅增加新表，不动旧程序每次打开时重建的两个唯一索引，见 RFC §7.1。
        conn.executescript(_FOLDER_SCHEMA)

    @property
    def embedder(self) -> Embeddings:
        return self._embedder

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=30.0)
        conn.execute("PRAGMA journal_mode=WAL")
        # SQLite 默认关闭外键，必须开启才能级联删除 corpus_chunks。
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    # ── 文档导入 ──

    async def ingest(
        self,
        filename: str,
        data: bytes,
        *,
        origin: str = "uploaded",
        doc_key: str | None = None,
    ) -> Ingested:
        """提取、切块、嵌入并保存文档版本，对相同内容保持幂等。

        只有明确传入 doc_key 才表示替换已有逻辑文档；省略时创建新文档。
        文件名只是版本展示信息，不能据此推断同一文档。
        提取和去重在线程中运行，避免大 PDF 阻塞事件循环；异步嵌入留在调用方循环；写入再回线程。
        origin 区分人工上传和自动保存的生成报告。"""
        prepared = await asyncio.to_thread(self._prepare_sync, filename, data, doc_key)
        if isinstance(prepared, Ingested):  # 重复内容已跳过，无需再生成嵌入。
            return prepared
        doc_id, text, page_starts, kind, digest, chunks, extraction = prepared

        vectors = await self._embed_chunks([c.text for c in chunks])
        width = len(vectors[0])
        if any(len(v) != width for v in vectors):
            raise RuntimeError("Embedder returned vectors of inconsistent width.")

        doc_key, version = await asyncio.to_thread(
            self._write_sync,
            doc_id,
            filename,
            text,
            page_starts,
            kind,
            digest,
            chunks,
            vectors,
            data,
            origin,
            doc_key,
            extraction.segments,
            extraction.mode,
        )
        logger.info(
            "corpus_ingested",
            filename=filename,
            doc_id=doc_id,
            chunks=len(chunks),
            model=self._embedder.model_id,
        )
        return Ingested(
            doc_id=doc_id,
            filename=filename,
            chunks_written=len(chunks),
            doc_key=doc_key,
            version=version,
        )

    def _prepare_sync(
        self, filename: str, data: bytes, doc_key: str | None = None
    ) -> Ingested | tuple:
        extraction = extract(filename, data)
        text, page_starts, kind = extraction.text, extraction.page_starts, extraction.kind
        digest = hashlib.sha256(data).hexdigest()

        with self._connect() as conn:
            if doc_key is not None:
                # 先确认被替换文档存在，再花费嵌入成本；未知 doc_key 不能当作新文档。
                if not conn.execute(
                    "SELECT 1 FROM corpus_documents WHERE doc_key = ?", (doc_key,)
                ).fetchone():
                    raise UnknownDocumentKey(doc_key)
                # 内容身份由文档范围内的 SHA 决定；当前版本字节相同就不是新版本，与此次文件名无关。
                existing = conn.execute(
                    "SELECT id FROM corpus_documents "
                    "WHERE doc_key = ? AND sha256 = ? AND superseded_at IS NULL",
                    (doc_key, digest),
                ).fetchone()
            else:
                existing = conn.execute(
                    "SELECT id FROM corpus_documents WHERE filename = ? AND sha256 = ?",
                    (filename, digest),
                ).fetchone()
        if existing:
            # 相同字节不重复导入，避免增加无信息的版本、块和嵌入费用。
            return Ingested(
                doc_id=existing[0],
                filename=filename,
                skipped=True,
                reason="identical content already ingested",
            )

        chunks = chunk_document(text, page_starts=page_starts)
        if not chunks:
            return Ingested(filename=filename, skipped=True, reason="no chunks produced")
        return str(uuid.uuid4()), text, page_starts, kind, digest, chunks, extraction

    def _write_sync(
        self,
        doc_id: str,
        filename: str,
        text: str,
        page_starts: list[int],
        kind: str,
        digest: str,
        chunks: list,
        vectors: list[list[float]],
        blob: bytes | None = None,
        origin: str = "uploaded",
        doc_key: str | None = None,
        segments: list | None = None,
        extraction_mode: str | None = None,
    ) -> tuple[str, int]:
        # 版本分配、旧版替代、块写入和资料库计数在同一事务完成，
        # 避免读取者看到两个当前版本或与检索内容不符的计数。
        with self._connect() as conn:
            now = datetime.now(UTC).isoformat()
            key = doc_key or str(uuid.uuid4())
            if doc_key is not None:
                # 每文档至多一个当前版本；新版本存在时才在同一事务将旧版标为已替代。
                conn.execute(
                    "UPDATE corpus_documents SET superseded_at = ? "
                    "WHERE doc_key = ? AND superseded_at IS NULL",
                    (now, doc_key),
                )
            # 在 INSERT 内分配版本号，避免并发先读 MAX 再写得到相同编号；唯一索引作为最终保护。
            conn.execute(
                "INSERT INTO corpus_documents "
                "(id, filename, kind, sha256, text, page_starts, chunk_count, ingested_at, "
                "blob, origin, doc_key, segments, extraction_mode, version) "
                "SELECT ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, "
                "       COALESCE(MAX(version), 0) + 1 FROM corpus_documents WHERE doc_key = ?",
                (
                    doc_id,
                    filename,
                    kind,
                    digest,
                    text,
                    json.dumps(page_starts),
                    len(chunks),
                    now,
                    blob,
                    origin,
                    key,
                    json.dumps([list(seg) for seg in segments]) if segments else None,
                    extraction_mode,
                    key,
                ),
            )
            conn.executemany(
                "INSERT INTO corpus_chunks "
                '(document_id, chunk_index, start, "end", page, text, embedding, '
                "embedding_model) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    (
                        doc_id,
                        index,
                        chunk.start,
                        chunk.end,
                        chunk.page,
                        chunk.text,
                        _pack(vector),
                        self._embedder.model_id,
                    )
                    for index, (chunk, vector) in enumerate(zip(chunks, vectors, strict=True))
                ],
            )
            conn.execute("UPDATE corpus_meta SET version = version + 1")
            # 读取实际写入的版本号，不再自行计算，保持身份只有一个事实来源。
            version = conn.execute(
                "SELECT version FROM corpus_documents WHERE id = ?", (doc_id,)
            ).fetchone()[0]
        return (key, int(version))

    async def _embed_chunks(self, texts: list[str]) -> list[list[float]]:
        """分批生成嵌入；向量宽度或数量不符时让导入失败。"""
        vectors: list[list[float]] = []
        for start in range(0, len(texts), _EMBED_BATCH):
            batch = await self._embedder.embed(texts[start : start + _EMBED_BATCH])
            if len(batch) != len(texts[start : start + _EMBED_BATCH]):
                raise RuntimeError(
                    f"Embedder returned {len(batch)} vectors for "
                    f"{len(texts[start : start + _EMBED_BATCH])} chunks."
                )
            vectors.extend(batch)
        return vectors

    # ── 检索：Corpus 端口实现 ──

    async def search(
        self,
        query: str,
        max_results: int,
        *,
        document_ids: frozenset[str] | None = None,
        per_document: int = 1,
    ) -> list[dict]:
        # 查询嵌入是检索唯一的模型调用。仅资料库模式承诺本地处理时，
        # 必须检查嵌入器是否本地，不能依赖宿主配置或被 FakeEmbeddings 替换的测试来证明没有外发。
        from research_engine.eval_trace import trace_span

        with trace_span(
            "corpus.search",
            kind="retrieval",
            inputs={
                "query": query,
                "max_results": max_results,
                "document_ids": sorted(document_ids) if document_ids else None,
                "per_document": per_document,
            },
        ) as span:
            self._require_local_embedder_in_corpus_mode()
            query_vector = (await self._embedder.embed([query]))[0]
            result = await asyncio.to_thread(
                self._search_sync, query_vector, max_results, document_ids, per_document
            )
            span.observe(output=result)
            return result

    def _require_local_embedder_in_corpus_mode(self) -> None:
        """在承诺本地检索的仅资料库模式中拒绝远程嵌入，见 internal/12 M10。

        普通项目记忆允许使用托管嵌入，不受此限制。"""
        if not get_run_config().corpus_mode:
            return
        if getattr(self._embedder, "is_local", False):
            return
        raise EmbeddingsUnavailable(
            f"Corpus-only mode keeps retrieval on this machine, but the configured "
            f"embeddings provider '{self._embedder.model_id}' is remote — every corpus "
            f"search would send the query off this machine. Set EMBEDDINGS_PROVIDER=ollama "
            f"with a local OLLAMA_BASE_URL, or run without corpus-only mode."
        )

    def _search_sync(
        self,
        query_vector: list[float],
        max_results: int,
        document_ids: frozenset[str] | None = None,
        per_document: int = 1,
    ) -> list[dict]:
        import numpy as np

        if document_ids is not None:
            return self._rank(
                query_vector, max_results, self._selected_rows(document_ids), np, per_document
            )

        with self._connect() as conn:
            rows = conn.execute(
                'SELECT c.id, c.document_id, c.start, c."end", c.page, c.text, c.embedding, '
                "d.filename "
                "FROM corpus_chunks c JOIN corpus_documents d ON d.id = c.document_id "
                # 始终排除自动生成报告，避免下一轮把本项目的输出当作独立证据。
                # 已替代版本不参与新检索，但保留 read 能力，使历史引用继续可定位；排序和相关性阈值不变。
                "WHERE c.embedding_model = ? AND d.origin != 'generated' "
                "AND d.superseded_at IS NULL",
                (self._embedder.model_id,),
            ).fetchall()

        if not rows:
            # 分别诊断空库、嵌入模型不匹配、仅含生成报告或仅含已替代文档。
            # 诊断查询沿用主查询的其他过滤条件，仅放开正在检查的条件，避免给出自相矛盾的原因。
            with self._connect() as conn:
                superseded = conn.execute(
                    "SELECT COUNT(*) FROM corpus_chunks c "
                    "JOIN corpus_documents d ON d.id = c.document_id "
                    "WHERE c.embedding_model = ? AND d.origin != 'generated' "
                    "AND d.superseded_at IS NOT NULL",
                    (self._embedder.model_id,),
                ).fetchone()[0]
            if superseded:
                raise RuntimeError(
                    "Every document in this corpus has been superseded and none has a "
                    "current version. Retrieval only reads current versions — ingest a "
                    "document, or the run would cite content that was replaced."
                )
            with self._connect() as conn:
                other = conn.execute(
                    "SELECT c.embedding_model, COUNT(*) FROM corpus_chunks c "
                    "JOIN corpus_documents d ON d.id = c.document_id "
                    "WHERE d.origin != 'generated' AND d.superseded_at IS NULL GROUP BY 1"
                ).fetchall()
            if other:
                models = ", ".join(f"{m} ({n} chunks)" for m, n in other)
                raise RuntimeError(
                    f"Corpus was indexed with a different embedding model ({models}); "
                    f"current model is '{self._embedder.model_id}'. Re-ingest to re-index."
                )
            with self._connect() as conn:
                generated_only = conn.execute(
                    "SELECT COUNT(*) FROM corpus_documents WHERE origin = 'generated'"
                ).fetchone()[0]
            if generated_only:
                raise RuntimeError(
                    "The corpus has only auto-saved reports and no uploaded documents. "
                    "Generated reports are never used as evidence — upload a source "
                    "before running corpus-only research."
                )
            raise RuntimeError(
                "The corpus is empty. Ingest documents before running corpus-only research."
            )
        return self._rank(query_vector, max_results, rows, np)

    def _selected_rows(self, document_ids: frozenset[str]) -> list[tuple]:
        """读取指定文档版本的块，或返回明确失败原因。

        选择在运行开始时冻结版本，不能因为途中出现新版而排除旧选中版；生成报告始终不可作证据。
        ID 通过 json_each 单参数传递，避免大选择触发 SQLite 参数数量上限。
        诊断也只基于该选择，不拿用户无权读取的全库状态解释失败。"""
        ids = json.dumps(sorted(document_ids))
        with self._connect() as conn:
            rows = conn.execute(
                'SELECT c.id, c.document_id, c.start, c."end", c.page, c.text, c.embedding, '
                "d.filename "
                "FROM corpus_chunks c JOIN corpus_documents d ON d.id = c.document_id "
                "WHERE c.embedding_model = ? AND d.origin != 'generated' "
                "AND d.id IN (SELECT value FROM json_each(?))",
                (self._embedder.model_id, ids),
            ).fetchall()
            if rows:
                return rows
            present = conn.execute(
                "SELECT COUNT(*) FROM corpus_documents "
                "WHERE origin != 'generated' AND id IN (SELECT value FROM json_each(?))",
                (ids,),
            ).fetchone()[0]
            other = conn.execute(
                "SELECT c.embedding_model, COUNT(*) FROM corpus_chunks c "
                "JOIN corpus_documents d ON d.id = c.document_id "
                "WHERE d.origin != 'generated' AND d.id IN (SELECT value FROM json_each(?)) "
                "GROUP BY 1",
                (ids,),
            ).fetchall()
        if not present:
            raise RuntimeError(
                f"None of the {len(document_ids)} selected documents is in this corpus — "
                "they were deleted, or they are auto-saved reports, which are never evidence."
            )
        if other:
            models = ", ".join(f"{m} ({n} chunks)" for m, n in other)
            raise RuntimeError(
                f"The selected documents were indexed with a different embedding model "
                f"({models}); current model is '{self._embedder.model_id}'. Re-ingest them "
                "to re-index."
            )
        raise RuntimeError(f"The {present} selected document(s) produced no searchable text.")

    def _rank(
        self, query_vector: list[float], max_results: int, rows: list, np, per_document: int = 1
    ) -> list[dict]:
        matrix = np.stack([np.frombuffer(row[6], dtype=np.float32) for row in rows])
        if matrix.shape[1] != len(query_vector):
            raise RuntimeError(
                f"Stored vectors are {matrix.shape[1]}-wide but the query embedded to "
                f"{len(query_vector)} dimensions — the embedding model changed."
            )

        # 按余弦相似度比较；正交或更差的结果没有证据价值，距离阈值与记忆服务对应。
        q = np.asarray(query_vector, dtype=np.float32)
        q_norm = q / (np.linalg.norm(q) or 1.0)
        norms = np.linalg.norm(matrix, axis=1)
        norms[norms == 0] = 1.0
        scores = (matrix / norms[:, None]) @ q_norm

        ranked = np.argsort(-scores)[: max_results * 4]
        results: list[dict] = []
        hits_per_doc: dict[str, int] = {}
        for idx in ranked:
            if scores[idx] <= 0.0:
                break
            row = rows[int(idx)]
            doc_id = row[1]
            # 默认每文件每次搜索只取一块，避免把同文件多块当成独立来源；
            # 小范围选择可提高上限，否则单文件选择永远只能读到一段，见 RFC V3.1 §9.8。
            if hits_per_doc.get(doc_id, 0) >= per_document:
                continue
            hits_per_doc[doc_id] = hits_per_doc.get(doc_id, 0) + 1
            results.append(
                {
                    "title": row[7],
                    "url": corpus_url(doc_id, start=row[2], end=row[3], page=row[4]),
                    "snippet": row[5],
                }
            )
            if len(results) >= max_results:
                break
        return results

    async def search_generated_reports(self, query: str, max_results: int) -> list[dict]:
        """检索项目自动保存的生成报告，仅用于聊天依据。"""
        query_vector = (await self._embedder.embed([query]))[0]
        return await asyncio.to_thread(self._search_generated_sync, query_vector, max_results)

    def _search_generated_sync(self, query_vector: list[float], max_results: int) -> list[dict]:
        import numpy as np

        with self._connect() as conn:
            rows = conn.execute(
                'SELECT c.id, c.document_id, c.start, c."end", c.page, c.text, c.embedding, '
                "d.filename "
                "FROM corpus_chunks c JOIN corpus_documents d ON d.id = c.document_id "
                "WHERE c.embedding_model = ? AND d.origin = 'generated'",
                (self._embedder.model_id,),
            ).fetchall()

        if not rows:
            return []

        matrix = np.stack([np.frombuffer(row[6], dtype=np.float32) for row in rows])
        if matrix.shape[1] != len(query_vector):
            return []

        q = np.asarray(query_vector, dtype=np.float32)
        q_norm = q / (np.linalg.norm(q) or 1.0)
        norms = np.linalg.norm(matrix, axis=1)
        norms[norms == 0] = 1.0
        scores = (matrix / norms[:, None]) @ q_norm

        ranked = np.argsort(-scores)[: max_results * 4]
        results: list[dict] = []
        for idx in ranked:
            if scores[idx] <= 0.0:
                break
            row = rows[int(idx)]
            doc_id = row[1]
            results.append(
                {
                    "title": row[7],
                    "url": corpus_url(doc_id, start=row[2], end=row[3], page=row[4]),
                    "snippet": row[5],
                }
            )
            if len(results) >= max_results:
                break
        return results

    async def read(self, url: str) -> dict:
        return await asyncio.to_thread(self._read_sync, url)

    def _read_sync(self, url: str) -> dict:
        location = parse_corpus_url(url)
        if location is None:
            return {"url": url, "title": "", "text": "", "error": "not a valid corpus URL"}

        with self._connect() as conn:
            row = conn.execute(
                "SELECT filename, text, segments, kind FROM corpus_documents WHERE id = ?",
                (location.doc_id,),
            ).fetchone()
        if row is None:
            return {"url": url, "title": "", "text": "", "error": "document not found in corpus"}

        filename, text, segments, kind = row
        if location.start >= len(text) or location.end > len(text):
            # 引用区间已不适合该文档时拒绝读取，不能移位返回看似合理却引用错位置的文本。
            return {"url": url, "title": filename, "text": "", "error": "location out of range"}

        # 沿完整词边界扩展上下文。
        lo = max(location.start - _READ_CONTEXT_CHARS, 0)
        hi = min(location.end + _READ_CONTEXT_CHARS, len(text))
        while lo > 0 and not text[lo - 1].isspace():
            lo -= 1
        while hi < len(text) and not text[hi].isspace():
            hi += 1
        window = text[lo:hi]
        if len(window) > _READ_MAX_CHARS:
            window = window[:_READ_MAX_CHARS]

        label = segment_label(
            json.loads(segments) if segments else [], location.start, location.end
        )
        if not label and kind == "code":
            # 源码文件的行号由文本现场计算，无需逐行存储区段。
            first = text.count("\n", 0, location.start) + 1
            last = text.count("\n", 0, max(location.start, location.end - 1)) + 1
            label = f"line {first}" if first == last else f"lines {first}–{last}"
        where = (
            f"page {location.page}, chars {location.start}-{location.end}"
            if location.page is not None
            else f"{label}, chars {location.start}-{location.end}"
            if label
            else f"chars {location.start}-{location.end}"
        )
        return {
            "url": url,
            "title": f"{filename} ({where})",
            "text": window,
            "error": None,
        }

    # ── 资料库管理 ──

    async def identity(self) -> tuple[str, int]:
        """返回 corpus_id 和 corpus_version，表示资料库身份及当前状态。

        两个宿主共用此来源；运行打开资料库时记录该组合，以便追溯证据来自哪个资料库状态。"""
        return await asyncio.to_thread(self._identity_sync)

    def _identity_sync(self) -> tuple[str, int]:
        with self._connect() as conn:
            row = conn.execute("SELECT corpus_id, version FROM corpus_meta").fetchone()
        return (row[0], int(row[1]))

    async def documents(self) -> list[dict]:
        return await asyncio.to_thread(self._documents_sync)

    async def current_versions(self, doc_keys: list[str] | None = None) -> list[dict]:
        """返回指定逻辑文档的当前非生成版本。

        None 表示全库；没有当前上传版本的键不出现在结果中，由调用方决定是否报错。
        结构与冻结选择记录一致，见 RFC V3.1 §8.2。"""
        return await asyncio.to_thread(self._current_versions_sync, doc_keys)

    def _current_versions_sync(self, doc_keys: list[str] | None) -> list[dict]:
        query = (
            "SELECT id, doc_key, version, sha256, filename FROM corpus_documents "
            "WHERE origin != 'generated' AND superseded_at IS NULL"
        )
        params: tuple = ()
        if doc_keys is not None:
            query += " AND doc_key IN (SELECT value FROM json_each(?))"
            params = (json.dumps(sorted(set(doc_keys))),)
        with self._connect() as conn:
            rows = conn.execute(query + " ORDER BY filename, id", params).fetchall()
        return [
            {
                "document_id": r[0],
                "doc_key": r[1],
                "version": r[2],
                "sha256": r[3],
                "filename": r[4],
                "path": r[4],
            }
            for r in rows
        ]

    def _documents_sync(self) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT id, filename, kind, chunk_count, ingested_at, "
                "       doc_key, version, superseded_at, "
                # BLOB 的 length() 只读大小，不加载大文件。downloadable 由实际原件是否存在派生，
                # 旧版未保存原件的行不能显示可打开。
                "       length(blob), origin FROM corpus_documents "
                "ORDER BY ingested_at"
            ).fetchall()
        return [
            {
                "id": row[0],
                "filename": row[1],
                "kind": row[2],
                "chunk_count": row[3],
                "ingested_at": row[4],
                "doc_key": row[5],
                "version": row[6],
                # 是否当前可检索直接由 superseded_at 推导，不再存一个可能不一致的冗余状态。
                "is_current": row[7] is None,
                "size_bytes": row[8],
                "downloadable": row[8] is not None,
                "origin": row[9],
            }
            for row in rows
        ]

    async def blob(self, doc_id: str) -> tuple[bytes, str, str] | None:
        """返回原上传文件的字节、名称和类型；不存在或旧数据未保留原件时返回 None。

        不区分这两种缺失，避免泄露某个 ID 是否存在。"""
        return await asyncio.to_thread(self._blob_sync, doc_id)

    def _blob_sync(self, doc_id: str) -> tuple[bytes, str, str] | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT blob, filename, kind FROM corpus_documents WHERE id = ?", (doc_id,)
            ).fetchone()
        if row is None or row[0] is None:
            return None
        return bytes(row[0]), row[1], row[2]

    async def delete(self, doc_id: str) -> bool:
        return await asyncio.to_thread(self._delete_sync, doc_id)

    def _delete_sync(self, doc_id: str) -> bool:
        """仅删除一个版本，不代表删除整个逻辑文档。

        不重编号后续版本，不自动恢复前版，也不级联删除其他版本。
        版本号可能被历史引用使用；自动恢复旧版会重新发布用户已经移除的内容。"""
        with self._connect() as conn:
            cursor = conn.execute("DELETE FROM corpus_documents WHERE id = ?", (doc_id,))
            if cursor.rowcount:
                conn.execute("UPDATE corpus_meta SET version = version + 1")
        return cursor.rowcount > 0

    async def extracted(self, doc_id: str) -> dict | None:
        """返回版本的标准文本、区段和模式，即引用实际核对的内容。"""
        return await asyncio.to_thread(self._extracted_sync, doc_id)

    def _extracted_sync(self, doc_id: str) -> dict | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT filename, kind, text, segments, extraction_mode FROM corpus_documents "
                "WHERE id = ?",
                (doc_id,),
            ).fetchone()
        if row is None:
            return None
        return {
            "filename": row[0],
            "kind": row[1],
            "text": row[2],
            "segments": json.loads(row[3]) if row[3] else [],
            "extraction_mode": row[4] or "verbatim",
        }

    # ── 文件夹，见 RFC V3.1 §7.1 ──

    async def folders(self) -> list[dict]:
        return await asyncio.to_thread(self._folders_sync)

    def _folders_sync(self) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT id, parent_id, name FROM corpus_folders ORDER BY name, id"
            ).fetchall()
        return [{"id": r[0], "parent_id": r[1], "name": r[2]} for r in rows]

    async def memberships(self) -> list[dict]:
        return await asyncio.to_thread(self._memberships_sync)

    def _memberships_sync(self) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT doc_key, folder_id, display_name FROM corpus_memberships ORDER BY id"
            ).fetchall()
        return [{"doc_key": r[0], "folder_id": r[1], "display_name": r[2]} for r in rows]

    async def create_folder(self, name: str, parent_id: str | None = None) -> str:
        return await asyncio.to_thread(self._create_folder_sync, name, parent_id)

    def _create_folder_sync(self, name: str, parent_id: str | None) -> str:
        name = _folder_name(name)
        with self._connect() as conn:
            if (
                parent_id is not None
                and not conn.execute(
                    "SELECT 1 FROM corpus_folders WHERE id = ?", (parent_id,)
                ).fetchone()
            ):
                raise FolderNotFound(parent_id)
            existing = conn.execute(
                "SELECT id FROM corpus_folders WHERE name = ? AND parent_id IS ?",
                (name, parent_id),
            ).fetchone()
            if existing:
                return existing[0]
            folder_id = str(uuid.uuid4())
            conn.execute(
                "INSERT INTO corpus_folders (id, parent_id, name, created_at) VALUES (?, ?, ?, ?)",
                (folder_id, parent_id, name, datetime.now(UTC).isoformat()),
            )
            conn.execute("UPDATE corpus_meta SET version = version + 1")
        return folder_id

    async def ensure_folder_path(self, components: list[str]) -> str | None:
        """返回 components 指定的文件夹，缺失则逐级创建；根目录返回 None。"""
        parent: str | None = None
        for component in components:
            parent = await self.create_folder(component, parent)
        return parent

    async def update_folder(
        self,
        folder_id: str,
        *,
        name: str | None = None,
        parent_id: str | None = ...,
    ) -> None:
        await asyncio.to_thread(self._update_folder_sync, folder_id, name, parent_id)

    def _update_folder_sync(self, folder_id: str, name, parent_id) -> None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT parent_id, name FROM corpus_folders WHERE id = ?", (folder_id,)
            ).fetchone()
            if row is None:
                raise FolderNotFound(folder_id)
            new_parent = row[0] if parent_id is ... else parent_id
            new_name = row[1] if name is None else _folder_name(name)
            if new_parent is not None:
                if not conn.execute(
                    "SELECT 1 FROM corpus_folders WHERE id = ?", (new_parent,)
                ).fetchone():
                    raise FolderNotFound(new_parent)
                if new_parent in self._descendants(conn, [folder_id]):
                    raise ValueError(
                        "A folder cannot be moved into itself or one of its subfolders."
                    )
            clash = conn.execute(
                "SELECT 1 FROM corpus_folders WHERE name = ? AND parent_id IS ? AND id != ?",
                (new_name, new_parent, folder_id),
            ).fetchone()
            if clash:
                raise ValueError(f"A folder named {new_name!r} already exists there.")
            conn.execute(
                "UPDATE corpus_folders SET name = ?, parent_id = ? WHERE id = ?",
                (new_name, new_parent, folder_id),
            )
            conn.execute("UPDATE corpus_meta SET version = version + 1")

    @staticmethod
    def _descendants(conn, folder_ids: list[str]) -> set[str]:
        """返回指定文件夹及全部后代文件夹。"""
        if not folder_ids:
            return set()
        rows = conn.execute(
            "WITH RECURSIVE sub(id) AS ("
            "  SELECT value FROM json_each(?) "
            "  UNION SELECT f.id FROM corpus_folders f JOIN sub ON f.parent_id = sub.id"
            ") SELECT id FROM sub",
            (json.dumps(folder_ids),),
        ).fetchall()
        return {r[0] for r in rows}

    async def delete_folder(self, folder_id: str) -> dict:
        return await asyncio.to_thread(self._delete_folder_sync, folder_id)

    def _delete_folder_sync(self, folder_id: str) -> dict:
        """在同一事务中移除文件夹、子文件夹及成员关系。

        仅属于这些文件夹、移除后不再归属任何文件夹的文档连同所有版本删除。
        仍属其他文件夹或原本在根目录的文档不受影响；历史运行保留已存证据，再打开已删文档时报不可用。"""
        with self._connect() as conn:
            if not conn.execute(
                "SELECT 1 FROM corpus_folders WHERE id = ?", (folder_id,)
            ).fetchone():
                raise FolderNotFound(folder_id)
            doomed = self._descendants(conn, [folder_id])
            ids = json.dumps(sorted(doomed))
            keys = {
                r[0]
                for r in conn.execute(
                    "SELECT DISTINCT doc_key FROM corpus_memberships "
                    "WHERE folder_id IN (SELECT value FROM json_each(?))",
                    (ids,),
                )
            }
            conn.execute(
                "DELETE FROM corpus_memberships WHERE folder_id IN (SELECT value FROM json_each(?))",
                (ids,),
            )
            conn.execute(
                "DELETE FROM corpus_folders WHERE id IN (SELECT value FROM json_each(?))", (ids,)
            )
            orphaned = [
                key
                for key in keys
                if not conn.execute(
                    "SELECT 1 FROM corpus_memberships WHERE doc_key = ?", (key,)
                ).fetchone()
            ]
            removed = 0
            for key in orphaned:
                removed += conn.execute(
                    "DELETE FROM corpus_documents WHERE doc_key = ?", (key,)
                ).rowcount
            conn.execute("UPDATE corpus_meta SET version = version + 1")
        return {
            "folders_deleted": len(doomed),
            "documents_deleted": len(orphaned),
            "versions_deleted": removed,
        }

    async def add_membership(self, doc_key: str, folder_id: str, display_name: str) -> None:
        await asyncio.to_thread(self._add_membership_sync, doc_key, folder_id, display_name)

    def _add_membership_sync(self, doc_key: str, folder_id: str, display_name: str) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO corpus_memberships "
                "(doc_key, folder_id, display_name, added_at) VALUES (?, ?, ?, ?)",
                (doc_key, folder_id, display_name, datetime.now(UTC).isoformat()),
            )

    async def doc_key_of(self, doc_id: str) -> str | None:
        return await asyncio.to_thread(self._doc_key_of_sync, doc_id)

    def _doc_key_of_sync(self, doc_id: str) -> str | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT doc_key FROM corpus_documents WHERE id = ?", (doc_id,)
            ).fetchone()
        return row[0] if row else None

    async def sha256_of(self, doc_id: str) -> str | None:
        """返回指定版本存储的哈希；不存在时返回 None。"""
        return await asyncio.to_thread(self._sha256_of_sync, doc_id)

    def _sha256_of_sync(self, doc_id: str) -> str | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT sha256 FROM corpus_documents WHERE id = ?", (doc_id,)
            ).fetchone()
        return row[0] if row else None

    async def folder_doc_keys(self, folder_ids: list[str]) -> set[str] | None:
        """返回这些文件夹及后代包含的逻辑文档；任一文件夹未知时返回 None。"""
        return await asyncio.to_thread(self._folder_doc_keys_sync, folder_ids)

    def _folder_doc_keys_sync(self, folder_ids: list[str]) -> set[str] | None:
        with self._connect() as conn:
            known = {
                r[0]
                for r in conn.execute(
                    "SELECT id FROM corpus_folders WHERE id IN (SELECT value FROM json_each(?))",
                    (json.dumps(folder_ids),),
                )
            }
            if known != set(folder_ids):
                return None
            every = self._descendants(conn, folder_ids)
            return {
                r[0]
                for r in conn.execute(
                    "SELECT DISTINCT doc_key FROM corpus_memberships "
                    "WHERE folder_id IN (SELECT value FROM json_each(?))",
                    (json.dumps(sorted(every)),),
                )
            }

    async def status(self) -> dict:
        return await asyncio.to_thread(self._status_sync)

    def _status_sync(self) -> dict:
        with self._connect() as conn:
            meta = conn.execute("SELECT corpus_id, version FROM corpus_meta").fetchone()
            docs = conn.execute("SELECT COUNT(*) FROM corpus_documents").fetchone()[0]
            by_model = conn.execute(
                "SELECT embedding_model, COUNT(*) FROM corpus_chunks GROUP BY 1 ORDER BY 1"
            ).fetchall()
        return {
            "documents": docs,
            "chunks_by_model": {model: count for model, count in by_model},
            "current_model": self._embedder.model_id,
            "corpus_id": meta[0],
            "corpus_version": int(meta[1]),
        }


def _pack(vector: list[float]) -> bytes:
    import numpy as np

    return np.asarray(vector, dtype=np.float32).tobytes()
