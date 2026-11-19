"""项目记忆：写入获批报告，再按项目读取知识。

摄取必须检查 COMPLETED，不能让草稿或拒绝报告进入检索。来源 ID 同时支持会话和研究运行，不能只外键关联旧会话表。
隔离必须在 SQL 查询中过滤 project_id，再把文字交给模型；提示词限制不能替代安全边界（docs/06 §4）。"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime

import structlog
from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.memory_chunk import MemoryChunk
from app.models.research import ResearchRun
from app.models.session import Session, SessionStatus
from research_engine.chunking import chunk_report
from research_engine.embeddings import EmbeddingsUnavailable
from research_engine.ports import Embeddings

logger = structlog.get_logger()

# 每轮聊天取多少记忆片段，也是上下文费用开关；8 段约 300 Token 的片段约占 2400 Token（docs/14 §5）。
DEFAULT_TOP_K = 8

# 余弦距离上限仅剔除明显不相关的片段；归一化向量中不相关接近 1，不能凭感觉设过严阈值。
#
# 此前未经真实向量基准验证就收紧到 0.45，反而把确有答案的片段过滤掉。
# 调整检索阈值必须有基准说明收益和损失，不能让猜测变成“无项目记忆”的假测量。
MAX_COSINE_DISTANCE = 1.0


def is_available(db: AsyncSession) -> bool:
    """统一判断主机是否支持项目记忆。

    目前依赖 pgvector，桌面没有此能力是产品差异，不是服务故障；调用方不能各自复制数据库方言判断。"""
    return db.bind is not None and db.bind.dialect.name == "postgresql"


async def purge_report(db: AsyncSession, report_id: uuid.UUID) -> int:
    """显式删除某报告的全部记忆片段；无记忆能力时不操作。

    桌面没有 memory_chunks 表，不能通过无条件 ORM 级联去查询它。必须先检查能力再清理，同时保证删除研究后不会残留可被聊天引用的正文。"""
    if not is_available(db):
        return 0
    result = await db.execute(delete(MemoryChunk).where(MemoryChunk.source_report_id == report_id))
    return result.rowcount or 0


@dataclass(frozen=True)
class IngestResult:
    """一次摄取的结果，skipped 表示成功跳过，不是失败。"""

    chunks_written: int = 0
    skipped: bool = False
    reason: str | None = None


@dataclass(frozen=True)
class SourceReport:
    """足够定位引用报告的简要信息，不直接暴露 ORM 行。

    报告可来自会话或研究运行，统一结构避免调用方错误依赖其中一种表的字段。"""

    id: uuid.UUID
    title: str
    created_at: datetime
    # 来源类型为 run 或 session；引用需要据此跳到正确路由，不能生成打不开的标记。
    kind: str


@dataclass(frozen=True)
class Retrieved:
    """检索片段及其来源报告，可直接用于 R 编号引用。"""

    chunk: MemoryChunk
    distance: float
    report: SourceReport


async def ingest_report(
    db: AsyncSession,
    *,
    project_id: uuid.UUID,
    report_id: uuid.UUID,
    report_markdown: str,
    embedder: Embeddings,
    force: bool = False,
) -> IngestResult:
    """将一份已批准报告切块并写入项目记忆。

    批准由会话和运行各入口先检查，本函数共用切块、批量向量化及替换规则。当前模型下已索引就跳过，避免重复批准产生重复数据和费用；force=True 用于模型切换后的重建。
    无向量服务时抛 EmbeddingsUnavailable，由调用方决定是否致命；已完成研究只披露索引缺口，不能因此改为失败。"""
    report = (report_markdown or "").strip()
    if not report:
        return IngestResult(skipped=True, reason="no final report text")

    if not force:
        existing = (
            await db.execute(
                select(func.count())
                .select_from(MemoryChunk)
                .where(
                    MemoryChunk.source_report_id == report_id,
                    MemoryChunk.embedding_model == embedder.model_id,
                )
            )
        ).scalar_one()
        if existing:
            return IngestResult(skipped=True, reason="already indexed with this model")

    chunks = chunk_report(report)
    if not chunks:
        return IngestResult(skipped=True, reason="report produced no chunks")

    # 整份报告批量调用一次，不按片段逐个请求。
    vectors = await embedder.embed(chunks)
    if len(vectors) != len(chunks):
        raise EmbeddingsUnavailable(
            f"Embedder returned {len(vectors)} vectors for {len(chunks)} chunks."
        )

    # 重建时整体替换旧片段，避免新旧切块数量不同而残留仍可被检索的旧尾部。
    await db.execute(delete(MemoryChunk).where(MemoryChunk.source_report_id == report_id))
    db.add_all(
        [
            MemoryChunk(
                project_id=project_id,
                source_report_id=report_id,
                chunk_index=index,
                text=text,
                embedding=vector,
                embedding_model=embedder.model_id,
            )
            for index, (text, vector) in enumerate(zip(chunks, vectors, strict=True))
        ]
    )
    await db.commit()
    logger.info(
        "memory_ingested",
        project_id=str(project_id),
        report_id=str(report_id),
        chunks=len(chunks),
        model=embedder.model_id,
    )
    return IngestResult(chunks_written=len(chunks))


async def ingest_session(
    db: AsyncSession,
    session: Session,
    embedder: Embeddings,
    *,
    force: bool = False,
) -> IngestResult:
    """先检查会话报告批准状态，再执行共用摄取。"""
    if session.status != SessionStatus.COMPLETED:
        # 核心批准门禁，后续步骤仅负责执行机制。
        return IngestResult(skipped=True, reason=f"status is {session.status}, not COMPLETED")
    return await ingest_report(
        db,
        project_id=session.project_id,
        report_id=session.id,
        report_markdown=session.final_report or "",
        embedder=embedder,
        force=force,
    )


async def ingest_run(
    db: AsyncSession,
    run: ResearchRun,
    report_markdown: str,
    embedder: Embeddings,
    *,
    force: bool = False,
) -> IngestResult:
    """先检查研究运行批准状态，再执行共用摄取。

    正文由调用方传入已批准修订，不能从 run 再查最新版本，否则可能索引到用户没审核过的另一版。"""
    if run.status != "COMPLETED":
        return IngestResult(skipped=True, reason=f"status is {run.status}, not COMPLETED")
    return await ingest_report(
        db,
        project_id=run.project_id,
        report_id=run.id,
        report_markdown=report_markdown,
        embedder=embedder,
        force=force,
    )


async def _resolve_reports(db: AsyncSession, ids: list[uuid.UUID]) -> dict[uuid.UUID, SourceReport]:
    """分别从会话和运行表查询报告标题与日期。

    来源 ID 是多态，用两次小型 IN 查询；两表均无此报告则不返回，调用方丢弃其片段，不能引用不存在的报告。"""
    if not ids:
        return {}
    found: dict[uuid.UUID, SourceReport] = {}
    for row in (
        (await db.execute(select(ResearchRun).where(ResearchRun.id.in_(ids)))).scalars().all()
    ):
        found[row.id] = SourceReport(
            id=row.id, title=row.question, created_at=row.created_at, kind="run"
        )
    for row in (await db.execute(select(Session).where(Session.id.in_(ids)))).scalars().all():
        found.setdefault(
            row.id,
            SourceReport(id=row.id, title=row.prompt, created_at=row.created_at, kind="session"),
        )
    return found


async def retrieve(
    db: AsyncSession,
    *,
    index,
    project_id: uuid.UUID,
    query_vector: list[float],
    embedding_model: str,
    limit: int = DEFAULT_TOP_K,
) -> list[Retrieved]:
    """只在一个项目内检索最近片段，并附带来源报告。

    归属检查与 project_id 过滤构成隔离边界；同时按 embedding_model 过滤，禁止不可比模型的向量混排。"""
    # 数据库专有的向量距离查询交给 MemoryIndex 端口。
    # 项目与模型两个过滤条件都是正确性边界，不是可省略的优化。
    rows = await index.nearest(
        db,
        project_id=project_id,
        query_vector=query_vector,
        embedding_model=embedding_model,
        limit=limit,
    )

    reports = await _resolve_reports(db, [chunk.source_report_id for chunk, _ in rows])
    # 来源报告已删除的片段直接丢弃，不能让用户收到无法打开的引用。
    return [
        Retrieved(chunk=chunk, distance=float(dist), report=reports[chunk.source_report_id])
        for chunk, dist in rows
        if chunk.source_report_id in reports
    ]


@dataclass(frozen=True)
class MemoryStatus:
    """项目记忆状态（docs/14 §8）。

    pending_reports 由已批准报告减去已索引报告计算，不额外维护易失同步的状态；重建成功自然缩小缺口。"""

    chunk_count: int
    indexed_reports: int
    approved_reports: int
    pending_reports: int
    current_model: str
    models: list[dict]
    last_ingest_at: datetime | None
    stale_models: list[str]


async def status(db: AsyncSession, *, project_id: uuid.UUID, current_model: str) -> MemoryStatus:
    """展示真实已索引内容和待处理缺口，让记忆状态可理解。"""
    norm_pid = uuid.UUID(str(project_id))
    by_model = (
        await db.execute(
            select(
                MemoryChunk.embedding_model,
                func.count(),
                func.count(func.distinct(MemoryChunk.source_report_id)),
                func.max(MemoryChunk.created_at),
            )
            .where(MemoryChunk.project_id == norm_pid)
            .group_by(MemoryChunk.embedding_model)
        )
    ).all()

    session_approved = (
        await db.execute(
            select(func.count())
            .select_from(Session)
            .where(Session.project_id == norm_pid, Session.status == SessionStatus.COMPLETED)
        )
    ).scalar_one()

    run_approved = (
        await db.execute(
            select(func.count())
            .select_from(ResearchRun)
            .where(ResearchRun.project_id == norm_pid, ResearchRun.status == "COMPLETED")
        )
    ).scalar_one()

    approved = session_approved + run_approved

    pg_chunks = sum(chunks for _, chunks, _, _ in by_model)
    pg_indexed = sum(reports for model, _, reports, _ in by_model if model == current_model)

    # 这里只统计项目记忆，不能与独立语料库取最大值或混算。
    # 否则记忆只有一份却会显示索引两份，冒报实际检索能力。
    pending = max(approved - pg_indexed, 0)

    models_list = [
        {"embedding_model": model, "chunks": chunks, "reports": reports}
        for model, chunks, reports, _ in sorted(by_model, key=lambda row: row[0])
    ]

    latest_ingest = max((ts for *_, ts in by_model if ts), default=None)
    stale = sorted({model for model, *_ in by_model if model != current_model})

    return MemoryStatus(
        chunk_count=pg_chunks,
        indexed_reports=pg_indexed,
        approved_reports=approved,
        pending_reports=pending,
        current_model=current_model,
        models=models_list,
        last_ingest_at=latest_ingest,
        stale_models=stale,
    )
