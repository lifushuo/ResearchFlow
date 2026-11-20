"""研究会话接口（docs/05 §3，docs/02 §2/§5）。

Cookie 鉴权兼容浏览器 EventSource。SSE 先按 Last-Event-ID 回放持久日志，再监听 Redis，防止断线丢消息。批准会记录审计，并从检查点恢复流程。"""

from __future__ import annotations

import hashlib
import json
import uuid
from collections.abc import AsyncGenerator
from datetime import UTC, datetime

import structlog
from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from fastapi.responses import StreamingResponse
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.v1.projects import resolve_project
from app.db.base import AsyncSessionLocal, get_db
from app.db.redis import get_redis
from app.dependencies import enforce_research_rate_limit, get_current_user
from app.logconfig import bind_session_context
from app.models.agent_log import AgentLog
from app.models.audit_log import AuditLog
from app.models.session import Session, SessionStatus
from app.models.user import User
from app.ports import CorpusLocator, TerminalEventEmitter
from app.run_dispatch import SessionDispatcher
from app.schemas.corpus import ExtractedTextResponse
from app.schemas.research import (
    ApprovalRequest,
    OutlineSectionSchema,
    OutlineTemplateSchema,
    PlanDecisionRequest,
    PlanResponse,
    PlanTaskSchema,
    ResearchStartRequest,
    ResearchStartResponse,
    SessionDetail,
    SessionListResponse,
    SessionSummary,
)
from app.services import (
    checkpoints,
    cited_documents,
    corpus_ingest,
    export,
    memory,
    model_routing,
    usage,
)
from app.services.document_headers import download_headers, media_type_for
from app.services.event_stream import sse_frames
from app.services.host_ports import get_corpus_locator, get_terminal_event_emitter
from app.services.sse import SSE_HEADERS
from app.workers.dispatch import get_session_dispatcher
from research_engine import bundle
from research_engine.events import make_event

# 历史回放只在真正终态停止，实时监听还在人工关卡停止，因为图等待人工操作时不会继续发消息。
# 两种主机和两种研究入口共用这些列表，区别原因见 app/services/event_stream.py。
_REPLAY_STOP_EVENTS = ("COMPLETED", "FAILED")
_TERMINAL_EVENTS = ("COMPLETED", "FAILED", "HITL_READY", "PLAN_READY")

logger = structlog.get_logger()
router = APIRouter(prefix="/research", tags=["Research"])


@router.post("", response_model=ResearchStartResponse, status_code=202)
async def start_research(
    payload: ResearchStartRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
    _rl: None = Depends(enforce_research_rate_limit),
    dispatcher: SessionDispatcher = Depends(get_session_dispatcher),
):
    # 入队前检查月度 Token 上限，0 表示不限。超额时立即返回清楚的 402，而不是等任务执行中途失败。
    if current_user.monthly_token_limit > 0:
        used = await usage.monthly_tokens(db, current_user.id)
        if used >= current_user.monthly_token_limit:
            raise HTTPException(
                status.HTTP_402_PAYMENT_REQUIRED,
                detail=(
                    f"Monthly token limit reached ({used:,} of "
                    f"{current_user.monthly_token_limit:,}). It resets on the 1st, or you "
                    "can add your own API key in Settings."
                ),
            )

    # 在创建会话前校验本次模型选择，避免任务跑起来后才发现路由无效。
    # None 表示沿用已保存设置，之后按用户、部署顺序解析。
    routing = None
    if payload.model_routing:
        try:
            routing = model_routing.validate(payload.model_routing)
        except model_routing.InvalidRouting as e:
            raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(e)) from e

    # 解析目标项目并检查归属；未指定则使用默认项目，让新用户可以直接开始。
    project = await resolve_project(db, current_user.id, payload.project_id)

    session = Session(
        user_id=current_user.id,
        project_id=project.id,
        prompt=payload.query,
        status=SessionStatus.PENDING,
        research_depth=payload.depth,
        model_routing=routing,
        # 这些字段虽然被请求结构接受，也必须真正存入会话记录。
        # 否则会退回列默认值，导致用户选择仅用上传语料，实际却进行了网页搜索。
        corpus_mode=payload.corpus_mode,
        demo=payload.demo,
        # 计划关卡同样必须持久化，_run_config_for 恢复时会读取这些选项（internal/07 Phase 4）。
        # 只接受请求却不存记录，会让执行行为违背用户选择。
        skip_plan_gate=payload.skip_plan_gate,
        topic_seeds=payload.topic_seeds or None,
        outline_template=payload.outline_template,
    )
    db.add(session)
    await db.commit()
    await db.refresh(session)

    await dispatcher.start(str(session.id), str(current_user.id))
    # 绑定本次会话的 correlation_id，让触发请求、Celery 任务与引擎日志可以串联。
    bind_session_context(str(session.id), user_id=str(current_user.id))
    logger.info("research_started", session_id=str(session.id))
    return ResearchStartResponse(session_id=session.id, status=session.status)


@router.get("", response_model=SessionListResponse)
async def list_sessions(
    page: int = 1,
    limit: int = 20,
    archived: bool = False,
    project_id: uuid.UUID | None = None,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """列出当前用户的会话，可按项目筛选。

    活动和归档列表分开，默认不混入归档会话；项目筛选前必须验证归属，避免越权。"""
    limit = max(1, min(limit, 100))
    filters = [
        Session.user_id == current_user.id,
        Session.archived_at.is_not(None) if archived else Session.archived_at.is_(None),
    ]
    if project_id is not None:
        project = await resolve_project(db, current_user.id, project_id)
        filters.append(Session.project_id == project.id)
    base = select(Session).where(*filters)
    total = (await db.execute(select(func.count()).select_from(base.subquery()))).scalar_one()
    rows = (
        (
            await db.execute(
                base.order_by(Session.created_at.desc()).offset((page - 1) * limit).limit(limit)
            )
        )
        .scalars()
        .all()
    )
    return SessionListResponse(
        sessions=[SessionSummary.model_validate(s) for s in rows],
        total=total,
        page=page,
        limit=limit,
    )


@router.get("/outline-templates", response_model=list[OutlineTemplateSchema])
async def list_outline_templates(_: User = Depends(get_current_user)):
    """返回计划审核可选的报告结构（internal/07 Phase 4）。

    直接读取引擎 outlines，保证预览与综合器收到的章节一致。静态路由放在 UUID 路由前，避免被误当会话 ID 而返回 422。"""
    from research_engine import outlines

    return [OutlineTemplateSchema.model_validate(t) for t in outlines.catalog()]


@router.get("/{session_id}", response_model=SessionDetail)
async def get_session(
    session_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    session = await _authorized_session(db, session_id, current_user.id)
    from app.services.model_usage import load_model_usage

    return SessionDetail.model_validate(session).model_copy(
        update={"token_usage": await load_model_usage(db, session.id)}
    )


async def _cited_session_store(db, session_id, document_id, current_user, corpus, also_in):
    """会话版引用语料定位。会话没有冻结语料身份或文件选择，因此仅检查归属及来源是否确实被该会话记录。"""
    session = await _authorized_session(db, session_id, current_user.id)
    store = await cited_documents.store_holding(
        corpus,
        document_id,
        cited=cited_documents.recorded_ids(
            s.get("url") for s in (session.sources or []) if isinstance(s, dict)
        ),
        hashes={},
        corpus_id=None,
        projects=[session.project_id, *also_in],
    )
    if store is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail=corpus_ingest.DOCUMENT_NOT_FOUND)
    return store


@router.get("/{session_id}/cited-documents/{document_id}", response_model=ExtractedTextResponse)
async def get_cited_document_text(
    session_id: uuid.UUID,
    document_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
    corpus: CorpusLocator = Depends(get_corpus_locator),
    also_in: list[uuid.UUID] = Depends(cited_documents.no_other_projects),
):
    """读取该会话引用的文档版本的提取正文（RFC V3.1 §7.3）。"""
    store = await _cited_session_store(db, session_id, document_id, current_user, corpus, also_in)
    found = await store.extracted(document_id)
    if found is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail=corpus_ingest.DOCUMENT_NOT_FOUND)
    return ExtractedTextResponse(**found)


@router.get("/{session_id}/cited-documents/{document_id}/download")
async def download_cited_document(
    session_id: uuid.UUID,
    document_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
    corpus: CorpusLocator = Depends(get_corpus_locator),
    also_in: list[uuid.UUID] = Depends(cited_documents.no_other_projects),
):
    """读取该会话引用的文档版本的原始文件。"""
    store = await _cited_session_store(db, session_id, document_id, current_user, corpus, also_in)
    found = await store.blob(document_id)
    if found is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail=corpus_ingest.DOCUMENT_NOT_FOUND)
    data, filename, kind = found
    return Response(
        content=data, media_type=media_type_for(kind), headers=download_headers(kind, filename)
    )


@router.get("/{session_id}/stream")
async def stream_events(
    session_id: uuid.UUID,
    request: Request,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
    redis=Depends(get_redis),
):
    await _authorized_session(db, session_id, current_user.id)
    last_event_id = request.headers.get("last-event-id")
    after_id = int(last_event_id) if last_event_id and last_event_id.isdigit() else 0

    async def gen() -> AsyncGenerator[str]:
        from app import adapters

        channel = f"session:{session_id}:events"
        pubsub = redis.pubsub()
        # 先订阅再拍历史快照，避免两步之间发布的事件在历史和实时流中都丢失。
        # 重复事件按持久 ID 去重，快速恢复后立即完成的任务也不会被漏掉。
        await pubsub.subscribe(channel)
        try:
            # 订阅之后用新数据库会话读取历史；请求自己的会话会在响应进入流式阶段时关闭，不能继续使用。
            async with AsyncSessionLocal() as sdb:
                backlog = (
                    (
                        await sdb.execute(
                            select(AgentLog)
                            .where(AgentLog.session_id == session_id, AgentLog.id > after_id)
                            .order_by(AgentLog.id.asc())
                        )
                    )
                    .scalars()
                    .all()
                )

            # 监听循环由 app/services/event_stream.py 统一实现。
            # 本路由只保留实际不同的历史查询和 Redis 订阅方式。
            async for frame in sse_frames(
                connected={"type": "connected"},
                backlog=[(row.id, row.payload) for row in backlog],
                live=adapters.redis_event_stream(pubsub),
                replay_stop=_REPLAY_STOP_EVENTS,
                terminal_stop=_TERMINAL_EVENTS,
                already_done=False,
                seen_from=after_id,
            ):
                yield frame
        finally:
            await pubsub.unsubscribe(channel)
            await pubsub.aclose()

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers=SSE_HEADERS,
    )


# 演示标记规则统一位于不依赖主机的 research_engine.bundle，确保桌面 Markdown 导出一致。
# 保留 _DEMO_STAMP 别名，兼容既有引用位置。
_DEMO_STAMP = bundle.DEMO_STAMP_MD


def _report_or_404(session: Session, *, stamp_demo: bool = True) -> str:
    """取得待导出的报告；演示会话默认加上演示说明。

    所有文字导出走同一入口，避免新增导出格式时忘记标记，使演示被误认为真实研究。
    证据包唯一例外使用 stamp_demo=False：批准链验证正文哈希，批准后插入说明会破坏它。证据包改以受哈希保护的 demo 字段说明身份。"""
    report = session.final_report or session.draft_report
    if not report:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="No report available to export.")
    return bundle.stamp_demo_md(report, demo=bool(session.demo) and stamp_demo)


@router.get("/{session_id}/export.md")
async def export_markdown(
    session_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    session = await _authorized_session(db, session_id, current_user.id)
    report = _report_or_404(session) + export.render_model_attribution_md(session.model_routing)
    filename = f"research-{str(session.id)[:8]}.md"
    return Response(
        content=report,
        media_type="text/markdown; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.get("/{session_id}/export.pdf")
async def export_pdf(
    session_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    session = await _authorized_session(db, session_id, current_user.id)
    report = _report_or_404(session)
    title = (session.prompt or "Research Report")[:120]
    try:
        pdf = export.render_pdf(
            report, session.sources or [], title=title, model_routing=session.model_routing
        )
    except RuntimeError as e:
        # 当前环境缺少 PDF 原生库；正式镜像应按 docs/09 §1 包含这些依赖。
        raise HTTPException(status.HTTP_501_NOT_IMPLEMENTED, detail=str(e)) from e
    filename = f"research-{str(session.id)[:8]}.pdf"
    return Response(
        content=pdf,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.get("/{session_id}/export.bundle.json")
async def export_bundle_json(
    session_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    session = await _authorized_session(db, session_id, current_user.id)
    if session.status != SessionStatus.COMPLETED:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            detail="Bundle export is only available for COMPLETED sessions.",
        )

    # 证据包正文不追加演示标记，原因见 _report_or_404。
    # 演示身份在受 bundle_hash 保护的 demo 字段中披露，避免破坏正文批准链。
    report = _report_or_404(session, stamp_demo=False)

    agent_logs = (
        (
            await db.execute(
                select(AgentLog)
                .where(AgentLog.session_id == session.id)
                .order_by(AgentLog.id.asc())
            )
        )
        .scalars()
        .all()
    )
    trace = [log.payload for log in agent_logs]

    audit_logs = (
        (
            await db.execute(
                select(AuditLog)
                .where(AuditLog.session_id == session.id)
                .order_by(AuditLog.id.asc())
            )
        )
        .scalars()
        .all()
    )
    approval_chain = [
        {
            "action": al.action,
            "feedback": al.feedback,
            "draft_hash": al.draft_hash,
            "timestamp": al.created_at.isoformat(),
        }
        for al in audit_logs
    ]

    state = await checkpoints.get_thread_state(str(session.id))
    evidence = state.get("evidence", [])
    contradictions = state.get("contradictions", [])

    from research_engine import bundle

    manifest = bundle.assemble(
        session_id=str(session.id),
        query=session.prompt,
        report=report,
        evidence=evidence,
        sources=session.sources or [],
        contradictions=contradictions,
        models=session.model_routing or {},
        cost_usd=float(session.total_cost_usd),
        tokens_input=session.total_tokens_input,
        tokens_output=session.total_tokens_output,
        elapsed_seconds=float(session.elapsed_seconds) if session.elapsed_seconds else None,
        research_depth=session.research_depth,
        approval_chain=approval_chain,
        trace=trace,
        trace_available=True,
        demo=session.demo,
    )

    filename = f"research-{str(session.id)[:8]}.bundle.json"
    return Response(
        content=bundle.serialize(manifest),
        media_type="application/json",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


def _plan_response(session: Session) -> PlanResponse:
    plan = session.plan_json or {}
    outline = session.outline_json or {}
    return PlanResponse(
        session_id=session.id,
        status=session.status,
        tasks=[PlanTaskSchema.model_validate(t) for t in (plan.get("tasks") or [])],
        outline=[OutlineSectionSchema.model_validate(s) for s in (outline.get("sections") or [])],
        approved_at=session.plan_approved_at,
    )


@router.get("/{session_id}/plan", response_model=PlanResponse)
async def get_plan(
    session_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """读取本次实际使用的研究计划（internal/07 Phase 4）。

    plan_json 为空时返回 404，不能伪造空任务列表；未到达或跳过关卡表示没有可展示计划，不代表规划器实测提案为零项。"""
    session = await _authorized_session(db, session_id, current_user.id)
    if session.plan_json is None:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND,
            detail="This session has no research plan to review — it did not use the design gate.",
        )
    return _plan_response(session)


@router.post("/{session_id}/plan", response_model=PlanResponse)
async def submit_plan(
    session_id: uuid.UUID,
    payload: PlanDecisionRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
    dispatcher: SessionDispatcher = Depends(get_session_dispatcher),
):
    """批准并可编辑研究计划，然后继续执行。

    必须确认当前停在计划关卡，否则把计划负载交给报告关卡会被误解成拒绝。
    先保存人工决定再入队恢复，保证即使 Worker 随后崩溃，审核事实仍可追溯。"""
    session = await _authorized_session(db, session_id, current_user.id)
    if session.status != SessionStatus.AWAITING_PLAN:
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            detail=f"Session must be AWAITING_PLAN (currently {session.status}).",
        )

    proposed = (session.plan_json or {}).get("tasks") or []
    # None 表示未编辑，沿用提案；[] 表示排除全部任务，应返回 422。
    tasks = [t.model_dump() for t in payload.tasks] if payload.tasks is not None else proposed
    outline = (
        [s.model_dump() for s in payload.outline]
        if payload.outline is not None
        else (session.outline_json or {}).get("sections") or []
    )

    kept = [t for t in tasks if t.get("include", True)]
    if not kept:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Keep at least one task — a plan with nothing in it researches nothing.",
        )

    # 保存实际获批的 kept，不是未筛选提案；后续查询应看到报告真正执行的研究设计。
    session.plan_json = {"tasks": kept}
    session.outline_json = {"sections": outline}
    session.plan_approved_at = datetime.now(UTC)
    db.add(
        AuditLog(
            session_id=session.id,
            user_id=current_user.id,
            action="plan_approved",
            feedback=None,
            # 对人工批准的计划计算哈希，与草稿审核规则一致，使证据包能携带两次决定。
            draft_hash=hashlib.sha256(
                json.dumps({"tasks": kept, "outline": outline}, sort_keys=True).encode("utf-8")
            ).hexdigest(),
        )
    )
    session.status = SessionStatus.RUNNING
    await db.commit()
    await db.refresh(session)

    await dispatcher.resume_plan(
        str(session.id), str(current_user.id), {"tasks": kept, "outline": outline}
    )
    bind_session_context(str(session.id), user_id=str(current_user.id))
    logger.info("research_plan_approved", session_id=str(session.id), task_count=len(kept))
    return _plan_response(session)


@router.post("/{session_id}/approve", status_code=200)
async def approve_or_rework(
    session_id: uuid.UUID,
    payload: ApprovalRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
    dispatcher: SessionDispatcher = Depends(get_session_dispatcher),
):
    session = await _authorized_session(db, session_id, current_user.id)
    if session.status != SessionStatus.AWAITING_APPROVAL:
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            detail=f"Session must be AWAITING_APPROVAL (currently {session.status}).",
        )
    if not payload.approved and session.rework_count >= 3:
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            detail="Rework limit reached. Approve or abandon this session.",
        )

    draft_hash = hashlib.sha256((session.draft_report or "").encode("utf-8")).hexdigest()
    db.add(
        AuditLog(
            session_id=session.id,
            user_id=current_user.id,
            action="approved" if payload.approved else "rework_requested",
            feedback=None if payload.approved else payload.feedback,
            draft_hash=draft_hash,
        )
    )
    session.status = SessionStatus.RUNNING
    await db.commit()

    await dispatcher.resume_review(
        str(session.id), str(current_user.id), payload.approved, payload.feedback
    )
    bind_session_context(str(session.id), user_id=str(current_user.id))
    logger.info(
        "research_resumed",
        session_id=str(session.id),
        approved=payload.approved,
    )
    return {"message": "Approved. Finalizing." if payload.approved else "Rework requested."}


@router.post("/{session_id}/cancel", response_model=SessionSummary)
async def cancel_session(
    session_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
    emit_terminal_event: TerminalEventEmitter = Depends(get_terminal_event_emitter),
):
    """请求停止在途会话，并持久记录终止事件。

    emit_terminal_event 先落盘 FAILED 再发布，断线后重连也能回放取消事件；直接 publish_event 只对当时在线的客户端可见。原因使用 data.reason，沿用既有生命周期事件结构。"""
    session = await _authorized_session(db, session_id, current_user.id)
    if session.status not in (SessionStatus.RUNNING, SessionStatus.PENDING):
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            detail=f"Cannot stop a session with status {session.status}.",
        )

    session.status = SessionStatus.FAILED
    session.error_message = "Research stopped by user."
    # 把取消持久化，后续结果不能推翻它。
    # 此前带一小时 TTL 的 Redis 取消键没有读取方，也无法可靠跨重启；现在 _persist_outcome 读取数据库决定。
    session.cancelled_at = datetime.now(UTC)
    await db.commit()
    await db.refresh(session)

    await emit_terminal_event(
        db, str(session_id), make_event("FAILED", data={"reason": "Research stopped by user."})
    )

    logger.info("research_stopped_by_user", session_id=str(session_id))
    return SessionSummary.model_validate(session)


@router.post("/{session_id}/archive", response_model=SessionSummary)
async def archive_session(
    session_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """从活动列表归档会话；可以恢复，不丢失数据。"""
    session = await _authorized_session(db, session_id, current_user.id)
    if session.archived_at is None:
        session.archived_at = datetime.now(UTC)
        await db.commit()
        await db.refresh(session)
    return SessionSummary.model_validate(session)


@router.post("/{session_id}/unarchive", response_model=SessionSummary)
async def unarchive_session(
    session_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    session = await _authorized_session(db, session_id, current_user.id)
    if session.archived_at is not None:
        session.archived_at = None
        await db.commit()
        await db.refresh(session)
    return SessionSummary.model_validate(session)


@router.delete("/{session_id}", status_code=204)
async def delete_session(
    session_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """永久删除会话及其派生内容。

    代理日志、聊天和审计通过外键级联删除；LangGraph 检查点属于独立表，必须按 thread ID 显式删除，避免残留完整研究状态和网页正文。"""
    session = await _authorized_session(db, session_id, current_user.id)
    if session.status == SessionStatus.RUNNING:
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            detail="This session is still running. Wait for it to finish before deleting.",
        )

    # 项目记忆需要显式且有条件清理，规则见 memory.purge_report。
    # 不能配置无条件级联，因为桌面主机没有这张记忆表。
    await memory.purge_report(db, session_id)
    await db.delete(session)
    await db.commit()

    try:
        await checkpoints.delete_thread(str(session_id))
    except Exception as e:  # noqa: BLE001 — 用户记录已删除；后续清理失败只记录日志
        logger.warning("checkpoint_cleanup_failed", session_id=str(session_id), error=str(e))

    logger.info("session_deleted", session_id=str(session_id))
    return Response(status_code=status.HTTP_204_NO_CONTENT)


async def _authorized_session(
    db: AsyncSession, session_id: uuid.UUID, user_id: uuid.UUID
) -> Session:
    session = (
        await db.execute(
            select(Session).where(Session.id == session_id, Session.user_id == user_id)
        )
    ).scalar_one_or_none()
    if not session:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="Session not found.")
    return session
