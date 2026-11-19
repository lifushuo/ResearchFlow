"""项目聊天：基于项目内获批研究检索和带引用回答（docs/14 §5、§8）。

隔离由 memory.retrieve 的 project_id 查询条件保证，不靠提示词承诺。先校验归属并过滤项目，再把内容交给模型。
检索片段源自网页，始终是不可信输入，统一包在 untrusted_web_content 中，避免历史页面注入在聊天中重新生效。"""

from __future__ import annotations

import json
import uuid
from collections.abc import AsyncGenerator
from datetime import UTC, datetime

import structlog
from fastapi import APIRouter, Depends, HTTPException, Response, status
from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app import adapters
from app.api.v1.projects import resolve_project
from app.db.base import get_db
from app.dependencies import enforce_chat_rate_limit, get_current_user
from app.models.chat_message import ChatMessage
from app.models.chat_thread import ChatThread, derive_title
from app.models.project import Project
from app.models.user import User
from app.schemas.chat import (
    MemoryStatusResponse,
    ThreadCreateRequest,
    ThreadListResponse,
    ThreadMessageRequest,
    ThreadMessageSchema,
    ThreadResponse,
)
from app.services import chat_scope, crypto, memory
from app.services.chat_history import recent_turns
from app.services.sse import SSE_HEADERS
from research_engine.embeddings import EmbeddingsUnavailable
from research_engine.llm_factory import get_llm, reset_user_keys, set_user_keys, text_of
from research_engine.prompt_composition import system_prompt

logger = structlog.get_logger()
router = APIRouter(tags=["Project chat"])


async def _owned_thread(db: AsyncSession, thread_id: uuid.UUID, user_id: uuid.UUID) -> ChatThread:
    """通过项目归属验证聊天线程，非当前用户拥有或不存在都返回 404，避免泄露他人线程。"""
    thread = (
        await db.execute(
            select(ChatThread)
            .join(Project, Project.id == ChatThread.project_id)
            .where(ChatThread.id == thread_id, Project.user_id == user_id)
        )
    ).scalar_one_or_none()
    if not thread:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="Thread not found.")
    return thread


async def _message_counts(db: AsyncSession, thread_ids: list[uuid.UUID]) -> dict[uuid.UUID, int]:
    if not thread_ids:
        return {}
    rows = await db.execute(
        select(ChatMessage.thread_id, func.count())
        .where(ChatMessage.thread_id.in_(thread_ids))
        .group_by(ChatMessage.thread_id)
    )
    return {tid: n for tid, n in rows.all()}


# 聊天线程管理


@router.get("/projects/{project_id}/threads", response_model=ThreadListResponse)
async def list_threads(
    project_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    project = await resolve_project(db, current_user.id, project_id)
    rows = (
        (
            await db.execute(
                select(ChatThread)
                .where(ChatThread.project_id == project.id)
                .order_by(ChatThread.last_message_at.desc())
            )
        )
        .scalars()
        .all()
    )
    counts = await _message_counts(db, [t.id for t in rows])
    return ThreadListResponse(
        threads=[
            ThreadResponse(
                id=t.id,
                project_id=t.project_id,
                title=t.title,
                message_count=counts.get(t.id, 0),
                created_at=t.created_at,
                last_message_at=t.last_message_at,
            )
            for t in rows
        ],
        total=len(rows),
    )


@router.post("/projects/{project_id}/threads", response_model=ThreadResponse, status_code=201)
async def create_thread(
    project_id: uuid.UUID,
    payload: ThreadCreateRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    project = await resolve_project(db, current_user.id, project_id)
    thread = ChatThread(project_id=project.id, title=(payload.title or "New chat").strip())
    db.add(thread)
    await db.commit()
    await db.refresh(thread)
    return ThreadResponse(
        id=thread.id,
        project_id=thread.project_id,
        title=thread.title,
        message_count=0,
        created_at=thread.created_at,
        last_message_at=thread.last_message_at,
    )


@router.delete("/threads/{thread_id}", status_code=204)
async def delete_thread(
    thread_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    thread = await _owned_thread(db, thread_id, current_user.id)
    await db.delete(thread)  # 消息随线程级联删除。
    await db.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.get("/threads/{thread_id}/messages", response_model=list[ThreadMessageSchema])
async def thread_history(
    thread_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    await _owned_thread(db, thread_id, current_user.id)
    return (
        (
            await db.execute(
                select(ChatMessage)
                .where(ChatMessage.thread_id == thread_id)
                .order_by(ChatMessage.created_at.asc())
            )
        )
        .scalars()
        .all()
    )


# 带引用的回答


def _grounding(retrieved: list[memory.Retrieved]) -> tuple[str, list[dict]]:
    """按检索顺序给片段编号，并建立引用映射；R1 是最近匹配。

    在模型回答前生成，因为检索到哪些内容是已知事实，与模型最终选择引用哪些内容不同。"""
    blocks: list[str] = []
    citations: list[dict] = []
    for index, hit in enumerate(retrieved, start=1):
        marker = f"R{index}"
        report = hit.report
        blocks.append(
            f"[{marker}] From the approved report {report.title!r} "
            f"({report.created_at:%Y-%m-%d}):\n{hit.chunk.text}"
        )
        citations.append(
            {
                "marker": marker,
                # 使用 report_id 而非 session_id；记忆可能来自会话或研究运行，不能把两者都冒称为会话。
                "report_id": str(report.id),
                "report_kind": report.kind,
                "title": report.title,
                "created_at": report.created_at.isoformat(),
                "excerpt": hit.chunk.text,
            }
        )
    return "\n\n".join(blocks), citations


@router.post("/threads/{thread_id}/messages")
async def send_thread_message(
    thread_id: uuid.UUID,
    payload: ThreadMessageRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
    _rl: None = Depends(enforce_chat_rate_limit),
):
    """从项目获批研究回答问题，并提供引用。"""
    thread = await _owned_thread(db, thread_id, current_user.id)

    keys = crypto.user_provider_keys(current_user)
    scope = payload.scope
    # 先判断范围是否需要向量。纯网页追问不读项目记忆，不应因未配置向量模型而无法聊天。
    needs_embedder = scope in ("report", "corpus", "everything")
    embedder = await adapters.embeddings_for(keys) if needs_embedder else None

    citations: list = []
    excerpts = ""
    if scope in ("report", "everything"):
        # 先向量化和检索再保存用户消息，避免供应商不可用时留下永远没有回答的消息。
        try:
            query_vector = (await embedder.embed([payload.message]))[0]
        except EmbeddingsUnavailable as e:
            # 无法取得获批研究证据时返回 503，不能换成没有依据的回答。
            raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(e)) from e

        retrieved = await memory.retrieve(
            db,
            index=adapters.PgVectorMemoryIndex(),
            project_id=thread.project_id,
            query_vector=query_vector,
            embedding_model=embedder.model_id,
        )
        excerpts, citations = _grounding(retrieved)

    # 只在所选范围确实读取语料时解析文件路径；项目没有语料时 for_project 返回 None。
    corpus_store = None
    if scope in ("report", "corpus", "everything"):
        from app.adapters import ServerCorpusLocator

        corpus_store = await ServerCorpusLocator().for_project(thread.project_id, keys=keys)

    def _corpus_store():
        return corpus_store

    if scope in ("report", "everything") and not excerpts:
        store = _corpus_store()
        if store:
            try:
                gen_hits = await store.search_generated_reports(payload.message, 8)
                if gen_hits:
                    gen_blocks = []
                    for idx, hit in enumerate(gen_hits, 1):
                        gen_blocks.append(
                            f"[{idx}] From approved report {hit['title']}:\n{hit['snippet']}"
                        )
                        citations.append(
                            {
                                "marker": idx,
                                "session_id": hit.get("document_id", ""),
                                "title": hit.get("title", "Approved Report"),
                                "created_at": datetime.now(UTC).isoformat(),
                                "excerpt": hit.get("snippet", ""),
                            }
                        )
                    excerpts = "\n\n".join(gen_blocks)
            except Exception as e:
                logger.warning("corpus_generated_reports_search_failed", error=str(e))

    try:
        grounding = await chat_scope.gather(
            scope,
            query=payload.message,
            report=None,
            report_sources=[],
            memory_excerpts=excerpts,
            store_factory=_corpus_store,
        )
    except EmbeddingsUnavailable as e:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, detail=str(e)) from e

    now = datetime.now(UTC)
    db.add(ChatMessage(thread_id=thread_id, role="user", content=payload.message))
    if thread.title == "New chat":
        thread.title = derive_title(payload.message)
    thread.last_message_at = now
    await db.commit()

    history = await recent_turns(db, ChatMessage.thread_id == thread_id)

    # 保留 PROJECT_CHAT_PROMPT 开头的拒答规则；扩大范围时尤其不能用模型自身知识冒充项目证据（docs/14 §9）。
    system = (
        f"{system_prompt('chat.project')}\n\n"
        f"{chat_scope.system_suffix(grounding)}\n\n"
        f"<untrusted_web_content>\n"
        f"{grounding.text or '(nothing has been approved yet)'}\n"
        f"</untrusted_web_content>"
    )
    messages: list = [SystemMessage(content=system)]
    for m in history:
        messages.append(
            HumanMessage(content=m.content) if m.role == "user" else AIMessage(content=m.content)
        )

    async def gen() -> AsyncGenerator[str]:
        keys_token = set_user_keys(keys)
        acc = ""
        try:
            llm = get_llm("chat")  # 没有密钥时返回能指导用户处理的错误。
            # 先发送引用映射，方便客户端随正文逐步渲染引用。
            yield (
                "data: "
                + json.dumps(
                    {
                        "type": "connected",
                        "citations": citations,
                        "scope": grounding.scope,
                        "sources": grounding.sources,
                        "notes": grounding.notes,
                    }
                )
                + "\n\n"
            )
            async for chunk in llm.astream(messages):
                text = text_of(chunk)
                if text:
                    acc += text
                    yield f"data: {json.dumps({'type': 'chunk', 'text': text})}\n\n"

            # 只保存回答实际使用的引用；检索到但未使用的片段不算引用，避免展示虚假的来源背书（docs/27）。
            used = [c for c in citations if f"[{c['marker']}]" in acc]
            msg = ChatMessage(
                thread_id=thread_id, role="assistant", content=acc, citations=used or None
            )
            db.add(msg)
            await db.commit()
            await db.refresh(msg)
            yield (
                "data: "
                + json.dumps({"type": "done", "message_id": str(msg.id), "citations": used})
                + "\n\n"
            )
        except Exception as e:  # noqa: BLE001
            logger.warning("thread_chat_failed", thread_id=str(thread_id), error=str(e))
            yield f"data: {json.dumps({'type': 'error', 'detail': 'An internal error occurred. Please try again.'})}\n\n"
        finally:
            reset_user_keys(keys_token)

    return StreamingResponse(gen(), media_type="text/event-stream", headers=SSE_HEADERS)


# 项目记忆状态


@router.get("/projects/{project_id}/memory/status", response_model=MemoryStatusResponse)
async def memory_status(
    project_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """返回项目已索引的内容及尚未索引的缺口（docs/14 §8）。"""
    project = await resolve_project(db, current_user.id, project_id)
    embedder = await adapters.embeddings_for(crypto.user_provider_keys(current_user))
    result = await memory.status(db, project_id=project.id, current_model=embedder.model_id)
    return MemoryStatusResponse(
        available=embedder.model_id != "none",
        chunk_count=result.chunk_count,
        indexed_reports=result.indexed_reports,
        approved_reports=result.approved_reports,
        pending_reports=result.pending_reports,
        current_model=result.current_model,
        models=result.models,
        stale_models=result.stale_models,
        last_ingest_at=result.last_ingest_at,
    )
