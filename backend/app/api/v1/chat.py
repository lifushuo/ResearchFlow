"""基于已完成报告的追问接口（docs/04 §5，docs/05 §3）。

历史助手回复恢复为 AIMessage，不能升级成 SystemMessage。使用 SSE 输出，报告与来源作为证据输入，并标记为不可信内容（docs/06 §4）。"""

from __future__ import annotations

import json
import uuid
from collections.abc import AsyncGenerator

import structlog
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.base import get_db
from app.dependencies import enforce_chat_rate_limit, get_current_user
from app.models.chat_message import ChatMessage
from app.models.session import Session, SessionStatus
from app.models.user import User
from app.schemas.research import ChatMessageSchema, ChatRequest
from app.services import chat_scope, crypto
from app.services.chat_history import recent_turns
from app.services.run_config import chat_prompt_context
from app.services.sse import SSE_HEADERS
from research_engine.embeddings import EmbeddingsUnavailable
from research_engine.llm_factory import get_llm, reset_user_keys, set_user_keys, text_of
from research_engine.prompt_composition import system_prompt

logger = structlog.get_logger()

router = APIRouter(prefix="/research/{session_id}/chat", tags=["Chat"])


async def _authorized_completed_session(db, session_id, user_id) -> Session:
    session = (
        await db.execute(
            select(Session).where(Session.id == session_id, Session.user_id == user_id)
        )
    ).scalar_one_or_none()
    if not session:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="Session not found.")
    return session


@router.get("", response_model=list[ChatMessageSchema])
async def get_history(
    session_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    await _authorized_completed_session(db, session_id, current_user.id)
    rows = (
        (
            await db.execute(
                select(ChatMessage)
                .where(ChatMessage.session_id == session_id)
                .order_by(ChatMessage.created_at.asc())
            )
        )
        .scalars()
        .all()
    )
    return rows


@router.post("")
async def send_message(
    session_id: uuid.UUID,
    payload: ChatRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
    _rl: None = Depends(enforce_chat_rate_limit),
):
    session = await _authorized_completed_session(db, session_id, current_user.id)
    if session.status != SessionStatus.COMPLETED:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST, detail="Chat is available only for completed sessions."
        )

    db.add(ChatMessage(session_id=session_id, role="user", content=payload.message))
    await db.commit()

    history = await recent_turns(db, ChatMessage.session_id == session_id)

    # 先解析用户自带密钥，再收集语料证据；语料向量化应与研究一样使用用户自己的密钥。
    user_keys = crypto.user_provider_keys(current_user)

    # 只在需要语料的范围解析存储；报告范围不应访问语料文件系统或调用向量模型。
    # store_factory 按 chat_scope.gather 契约保留为可调用对象，实际路径由定位器负责。
    store = None
    if payload.scope in ("corpus", "everything"):
        from app import adapters

        store = await adapters.ServerCorpusLocator().for_project(session.project_id, keys=user_keys)

    def _corpus_store():
        return store

    try:
        grounding = await chat_scope.gather(
            payload.scope,
            query=payload.message,
            report=session.final_report,
            report_sources=session.sources or [],
            memory_excerpts=None,
            store_factory=_corpus_store,
        )
    except EmbeddingsUnavailable as e:
        # 用户明确选择了语料范围，若执行会外发语料，应返回 400 说明原因，不能偷偷换来源回答。
        raise HTTPException(status.HTTP_400_BAD_REQUEST, detail=str(e)) from e

    # 聊天没有需要前后保持一致的运行快照，所以每轮实时读取提示词偏好。
    # 仍统一经过 system_prompt 应用覆盖策略，不能在这里直接拼接。
    with chat_prompt_context(current_user):
        chat_system = system_prompt("chat.general")

    system = (
        f"{chat_system}\n\n"
        f"{chat_scope.system_suffix(grounding)}\n\n"
        f"<untrusted_web_content>\n{grounding.text}\n</untrusted_web_content>"
    )
    messages: list = [SystemMessage(content=system)]
    for m in history:
        messages.append(
            HumanMessage(content=m.content) if m.role == "user" else AIMessage(content=m.content)
        )

    async def gen() -> AsyncGenerator[str]:
        keys_token = set_user_keys(user_keys)
        acc = ""
        try:
            llm = get_llm("chat")  # 缺少密钥时抛出可操作的错误提示。
            # 先发送来源，便于前端随流式正文渲染引用，并解析网页回答的 [n]。
            # 同时返回实际 scope，明确回答使用哪个范围的证据。
            yield (
                "data: "
                + json.dumps(
                    {
                        "type": "connected",
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
            msg = ChatMessage(session_id=session_id, role="assistant", content=acc)
            db.add(msg)
            await db.commit()
            await db.refresh(msg)
            yield f"data: {json.dumps({'type': 'done', 'message_id': str(msg.id)})}\n\n"
        except Exception as e:  # noqa: BLE001
            logger.warning("chat_failed", session_id=str(session_id), error=str(e), exc_info=True)
            yield f"data: {json.dumps({'type': 'error', 'detail': 'An internal error occurred. Please try again.'})}\n\n"
        finally:
            reset_user_keys(keys_token)

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers=SSE_HEADERS,
    )
