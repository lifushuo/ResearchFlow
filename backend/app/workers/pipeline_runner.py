"""研究流程的服务端适配器（docs/02 §2/§6，docs/13 §4—§5）。

实际流程由 research_engine.runner 编排；它不依赖 Postgres、Redis、Celery 或 ORM。
本模块负责防重复执行的 Redis 令牌锁、整轮共用的数据库会话、Postgres 检查点、代理日志和 Redis 事件/搜索缓存。
先持久化并提交结果，再发布生命周期事件，避免客户端收到 COMPLETED 后查到旧 RUNNING。人工批准和返工从关卡恢复，不重跑已经付费的研究。"""

from __future__ import annotations

import uuid
from dataclasses import replace

import structlog
from sqlalchemy import select

from app import adapters
from app.config import settings
from app.db.base import AsyncSessionLocal, engine
from app.db.redis import (
    acquire_session_lock,
    close_redis_pool,
    init_redis_pool,
    release_session_lock,
)
from app.models.session import Session, SessionStatus
from app.models.user import User
from app.runtime import run_config_from_settings
from app.services import crypto, model_routing
from app.services.run_config import (
    OVERRIDES_APPLIED,
    apply_demo_rule,
    preference_overrides,
    snapshot_overrides,
)
from app.services.session_events import lifecycle_event
from app.session_lifecycle import mark_running
from research_engine import citation_rate, events, runner
from research_engine.runconfig import RunConfig
from research_engine.runner import RunOutcome

logger = structlog.get_logger()


def _checkpointer_dsn() -> str:
    # LangGraph 的 Postgres 保存器使用 psycopg 连接串，不使用 asyncpg 格式。
    return settings.database_url.replace("postgresql+asyncpg://", "postgresql://")


async def _user_provider_keys(db, user_id: str) -> dict[str, str]:
    """获取用户自带的模型密钥。

    返回供应商到密钥的映射；未设置或无法解密时返回空字典，回退部署密钥。只有解密失败值得记录警告，用户从未设置密钥不是错误。"""
    user = (
        await db.execute(select(User).where(User.id == uuid.UUID(user_id)))
    ).scalar_one_or_none()
    if user is None:
        return {}
    keys = crypto.user_provider_keys(user)
    if not keys and user.api_key_encrypted and user.api_key_provider:
        logger.warning("byok_key_undecryptable", user_id=user_id, provider=user.api_key_provider)
    return keys


async def _run_config_for(db, session: Session, user_id: str) -> RunConfig:
    """解析并冻结本轮模型路由，构建引擎配置。

    优先级为会话记录、用户设置、部署默认值。恢复时沿用会话快照，防止同一份报告的前后两段在用户修改配置后改用不同模型。"""
    user = (
        await db.execute(select(User).where(User.id == uuid.UUID(user_id)))
    ).scalar_one_or_none()

    routing = model_routing.resolve(
        session_routing=session.model_routing,
        user_routing=(user.model_routing if user else None),
    )

    if session.model_routing != routing:
        session.model_routing = routing
        await db.commit()

    base = run_config_from_settings()
    overrides = preference_overrides(user)
    # 计划关卡配置必须从会话记录读取，因为恢复时原请求已经结束。
    # 对应 AWAITING_PLAN、runner.resume(plan=…)、resume_plan_gate 任务和计划接口（internal/07 Phase 4）。
    # CLI 与评测仍沿用 RunConfig 的跳过关卡默认值；托管运行在这里明确自己的选择。桌面对应 sidecar._drive_session。
    overrides |= {
        "skip_plan_gate": bool(session.skip_plan_gate),
        "topic_seeds": tuple(session.topic_seeds or ()),
        "outline_template": session.outline_template,
        # 只安装语料库端口还不够，必须同时告诉引擎启用语料模式。
        # 检索器与 read_webpage 都读取此字段；与 run_execution.run_config_for_run 保持一致。
        "corpus_mode": bool(session.corpus_mode),
    }
    # 没有调用真实供应商的脚本模型运行必须标记为演示。
    # 两种主机共用 app/services/run_config.py 的规则，不能各写分支靠人工同步。
    config, needs_stamp = apply_demo_rule(
        base, row_demo=bool(session.demo), host_is_scripted=base.llm_mode == "fake"
    )
    if needs_stamp:
        session.demo = True
        await db.commit()
    return replace(config, models=routing, **overrides)


async def _execute(
    session_id: str,
    user_id: str,
    *,
    resume: tuple[bool, str | None] | None,
    plan: dict | None = None,
) -> None:
    """取得会话锁、连接主机资源，并选择新执行或关卡恢复。

    首次运行、计划关卡恢复、报告关卡恢复共用同一把锁、同一 Postgres 检查点和语料检查，防止三种入口同时驱动同一会话。
    锁最先取得、最后释放。消息重投导致锁忙时直接返回，让持锁调用完成保存。语料检查先于打开检查点执行，提前明确报告缺少本地向量模型或语料库的问题。"""
    lock_token = uuid.uuid4().hex
    await init_redis_pool()
    try:
        if not await acquire_session_lock(
            session_id, lock_token, ttl=settings.celery_task_timeout_seconds + 60
        ):
            logger.warning("session_lock_busy", session_id=session_id)
            return

        try:
            async with AsyncSessionLocal() as db:
                session = (
                    await db.execute(select(Session).where(Session.id == uuid.UUID(session_id)))
                ).scalar_one_or_none()
                if session is None:
                    logger.error("session_not_found", session_id=session_id)
                    return

                # 条件更新为 RUNNING；若分发后用户已取消，就不覆盖取消状态。规则在 app/session_lifecycle.py。
                # 当前取消是协作标记，在途图仍可能继续，最终结果写入器负责保留取消决定。
                await mark_running(db, session.id)
                # 会话路径按设计不应用提示词覆盖。若用户配置了覆盖，要在启动事务中明确记录未应用，不能悄悄忽略。
                # 只在首次开始时标记，恢复继承原披露状态。桌面对应 sidecar._drive_session。
                if resume is None and plan is None:
                    owner = (
                        await db.execute(select(User).where(User.id == uuid.UUID(user_id)))
                    ).scalar_one_or_none()
                    _, override_status = snapshot_overrides(owner)
                    session.prompt_overrides_not_applied = override_status == OVERRIDES_APPLIED
                await db.commit()

                sink = adapters.agent_log_sink(db, session_id)
                ports = {
                    # 用户自带密钥只在本次运行上下文生效，避免并发任务串用；空配置回退服务端密钥。
                    "provider_keys": await _user_provider_keys(db, user_id),
                    "event_sink": sink,
                    "cache": adapters.RedisCache(),
                    # 每次运行独立设置模型路由（docs/12 M8），不修改进程默认配置，保证不同模型的并发运行相互隔离。
                    "run_config": await _run_config_for(db, session, user_id),
                }

                from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

                if session.corpus_mode:
                    embedder = await adapters.embeddings_for(ports["provider_keys"])
                    # 防止语料外发：语料模式必须使用本地向量模型。
                    if not (
                        embedder.model_id.startswith("ollama:")
                        or embedder.model_id.startswith("local:")
                    ):
                        session.status = SessionStatus.FAILED
                        session.error_message = f"Corpus mode requires a local embedder (Ollama), but got {embedder.model_id}."
                        await db.commit()
                        await sink(
                            session_id,
                            events.make_event("FAILED", data={"reason": session.error_message}),
                        )
                        return

                    # API 与 Worker 共用同一个语料路径定位器，避免不同进程和工作目录把同一项目解析到不同位置，导致上传后研究却找不到文件。
                    store = await adapters.ServerCorpusLocator().for_project(
                        session.project_id, keys=ports["provider_keys"]
                    )
                    if store is None:
                        session.status = SessionStatus.FAILED
                        session.error_message = f"Corpus database not found for project {session.project_id}. Ingest documents first."
                        await db.commit()
                        await sink(
                            session_id,
                            events.make_event("FAILED", data={"reason": session.error_message}),
                        )
                        return

                    ports["corpus"] = store

                async with AsyncPostgresSaver.from_conn_string(_checkpointer_dsn()) as saver:
                    await saver.setup()
                    if plan is not None:
                        # 从计划审核关卡恢复（internal/07 Phase 4）。
                        outcome = await runner.resume(
                            checkpointer=saver, session_id=session_id, plan=plan, **ports
                        )
                    elif resume is None:
                        outcome = await runner.run(
                            checkpointer=saver,
                            session_id=session_id,
                            user_id=user_id,
                            query=session.prompt,
                            depth=session.research_depth,
                            **ports,
                        )
                    else:
                        approved, feedback = resume
                        outcome = await runner.resume(
                            checkpointer=saver,
                            session_id=session_id,
                            approved=approved,
                            feedback=feedback,
                            **ports,
                        )

                # session 是执行前加载的旧对象，expire_on_commit=False 会让它一直保留旧属性。
                # 必须重读数据库才能看到执行中发生的取消，避免把已停止的会话改回待审核。
                await db.refresh(session, ["cancelled_at"])
                await _persist_outcome(
                    db, session, session_id, outcome, sink, ports["provider_keys"]
                )
        finally:
            await release_session_lock(session_id, lock_token)
    finally:
        await close_redis_pool()
        await engine.dispose()


async def _ingest_into_project_memory(db, session: Session, provider_keys: dict[str, str]) -> None:
    """把获批报告写入项目记忆（docs/14 §2）。

    只在批准后写入，让未审核草稿和被拒绝报告不会进入检索。先发布 COMPLETED 再建立索引，避免向量化耗时延迟用户看到完成。
    索引失败不向外抛出；报告已经成功提交，不能因向量服务故障倒退为失败。缺口由 memory/status 报告，并在重建索引时修复。"""
    try:
        from app import adapters
        from app.services import memory

        # 先检查主机是否支持记忆表，与 run_lifecycle.delete_run 一致。
        # 不支持时贸然写入会失败并回滚，使调用方持有的会话对象也失效，反而影响已完成运行。
        if not memory.is_available(db):
            logger.info("memory_ingest_skipped", session_id=str(session.id), reason="unavailable")
            return

        embedder = await adapters.embeddings_for(provider_keys)
        result = await memory.ingest_session(db, session, embedder)
        if result.skipped:
            logger.info("memory_ingest_skipped", session_id=str(session.id), reason=result.reason)
    except Exception as e:  # noqa: BLE001 — 索引失败不能推翻已经完成的运行
        await db.rollback()
        logger.warning(
            "memory_ingest_failed",
            session_id=str(session.id),
            project_id=str(session.project_id),
            error=str(e),
        )


async def _ingest_report_into_corpus(session: Session, provider_keys: dict[str, str]) -> None:
    """自动将获批报告存入项目语料库，规则见 app/services/report_corpus.py。

    与项目记忆使用相同触发时机，失败不影响已提交运行。写入使用部署配置的向量模型；与语料内搜索不同，摄取步骤允许远程向量化，与上传入口保持一致。"""
    try:
        from app import adapters
        from app.services.report_corpus import ingest_report

        store = await adapters.ServerCorpusLocator().ensure(session.project_id, keys=provider_keys)
        await ingest_report(store, report_id=str(session.id), report_markdown=session.final_report)
    except Exception as e:  # noqa: BLE001 — 失败处理规则见 report_corpus.ingest_report
        logger.warning(
            "report_corpus_ingest_setup_failed",
            session_id=str(session.id),
            project_id=str(session.project_id),
            error=str(e),
        )


async def _persist_outcome(
    db,
    session: Session,
    session_id: str,
    outcome: RunOutcome,
    sink,
    provider_keys: dict[str, str] | None = None,
) -> None:
    """严格按顺序执行：保存结果、提交事务、发布生命周期事件。"""
    session.total_cost_usd = outcome.cost_usd
    session.total_tokens_input = outcome.tokens_input
    session.total_tokens_output = outcome.tokens_output
    session.rework_count = outcome.rework_count

    # 用户取消后必须保持取消。当前取消不会立即中断图，在途结果可能几分钟后才到达。
    # 若不检查，就会把已取消会话改回待审核或完成，甚至把用户放弃的报告加入项目记忆。
    #
    # 已消耗的 Token 和费用仍要提交，不能因取消而丢失真实账目。
    # 状态、报告、来源与生命周期结论则不再覆盖，用户的取消决定优先。
    #
    # 此规则另有桌面 sidecar._apply_outcome 和运行 run_execution.persist_outcome 两处实现，修改时三处同步。
    if session.is_cancelled:
        await db.commit()
        logger.info(
            "outcome_discarded_session_cancelled",
            session_id=session_id,
            outcome_status=outcome.status,
            cost_usd=round(outcome.cost_usd, 4),
        )
        return

    session.sources = outcome.sources

    if outcome.status == "awaiting_plan":
        # 先保存计划提案再发布 PLAN_READY，保证客户端立即读取计划时能看到对应数据。
        # 提案还未获批，plan_approved_at 保留空值，等计划审核接口填写。
        session.status = SessionStatus.AWAITING_PLAN
        session.plan_json = {"tasks": outcome.plan_tasks}
        session.outline_json = {"sections": outcome.plan_outline}
        await db.commit()
        await sink(session_id, lifecycle_event(outcome))
        return

    if outcome.status == "awaiting_approval":
        session.status = SessionStatus.AWAITING_APPROVAL
        session.draft_report = outcome.draft_report
        await db.commit()
        await sink(session_id, lifecycle_event(outcome))
        return

    if outcome.status == "failed":
        session.status = SessionStatus.FAILED
        session.error_message = outcome.error
        await db.commit()
        await sink(session_id, lifecycle_event(outcome))
        return

    session.status = SessionStatus.COMPLETED
    session.final_report = outcome.final_report
    session.elapsed_seconds = outcome.elapsed_seconds
    # 报告定稿时计算一次，不在每次查询列表时重算；桌面对应 sidecar._apply_outcome。
    session.citation_resolution_rate = citation_rate.resolution_rate(
        outcome.final_report or "", outcome.sources
    )
    await db.commit()
    await sink(session_id, lifecycle_event(outcome))
    # 已批准研究在这里进入项目记忆（docs/14 §2）。
    await _ingest_into_project_memory(db, session, provider_keys or {})
    # 同时加入项目语料库，规则见 report_corpus.py（docs/12 M10）。
    await _ingest_report_into_corpus(session, provider_keys or {})


async def run_pipeline(session_id: str, user_id: str) -> None:
    await _execute(session_id, user_id, resume=None)


async def resume_pipeline(
    session_id: str, user_id: str, approved: bool, feedback: str | None
) -> None:
    await _execute(session_id, user_id, resume=(approved, feedback))


async def resume_plan(session_id: str, user_id: str, plan: dict) -> None:
    """用审核者的编辑内容，恢复暂停在研究计划关卡的会话。"""
    await _execute(session_id, user_id, resume=None, plan=plan)
