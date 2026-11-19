"""Celery 对异步研究引擎的轻量任务封装。

调用 app.workers.pipeline_runner；意外失败时标记会话失败。不自动重试整套研究，因为重复执行会重复消耗模型费用；崩溃必须明确从检查点恢复。"""

from __future__ import annotations

import structlog

from app.logconfig import bind_research_run_context, bind_session_context, clear_run_context
from app.workers import event_loop
from app.workers.celery_app import celery_app

logger = structlog.get_logger()


@celery_app.task(name="run_agent_pipeline")
def run_agent_pipeline(session_id: str, user_id: str) -> None:
    from app.workers.pipeline_runner import run_pipeline

    # 先清空旧上下文：Worker 是长生命周期进程，上一轮身份不能泄漏到下一轮日志。
    # 本次 correlation_id 等于 session_id，并随任务和引擎日志传递。
    clear_run_context()
    bind_session_context(session_id, user_id=user_id)
    logger.info("pipeline_task_started")
    try:
        event_loop.run(run_pipeline(session_id, user_id))
        logger.info("pipeline_task_finished")
    except Exception as exc:  # noqa: BLE001
        logger.error("pipeline_task_failed", error=str(exc), exc_info=True)
        event_loop.run(_mark_failed(session_id, str(exc)))


@celery_app.task(name="resume_agent_pipeline")
def resume_agent_pipeline(
    session_id: str, user_id: str, approved: bool, feedback: str | None = None
) -> None:
    from app.workers.pipeline_runner import resume_pipeline

    clear_run_context()
    bind_session_context(session_id, user_id=user_id, approved=approved)
    logger.info("resume_task_started")
    try:
        event_loop.run(resume_pipeline(session_id, user_id, approved, feedback))
        logger.info("resume_task_finished")
    except Exception as exc:  # noqa: BLE001
        logger.error("resume_task_failed", error=str(exc), exc_info=True)
        event_loop.run(_mark_failed(session_id, str(exc)))


@celery_app.task(name="resume_plan_gate")
def resume_plan_gate(session_id: str, user_id: str, plan: dict) -> None:
    """恢复暂停在研究计划关卡的会话（internal/07 Phase 4）。

    单独设任务，不复用报告关卡恢复入口：两者负载和中断类型不同，不能猜测。排队消息可能跨部署存在，随意扩展旧任务参数也会使旧消息产生歧义。"""
    from app.workers.pipeline_runner import resume_plan

    clear_run_context()
    bind_session_context(session_id, user_id=user_id)
    logger.info("plan_resume_task_started", task_count=len((plan or {}).get("tasks") or []))
    try:
        event_loop.run(resume_plan(session_id, user_id, plan))
        logger.info("plan_resume_task_finished")
    except Exception as exc:  # noqa: BLE001
        logger.error("plan_resume_task_failed", error=str(exc), exc_info=True)
        event_loop.run(_mark_failed(session_id, str(exc)))


async def _mark_failed(session_id: str, error: str) -> None:
    import uuid

    from sqlalchemy import select

    from app.db.base import AsyncSessionLocal, engine
    from app.db.redis import close_redis_pool, init_redis_pool, publish_event
    from app.models.session import Session, SessionStatus

    await init_redis_pool()
    try:
        async with AsyncSessionLocal() as db:
            session = (
                await db.execute(select(Session).where(Session.id == uuid.UUID(session_id)))
            ).scalar_one_or_none()
            if session:
                session.status = SessionStatus.FAILED
                session.error_message = error[:500]
                await db.commit()
        await publish_event(session_id, {"type": "FAILED", "data": {"reason": error[:500]}})
    finally:
        await close_redis_pool()
        await engine.dispose()


# 研究运行任务
#
# 研究运行与会话使用不同任务名；排队消息可能跨部署保留，不能通过修改旧任务含义来混用两个领域。
#
# name= 的字符串是消息协议，不跟随 Python 函数名随意改动。
# 消息代理中的旧消息已经保存该名称，改名会使升级时队列中的任务失去处理入口；与 Alembic revision ID 一样，它属于已部署状态。


@celery_app.task(name="run_research_pipeline")
def run_research_pipeline(run_id: str, user_id: str) -> None:
    from app.run_execution import execute_run

    clear_run_context()
    bind_research_run_context(run_id, user_id=user_id)
    logger.info("run_pipeline_task_started")
    try:
        event_loop.run(execute_run(run_id))
        logger.info("run_pipeline_task_finished")
    except Exception as exc:  # noqa: BLE001
        logger.error("run_pipeline_task_failed", error=str(exc), exc_info=True)
        event_loop.run(_mark_run_failed(run_id, str(exc)))


@celery_app.task(name="resume_research_pipeline")
def resume_research_pipeline(
    run_id: str, user_id: str, approved: bool, feedback: str | None = None
) -> None:
    from app.run_execution import execute_run

    clear_run_context()
    bind_research_run_context(run_id, user_id=user_id, approved=approved)
    logger.info("run_resume_task_started")
    try:
        event_loop.run(execute_run(run_id, resume=(approved, feedback)))
        logger.info("run_resume_task_finished")
    except Exception as exc:  # noqa: BLE001
        logger.error("run_resume_task_failed", error=str(exc), exc_info=True)
        event_loop.run(_mark_run_failed(run_id, str(exc)))


@celery_app.task(name="resume_research_plan_gate")
def resume_research_plan_gate(run_id: str, user_id: str, plan: dict) -> None:
    from app.run_execution import execute_run

    clear_run_context()
    bind_research_run_context(run_id, user_id=user_id)
    logger.info("run_plan_resume_task_started")
    try:
        event_loop.run(execute_run(run_id, plan=plan))
        logger.info("run_plan_resume_task_finished")
    except Exception as exc:  # noqa: BLE001
        logger.error("run_plan_resume_task_failed", error=str(exc), exc_info=True)
        event_loop.run(_mark_run_failed(run_id, str(exc)))


async def _mark_run_failed(run_id: str, error: str) -> None:
    """即使适配器外崩溃，也要把运行标为 FAILED，不能永远停在 RUNNING。

    与会话 _mark_failed 一样，使用独立会话管理数据库和 Redis，并自行处理异常，避免记录失败时又报错，丢掉最初故障。"""
    import uuid

    from sqlalchemy import select

    from app import run_lifecycle
    from app.db.base import AsyncSessionLocal, engine
    from app.db.redis import close_redis_pool, init_redis_pool, publish_event
    from app.models.research import ResearchRun

    await init_redis_pool()
    try:
        async with AsyncSessionLocal() as db:
            run = (
                await db.execute(select(ResearchRun).where(ResearchRun.id == uuid.UUID(run_id)))
            ).scalar_one_or_none()
            if run:
                # 通过 record_failure 统一处理，不在这里直接写状态。
                # 它会保留已取消运行的终态；否则写 FAILED 会违反 ck_run_cancelled，使本应不抛异常的失败处理器再次报错。
                await run_lifecycle.record_failure(db, run, error[:500])
                await db.commit()
        await publish_event(run_id, {"type": "FAILED", "data": {"reason": error[:500]}})
    finally:
        await close_redis_pool()
        await engine.dispose()
