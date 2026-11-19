"""服务端的 Celery 运行调度实现，位于 Worker 层，不放进两个主机共用的协议模块。

所有任务导入延迟到调用时，避免桌面导入共享路由时拉入未打包的 Celery。"""

from __future__ import annotations

from app.run_dispatch import RunDispatcher, SessionDispatcher


class CeleryDispatcher:
    """通过消息代理把研究运行交给 Worker。

    任务模块带入 Celery 和整套执行依赖，必须延迟导入，保证桌面可导入共用路由。"""

    async def start(self, run_id: str, user_id: str) -> None:
        from app.workers.tasks import run_research_pipeline

        run_research_pipeline.delay(run_id, user_id)

    async def resume_plan(self, run_id: str, user_id: str, plan: dict) -> None:
        from app.workers.tasks import resume_research_plan_gate

        resume_research_plan_gate.delay(run_id, user_id, plan)

    async def rework(self, run_id: str, user_id: str, feedback: str | None) -> None:
        from app.workers.tasks import resume_research_pipeline

        resume_research_pipeline.delay(run_id, user_id, False, feedback)


def get_run_dispatcher() -> RunDispatcher:
    """FastAPI 注入当前部署的运行调度机制。

    桌面直接传自己的 dispatcher，与 db、user 参数一样，处理函数本身不区分主机。"""
    return CeleryDispatcher()


class CelerySessionDispatcher:
    """服务端会话调度，和 CeleryDispatcher 一样延迟导入任务。"""

    async def start(self, session_id: str, user_id: str) -> None:
        from app.workers.tasks import run_agent_pipeline

        run_agent_pipeline.delay(session_id, user_id)

    async def resume_plan(self, session_id: str, user_id: str, plan: dict) -> None:
        from app.workers.tasks import resume_plan_gate

        resume_plan_gate.delay(session_id, user_id, plan)

    async def resume_review(
        self, session_id: str, user_id: str, approved: bool, feedback: str | None
    ) -> None:
        from app.workers.tasks import resume_agent_pipeline

        resume_agent_pipeline.delay(session_id, user_id, approved, feedback)


def get_session_dispatcher() -> SessionDispatcher:
    """为 FastAPI 注入当前部署的会话驱动方式。"""
    return CelerySessionDispatcher()
