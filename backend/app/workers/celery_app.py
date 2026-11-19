"""Celery 后台任务应用的配置（docs/02 §6）。"""

import structlog
from celery import Celery
from celery.signals import worker_process_init, worker_process_shutdown

from app.config import settings
from app.logconfig import configure_logging
from app.runtime import install_process_default

logger = structlog.get_logger(__name__)

# Worker 不会导入 app.main，因此也要在这里配置 structlog。
# 与 API 共用上下文合并规则，才能用 correlation_id 串起 API 到 Celery 的日志。
configure_logging(json_output=settings.is_production)

# 研究引擎不读取 app.config；在导入图和处理任务之前，由 Worker 安装基础运行配置（docs/13 §2）。
install_process_default()

celery_app = Celery(
    "research_worker",
    broker=settings.redis_url,
    backend=settings.redis_url,
    include=["app.workers.tasks"],
)

# 硬超时来自配置；软超时由它推算，避免调整后出现软时限晚于硬时限。
_HARD = settings.celery_task_timeout_seconds
_SOFT = max(60, _HARD - 60)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    task_ignore_result=True,
    # 研究流程不能重复执行而保证结果不变，自动重发可能再次收费。
    # 崩溃后必须明确从 LangGraph 检查点恢复，不能让消息代理自动重投（docs/02 §6）。
    task_acks_late=False,
    worker_prefetch_multiplier=1,
    task_soft_time_limit=_SOFT,
    task_time_limit=_HARD,
)


def preload_pipeline_imports() -> None:
    """启动时预先导入 LangGraph、研究图和检查点组件，避免首次任务才承担导入开销。

    CI 曾测得相同输入第一次耗时 35.3 秒、第二次 0.26 秒，主要差别是导入。进程能响应 celery inspect ping，也不代表这些组件已就绪，用户会看到任务长时间没有进展。
    预加载失败只记日志，不阻止 Worker 启动，让实际任务带着自己的上下文报告错误。每个子进程都执行；重复导入由 sys.modules 缓存。"""
    try:
        from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver  # noqa: F401

        import research_engine.graph  # noqa: F401 — 预先加载 LangGraph 和模型供应商客户端
    except Exception as e:  # noqa: BLE001 — 预加载失败不能阻止 Worker 启动
        logger.warning("worker_preload_failed", error=str(e))
    else:
        logger.info("worker_preload_complete")


@worker_process_init.connect
def _preload_on_worker_boot(**_: object) -> None:
    """每个 fork 子进程独立预加载；prefork 使用写时复制，但导入仍各自执行。"""
    preload_pipeline_imports()


@worker_process_shutdown.connect
def _close_worker_loop(**_: object) -> None:
    """子进程退出时，关闭它唯一的事件循环 app.workers.event_loop。"""
    from app.workers import event_loop

    event_loop.shutdown()
