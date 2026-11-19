"""API 与 Celery 共用的结构化日志配置。

把工作单元身份绑定到 contextvars，让请求、后台任务和引擎日志通过同一 correlation_id 串联。
correlation_id 是关联键；session_id 只能指 sessions.id，run_id 只能指 research_runs.id。分别提供绑定函数，避免把运行冒记为会话。已有 ID 可跨任务传递，无需新造关联标识。"""

from __future__ import annotations

import structlog


def configure_logging(*, json_output: bool) -> None:
    """安装进程共用日志配置，重复调用结果一致。

    merge_contextvars 将入口绑定的身份自动加入后续日志；Worker 为每个任务复制独立上下文，避免串用。"""
    renderer = (
        structlog.processors.JSONRenderer() if json_output else structlog.dev.ConsoleRenderer()
    )
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            renderer,
        ],
        cache_logger_on_first_use=True,
    )


def bind_session_context(session_id: str, **extra) -> None:
    """绑定会话日志身份；在请求触发点和 Celery 入口调用。

    后续日志自动携带等于 session_id 的 correlation_id，无需层层传参。"""
    structlog.contextvars.bind_contextvars(
        correlation_id=session_id, session_id=session_id, **extra
    )


def bind_research_run_context(run_id: str, **extra) -> None:
    """绑定研究运行日志身份，字段必须是 run_id，不能写成 session_id。

    否则日志会声称运行属于一个不存在的会话，既无法关联又会误导排查。"""
    structlog.contextvars.bind_contextvars(correlation_id=run_id, run_id=run_id, **extra)


def clear_run_context() -> None:
    """清空绑定上下文；Celery 每个任务开始时调用，避免长生命周期进程沿用上一个任务的日志身份。"""
    structlog.contextvars.clear_contextvars()
