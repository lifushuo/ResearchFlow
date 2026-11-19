"""供 FastAPI 注入服务端 app.ports 实现的工厂（docs/13 §5.2）。

路由会在桌面请求时导入，因此这里不能在模块顶部导入 adapters，否则会连带引入桌面未打包的 Redis 和配置。
服务端实现都延迟到函数调用时导入；桌面直接传自己的端口，因此不会执行这些导入。"""

from __future__ import annotations

from app.ports import CheckpointDeleter, CorpusLocator, TerminalEventEmitter


def get_corpus_locator() -> CorpusLocator:
    """注入部署的语料定位器。

    避免在共用路由里直接 new ServerCorpusLocator，让桌面可传自己的布局实现，路由无需辨认主机。"""
    from app.adapters import ServerCorpusLocator

    return ServerCorpusLocator()


def get_checkpoint_deleter() -> CheckpointDeleter:
    """注入部署的检查点删除方式。

    checkpoints.delete_thread 已匹配异步协议，直接作为实现，不再包装多余类。"""
    from app.services import checkpoints

    return checkpoints.delete_thread


def get_terminal_event_emitter() -> TerminalEventEmitter:
    """注入图执行之外的单条事件写入与发布方式。

    复用 agent_log_sink 的先提交再发布及附带 ID 规则，不再复制消息顺序。"""
    from app.adapters import agent_log_sink

    async def _emit(db, session_id: str, event: dict) -> None:
        await agent_log_sink(db, session_id)(session_id, event)

    return _emit
