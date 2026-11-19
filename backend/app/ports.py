"""主机间确实不同的基础设施端口，只声明协议，不导入任何实现。

共用 ORM 的持久化规则无需再包装一层仓储；需要适配的是消息流、配置来源、密钥存储、语料位置、路由存储及记忆能力。每个端口都要明确生命周期、错误、事务和并发契约。"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Protocol, runtime_checkable

from research_engine.ports import Corpus


@runtime_checkable
class CorpusLocator(Protocol):
    """定位并打开项目语料。

    文件布局属于主机实现，不应在各路由复制路径规则。for_project 在无语料时返回 None，ensure 则用于确实需要创建的摄取操作。
    keys 是请求级向量模型密钥，不固定在定位器上，避免并发用户串用。"""

    async def for_project(
        self, project_id: uuid.UUID, *, keys: dict[str, str] | None = None
    ) -> Corpus | None: ...

    async def ensure(
        self, project_id: uuid.UUID, *, keys: dict[str, str] | None = None
    ) -> Corpus: ...

    def paths_to_delete(self, project_id: uuid.UUID) -> list[Path]:
        """列出语料库的全部组成文件，包括 SQLite 的 WAL、SHM。

        只删 .sqlite 可能让未合并日志中的文档和向量残留。"""
        ...

    def delete(self, project_id: uuid.UUID) -> None: ...


@runtime_checkable
class MemoryIndex(Protocol):
    """项目记忆近邻检索端口。

    服务端使用 pgvector，桌面按设计没有此能力；切块、索引状态等共用业务规则仍放在 memory.py。
    available 由端口统一报告，不由调用方猜数据库方言。无能力时抛 CapabilityUnavailable，不能返回空列表冒充已检索但无结果。"""

    @property
    def available(self) -> bool: ...

    async def nearest(
        self,
        db,
        *,
        project_id: uuid.UUID,
        query_vector: list[float],
        embedding_model: str,
        limit: int,
    ) -> list[tuple[object, float]]:
        """返回按距离排序的片段与距离对。

        必须同时过滤 project_id 与 embedding_model，既防跨项目泄露，也防把不同模型的不可比向量混排。"""
        ...


@runtime_checkable
class CheckpointDeleter(Protocol):
    """清除一个会话或运行线程的全部检查点。

    服务端每次新建 Postgres 保存器连接，桌面复用长生命周期 SQLite 保存器，以同一异步可调用协议适配。
    两端实现都按约定记录清理失败并继续，因为业务行已删除，不能为清理失败倒退删除结果。"""

    async def __call__(self, thread_id: str) -> None: ...


@runtime_checkable
class TerminalEventEmitter(Protocol):
    """用于图外结束会话的操作：先持久化一条事件，再实时发布。

    图内 EventSink 绑定整轮执行，这个端口只服务取消等单次操作。服务端复用传入数据库会话，桌面可用会话工厂自行开作用域；保留 db 参数以兼容两端机制。"""

    async def __call__(self, db, session_id: str, event: dict) -> None: ...
