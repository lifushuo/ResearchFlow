"""引擎端口，见 docs/13 §4、docs/12 M6。

只定义协议，由宿主提供数据库、缓存、事件和嵌入实现，使同一张图可在服务端和桌面运行。
密钥在运行前作为普通映射传入，无需多包一层查找接口；
运行锁属于宿主调度，不属于图；检查点则直接采用 LangGraph saver 接口。"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class EventSink(Protocol):
    """接收一个流水线事件，通过 events.set_emitter 安装。

    服务端落库并发布 Redis；桌面可在进程内转发。"""

    async def __call__(self, session_id: str, event: dict) -> None: ...


@runtime_checkable
class Cache(Protocol):
    """尽力而为的搜索键值缓存，通过 cache.set_cache 安装。

    允许失败或不缓存，检索不能因缓存故障被中断。"""

    async def get(self, key: str) -> str | None: ...

    async def set(self, key: str, value: str, ttl: int) -> None: ...


@runtime_checkable
class Embeddings(Protocol):
    """把文本转成项目记忆向量，见 docs/14 §4。

    引擎只规定协议，宿主选择客户端。model_id 和 dimensions 都是契约：
    不同模型的向量即使维数相同也不可混排，存储和检索必须按模型隔离。"""

    @property
    def model_id(self) -> str: ...

    @property
    def dimensions(self) -> int: ...

    async def embed(self, texts: list[str]) -> list[list[float]]: ...


@runtime_checkable
class Corpus(Protocol):
    """作为检索来源的封闭文档库，见 internal/12 M10。

    search 返回 title、url、snippet，read 将 URL 还原到精确文档位置，保持执行器无需分支。
    由 corpus.set_corpus 安装；corpus_mode 启用后仅使用此端口并拒绝普通网页地址。
    该限制控制检索来源，代理模型是否外发另由路由决定。"""

    async def search(self, query: str, max_results: int) -> list[dict]: ...

    async def read(self, url: str) -> dict: ...
