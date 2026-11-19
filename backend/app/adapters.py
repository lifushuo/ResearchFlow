"""服务端对研究引擎端口的实现（docs/13 §4）。

引擎在 ports.py 定义接口，本模块提供 Postgres、Redis 和项目记忆向量化适配；桌面端改用 SQLite 与进程内实现，引擎无需感知。采用结构化协议匹配而非继承，由运行器测试验证。
与仅负责配置映射的 app/runtime.py 分开。"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from pathlib import Path  # noqa: F401 — 类型注解需要此导入

import httpx
import structlog

from app.config import settings
from app.db.redis import publish_event
from app.models.agent_log import AgentLog
from research_engine.embeddings import (
    EmbeddingsUnavailable,
    FakeEmbeddings,
    NoEmbeddings,
    is_local_endpoint,
)
from research_engine.llm_factory import map_local_host
from research_engine.ports import Embeddings, EventSink

logger = structlog.get_logger()


class RedisCache:
    """基于共用 Redis 池缓存搜索结果。

    缓存只是优化，检索器会处理其异常；Redis 不可用时应退化为无缓存，不能让研究失败。"""

    async def get(self, key: str) -> str | None:
        from app.db.redis import get_redis

        return await get_redis().get(key)

    async def set(self, key: str, value: str, ttl: int) -> None:
        from app.db.redis import get_redis

        await get_redis().set(key, value, ex=ttl)


def agent_log_sink(db, session_id: str) -> EventSink:
    """先将事件提交到 agent_logs，再发布到 Redis，并携带数据库事件 ID。

    客户端凭 Last-Event-ID 从持久记录恢复，避免断线丢事件（docs/05 §4）。整个运行使用调用方提供的数据库会话。
    AsyncSession 不能并发使用，因此每次运行用锁串行完成写入和发布；锁也覆盖发布，确保 Redis 顺序与提交顺序一致，防止重连时游标跨过尚未收到的旧事件。"""
    session_uuid = uuid.UUID(session_id)
    write_lock = asyncio.Lock()

    async def sink(sid: str, event: dict) -> None:  # noqa: ARG001 - 保留端口要求的参数形式
        row = AgentLog(
            session_id=session_uuid,
            event_type=event.get("type", "agent_log"),
            agent_name=event.get("agent"),
            payload=event,
        )
        async with write_lock:
            db.add(row)
            await db.flush()
            event["id"] = row.id
            await db.commit()
            await publish_event(session_id, event)

    return sink


async def redis_event_stream(pubsub):
    """将服务端实时消息转换成 sse_frames 需要的 ID、负载对。

    无法解析的消息跳过，不应因某个发布者发错内容而关闭全部监听连接。"""
    async for message in pubsub.listen():
        if message["type"] != "message":
            continue
        try:
            payload = json.loads(message["data"])
        except json.JSONDecodeError, TypeError:
            continue
        yield payload.get("id"), payload


class PgVectorMemoryIndex:
    """在 memory_chunks 上实现 MemoryIndex，使用 pgvector 余弦距离。

    把数据库方言相关表达式集中在适配层，避免共用 memory 服务和桌面端依赖 Postgres。"""

    available = True

    async def nearest(self, db, *, project_id, query_vector, embedding_model, limit):
        from sqlalchemy import select

        from app.models.memory_chunk import MemoryChunk
        from app.services.memory import MAX_COSINE_DISTANCE

        distance = MemoryChunk.embedding.cosine_distance(query_vector).label("distance")
        rows = (
            await db.execute(
                select(MemoryChunk, distance)
                .where(
                    MemoryChunk.project_id == project_id,
                    MemoryChunk.embedding_model == embedding_model,
                    distance <= MAX_COSINE_DISTANCE,
                )
                .order_by(distance)
                .limit(limit)
            )
        ).all()
        return [(chunk, float(dist)) for chunk, dist in rows]


class ServerCorpusLocator:
    """服务端语料定位器：每个项目在 corpus_path 下有独立 SQLite 文件。

    必须使用已解析的 corpus_path，不能使用可能相对工作目录的 corpus_dir；API 与 Worker 工作目录不同，否则会出现上传成功但研究找不到文档。"""

    def __init__(self, embedder_factory=None) -> None:
        # 通过注入向量模型，让测试不访问真实供应商，也便于两种主机使用相同模型验证契约。
        self._embedder_factory = embedder_factory or embeddings_for

    def _root(self):
        from app.config import settings

        return settings.corpus_path

    def path_for(self, project_id) -> Path:
        return self._root() / f"corpus_{project_id}.sqlite"

    def paths_to_delete(self, project_id) -> list[Path]:
        stem = self.path_for(project_id)
        return [
            stem.with_name(stem.name.replace(".sqlite", suffix))
            for suffix in (".sqlite", ".sqlite-wal", ".sqlite-shm")
        ]

    async def _open(self, project_id, keys):
        from research_engine.corpus import CorpusStore

        factory = self._embedder_factory
        embedder = factory(keys)
        if hasattr(embedder, "__await__"):
            embedder = await embedder
        return CorpusStore(self.path_for(project_id), embedder)

    async def for_project(self, project_id, *, keys=None):
        return (
            None if not self.path_for(project_id).exists() else await self._open(project_id, keys)
        )

    async def ensure(self, project_id, *, keys=None):
        self._root().mkdir(parents=True, exist_ok=True)
        return await self._open(project_id, keys)

    def delete(self, project_id) -> None:
        """尽力删除语料；语料文件清理失败不能推翻项目删除。"""
        for path in self.paths_to_delete(project_id):
            try:
                path.unlink(missing_ok=True)
            except OSError as e:  # noqa: PERF203
                logger.warning("corpus_cleanup_failed", project_id=str(project_id), error=str(e))


# 向量化适配（docs/14 §4）
#
# memory_chunks.embedding 的固定维度。所有供应商按此维度输出，切换时只需重建索引，不必迁移列。
# 维度相同不代表向量含义相同，因此每条记录还要保存生成它的模型。
EMBEDDING_DIMENSIONS = 768

_DEFAULT_EMBEDDING_MODELS = {
    "ollama": "nomic-embed-text",
    "google": "text-embedding-004",
    "openai": "text-embedding-3-small",
}

# 本地向量化通常很快，但冷启动要加载模型；首次调用需留出时间，避免被当成聊天卡死。
_EMBED_TIMEOUT_SECONDS = 60.0

# 短暂缓存 Ollama 可用性探测；本地服务没启动时，不必让每条聊天消息都等一次探测超时。
_PROBE_CACHE_SECONDS = 300.0
_probe_cache: tuple[float, str | None] | None = None


def _check_width(vectors: list[list[float]], model_id: str) -> list[list[float]]:
    """提前拒绝维度不符合记忆列要求的向量，并报告模型及预期、实际维度。

    数据库最终也会拒绝，但在适配层检查能更早指出配置问题，避免只看到模糊的写入失败。"""
    for vector in vectors:
        if len(vector) != EMBEDDING_DIMENSIONS:
            raise EmbeddingsUnavailable(
                f"Embedding model '{model_id}' returned {len(vector)}-dimensional "
                f"vectors, but project memory stores {EMBEDDING_DIMENSIONS}. Configure "
                f"EMBEDDINGS_MODEL to a {EMBEDDING_DIMENSIONS}-dimensional model, or "
                f"migrate the column and re-index everything."
            )
    return vectors


class OllamaEmbeddings:
    """通过 Ollama 的 OpenAI 兼容接口获取本地向量。

    协议简单，直接使用 httpx，避免为此引入完整模型客户端。"""

    def __init__(self, model: str, base_url: str) -> None:
        self._model = model
        # 独立 httpx 向量客户端也要复用 localhost→host.docker.internal 的容器地址映射。
        # 它不经过 llm_factory，不能假定已自动映射；否则聊天正常，语料上传和记忆写入却会连接到容器自身而失败。
        self._base_url = map_local_host(base_url.rstrip("/"))

    @property
    def is_local(self) -> bool:
        # OLLAMA_BASE_URL 可配置为远程地址，是否本地必须看实际端点（docs/12 M10）。
        return is_local_endpoint(self._base_url)

    @property
    def model_id(self) -> str:
        return f"ollama:{self._model}"

    @property
    def dimensions(self) -> int:
        return EMBEDDING_DIMENSIONS

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        try:
            async with httpx.AsyncClient(timeout=_EMBED_TIMEOUT_SECONDS) as client:
                response = await client.post(
                    f"{self._base_url}/embeddings",
                    json={"model": self._model, "input": texts},
                )
                response.raise_for_status()
                payload = response.json()
        except Exception as exc:  # noqa: BLE001 — 这里任何请求失败都表示本次无法生成向量
            raise EmbeddingsUnavailable(
                f"Local embedding server at {self._base_url} did not answer: {exc}. "
                f"Check that Ollama is running and `ollama pull {self._model}` has been run."
            ) from exc

        # 按照响应 index 对齐，不按返回先后配对；错配向量会悄悄损坏检索结果。
        rows = sorted(payload.get("data") or [], key=lambda row: row.get("index", 0))
        vectors = [row.get("embedding") or [] for row in rows]
        if len(vectors) != len(texts):
            raise EmbeddingsUnavailable(
                f"Embedding server returned {len(vectors)} vectors for {len(texts)} inputs."
            )
        return _check_width(vectors, self.model_id)


class HostedEmbeddings:
    """使用用户自带密钥或部署密钥调用远程向量模型。

    包装 LangChain 客户端，复用供应商 SDK 的批量限制和重试机制。"""

    def __init__(self, provider: str, model: str, api_key: str) -> None:
        self._provider = provider
        self._model = model
        self._api_key = api_key

    # 远程向量化必然外发数据。
    # 仅语料库模式即使不搜索网页，查询向量也可能访问 google/openai，所以隔离模式必须直接拒绝此适配器。
    is_local = False

    @property
    def model_id(self) -> str:
        return f"{self._provider}:{self._model}"

    @property
    def dimensions(self) -> int:
        return EMBEDDING_DIMENSIONS

    def _client(self):
        if self._provider == "google":
            from langchain_google_genai import GoogleGenerativeAIEmbeddings

            # Google 模型名称可能带 models/ 前缀，两种写法都接受，兼容供应商文档中的配置形式。
            name = self._model if self._model.startswith("models/") else f"models/{self._model}"
            return GoogleGenerativeAIEmbeddings(model=name, google_api_key=self._api_key)

        from langchain_openai import OpenAIEmbeddings

        # text-embedding-3 系列支持维度截取，统一输出存储列所需维度，无需为它单独扩列。
        return OpenAIEmbeddings(
            model=self._model, api_key=self._api_key, dimensions=EMBEDDING_DIMENSIONS
        )

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        try:
            vectors = await self._client().aembed_documents(texts)
        except Exception as exc:  # noqa: BLE001
            raise EmbeddingsUnavailable(
                f"Embedding provider '{self._provider}' failed: {exc}"
            ) from exc
        return _check_width(vectors, self.model_id)


async def _local_embedding_model() -> str | None:
    """探测本地 Ollama 是否安装向量模型。

    结果短暂缓存；没有本地服务的部署不应在每轮聊天前都承担完整探测超时。"""
    global _probe_cache
    now = time.monotonic()
    if _probe_cache is not None and _probe_cache[0] > now:
        return _probe_cache[1]

    from app.services import local_llm

    status = await local_llm.probe()
    configured = settings.embeddings_model
    installed = [m.name for m in status.models if m.is_embedding]
    # 明确配置的模型确实存在时优先使用，否则取服务端报告的第一个向量模型。
    chosen = None
    if status.reachable:
        if configured and any(
            name.split(":", 1)[0] == configured.split(":", 1)[0] for name in installed
        ):
            chosen = configured
        elif installed:
            chosen = installed[0]

    _probe_cache = (now + _PROBE_CACHE_SECONDS, chosen)
    return chosen


def reset_embeddings_probe_cache() -> None:
    """清空 Ollama 探测缓存，用于测试或配置变更。"""
    global _probe_cache
    _probe_cache = None


async def embeddings_for(provider_keys: dict[str, str] | None = None) -> Embeddings:
    """构建部署的向量模型，优先使用调用方的自带密钥。

    provider_keys 与研究流程格式一致，保证用户的记忆向量化也使用其自己的密钥。无可用配置时返回 NoEmbeddings，是否终止由调用方决定。"""
    keys = provider_keys or {}
    choice = settings.embeddings_provider

    # 先检查 fake 模式，再检查供应商配置；演示承诺适用于整次运行（docs/17 §6.2）。
    # 否则即使研究使用脚本模型，上传向量化仍可能读取 .env 的真密钥并收费。
    if settings.llm_mode == "fake":
        return FakeEmbeddings()

    if choice == "none":
        return NoEmbeddings()

    def hosted(provider: str) -> Embeddings | None:
        key = keys.get(provider) or getattr(settings, f"{provider}_api_key", "")
        if not key:
            return None
        model = settings.embeddings_model or _DEFAULT_EMBEDDING_MODELS[provider]
        return HostedEmbeddings(provider, model, key)

    if choice == "ollama":
        model = await _local_embedding_model() or (
            settings.embeddings_model or _DEFAULT_EMBEDDING_MODELS["ollama"]
        )
        return OllamaEmbeddings(model, settings.ollama_base_url)

    if choice in ("google", "openai"):
        return hosted(choice) or NoEmbeddings()

    # 自动选择时优先本地，再选择用户或部署配置了密钥的远程供应商。
    local = await _local_embedding_model()
    if local:
        return OllamaEmbeddings(local, settings.ollama_base_url)
    for provider in ("google", "openai"):
        built = hosted(provider)
        if built is not None:
            return built
    return NoEmbeddings()
