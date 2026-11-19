"""Embeddings 端口的保守默认实现，见 docs/14 §4。

端口定义在 ports 中，这里提供错误类型和未配置时明确拒绝的实现。
调用方直接传入适配器，不另设没有必要的全局 ContextVar。
缺少缓存可以重新抓取，但缺少嵌入若静默返回空向量，会让导入和检索无声失效，
最后误导聊天认为项目没有知识，因此必须在调用处明确报错。"""

from __future__ import annotations

import hashlib
import math

import httpx

_EMBED_TIMEOUT_SECONDS = 60.0

# 本机地址集合包含 host.docker.internal，因为容器内 localhost 会经 map_local_host 改写为该地址。
_LOCAL_HOSTS = frozenset(
    {"localhost", "127.0.0.1", "::1", "[::1]", "0.0.0.0", "host.docker.internal"}
)


def is_local_endpoint(base_url: str) -> bool:
    """判断 base_url 是否指向本机。

    仅资料库模式的查询嵌入必须保持本地；代理自身的模型调用由另一套路由配置控制。
    两个宿主的所有嵌入适配器共用此判断，避免各自实现后不一致。"""
    from urllib.parse import urlsplit

    try:
        host = urlsplit(base_url).hostname or ""
    except ValueError:  # URL 格式错误按远程处理，不能因解析失败绕过外发限制。
        return False
    return host.lower() in _LOCAL_HOSTS


class EmbeddingsUnavailable(RuntimeError):
    """嵌入提供方未配置或无法访问。

    调用方决定后果：导入记录未索引，但不回溯否定已完成研究；
    聊天则拒绝回答，不能把嵌入故障说成项目没有相关知识。"""


class NoEmbeddings:
    """没有配置嵌入提供方时使用，调用会明确失败。"""

    model_id = "none"
    dimensions = 0
    # 在任何 I/O 前就抛错，不可能外发；显式声明本地，避免隔离保护错误拦截。
    is_local = True

    async def embed(self, texts: list[str]) -> list[list[float]]:  # noqa: ARG002
        raise EmbeddingsUnavailable(
            "No embeddings provider is configured, so project memory cannot be read or "
            "written. Install Ollama and pull an embedding model "
            "(`ollama pull nomic-embed-text`), or set EMBEDDINGS_PROVIDER together with "
            "that provider's API key."
        )


class FakeEmbeddings:
    """fake 模式的进程内确定性嵌入，见 docs/17 §6.2。

    演示承诺无需密钥、无需网络；嵌入也必须使用假实现，避免宿主适配器意外调用付费 API。
    同一文本始终产生同一向量，跨重启结果稳定。
    这些向量不具备语义理解，只用于走通真实导入与搜索流程，不能证明检索相关性。"""

    #: local: 标明进程内执行，fake 标明实际使用脚本实现，不能冒充真实模型。
    model_id = "local:fake-deterministic"
    #: 使用 768 维以匹配 memory_chunks.embedding；否则资料库可写入，项目记忆却会静默索引失败。
    dimensions = 768
    #: 不进行任何 I/O，显式声明本地，避免被默认按远程处理的隔离保护误拦。
    is_local = True

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(t) for t in texts]

    def _vector(self, text: str) -> list[float]:
        # 对文本按各维索引计算 blake2b，扩展成 dimensions 个浮点数，不受单个摘要长度限制。
        raw = [
            int.from_bytes(hashlib.blake2b(f"{i}:{text}".encode(), digest_size=4).digest(), "big")
            for i in range(self.dimensions)
        ]
        # 以零为中心，让向量既可相近也可相反；全正数会使余弦相似度普遍偏高。
        centred = [v / 2**31 - 1.0 for v in raw]
        norm = math.sqrt(sum(x * x for x in centred)) or 1.0
        return [x / norm for x in centred]


class LocalEmbeddings:
    """本地 OpenAI 兼容服务的嵌入客户端。

    不导入服务端适配器，避免把 app.config 和 Redis 带入桌面包。
    使用简单 httpx 协议，保持与服务端相同的返回顺序保证。"""

    def __init__(self, model: str, base_url: str, dimensions: int = 768) -> None:
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._dimensions = dimensions

    @property
    def is_local(self) -> bool:
        # 名称虽是 Local，配置仍可能指向远程机器，必须检查实际地址。
        return is_local_endpoint(self._base_url)

    @property
    def model_id(self) -> str:
        # 与服务端 OllamaEmbeddings 共用模型 ID 格式，保证跨宿主读取时能确认同一嵌入模型。
        return f"ollama:{self._model}"

    @property
    def dimensions(self) -> int:
        return self._dimensions

    async def embed(self, texts: list[str]) -> list[list[float]]:
        from research_engine.eval_trace import trace_span

        with trace_span(
            "embedding.local", kind="embedding", inputs={"texts": texts, "model": self.model_id}
        ) as span:
            result = await self._embed(texts)
            span.observe(
                output={
                    "vector_count": len(result),
                    "dimensions": self._dimensions,
                    "vectors_omitted": "diagnostic content excludes numerical embedding arrays",
                }
            )
            return result

    async def _embed(self, texts: list[str]) -> list[list[float]]:
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
        except Exception as exc:  # noqa: BLE001 — 各类失败均表示本次无法生成嵌入
            raise EmbeddingsUnavailable(
                f"Local embedding server at {self._base_url} did not answer: {exc}. "
                f"Check that Ollama is running and `ollama pull {self._model}` has been run."
            ) from exc

        # 按 index 排序，不能依赖到达顺序；文本与向量错配会静默破坏检索。
        rows = sorted(payload.get("data") or [], key=lambda row: row.get("index", 0))
        vectors = [row.get("embedding") or [] for row in rows]
        if len(vectors) != len(texts):
            raise EmbeddingsUnavailable(
                f"Embedding server returned {len(vectors)} vectors for {len(texts)} inputs."
            )
        for vector in vectors:
            if len(vector) != self._dimensions:
                raise EmbeddingsUnavailable(
                    f"Embedding model '{self._model}' returned {len(vector)}-dimensional "
                    f"vectors, but this store is configured for {self._dimensions}."
                )
        return vectors
