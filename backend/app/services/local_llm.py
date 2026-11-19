"""发现本地 Ollama 服务及已安装模型（docs/12 M15）。

无需密钥不代表服务正在运行，探测让界面在研究开始前得知真实可用性。直接用 httpx 调 REST，不创建模型客户端，也不消耗模型 Token。"""

from __future__ import annotations

import json
import re
import shutil
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import httpx
import structlog

# Ollama 默认地址从引擎 RunConfig 读取，不从 app.config.settings 读取。
# 后者导入时要求数据库和 JWT 配置，桌面没有这些配置，延迟导入也只是把崩溃推迟到首次探测。
from research_engine import catalog
from research_engine.llm_factory import map_local_host
from research_engine.runconfig import get_run_config

logger = structlog.get_logger()


def _log_probe_failure(event: str, **fields: object) -> None:
    """尽力记录探测或下载失败，日志本身也不能破坏“不抛异常”的契约。

    Windows 的 structlog 输出可能再次抛 I/O 错误；原错误已通过返回状态交给用户，因此可以放弃该日志行，不能为写日志使设置页失败。"""
    try:
        logger.info(event, **fields)
    except Exception:  # noqa: BLE001 — 日志写入不能成为 probe/pull 抛异常的原因
        pass


# 本地探测需快速返回；超时则报告不可用，不能拖慢设置页。
_PROBE_TIMEOUT_SECONDS = 3.0

# 模型参数量最低建议，面向执行器结构化证据步骤。
# 实测小模型可能会规划和调用搜索，却无法返回严格可解析的证据 JSON；界面应事先披露（docs/12 M15）。
_MIN_RESEARCH_PARAMS_B = 14.0

# 从标签提取参数量，如 7b、14b、70b；必须检查边界，不能把 14b 中的 4b 误识别成小模型。
_PARAM_RE = re.compile(r"(?:^|[^0-9.])(\d+(?:\.\d+)?)\s*b(?![a-z0-9])", re.IGNORECASE)

# 没有明确参数量时使用的定性大小标识。
_SMALL_WORDS = ("mini", "small", "tiny")

# 向量模型不能聊天或调用工具，需明确标识，不能作为普通代理角色选项。
_EMBEDDING_HINTS = ("embed", "bge-", "gte-", "minilm")


# 已知在 Ollama 中不支持工具调用的模型。
_NO_TOOL_HINTS = ("deepseek-r1", "deepseek-coder", "phi3:mini", "tinyllama")


def _supports_tools(name: str) -> bool:
    lowered = name.lower()
    return not any(hint in lowered for hint in _NO_TOOL_HINTS)


@dataclass
class LocalModel:
    """本地服务已安装的单个模型。"""

    name: str  # Ollama 原始标签，例如 qwen2.5:7b。
    size_bytes: int | None = None
    # 可映射到目录时，记录此模型实际使用的 provider:model 路由。
    route: str | None = None
    in_catalog: bool = False
    # 名称所示参数量可能难以完成研究流程时为 True。
    likely_underpowered: bool = False
    # 向量模型为 True；可用于检索，不能承担代理角色。
    is_embedding: bool = False
    # 是否支持工具调用，Executor 必须具备此能力。
    supports_tools: bool = True
    # 标签明确标注时解析参数量，单位十亿。
    params_b: float | None = None


# 区分未安装与已安装但未运行：前者提示安装，后者提供启动按钮（internal/07 Phase 2b）。
InstallState = Literal["running", "installed_not_running", "not_installed"]

# 安装器可能未加入 PATH 的常见程序位置。
_OLLAMA_BINARY_CANDIDATES = (
    "/usr/local/bin/ollama",
    "/opt/homebrew/bin/ollama",
    "/usr/bin/ollama",
    "/Applications/Ollama.app/Contents/Resources/ollama",
)


def resolve_binary() -> str | None:
    """查找本机 ollama 可执行文件的真实路径，除 PATH 外也检查系统常用安装目录。

    找不到返回 None；桌面启动服务需要路径本身，而不只是是否存在。"""
    on_path = shutil.which("ollama")
    if on_path:
        return on_path
    windows_candidate = Path.home() / "AppData" / "Local" / "Programs" / "Ollama" / "ollama.exe"
    for candidate in (*_OLLAMA_BINARY_CANDIDATES, str(windows_candidate)):
        if Path(candidate).exists():
            return candidate
    return None


def _binary_installed() -> bool:
    """检查本机是否安装 ollama，不只依赖 PATH。"""
    return resolve_binary() is not None


@dataclass
class LocalLLMStatus:
    """设置界面说明本地模型连接所需的完整状态。"""

    configured_base_url: str
    reachable: bool
    models: list[LocalModel] = field(default_factory=list)
    error: str | None = None
    # 面向普通用户的可执行下一步建议。
    hint: str | None = None
    install_state: InstallState = "not_installed"

    @property
    def usable(self) -> bool:
        return self.reachable and bool(self.models)


def _api_root(base_url: str) -> str:
    """从兼容接口地址推导 Ollama REST 根地址。

    聊天配置通常以 /v1 结尾，而模型列表在 /api/tags；统一去掉后缀，无需用户配置两份必须一致的地址。"""
    return base_url.rstrip("/").removesuffix("/v1")


def _parse_params_b(name: str) -> float | None:
    """从 Ollama 标签读取十亿参数量；未标注则返回 None。"""
    match = _PARAM_RE.search(name)
    return float(match.group(1)) if match else None


def _is_embedding(name: str) -> bool:
    lowered = name.lower()
    return any(hint in lowered for hint in _EMBEDDING_HINTS)


def _is_underpowered(name: str) -> bool:
    """根据明确参数量或定性大小词判断是否可能不足以完成研究。

    完全没有大小信息就不警告，不能猜测后误劝用户放弃可能足够的模型。"""
    lowered = name.lower()
    params = _parse_params_b(lowered)
    if params is not None:
        return params < _MIN_RESEARCH_PARAMS_B
    return any(word in lowered for word in _SMALL_WORDS)


def _match_catalog_route(name: str) -> tuple[str, bool]:
    """返回精确安装标签的路由，并单独报告是否属于已知目录家族。

    不能用目录家族名替代真实标签，否则不同大小被合并，可能调用不存在的 latest，或悄悄运行与用户所选不同的模型。目录仅提供已知计价和能力信息，不决定实际拨号名称。"""
    family = name.split(":", 1)[0]
    spec = catalog.get(family)
    in_catalog = spec is not None and spec.provider == "ollama"
    return f"ollama:{name}", in_catalog


async def probe(base_url: str | None = None) -> LocalLLMStatus:
    """探测配置的 Ollama 服务器及模型，故障不向外抛出。

    界面保留原 configured 地址，容器请求使用 map_local_host 后的 dial_url，与实际模型调用保持一致，不要求用户手动改容器地址。"""
    configured = base_url or get_run_config().ollama_base_url
    dial_url = map_local_host(configured)
    url = f"{_api_root(dial_url)}/api/tags"

    try:
        async with httpx.AsyncClient(timeout=_PROBE_TIMEOUT_SECONDS) as client:
            response = await client.get(url)
            response.raise_for_status()
            payload = response.json()
    except Exception as exc:  # noqa: BLE001 — 所有故障都转成用户可处理的不可达状态
        _log_probe_failure(
            "local_llm_probe_failed", configured=configured, dial_url=url, error=str(exc)
        )
        # 地址已自动完成容器映射，不能失败后再提示用户手动重复同一操作。
        installed = _binary_installed()
        return LocalLLMStatus(
            configured_base_url=configured,
            reachable=False,
            error=str(exc),
            hint=(
                "Ollama is installed but not running. Start it, then reload."
                if installed
                else "No local model server answered at this address. Start Ollama with "
                "`ollama serve`, then reload."
            ),
            install_state="installed_not_running" if installed else "not_installed",
        )

    models: list[LocalModel] = []
    for entry in payload.get("models", []) or []:
        name = entry.get("name") or entry.get("model") or ""
        if not name:
            continue
        route, in_catalog = _match_catalog_route(name)
        embedding = _is_embedding(name)
        models.append(
            LocalModel(
                name=name,
                size_bytes=entry.get("size"),
                route=route,
                in_catalog=in_catalog,
                supports_tools=_supports_tools(name),
                # 向量模型不是聊天能力较弱，而是完全不能聊天。
                likely_underpowered=(not embedding and _is_underpowered(name)),
                is_embedding=embedding,
                params_b=_parse_params_b(name),
            )
        )
    models.sort(key=lambda m: m.name)

    chat_models = [m for m in models if not m.is_embedding]

    hint = None
    if not models:
        hint = (
            "The server is running but has no models. Pull an embedding model for "
            "corpus retrieval, e.g. `ollama pull nomic-embed-text`. For research "
            "runs, use a hosted provider (Anthropic, Google, or OpenAI)."
        )
    elif not chat_models:
        hint = (
            "Only embedding models are installed — that is the recommended setup. "
            "Embedding models power corpus retrieval locally. For the research "
            "pipeline and chat, use a hosted provider for best quality."
        )
    elif all(m.likely_underpowered for m in chat_models):
        hint = (
            f"Every installed chat model looks small (under {_MIN_RESEARCH_PARAMS_B:.0f}B). "
            "Local models at this size usually fail the research pipeline's "
            "structured-evidence step. Use a hosted provider (Anthropic, Google, "
            "or OpenAI) for research runs."
        )

    return LocalLLMStatus(
        configured_base_url=configured,
        reachable=True,
        models=models,
        hint=hint,
        install_state="running",
    )


@dataclass
class PullProgress:
    """Ollama /api/pull 流中的一条下载进度。"""

    status: str
    completed: int | None = None
    total: int | None = None
    error: str | None = None


async def pull(model: str, base_url: str | None = None) -> AsyncIterator[PullProgress]:
    """流式获取模型下载进度（internal/07 Phase 2b）。

    传输失败或非 200 转换为一条 status=error 的 PullProgress，不向外抛出，保持与 probe 一致的状态契约。"""
    configured = base_url or get_run_config().ollama_base_url
    dial_url = map_local_host(configured)
    url = f"{_api_root(dial_url)}/api/pull"

    try:
        async with httpx.AsyncClient(timeout=None) as client:
            async with client.stream("POST", url, json={"name": model, "stream": True}) as resp:
                if resp.status_code != 200:
                    body = await resp.aread()
                    yield PullProgress(
                        status="error",
                        error=f"HTTP {resp.status_code}: {body.decode(errors='replace')[:200]}",
                    )
                    return
                async for line in resp.aiter_lines():
                    if not line.strip():
                        continue
                    try:
                        payload = json.loads(line)
                    except Exception:  # noqa: BLE001 — 单条非 JSON 消息不应中断整次进度读取
                        continue
                    yield PullProgress(
                        status=payload.get("status", ""),
                        completed=payload.get("completed"),
                        total=payload.get("total"),
                        error=payload.get("error"),
                    )
    except Exception as exc:  # noqa: BLE001 — 传输失败统一报告为没有响应
        _log_probe_failure("local_llm_pull_failed", model=model, error=str(exc))
        yield PullProgress(status="error", error=str(exc))
