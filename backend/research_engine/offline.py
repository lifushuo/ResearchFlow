"""Offline 保证本轮所有模型调用留在本机，见 RFC V3.1 §6.2。

这与只用选中文件是不同承诺，分别检查记录。
检查五个角色路由：Ollama 或 custom 必须实际指向本机，其余供应商属于远程服务。
复用地址改写和本地判断，不复制第三套规则。
Offline 同时采用仅资料库模式，因此网络工具已拒绝外网、资料库已拒绝远程嵌入。
fake 模式不调用真实模型，不因路由被拒绝。"""

from __future__ import annotations

from research_engine.embeddings import is_local_endpoint
from research_engine.runconfig import ROLES, RunConfig

#: 所有拒绝消息共用前缀，供宿主与测试识别。
OFFLINE_UNAVAILABLE = "offline_unavailable"


def violations(cfg: RunConfig, *, custom_base_url: str = "") -> list[str]:
    """逐角色列出会外发的模型路由；全部本地则为空。"""
    if cfg.llm_mode == "fake":
        return []
    from research_engine.llm_factory import map_local_host

    out: list[str] = []
    for role in ROLES:
        route = cfg.model_for(role)
        provider, _, _ = route.partition(":")
        if provider == "ollama":
            endpoint = map_local_host(cfg.ollama_base_url)
            if not is_local_endpoint(endpoint):
                out.append(f"{role} uses Ollama at {endpoint}, which is not on this machine")
        elif provider == "custom":
            endpoint = map_local_host(custom_base_url) if custom_base_url else ""
            if not endpoint:
                out.append(f"{role} uses a custom endpoint with no address saved")
            elif not is_local_endpoint(endpoint):
                out.append(f"{role} uses a custom endpoint at {endpoint}, not on this machine")
        else:
            out.append(f"{role} uses {route}, a cloud provider")
    return out


def refusal(cfg: RunConfig, *, custom_base_url: str = "") -> str | None:
    """返回不能满足 Offline 的原因；符合时返回 None。"""
    if not cfg.offline:
        return None
    found = violations(cfg, custom_base_url=custom_base_url)
    if not found:
        return None
    return (
        f"{OFFLINE_UNAVAILABLE}: Offline research needs every model to run on this computer. "
        + "; ".join(found)
        + ". Choose local models in Settings → Models, or turn Offline off."
    )
