"""两个宿主共用的模型路由校验。

Ollama、custom、OpenRouter 的模型 ID 由端点定义，不能强制出现在本地目录。
过去服务端已允许而桌面仍拒绝，导致精确本地 tag 和网关模型不可选。
此纯规则只依赖目录和角色表，放在引擎层；服务端重新导出，桌面委托调用，避免再维护两份。"""

from __future__ import annotations

from research_engine import catalog
from research_engine.runconfig import ROLES


class InvalidRouting(ValueError):
    """不能持久化的非法路由映射，继承 ValueError 以兼容宿主错误处理。"""


#: 模型 ID 由端点定义、而非本地目录定义的供应商。
#:
#: 对这些供应商只检查路由形状，否则合法本地 tag 和网关模型都会被拒绝。
#: 未知定价使本地费用上限无效，应在供应商侧设限，不能把未知费用展示为实测零。
ENDPOINT_DEFINED_PROVIDERS = ("ollama", "custom", "openrouter")


def validate(routing: dict) -> dict[str, str]:
    """规范化并校验用户路由。

    不静默修复错误；目录内却缺价格的模型明确拒绝，避免该角色预算保护失效。"""
    if not isinstance(routing, dict):
        raise InvalidRouting("Model routing must be an object keyed by agent role.")

    unknown_roles = sorted(set(routing) - set(ROLES))
    if unknown_roles:
        raise InvalidRouting(f"Unknown agent role(s): {unknown_roles}. Valid roles: {list(ROLES)}.")
    missing_roles = sorted(set(ROLES) - set(routing))
    if missing_roles:
        raise InvalidRouting(f"Every role needs a model. Missing: {missing_roles}.")

    cleaned: dict[str, str] = {}
    for role, route in routing.items():
        if not isinstance(route, str) or ":" not in route:
            raise InvalidRouting(f"{role}: expected a 'provider:model' string, got {route!r}.")
        # 只按第一个冒号拆分；ollama:qwen2.5:7b 的模型部分必须保留 qwen2.5:7b。
        provider, _, model_id = route.partition(":")
        if provider not in catalog.KNOWN_PROVIDERS:
            raise InvalidRouting(
                f"{role}: unknown provider '{provider}'. "
                f"Known: {', '.join(catalog.KNOWN_PROVIDERS)}."
            )
        if provider in ENDPOINT_DEFINED_PROVIDERS:
            if not model_id:
                raise InvalidRouting(f"{role}: '{provider}:' names no model.")
            # 端点定义的精确 tag 原样保留，不能擅自规范化成另一个模型。
            cleaned[role] = route
            continue
        spec = catalog.get(model_id)
        if spec is None:
            raise InvalidRouting(f"{role}: '{model_id}' is not in the model catalog.")
        if not spec.priced:
            raise InvalidRouting(
                f"{role}: '{model_id}' has no configured price, so spending on it could "
                "not be capped. Add its price to the catalog before routing to it."
            )
        cleaned[role] = spec.route  # 目录模型使用标准形式，便于持久化值相互比较。
    return cleaned
