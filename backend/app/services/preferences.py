"""两个主机共用的用户偏好合并逻辑。

请求只包含当前设置分区，必须按字段合并，不能清空其他分区。prompt_overrides 还需按角色再合并一层；角色值为 null 表示重置该角色，缺失则不修改。
结构校验由 UserPreferences 提前完成，本函数不重复校验。"""

from __future__ import annotations

from typing import Any

# 值为映射的偏好需深一层合并；新增同类偏好也应集中登记，不能在调用点加私有分支。
_NESTED_MAP_PREFERENCES = ("prompt_overrides",)


def merge_preferences(stored: dict[str, Any] | None, incoming: dict[str, Any]) -> dict[str, Any]:
    """返回应用 incoming 后的偏好，不修改两个输入对象。

    incoming 只含客户端明确设置的字段，未提及不等同于 None，而是保持原值。"""
    merged = dict(stored or {})
    for key, value in incoming.items():
        if key not in _NESTED_MAP_PREFERENCES:
            merged[key] = value
            continue
        if value is None:
            # 用户明确清空整份提示词覆盖映射。
            merged.pop(key, None)
            continue
        nested = dict(merged.get(key) or {})
        for inner_key, inner_value in value.items():
            if inner_value is None:
                nested.pop(inner_key, None)  # 将该角色重置为项目默认提示词。
            else:
                nested[inner_key] = inner_value
        # 空映射等同未配置，最后一项重置后统一为空，避免与从未配置的用户出现无意义差别。
        if nested:
            merged[key] = nested
        else:
            merged.pop(key, None)
    return merged
