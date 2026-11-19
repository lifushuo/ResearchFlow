"""标记路由委托给哪个标准处理函数，只依赖标准库，供两种主机共用。

@delegates_to 记录 owner，让检查可以确认两个主机指向同一个函数对象，而非看似相同的两份实现。标记只声明归属，实际行为仍需两端契约回归验证。"""

from __future__ import annotations

import importlib
from collections.abc import Callable
from typing import TypeVar

F = TypeVar("F", bound=Callable)


def delegates_to(owner: str) -> Callable[[F], F]:
    """记录路由包装器委托的 owner，不增加行为。

    owner 使用 module:function 字符串，避免装饰时立即导入服务端依赖链；桌面只在请求需要时解析共享处理函数。"""

    def mark(route: F) -> F:
        route.__canonical__ = owner
        return route

    return mark


def canonical_owner(route: Callable) -> Callable:
    """按需解析路由委托的标准函数；没有委托标记则返回路由自身。

    解析放在测试和工具调用时，不在装饰阶段导入。"""
    reference = getattr(route, "__canonical__", None)
    if reference is None:
        return route
    module_path, _, name = reference.partition(":")
    return getattr(importlib.import_module(module_path), name)
