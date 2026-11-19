"""研究运行的主机调度协议。

业务规则集中在路由：先提交再分发，恢复决定与 RUNNING 同事务，拒绝草稿只回到综合器。不同主机只替换机制：服务端走 Redis/Celery，桌面走进程内任务。
这里只声明协议，Celery 实现在 workers/dispatch.py，桌面实现在 sidecar。不能为复用调度迫使桌面导入未打包的 Celery。
明确区分开始、计划恢复、返工三种操作，不能用含义模糊的 enqueue 猜测关卡。"""

from __future__ import annotations

from typing import Protocol


class RunDispatcher(Protocol):
    """开始和恢复研究运行，每种主机提供自己的机制。"""

    async def start(self, run_id: str, user_id: str) -> None:
        """从头启动新创建的研究运行。"""
        ...

    async def resume_plan(self, run_id: str, user_id: str, plan: dict) -> None:
        """使用获批计划，恢复停在设计关卡的研究运行。"""
        ...

    async def rework(self, run_id: str, user_id: str, feedback: str | None) -> None:
        """将草稿被拒绝的研究运行送回综合器返工。"""
        ...


class SessionDispatcher(Protocol):
    """旧会话的开始与恢复协议，与 RunDispatcher 对应。

    历史会话仍承载用户报告和单报告聊天，不能删除。调度要从路由直接调用 Celery 改为可替换机制，才能在无代理的桌面端运行。
    同样拆成三个明确操作，避免新增关卡后把恢复负载送错位置。"""

    async def start(self, session_id: str, user_id: str) -> None:
        """从头启动新创建的会话。"""
        ...

    async def resume_plan(self, session_id: str, user_id: str, plan: dict) -> None:
        """用获批计划恢复设计关卡的会话。"""
        ...

    async def resume_review(
        self, session_id: str, user_id: str, approved: bool, feedback: str | None
    ) -> None:
        """恢复停在草稿关卡的会话。

        approved 必须明确传入：True 定稿，False 返回综合器。运行接口通过审核记录定稿，其对应操作名为 rework，不能混淆两种契约。"""
        ...
