"""最终产物授权的统一业务规则：只有 APPROVED 的 REPORT 审核可授权 ResearchArtifact，计划批准不可以。

四层各有职责：数据库约束保护所有写入者；本模块供业务调用；run_bundle 的序列化区分报告批准和计划批准；独立校验器检查外部证据包。
序列化不能把所有批准都映射为 approved，否则计划批准会骗过只识别该动作的校验器。
本模块放在 app 顶层供两个主机共用，不能各自复制授权判断。"""

from __future__ import annotations

import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.review import Review

# 授权产物所需的关卡与决定组合，调用点不得另写字面量。
AUTHORIZING_GATE = "REPORT"
AUTHORIZING_DECISION = "APPROVED"


class NotAuthorizing(Exception):
    """提供的审核记录不能授权该产物。"""


def may_authorize_artifact(review: Review) -> bool:
    """仅在报告关卡作出批准决定时返回 True。"""
    return review.gate == AUTHORIZING_GATE and review.decision == AUTHORIZING_DECISION


async def approving_report_review(db: AsyncSession, run_id: uuid.UUID) -> Review | None:
    """查找能授权本次产物的报告批准，没有则返回 None。

    唯一约束保证每个修订最多一个批准；按 sequence 排序，使多轮返工时选择确定。"""
    return (
        (
            await db.execute(
                select(Review)
                .where(
                    Review.run_id == run_id,
                    Review.gate == AUTHORIZING_GATE,
                    Review.decision == AUTHORIZING_DECISION,
                )
                .order_by(Review.sequence.desc())
            )
        )
        .scalars()
        .first()
    )


def artifact_authorization_values(review: Review) -> dict:
    """生成产物证明授权所需的三个冗余字段。

    若误传计划批准，应在这里明确拒绝，而不是等数据库约束报错才发现。"""
    if not may_authorize_artifact(review):
        raise NotAuthorizing(
            f"review {review.id} is {review.gate}/{review.decision}; an artifact requires "
            f"{AUTHORIZING_GATE}/{AUTHORIZING_DECISION}"
        )
    return {
        "review_id": review.id,
        "review_decision": review.decision,
        "review_gate": review.gate,
    }


async def approval_chain(db: AsyncSession, run_id: uuid.UUID) -> list[Review]:
    """按 sequence 返回本运行两个关卡的全部审核。

    计划审核和报告审核原本分别挂在不同表，必须通过 reviews.run_id 查询，否则会漏掉计划批准。"""
    return list(
        (
            await db.execute(
                select(Review).where(Review.run_id == run_id).order_by(Review.sequence.asc())
            )
        )
        .scalars()
        .all()
    )
