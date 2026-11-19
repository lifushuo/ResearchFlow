"""一次性维护命令，默认只预览，传 --apply 才写入（internal/07 Phase 7）。

backfill-citation-rate 只用已存报告和来源补算从未测量的引用解析率，不覆盖已有实测值。没有可引用结论时仍保留 NULL，不能编造零分。"""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import dataclass

import structlog

from app.models.session import Session, SessionStatus
from research_engine import citation_rate

logger = structlog.get_logger()


@dataclass(frozen=True)
class BackfillPlan:
    """一条拟修改记录及其目标值。保存行索引而非 ORM 对象，方便先展示方案、计数，再决定是否应用。"""

    index: int
    rate: float


def plan_backfill(sessions: list[Session]) -> list[BackfillPlan]:
    """仅根据已有数据制定引用解析率补算方案，不访问数据库。

    只处理 COMPLETED，不测未批准草稿；已有指标不覆盖；没有引用保持 NULL；引用全部无法解析才是实测 0.0。"""
    plans: list[BackfillPlan] = []
    for i, s in enumerate(sessions):
        if s.status != SessionStatus.COMPLETED:
            continue
        if s.citation_resolution_rate is not None:
            continue
        if not s.final_report:
            continue
        rate = citation_rate.resolution_rate(s.final_report, s.sources)
        if rate is None:
            continue
        plans.append(BackfillPlan(index=i, rate=rate))
    return plans


async def backfill_citation_rate(*, apply: bool) -> int:
    """为从未测量的已完成会话补算引用解析率，返回修改条数。"""
    from sqlalchemy import select

    from app.db.base import AsyncSessionLocal, engine

    try:
        async with AsyncSessionLocal() as db:
            rows = (
                (
                    await db.execute(
                        select(Session)
                        .where(
                            Session.status == SessionStatus.COMPLETED,
                            Session.citation_resolution_rate.is_(None),
                        )
                        .order_by(Session.created_at.asc())
                    )
                )
                .scalars()
                .all()
            )
            plans = plan_backfill(list(rows))

            for plan in plans:
                row = rows[plan.index]
                print(f"{'apply ' if apply else 'would '} {row.id}  →  {plan.rate:.4f}")
                if apply:
                    row.citation_resolution_rate = plan.rate

            skipped = len(rows) - len(plans)
            print(
                f"\n{len(plans)} session(s) measurable, {skipped} left unmeasured "
                f"(no citable claims, or no stored report)."
            )
            if apply:
                await db.commit()
                print("Written.")
            else:
                print("Dry run — nothing written. Re-run with --apply.")
            return len(plans)
    finally:
        await engine.dispose()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m app.maintenance")
    sub = parser.add_subparsers(dest="command", required=True)
    backfill = sub.add_parser(
        "backfill-citation-rate",
        help="Recompute citation_resolution_rate for completed sessions that lack one.",
    )
    backfill.add_argument(
        "--apply",
        action="store_true",
        help="Write the changes. Without this the command only prints what it would do.",
    )
    args = parser.parse_args(argv)

    if args.command == "backfill-citation-rate":
        asyncio.run(backfill_citation_rate(apply=args.apply))
    return 0


if __name__ == "__main__":  # pragma: no cover — 命令行入口
    raise SystemExit(main())
