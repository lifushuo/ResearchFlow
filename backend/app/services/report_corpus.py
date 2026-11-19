"""将获批报告自动写入项目语料库，供会话、运行和桌面共用。

调用方提供已打开的 CorpusStore，本函数不解析主机路径，只以 generated 来源写入。只在批准时触发，避免草稿和拒绝内容进入语料。"""

from __future__ import annotations

import structlog

from research_engine.corpus import CorpusStore, Ingested

logger = structlog.get_logger()


async def ingest_report(
    store: CorpusStore,
    *,
    report_id: str,
    report_markdown: str | None,
) -> Ingested | None:
    """尽力保存，异常不向外抛出；索引缺口不能推翻已经批准的研究。

    report_id 可指会话或运行，不能一律称 session_id。文件名使用该 ID，存储按文件名与 SHA-256 去重，重复调用不新增同一文档，旧文件名保持兼容。"""
    if not report_markdown or not report_markdown.strip():
        return None
    try:
        result = await store.ingest(
            f"report-{report_id}.md",
            report_markdown.encode("utf-8"),
            origin="generated",
        )
        logger.info(
            "report_auto_ingested",
            report_id=report_id,
            skipped=result.skipped,
            reason=result.reason,
        )
        return result
    except Exception as e:  # noqa: BLE001 — 语料索引失败不能推翻已批准运行
        logger.warning("report_auto_ingest_failed", report_id=report_id, error=str(e))
        return None
