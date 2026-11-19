"""统计报告引用标记能否解析到真实来源，见 internal/07 Phase 7。

None 表示没有可测引用；0.0 表示出现过引用但全部无对应来源，二者不能混淆。
此实现由界面与 evals.metrics 共用，避免显示值与评测值不一致。
引用模式和参考文献边界统一来自 research_engine.claims。"""

from __future__ import annotations

from research_engine.claims import body_before_sources, extract_citations

__all__ = ["body_before_sources", "cited_indices", "resolution_rate"]


def cited_indices(text: str) -> list[int]:
    """按出现顺序提取正文引用编号，保留重复项。

    这是 claims.extract_citations 的薄封装，保留供既有调用方使用的名称。"""
    return extract_citations(text)


def resolution_rate(report: str, sources: list[dict] | None) -> float | None:
    """返回正文引用指向真实来源的比例；没有任何引用时返回 None，不能填 0。"""
    cites = cited_indices(body_before_sources(report))
    if not cites:
        return None
    valid = {
        s.get("index")
        for s in (sources or [])
        if isinstance(s, dict) and isinstance(s.get("index"), int)
    }
    resolved = sum(1 for n in cites if n in valid)
    return round(resolved / len(cites), 4)
