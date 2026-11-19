"""报告质量的纯结构指标（docs/08 §5）。不执行 I/O，也不调用模型；单元测试保证已提交的评测数字可信且可纵向比较。

引用支持率需要模型判断片段是否支持主张，因此实现在评测框架中；此模块只做确定性结构计算。

主张和引用的提取统一定义在 research_engine.claims，本模块仅重新导出。
引擎的引用忠实度检查与评测必须处理同一组句子，否则测量对象会发生偏差。
重新导出的是同一个对象而非副本；tests/test_claim_extraction_parity.py 通过对象身份检查阻止重复实现。"""

from __future__ import annotations

import re

from research_engine.claims import (
    CITE_RE,
    CONFLICTS_HEADING_RE,
    HEADING_RE,
    LIMITATIONS_HEADING_RE,
    SOURCES_HEADING_RE,
    body_before_sources,
    claim_lines,
    extract_citations,
    split_sentences,
)

# 为评测框架、基准及文档中的调用方重新导出这些名称。
# 显式列出公共接口，避免检查器将其误判为未使用的导入并删除。
__all__ = [
    "CITE_RE",
    "CONFLICTS_HEADING_RE",
    "HEADING_RE",
    "LIMITATIONS_HEADING_RE",
    "METRICS_VERSION",
    "SOURCES_HEADING_RE",
    "body_before_sources",
    "citation_stats",
    "claim_lines",
    "contradictions_surfaced",
    "extract_citations",
    "report_metrics",
    "source_indices",
    "split_sentences",
    "uncited_claim_count",
]


def source_indices(sources: list[dict]) -> set[int]:
    out: set[int] = set()
    for s in sources or []:
        idx = s.get("index") if isinstance(s, dict) else None
        if isinstance(idx, int):
            out.add(idx)
    return out


def citation_stats(text: str, sources: list[dict]) -> dict:
    """统计正文 [n] 引用标记的总数，以及能否解析到真实来源。

    resolution_rate 委托给 research_engine.citation_rate，会话记录也使用这一实现。
    同一指标若有两套算法，基准数字便会与界面不一致；应用层不能导入评测包，
    因此共享实现必须放在引擎中，参见 graph.py 中的未引用主张计数。"""
    from research_engine.citation_rate import resolution_rate

    cites = extract_citations(body_before_sources(text))
    valid = source_indices(sources)
    return {
        "total_citations": len(cites),
        "unresolved_citations": sum(1 for n in cites if n not in valid),
        # 没有引用时返回 None，而不是 1.0：无引用的报告不能算作引用完全正确。
        "resolution_rate": resolution_rate(text, sources),
    }


def uncited_claim_count(text: str) -> int:
    """统计不带 [n] 引用的断言句，作为缺少证据支持的主张数量的近似值。"""
    return sum(1 for c in claim_lines(text) if not CITE_RE.search(c))


def contradictions_surfaced(text: str) -> int:
    """统计报告明确呈现的冲突主张对数（docs/12 M11）。

    计数范围为引擎生成的“Conflicting evidence”标题下的编号项，遇到下一个标题即停止。
    这里只测量报告展示了多少冲突；冲突是否真实由 evals/contradiction_eval.py 的精选样例和模型评测。"""
    lines = (text or "").splitlines()
    inside = False
    count = 0
    for raw in lines:
        line = raw.strip()
        if HEADING_RE.match(line):
            inside = bool(CONFLICTS_HEADING_RE.match(line))
            continue
        if inside and re.match(r"^\d+[.)]\s", line):
            count += 1
    return count


# 指标定义变化时递增版本号，并记录到每份结果中，防止不同定义下的历史数字被直接比较。
#
# 1：原始定义，只识别 [n]，主张按原始行计数。
# 2：将 [1, 3] 拆分为独立引用，主张改按句子切分（docs/12 M5，缺陷 D1）。
#    两项变化都会扩大分母，因此 v2 与 v1 不可直接比较。
# 3：排除 Limitations 中的保留性表述（D5），它们不是事实主张；v3 支持率不可与 v2 比较。
# 4：排除引擎生成的冲突证据区块（docs/12 M11），其中引用冲突双方，并非作者的主张；
#    同时新增 contradictions_surfaced 指标。
# 5：Judge 仅接受当前批次内编号；不再让越界判定污染其他 Claim 或支持率分母。
METRICS_VERSION = 5


def report_metrics(text: str, sources: list[dict]) -> dict:
    """计算一份报告的结构质量指标。"""
    stats = citation_stats(text, sources)
    return {
        "word_count": len((text or "").split()),
        "source_count": len(sources or []),
        **stats,
        "uncited_claim_count": uncited_claim_count(text),
        "contradictions_surfaced": contradictions_surfaced(text),
    }
