"""Claim 与引用提取的唯一实现，见 docs/08 §5、docs/12 M5。

图内引用核验、评测 Judge 和证据包必须对 Claim 及引用采用同一定义，
否则评分测量的就不是流程实际处理的句子。
本模块放在引擎层，仅依赖标准库，evals 从这里导入，不能反向依赖评测包。
评测版本与运行聚合仍属于 evals.metrics。"""

from __future__ import annotations

import re

# 匹配引用标记并提取其中编号，包含 [1, 3] 这样的组合引用。
#
# 综合器经常输出组合引用。若只匹配单编号，会漏计引用并虚高解析率。
# extract_citations 将 [1, 3] 展开成两个编号，与 [1][3] 同等计数。
CITE_RE = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")

HEADING_RE = re.compile(r"^#{1,6}\s")
SOURCES_HEADING_RE = re.compile(r"^#{1,6}\s*(sources|references|citations|bibliography)\b", re.I)
# Limitations 描述证据未覆盖的范围，并非待证实的事实 Claim，因此与 Sources 一样排除。
LIMITATIONS_HEADING_RE = re.compile(r"^#{1,6}\s*limitations?\b", re.I)
# Conflicting evidence 由引擎确定性生成，引用双方矛盾观点。
# 它是在说明证据冲突，不能把这种说明当成研究者自己的事实断言评分。
CONFLICTS_HEADING_RE = re.compile(r"^#{1,6}\s*conflicting evidence\b", re.I)
LIST_MARKER_RE = re.compile(r"^(?:[-*+]\s+|\d+[.)]\s+)")
# 句末标点加空白作为边界，但排除常见缩写与单字母姓名。
# 宁可少切分，也避免把一个 Claim 拆成失去证据上下文的碎片。
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\[])")
ABBREV_TAIL_RE = re.compile(r"\b(?:e\.g|i\.e|vs|etc|Dr|Mr|Ms|Inc|Ltd|Fig|No|approx|cf)\.$", re.I)

#: 过短或不含字母的句子视为碎片；图内引用核验使用相同门槛。
MIN_CLAIM_CHARS = 15


def body_before_sources(text: str) -> str:
    """取参考文献章节之前的报告正文。

    来源列表不是事实 Claim，将它计入会让引用率无故趋近 1。"""
    lines = (text or "").splitlines()
    for i, raw in enumerate(lines):
        if SOURCES_HEADING_RE.match(raw.strip()):
            return "\n".join(lines[:i])
    return text or ""


def extract_citations(text: str) -> list[int]:
    """按顺序返回全部来源编号，保留重复项；组合引用与多个单独引用等价处理。"""
    out: list[int] = []
    for m in CITE_RE.finditer(text or ""):
        out.extend(int(part) for part in m.group(1).split(","))
    return out


def split_sentences(text: str) -> list[str]:
    """按句子切分正文，并重新合并被常见缩写误切开的部分。"""
    parts = SENTENCE_SPLIT_RE.split(text)
    if len(parts) < 2:
        return [text]
    merged: list[str] = [parts[0]]
    for part in parts[1:]:
        # 例如 Dr. Smith 中的句点不能切出新句子。
        if ABBREV_TAIL_RE.search(merged[-1]):
            merged[-1] = f"{merged[-1]} {part}"
        else:
            merged.append(part)
    return merged


def is_claim_sentence(sentence: str) -> bool:
    """判断文本片段是否足够长、是否包含可作为断言的文字。"""
    return len(sentence) >= MIN_CLAIM_CHARS and bool(re.search(r"[A-Za-z]", sentence))


def claim_lines(text: str) -> list[str]:
    """从报告正文逐句提取可独立判断的 Claim。

    综合器常把多句写在同一行，按行提取会把不同断言与来源混成一个评分对象。
    排除标题、列表标记、过短碎片及 Limitations，避免把诚实披露证据不足误判成引用失败。"""
    out: list[str] = []
    skipping = False
    for raw in (text or "").splitlines():
        line = raw.strip()
        if not line:
            continue
        if SOURCES_HEADING_RE.match(line):
            break  # 来源与参考文献章节不属于 Claim。
        if HEADING_RE.match(line):
            skipping = bool(LIMITATIONS_HEADING_RE.match(line)) or bool(
                CONFLICTS_HEADING_RE.match(line)
            )
            continue
        if skipping:
            continue
        content = LIST_MARKER_RE.sub("", line)
        for sentence in split_sentences(content):
            sentence = sentence.strip()
            if not is_claim_sentence(sentence):
                continue
            out.append(sentence)
    return out
