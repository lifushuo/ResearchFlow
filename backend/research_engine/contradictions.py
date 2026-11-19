"""证据矛盾检测，见 docs/12 M11。

在评审后、综合前运行一次，把不相容观点展示在报告和人工关卡，不自动裁决真假。
此模块负责分组、提示词输入组装、输出校验和确定性渲染；模型调用由图节点负责。
检测器不可用时明确记录，不编造冲突。未知来源 URL、同源冲突和重复项会被过滤；
输入证据标作不可信内容，最终冲突段由代码生成，避免模型省略或擅自消解矛盾。"""

from __future__ import annotations

import re

from research_engine.schemas import ContradictionPair

# 限制提示词大小，每轮只做一次有界检测，避免按来源两两调用导致成本平方增长。
MAX_SOURCES = 12
MAX_SNIPPETS_PER_SOURCE = 4
MAX_SNIPPET_CHARS = 500

# 识别参考文献标题，以便不论模型采用何种标题同义词，都能把冲突块放在来源列表之前。
_SOURCES_HEADING_RE = re.compile(r"^#{1,6}\s*(sources|references|citations|bibliography)\b", re.I)


def group_snippets_by_source(evidence: list[dict]) -> dict[str, list[str]]:
    """将证据按来源 URL 分组，保留各来源不同片段并限制数量。

    按首次出现排序，与综合器引用编号约定一致。"""
    by_source: dict[str, list[str]] = {}
    for e in evidence:
        url = (e.get("source_url") or "").strip()
        snippet = (e.get("snippet") or "").strip()
        if not url or not snippet:
            continue
        bucket = by_source.setdefault(url, [])
        if len(bucket) >= MAX_SNIPPETS_PER_SOURCE:
            continue
        trimmed = snippet[:MAX_SNIPPET_CHARS]
        if trimmed not in bucket:
            bucket.append(trimmed)
    # 仅取前 MAX_SOURCES 个来源，控制检测预算。
    return dict(list(by_source.items())[:MAX_SOURCES])


def build_detector_input(by_source: dict[str, list[str]]) -> str:
    """组装检测器的人类消息正文，包含来源及其逐字证据片段。"""
    blocks: list[str] = ["Sources and their verbatim snippets:"]
    for url, snippets in by_source.items():
        lines = [f"Source: {url}", "<untrusted_web_content>"]
        lines.extend(f'- "{s}"' for s in snippets)
        lines.append("</untrusted_web_content>")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def normalize_pairs(
    pairs: list[ContradictionPair], known_urls: set[str]
) -> list[ContradictionPair]:
    """有限修复已观察到的结构化字段错位，见 docs/12 M11。

    去掉 source_* 的 Source: 前缀；若 source_* 不是已知 URL 而 snippet_* 是，则交换两者。
    修复后仍必须 validate_pairs，不能放行幻觉 URL。"""
    out: list[ContradictionPair] = []
    for p in pairs:
        d = p.model_dump()
        for side in ("a", "b"):
            src_key, snip_key = f"source_{side}", f"snippet_{side}"
            # 去掉 Source: 前缀，不区分大小写。
            src = d[src_key].strip()
            if src.lower().startswith("source:"):
                src = src[len("source:") :].strip()
            d[src_key] = src
            # 来源字段不是已知 URL、而片段字段是时，交换两个字段。
            if d[src_key] not in known_urls and d[snip_key].strip() in known_urls:
                d[src_key], d[snip_key] = d[snip_key].strip(), d[src_key]
        out.append(ContradictionPair(**d))
    return out


def validate_pairs(pairs: list[ContradictionPair], by_source: dict[str, list[str]]) -> list[dict]:
    """把检测结果转换为可展示字典，丢弃不可信项。

    双方 URL 必须都曾提供给检测器；排除同一来源自身的冲突；
    同一对来源及观点无论左右顺序如何，都只保留一次。"""
    known = set(by_source)
    seen_keys: set[tuple] = set()
    out: list[dict] = []
    for p in pairs:
        if p.source_a not in known or p.source_b not in known or p.source_a == p.source_b:
            continue
        key = (frozenset((p.source_a, p.source_b)), frozenset((p.claim_a, p.claim_b)))
        if key in seen_keys:
            continue
        seen_keys.add(key)
        out.append(p.model_dump())
    return out


def render_block(contradictions: list[dict], index_of: dict[str, int]) -> str:
    """确定性生成追加到报告的 Markdown 冲突块。

    index_of 把来源 URL 映射到 [n]，保证双方观点都能追溯；措辞明确说明报告没有裁决冲突。"""
    lines = [
        "## Conflicting evidence",
        "",
        "The sources below make claims that cannot both be true. This report does **not**",
        "resolve these conflicts — they are surfaced as-is for the reviewer to adjudicate.",
        "",
    ]
    for i, c in enumerate(contradictions, start=1):
        a = index_of.get(c["source_a"], 0)
        b = index_of.get(c["source_b"], 0)
        nature = (c.get("nature") or "the two claims are incompatible as stated").strip()
        lines.append(f"{i}. [{a}] claims that {c['claim_a']} — [{b}] claims that {c['claim_b']}.")
        lines.append(f"   Nature of disagreement: {nature}")
        if c.get("snippet_a") and c.get("snippet_b"):
            lines.append(f'   Verbatim: "{c["snippet_a"]}" vs "{c["snippet_b"]}"')
        lines.append("")
    return "\n".join(lines).rstrip()


def insert_block(draft: str, block: str) -> str:
    """把冲突块放在参考文献之前、其他内容之后；找不到来源标题时追加到报告末尾。"""
    lines = draft.splitlines()
    for i, raw in enumerate(lines):
        if _SOURCES_HEADING_RE.match(raw.strip()):
            return "\n".join(lines[:i] + ["", block, ""] + lines[i:])
    return f"{draft.rstrip()}\n\n{block}"
