"""将追问范围统一转成聊天证据，供报告与项目聊天共用（internal/07 Phase 5）。

corpus 只保证检索不外发：关闭网页检索，拒绝远程向量化。但回答模型若走远程路由，问题和片段仍会传给它。
通过临时设置 corpus_mode 启用存储层已有防外发校验，不再复制一份安全判断。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import structlog

from research_engine.embeddings import EmbeddingsUnavailable
from research_engine.runconfig import get_run_config, reset_run_config, set_run_config

logger = structlog.get_logger()

# 对外范围值明确说明读取什么；使用 everything 而非含义模糊的 auto。
ChatScope = Literal["report", "corpus", "web", "everything"]

SCOPES: tuple[ChatScope, ...] = ("report", "corpus", "web", "everything")

# 每个检索来源每轮只贡献少量结果，追问无需承担整次研究的上下文和费用。
_K = 5


@dataclass(frozen=True)
class ScopeDescription:
    """范围选择器的说明与行为放在一起维护。

    excludes 明确禁止读取哪些来源，防止仅描述包含项却悄悄扩大范围。"""

    label: str
    reads: str
    excludes: str


_DESCRIPTIONS: dict[ChatScope, ScopeDescription] = {
    "report": ScopeDescription(
        label="This report",
        reads="the finished report and the sources it already cites",
        excludes="no new web search, no uploaded documents",
    ),
    "corpus": ScopeDescription(
        label="My corpus",
        # 两种主机从 3.1 起都按项目保存语料，所以项目隔离的说明同时成立。
        reads="only the documents uploaded to this project",
        # 模型路由与检索范围分别说明；问题和片段会发送到实际配置的聊天模型。
        excludes=(
            "no web search, and the question is never embedded off this machine; the "
            "question and excerpts still go to the provider chat is routed to, which is a "
            "cloud provider unless a local model was chosen"
        ),
    ),
    "web": ScopeDescription(
        label="Web",
        reads="a live web search, returning sources the report did not have",
        excludes="no uploaded documents",
    ),
    "everything": ScopeDescription(
        label="Everything",
        reads="finished research, your uploaded documents, and a live web search",
        excludes="nothing is held back",
    ),
}


def describe(scope: ChatScope) -> ScopeDescription:
    return _DESCRIPTIONS[scope]


@dataclass(frozen=True)
class Grounding:
    """保存本轮允许读取的内容与未能读取的原因。

    notes 向模型和用户披露无结果还是检索失败，避免空证据被模型误当成可以自行补充。"""

    scope: ChatScope
    text: str
    # 本轮新增来源从 1 编号，供正文 [n] 解析；未新增来源的范围为空。
    sources: list[dict] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def _wants(scope: ChatScope, source: str) -> bool:
    if scope == "everything":
        return True
    return scope == source


async def corpus_excerpts(*, store_factory, query: str, k: int = _K) -> list[dict]:
    """在防外发检查启用时检索项目上传文档。

    搜索期间设 corpus_mode=True，复用 CorpusStore 检查；远程向量模型触发 EmbeddingsUnavailable，必须交给用户决定，不能静默降级。"""
    store = store_factory()
    if store is None:
        return []
    token = set_run_config(replace_corpus_mode(get_run_config(), True))
    try:
        return await store.search(query, k)
    except EmbeddingsUnavailable:
        # 先原样重新抛出 EmbeddingsUnavailable，再处理 RuntimeError，因为前者是后者的子类。
        # 否则会吞掉外发拒绝，把需要用户决策的安全限制当普通无结果。
        raise
    except RuntimeError as e:
        # 语料为空或索引模型不同，对研究任务应明确失败；聊天则可返回没有可读内容的说明。
        # 两种原因仍应如实告知模型和调用方。
        raise CorpusUnreadable(str(e)) from e
    finally:
        reset_run_config(token)


class CorpusUnreadable(Exception):
    """语料存在但无法回答：内容为空，或索引由不同模型生成。"""


def replace_corpus_mode(cfg, value: bool):
    from dataclasses import replace

    return replace(cfg, corpus_mode=value)


def _format_sources(results: list[dict], start: int) -> tuple[str, list[dict]]:
    """从 start 为结果编号并生成提示词证据块。

    片段原样放在对应编号下，模型只可引用它实际看到的内容。"""
    lines: list[str] = []
    sources: list[dict] = []
    for offset, r in enumerate(results):
        index = start + offset
        url = r.get("url", "")
        title = r.get("title") or "(untitled)"
        content = (r.get("content") or r.get("snippet") or "").strip()
        lines.append(f'[{index}] {title}\n    Source: {url}\n    "{content}"')
        sources.append({"index": index, "url": url, "title": title, "snippet": content})
    return "\n".join(lines), sources


async def gather(
    scope: ChatScope,
    *,
    query: str,
    report: str | None,
    report_sources: list[dict],
    memory_excerpts: str | None,
    store_factory,
) -> Grounding:
    """为一轮聊天组装证据。

    report 与 memory_excerpts 分别对应报告正文和项目获批研究；两者都表示用户已批准的完成研究，所以共用 report 范围语义。"""
    blocks: list[str] = []
    notes: list[str] = []
    sources: list[dict] = []

    if _wants(scope, "report"):
        if report:
            blocks.append(f"--- THIS REPORT ---\n{report}")
            if report_sources:
                rendered, _ = _format_sources(report_sources, 1)
                blocks.append(f"--- SOURCES ALREADY CITED ---\n{rendered}")
        elif memory_excerpts:
            blocks.append(f"--- APPROVED RESEARCH IN THIS PROJECT ---\n{memory_excerpts}")
        else:
            notes.append("No finished research is available to read for this question.")

    if _wants(scope, "corpus"):
        try:
            results = await corpus_excerpts(store_factory=store_factory, query=query)
        except CorpusUnreadable as e:
            results = []
            # 保留具体原因；语料为空和索引模型不一致，不能都提示上传文件。
            notes.append(str(e))
        if results:
            rendered, added = _format_sources(results, len(sources) + 1)
            blocks.append(f"--- CORPUS EXCERPTS (your uploaded documents) ---\n{rendered}")
            sources.extend(added)
        elif not notes:
            # 项目本身没有内容与搜索没找到相关内容需要不同处理，不能混为一谈。
            notes.append(
                "This project has no uploaded corpus, so no documents were searched."
                if store_factory() is None
                else "The uploaded corpus returned no passages matching this question."
            )

    if _wants(scope, "web"):
        from research_engine import retrievers

        try:
            results = await retrievers.search(query, _K)
        except Exception as e:  # noqa: BLE001 — 必须披露错误，不能静默吞掉
            logger.warning("chat_web_search_failed", error=str(e))
            results = []
            notes.append(f"The web search could not run: {e}")
        if results:
            rendered, added = _format_sources(list(results), len(sources) + 1)
            blocks.append(f"--- WEB SEARCH RESULTS ---\n{rendered}")
            sources.extend(added)
        elif not notes:
            notes.append("The web search returned no results for this question.")

    return Grounding(scope=scope, text="\n\n".join(blocks), sources=sources, notes=notes)


def system_suffix(grounding: Grounding) -> str:
    """生成追加到聊天系统提示词的范围约束。

    不仅告诉用户，也告诉模型哪些内容没有提供，让它在证据范围不覆盖问题时正确拒答。"""
    described = describe(grounding.scope)
    parts = [
        f"RETRIEVAL SCOPE: {described.label}. You may use {described.reads} — "
        f"{described.excludes}. Say plainly when the answer is not in what you were "
        f"given; do not answer from your own knowledge, and do not apologise for the "
        f"scope the user chose."
    ]
    if grounding.notes:
        # 向模型明确缺失原因，不把缺少证据块误读为确实没有相关信息。
        parts.append(
            "GAPS IN WHAT YOU WERE GIVEN:\n" + "\n".join(f"- {n}" for n in grounding.notes)
        )
    return "\n\n".join(parts)
