"""报告的类型化派生视图，见 V2.1-a A7。

原始 Markdown 仍是唯一权威正文，report_hash、审批、产物与证据包都以它为准。
ReportDocument 仅用于理解结构，不能重新渲染后替代已批准文本，否则解析器变化会破坏哈希。
标题、列表与引用规则从 claims.py 导入，不能复制出第二套定义。"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

from research_engine.claims import HEADING_RE, LIST_MARKER_RE, extract_citations

#: 块类型或渲染规则变化时递增版本，供读取方识别采用的规则。
SCHEMA_VERSION = 1

BlockKind = Literal["heading", "paragraph", "list_item"]


class ReportBlock(BaseModel):
    """报告中的一个结构元素。

    只定义当前综合器会生成的三种类型，暂不为尚无生成流程的结构增加空字段。"""

    kind: BlockKind
    #: 去掉 Markdown 标记后的可见文本，渲染时再补回标题或列表标记。
    text: str
    order: int
    #: 标题级别为 1–6，其他块为 None。
    level: int | None = None
    #: 按出现顺序保存 [n] 编号，组合引用先展开；与引用解析率共用 claims.extract_citations。
    citation_markers: tuple[int, ...] = ()


class ReportMetadata(BaseModel):
    """报告身份的最小信息。

    尚未引入的工作流、代理、提示词或模型来源不在这里猜测记录。"""

    run_id: str
    revision_version: int
    question: str
    #: 时间由调用方传入，解析器不自行读时钟，保证同一输入得到相同输出。
    generated_at: str


class ReportDocument(BaseModel):
    """一份报告修订的类型化派生视图。"""

    schema_version: int = SCHEMA_VERSION
    blocks: tuple[ReportBlock, ...]
    metadata: ReportMetadata
    #: render_markdown 是否能逐字还原解析前的 Markdown。
    #:
    #: 记录还原精度而不强制要求精确。未知 Markdown 结构作为段落原文保留。
    #: EXACT 不等于质量通过，LOSSY 也不等于失败；二者都不能取代权威 report_markdown。
    render_fidelity: Literal["EXACT", "LOSSY"]


def _heading_level(line: str) -> int:
    return len(line) - len(line.lstrip("#"))


def parse_markdown(markdown: str, *, metadata: ReportMetadata) -> ReportDocument:
    """由报告派生类型化视图，纯函数且确定性。

    未知结构不丢弃、不猜测，原样保存为段落；render_fidelity 披露是否能准确还原。"""
    blocks: list[ReportBlock] = []
    paragraph: list[str] = []

    def flush() -> None:
        if paragraph:
            text = "\n".join(paragraph)
            blocks.append(
                ReportBlock(
                    kind="paragraph",
                    text=text,
                    order=len(blocks),
                    citation_markers=tuple(extract_citations(text)),
                )
            )
            paragraph.clear()

    for raw in (markdown or "").split("\n"):
        line = raw.rstrip()
        if not line.strip():
            flush()
            continue
        if HEADING_RE.match(line):
            flush()
            text = line.lstrip("#").strip()
            blocks.append(
                ReportBlock(
                    kind="heading",
                    text=text,
                    order=len(blocks),
                    level=_heading_level(line),
                    citation_markers=tuple(extract_citations(text)),
                )
            )
            continue
        if LIST_MARKER_RE.match(line):
            flush()
            text = LIST_MARKER_RE.sub("", line, count=1)
            blocks.append(
                ReportBlock(
                    kind="list_item",
                    text=text,
                    order=len(blocks),
                    citation_markers=tuple(extract_citations(text)),
                )
            )
            continue
        paragraph.append(line)
    flush()

    document = ReportDocument(
        blocks=tuple(blocks),
        metadata=metadata,
        render_fidelity="EXACT",  # 暂定值，以下方实际比较结果为准。
    )
    fidelity = "EXACT" if render_markdown(document) == (markdown or "") else "LOSSY"
    return document.model_copy(update={"render_fidelity": fidelity})


def render_markdown(document: ReportDocument) -> str:
    """把派生文档渲染回 Markdown。

    分隔规则按综合器现有输出：标题后正文单换行，连续标题间空行，列表项间单换行，其他块间空行。
    无法逐字还原时标为 LOSSY，不静默改写。report_hash 始终读取原始 report_markdown。"""
    if not document.blocks:
        return ""

    out: list[str] = []
    for index, block in enumerate(document.blocks):
        if block.kind == "heading":
            out.append("#" * (block.level or 1) + " " + block.text)
        elif block.kind == "list_item":
            out.append("- " + block.text)
        else:
            out.append(block.text)

        nxt = document.blocks[index + 1] if index + 1 < len(document.blocks) else None
        if nxt is None:
            out.append("\n")
        elif block.kind == "heading" and nxt.kind != "heading":
            out.append("\n")
        elif block.kind == "list_item" and nxt.kind == "list_item":
            out.append("\n")
        else:
            out.append("\n\n")
    return "".join(out)
