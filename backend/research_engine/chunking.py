"""把已批准的报告分成可检索片段，见 docs/14 §4。

纯文本输入输出，不依赖宿主、模型或 I/O，让服务端与桌面切分一致。
按标题和段落切分，避免固定字符截断句子；每块携带最近标题，保留上下文；
重叠区域对齐句子边界，让跨边界 Claim 在至少一个块中完整出现。
保留 [n] 引用标记，使聊天回答能追溯原报告来源，见 docs/14 §5。"""

from __future__ import annotations

import re
from bisect import bisect_left, bisect_right
from dataclasses import dataclass

# 约 1200 字符粗略对应 300 Token，兼顾完整论述与一次检索多块时的提示词成本。
TARGET_CHARS = 1200
OVERLAP_CHARS = 180
# 过短片段通常只是标题或零散行，应合并，避免检索得到没有信息的结果。
MIN_CHARS = 200

# 标题适合作为边界，但只有当前块足够独立时才切分；短章节合并以避免过多小块及嵌入费用。
_HEADING_FLUSH_RATIO = 0.6

_HEADING_RE = re.compile(r"^#{1,6}\s+\S")
# 句末标点可跟引号或括号，再接空白。Python re 不支持变长后顾，因此写成两个定长匹配。
# 缩写偶尔造成不理想切分，但不会丢失句子。
_SENTENCE_END_RE = re.compile(r"(?:(?<=[.!?])|(?<=[.!?][\"')\]]))\s+")


def _normalise(text: str) -> str:
    """去掉行尾空白，并把连续空行压成一个。"""
    lines = [line.rstrip() for line in text.replace("\r\n", "\n").split("\n")]
    out: list[str] = []
    for line in lines:
        if not line and out and not out[-1]:
            continue
        out.append(line)
    return "\n".join(out).strip()


def _blocks(text: str) -> list[str]:
    """按空行分段，再把标题行与后续正文分开。

    Markdown 标题后不一定有空行。若把整个区块当作标题，仅保留第一行，
    就会静默丢掉正文，导致项目记忆没有内容。因此标题与正文必须分别成为区块。"""
    out: list[str] = []
    for block in re.split(r"\n\s*\n", text):
        block = block.strip()
        if not block:
            continue
        run: list[str] = []
        for line in block.splitlines():
            if _HEADING_RE.match(line):
                if run:
                    out.append("\n".join(run).strip())
                    run = []
                out.append(line.strip())
            else:
                run.append(line)
        if run and "\n".join(run).strip():
            out.append("\n".join(run).strip())
    return out


def _split_long(block: str, limit: int) -> list[str]:
    """过长段落优先按句子切分，仍超长时再按词切分。"""
    pieces: list[str] = []
    current = ""
    for sentence in _SENTENCE_END_RE.split(block):
        if not sentence:
            continue
        candidate = f"{current} {sentence}".strip() if current else sentence
        if len(candidate) <= limit or not current:
            current = candidate
        else:
            pieces.append(current)
            current = sentence
    if current:
        pieces.append(current)

    # 单句也可能超长，例如表格行或 URL 列表，仍需遵守长度限制。
    out: list[str] = []
    for piece in pieces:
        while len(piece) > limit:
            cut = piece.rfind(" ", 0, limit)
            cut = cut if cut > limit // 2 else limit
            out.append(piece[:cut].strip())
            piece = piece[cut:].strip()
        if piece:
            out.append(piece)
    return out


def _tail_overlap(text: str, overlap: int) -> str:
    """取 text 末尾最多 overlap 字符的完整句子。"""
    if overlap <= 0 or not text:
        return ""
    window = text[-overlap:]
    parts = _SENTENCE_END_RE.split(window, maxsplit=1)
    # 优先从句子边界开始，找不到时退到词边界。
    if len(parts) > 1 and parts[1].strip():
        return parts[1].strip()
    space = window.find(" ")
    return window[space + 1 :].strip() if space != -1 else ""


def chunk_report(
    text: str,
    *,
    target_chars: int = TARGET_CHARS,
    overlap_chars: int = OVERLAP_CHARS,
    min_chars: int = MIN_CHARS,
) -> list[str]:
    """把报告切成带标题和重叠内容的片段。

    同一报告始终得到同样片段，配合 source_session_id、chunk_index 唯一键实现重复导入幂等。"""
    normalised = _normalise(text or "")
    if not normalised:
        return []

    heading = ""
    chunks: list[str] = []
    current = ""

    def flush() -> None:
        """保存当前片段；如果只有标题则跳过，避免产生没有正文信息的检索命中。"""
        nonlocal current
        body = current.strip()
        if body and not all(_HEADING_RE.match(line) for line in body.splitlines() if line.strip()):
            chunks.append(body)
        current = ""

    for block in _blocks(normalised):
        if _HEADING_RE.match(block):
            # 标题属于后续正文；仅当当前块足够长时才先结束它，见 _HEADING_FLUSH_RATIO。
            heading = block.splitlines()[0].strip()
            if len(current) >= target_chars * _HEADING_FLUSH_RATIO:
                flush()
                current = heading
            else:
                current = f"{current}\n\n{heading}" if current else heading
            continue

        # 切分大块时为续块的标题和重叠区域预留空间，否则最终块会超出 target_chars。
        room = max(target_chars // 2, target_chars - overlap_chars - len(heading) - 4)
        for piece in [block] if len(block) <= target_chars else _split_long(block, room):
            candidate = f"{current}\n\n{piece}" if current else piece
            if len(candidate) <= target_chars or not current:
                current = candidate
                continue

            flush()
            # 新块携带标题和一个句子的重叠内容，保留主题并避免从 Claim 中间开始。
            opener = heading if heading else ""
            carry = _tail_overlap(chunks[-1] if chunks else "", overlap_chars)
            current = "\n\n".join(part for part in (opener, carry, piece) if part)

    flush()

    # 过短尾块并入前一块；若整篇报告只有一个短块则仍保留。
    if len(chunks) > 1 and len(chunks[-1]) < min_chars:
        tail = chunks.pop()
        chunks[-1] = f"{chunks[-1]}\n\n{tail}"

    return chunks


# ── 资料库文档，见 internal/12 M10 ──
#
# 报告按标题重组；资料库文档则必须能定位原文，要求更严格。
# 每块都是提取文本的原样区间 [start, end)，不能拼接标题或改写开头。


@dataclass(frozen=True)
class DocumentChunk:
    """带精确原文位置的文档块。

    text 必须逐字等于 source[start:end]。PDF 页码从 1 开始，无分页文档为 None；
    字符偏移适用于所有文档，页码用于方便人工定位。"""

    text: str
    start: int
    end: int
    page: int | None


# 新块只能从段落开头或句末空白后开始，尽量避免切断 Claim。
_PIECE_BREAK_RE = re.compile(r"\n\s*\n|[.!?][\"')\]]*\s+")


def _split_points(text: str) -> list[int]:
    """返回允许作为文本片段起点的偏移位置。"""
    return [m.end() for m in _PIECE_BREAK_RE.finditer(text)]


def _page_for(page_starts: list[int], offset: int) -> int | None:
    """返回 offset 所在页，页码从 1 开始；没有分页结构时返回 None。"""
    if len(page_starts) <= 1:
        return None
    return bisect_right(page_starts, offset)


def chunk_document(
    text: str,
    *,
    page_starts: list[int] | None = None,
    target_chars: int = TARGET_CHARS,
    overlap_chars: int = OVERLAP_CHARS,
    min_chars: int = MIN_CHARS,
) -> list[DocumentChunk]:
    """把文档切成保留原文、带重叠和偏移量的块。

    同一文本产生同样区间；下一块起点落在前块末尾的重叠窗口内，并对齐合法边界。
    重叠内容仍逐字属于两个块各自的原文区间。"""
    body = text or ""
    first_nonws = len(body) - len(body.lstrip())
    if first_nonws >= len(body):
        return []

    pages = page_starts or []
    points = _split_points(body)
    chunks: list[DocumentChunk] = []
    start = first_nonws

    while start < len(body):
        if start + target_chars >= len(body):
            raw_end = len(body)
        else:
            # 优先选择窗口内最后一个合法边界；单句或表格行过长时按上限截断，保证能够推进。
            lo = bisect_right(points, start)
            hi = bisect_right(points, start + target_chars)
            raw_end = points[hi - 1] if hi > lo else start + target_chars

        # 仅通过调整 start/end 去除两侧空白，保持块文本严格等于 body[start:end]。
        end = raw_end
        while start < end and body[start].isspace():
            start += 1
        while end > start and body[end - 1].isspace():
            end -= 1
        if start >= end:
            break

        chunks.append(
            DocumentChunk(text=body[start:end], start=start, end=end, page=_page_for(pages, start))
        )

        if raw_end >= len(body):
            break

        # 从重叠窗口内第一个边界开始下一块；找不到时从当前块末尾继续，避免停滞。
        floor = max(raw_end - overlap_chars, start + 1)
        idx = bisect_left(points, floor)
        nxt = points[idx] if idx < len(points) else None
        start = nxt if nxt is not None and nxt < raw_end else raw_end

    # 过短尾块没有独立 Claim，将其原文区间合并到前块。
    if len(chunks) > 1 and (chunks[-1].end - chunks[-1].start) < min_chars:
        tail = chunks.pop()
        prev = chunks[-1]
        chunks[-1] = DocumentChunk(
            text=body[prev.start : tail.end], start=prev.start, end=tail.end, page=prev.page
        )

    return chunks
