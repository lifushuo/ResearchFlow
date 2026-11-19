"""提取资料库文档文本，见 internal/12 M10、docs/13 隐私分层。

输入字节，输出标准文本、页边界与精确定位所需偏移。
不支持的格式明确拒绝；PDF 解析器延迟导入，避免纯文本部署承担不必要依赖。"""

from __future__ import annotations

import io
import posixpath
import re
import zipfile
from dataclasses import dataclass, field

# 限制单文档大小，保证普通笔记本上的提取、切块和嵌入有界；这些是保护阈值。
MAX_DOCUMENT_BYTES = 25 * 1024 * 1024
MAX_PDF_PAGES = 2000

#: ZIP 格式在解析前检查声明解压大小，防止小上传包声明巨大解压内容，见 RFC V3.1 §11.2。
MAX_ARCHIVE_EXPANDED_BYTES = 200 * 1024 * 1024
MAX_ARCHIVE_RATIO = 100
MAX_ARCHIVE_MEMBERS = 10_000

VERBATIM, LINEARIZED = "verbatim", "linearized"


@dataclass(frozen=True)
class Extraction:
    """文档进入资料库后的标准表示。

    text 是固定解析器版本下的标准文本，引用必须与其逐字核对。
    page_starts 记录 PDF 页起点，其他文档为 [0]；segments 记录工作表、行、页、单元格或章节位置。
    结构转成的文本标为 linearized，界面需向引用读者披露，见 RFC D-13、D-14。"""

    text: str
    page_starts: list[int]
    kind: str
    segments: list[tuple[int, str]] = field(default_factory=list)
    mode: str = VERBATIM


@dataclass(frozen=True)
class Format:
    kind: str
    label: str
    #: 延迟解析器所需模块，用于核对桌面打包依赖；标准库模块记为 None。
    parser_module: str | None = None


#: 支持扩展名的唯一清单；错误提示、下载类型和上传选择器由测试核对，避免多处不一致。
FORMATS: dict[str, Format] = {
    ".pdf": Format("pdf", "PDF", "pypdf"),
    ".html": Format("html", "HTML", "bs4"),
    ".htm": Format("html", "HTML", "bs4"),
    ".md": Format("md", "Markdown"),
    ".markdown": Format("md", "Markdown"),
    ".txt": Format("txt", "plain text"),
    ".text": Format("txt", "plain text"),
    ".rst": Format("txt", "plain text"),
    ".json": Format("txt", "plain text"),
    # B 级格式使用固定版本的 MIT 解析器，要求所有目标平台有 cp314 轮子；表格仅读缓存值。
    ".docx": Format("docx", "Word document", "docx"),
    ".xlsx": Format("spreadsheet", "Excel workbook", "python_calamine"),
    ".xlsm": Format("spreadsheet", "Excel workbook", "python_calamine"),
    ".xls": Format("spreadsheet", "Excel workbook", "python_calamine"),
    ".ods": Format("spreadsheet", "OpenDocument spreadsheet", "python_calamine"),
    # C 级格式：PowerPoint、OpenDocument、EPUB 用标准库和已有 lxml 读取 ZIP/XML，
    # 避免额外引入图片库；RTF 使用纯 Python 的 striprtf。
    ".pptx": Format("pptx", "PowerPoint presentation"),
    ".odt": Format("odt", "OpenDocument text"),
    ".epub": Format("epub", "EPUB book"),
    ".rtf": Format("rtf", "Rich Text", "striprtf"),
    # A 级格式由标准库处理，见 RFC V3.1 §10.2。
    ".csv": Format("csv", "CSV"),
    ".tsv": Format("csv", "TSV"),
    ".ipynb": Format("ipynb", "Jupyter notebook"),
    **{
        ext: Format("code", "source code")
        for ext in (
            ".py",
            ".java",
            ".r",
            ".m",
            ".jl",
            ".c",
            ".h",
            ".cpp",
            ".hpp",
            ".cc",
            ".cs",
            ".js",
            ".ts",
            ".tsx",
            ".jsx",
            ".go",
            ".rs",
            ".rb",
            ".scala",
            ".kt",
            ".swift",
            ".sql",
            ".sh",
            ".ps1",
            ".yaml",
            ".yml",
            ".toml",
            ".ini",
            ".cfg",
            ".xml",
            ".tex",
            ".sty",
            ".cls",
            ".bib",
        )
    },
}

#: 表格文本超出上限就明确拒绝，不能仅索引部分内容却表现得像已读完整表格。
MAX_TABLE_ROWS = 1_048_576
MAX_TABLE_COLUMNS = 256
MAX_TABLE_CELLS = 2_000_000
#: 笔记本单元格的文本输出最多保留这么多字符，超出时明确标记截断。
MAX_NOTEBOOK_OUTPUT_CHARS = 4096


def _extension(filename: str) -> str:
    return "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""


def kind_for(filename: str) -> str:
    """根据文件名确定文档类型，不支持时明确列出支持格式并拒绝。"""
    ext = _extension(filename)
    fmt = FORMATS.get(ext)
    if fmt is None:
        labels = ", ".join(dict.fromkeys(f.label for f in FORMATS.values()))
        extensions = " ".join(sorted(FORMATS))
        raise ValueError(
            f"Unsupported document type {ext or '(no extension)'!r}. The corpus accepts "
            f"{labels} ({extensions}) — a format we cannot locate a quote inside cannot "
            "enter a citation-grade corpus."
        )
    return fmt.kind


def check_archive(data: bytes) -> None:
    """解析前读取 ZIP 中央目录，若声明展开大小超限则拒绝，不尝试实际解压验证。"""
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            members = archive.infolist()
    except zipfile.BadZipFile as exc:
        raise ValueError(f"Not a readable document: {exc}") from exc
    if len(members) > MAX_ARCHIVE_MEMBERS:
        raise ValueError(
            f"The file holds {len(members)} parts; the limit is {MAX_ARCHIVE_MEMBERS}."
        )
    total = sum(m.file_size for m in members)
    if total > MAX_ARCHIVE_EXPANDED_BYTES:
        raise ValueError(
            f"The file would expand to {total} bytes; the limit is {MAX_ARCHIVE_EXPANDED_BYTES}."
        )
    for m in members:
        if m.compress_size and m.file_size / m.compress_size > MAX_ARCHIVE_RATIO:
            raise ValueError(
                f"Part {m.filename!r} expands {m.file_size // m.compress_size}x; files that "
                f"compress more than {MAX_ARCHIVE_RATIO}x are refused."
            )


def extract(filename: str, data: bytes) -> Extraction:
    """提取单文档标准表示，无法读取时用 ValueError 说明原因。

    解析错误转换为上传拒绝而非服务端错误；构建缺少解析器时单独说明，不能归咎文件损坏。"""
    if len(data) > MAX_DOCUMENT_BYTES:
        raise ValueError(
            f"Document is {len(data)} bytes; the corpus accepts at most "
            f"{MAX_DOCUMENT_BYTES} bytes per file."
        )
    kind = kind_for(filename)
    try:
        if kind == "pdf":
            text, page_starts, kind = _extract_pdf(data)
            return Extraction(text, page_starts, kind)
        if kind == "html":
            text, page_starts, kind = _extract_html(data)
            return Extraction(text, page_starts, kind)
        if kind == "docx":
            return _extract_docx(data)
        if kind == "spreadsheet":
            return _extract_spreadsheet(data, _extension(filename))
        if kind == "pptx":
            return _extract_pptx(data)
        if kind == "odt":
            return _extract_odt(data)
        if kind == "epub":
            return _extract_epub(data)
        if kind == "rtf":
            return _extract_rtf(data)
        if kind == "csv":
            return _extract_table(data, delimiter="\t" if _extension(filename) == ".tsv" else None)
        if kind == "ipynb":
            return _extract_notebook(data)
        if kind == "code":
            # 保留源码原文，仅去掉 BOM 并统一 Windows 换行，保持引用偏移稳定。
            text = data.decode("utf-8-sig", errors="replace").replace("\r\n", "\n")
            if not text.strip():
                raise ValueError("Document contains no text.")
            return Extraction(text, [0], "code")
        text = data.decode("utf-8", errors="replace").replace("\r\n", "\n")
        if not text.strip():
            raise ValueError("Document contains no text.")
        return Extraction(text, [0], kind)
    except ValueError:
        raise
    except ImportError as exc:
        raise ValueError(
            f"This build cannot read {FORMATS[_extension(filename)].label} files "
            f"(missing {exc.name or 'parser'})."
        ) from exc
    except Exception as exc:  # noqa: BLE001 — 其他解析错误也应作为文件拒绝返回
        raise ValueError(f"Could not read this file: {type(exc).__name__}.") from exc


def extract_document(filename: str, data: bytes) -> tuple[str, list[int], str]:
    """保留 3.1 之前的返回结构 text、page_starts、kind，供既有调用方使用。"""
    result = extract(filename, data)
    return result.text, result.page_starts, result.kind


def _extract_pdf(data: bytes) -> tuple[str, list[int], str]:
    from pypdf import PdfReader

    try:
        reader = PdfReader(io.BytesIO(data))
    except Exception as exc:  # noqa: BLE001 — 解析失败表示无法作为 PDF 读取
        raise ValueError(f"Not a readable PDF: {exc}") from exc

    if reader.is_encrypted:
        # 加密 PDF 无法可靠定位引用，明确拒绝，不偷偷尝试空密码。
        raise ValueError("The PDF is encrypted. Remove the password before ingesting it.")
    if len(reader.pages) > MAX_PDF_PAGES:
        raise ValueError(
            f"PDF has {len(reader.pages)} pages; the corpus accepts at most "
            f"{MAX_PDF_PAGES} per file."
        )

    parts: list[str] = []
    page_starts: list[int] = []
    offset = 0
    for page in reader.pages:
        try:
            page_text = page.extract_text() or ""
        except Exception:  # noqa: BLE001 — 单页损坏不应使整份文档提取失败
            page_text = ""
        page_starts.append(offset)
        parts.append(page_text)
        offset += len(page_text) + 1  # 为下方通过换行连接页文本预留位置。

    text = "\n".join(parts)
    if not text.strip():
        raise ValueError("The PDF contains no extractable text (scanned images need OCR first).")
    return text, page_starts, "pdf"


def _extract_html(data: bytes) -> tuple[str, list[int], str]:
    """从 HTML 提取用于索引的可读正文。

    原件另存，预览在隔离 iframe 中展示；索引不保留无意义的标签。
    script/style 连内容一起删除，避免代码进入证据库。
    与 read_webpage 共用 bs4、lxml 解析方式，让上传页面和在线读取结果可比较。"""
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(data, "lxml")
    for tag in soup(["script", "style", "noscript", "template"]):
        tag.decompose()
    text = soup.get_text("\n")
    # 压缩去标签后留下的连续空行；偏移以确定性的标准文本为准。
    lines = [line.strip() for line in text.replace("\r\n", "\n").split("\n")]
    text = "\n".join(line for line in lines if line)
    if not text.strip():
        raise ValueError("Document contains no text.")
    return text, [0], "html"


def _extract_table(data: bytes, *, delimiter: str | None) -> Extraction:
    """CSV/TSV 每数据行生成一行带表头的文本，见 §10.2。

    每行重复表头，使单独引用仍有意义；行号采用表格本身编号，表头为第 1 行，空单元格省略。"""
    import csv

    raw = data.decode("utf-8-sig", errors="replace").replace("\r\n", "\n")
    if not raw.strip():
        raise ValueError("Document contains no text.")
    if delimiter is None:
        try:
            delimiter = csv.Sniffer().sniff(raw[:4096], delimiters=",;\t|").delimiter
        except csv.Error:
            delimiter = ","
    rows = list(csv.reader(io.StringIO(raw), delimiter=delimiter))
    rows = [r for r in rows if any(cell.strip() for cell in r)]
    if len(rows) < 2:
        raise ValueError("The table has a header row but no data rows.")
    header = [h.strip() or f"Column {i + 1}" for i, h in enumerate(rows[0])]
    width = max(len(r) for r in rows)
    if (
        len(rows) > MAX_TABLE_ROWS
        or width > MAX_TABLE_COLUMNS
        or len(rows) * width > MAX_TABLE_CELLS
    ):
        raise ValueError(
            f"The table is {len(rows)} rows by {width} columns; the corpus reads at most "
            f"{MAX_TABLE_ROWS} rows, {MAX_TABLE_COLUMNS} columns and {MAX_TABLE_CELLS} cells."
        )
    parts: list[str] = []
    segments: list[tuple[int, str]] = []
    offset = 0
    for number, row in enumerate(rows[1:], start=2):
        cells = [
            f"{header[i] if i < len(header) else f'Column {i + 1}'}: {value.strip()}"
            for i, value in enumerate(row)
            if value.strip()
        ]
        line = f"Row {number} — " + "; ".join(cells)
        segments.append((offset, f"row {number}"))
        parts.append(line)
        offset += len(line) + 1
    return Extraction("\n".join(parts), [0], "csv", segments, LINEARIZED)


def _extract_notebook(data: bytes) -> Extraction:
    """按顺序输出 Jupyter 单元格，以单元格号和类型作标题。

    保留 stream、text/plain 输出并限制长度；图片等富输出省略但计数，避免暗示已读取。"""
    import json

    try:
        notebook = json.loads(data.decode("utf-8-sig"))
        cells = notebook["cells"]
        if not isinstance(cells, list):
            raise TypeError("cells is not a list")
    except (ValueError, KeyError, TypeError) as exc:
        raise ValueError(f"Not a readable notebook: {exc}") from exc

    def joined(value) -> str:
        return "".join(value) if isinstance(value, list) else str(value or "")

    parts: list[str] = []
    segments: list[tuple[int, str]] = []
    offset = 0
    for number, cell in enumerate(cells, start=1):
        if not isinstance(cell, dict):
            continue
        kind = "code" if cell.get("cell_type") == "code" else "markdown"
        body = joined(cell.get("source")).replace("\r\n", "\n").rstrip()
        texts, dropped = [], 0
        for output in cell.get("outputs") or []:
            if not isinstance(output, dict):
                continue
            if output.get("output_type") == "stream":
                texts.append(joined(output.get("text")))
            elif "text/plain" in (output.get("data") or {}):
                texts.append(joined(output["data"]["text/plain"]))
                dropped += len(output["data"]) - 1
            elif output.get("data"):
                dropped += len(output["data"])
        heading = f"## Cell {number} ({kind})"
        if dropped:
            heading += f" — {dropped} non-text output(s) not shown"
        block = heading + "\n" + body
        out = "\n".join(t.rstrip() for t in texts if t.strip())
        if out:
            if len(out) > MAX_NOTEBOOK_OUTPUT_CHARS:
                out = out[:MAX_NOTEBOOK_OUTPUT_CHARS] + " [output cut]"
            block += "\nOutput:\n" + out
        segments.append((offset, f"cell {number}"))
        parts.append(block)
        offset += len(block) + 2
    text = "\n\n".join(parts)
    if not text.strip():
        raise ValueError("The notebook has no cells.")
    return Extraction(text, [0], "ipynb", segments, LINEARIZED)


#: 加密 Office 文件可能是 OLE 而非 ZIP；通过文件头识别并明确报加密，不误报普通损坏。
_OLE_MAGIC = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"


def _refuse_encrypted_office(data: bytes) -> None:
    if data[:8] == _OLE_MAGIC:
        raise ValueError(
            "The file is password-protected (or a pre-2007 .doc). Save an unprotected "
            ".docx/.xlsx copy and upload that."
        )


def _extract_docx(data: bytes) -> Extraction:
    """按顺序提取 Word 标题、段落和表格。

    标题保留层级；读取接受修订后的文字，排除删除内容；表格按带表头的行输出。
    不读取脚注、批注、页眉和页脚，该限制已在文档披露。"""
    _refuse_encrypted_office(data)
    check_archive(data)
    import docx
    from docx.oxml.ns import qn

    document = docx.Document(io.BytesIO(data))
    parts: list[str] = []
    segments: list[tuple[int, str]] = []
    offset = 0
    path: list[str] = []
    tables = 0

    def add(text: str, label: str) -> None:
        nonlocal offset
        segments.append((offset, label))
        parts.append(text)
        offset += len(text) + 1

    for block in document.element.body.iterchildren():
        if block.tag == qn("w:p"):
            text = "".join(t.text or "" for t in block.iter(qn("w:t"))).strip()
            if not text:
                continue
            style = block.find(qn("w:pPr"))
            style_id = ""
            if style is not None and style.find(qn("w:pStyle")) is not None:
                style_id = style.find(qn("w:pStyle")).get(qn("w:val")) or ""
            level = (
                int(style_id[7:])
                if style_id.startswith("Heading") and style_id[7:].isdigit()
                else 0
            )
            if level:
                path = path[: level - 1] + [text]
                add("#" * level + " " + text, " › ".join(path))
            else:
                add(text, " › ".join(path) or "body")
        elif block.tag == qn("w:tbl"):
            tables += 1
            rows = [
                [
                    "".join(t.text or "" for t in cell.iter(qn("w:t"))).strip()
                    for cell in row.iter(qn("w:tc"))
                ]
                for row in block.iter(qn("w:tr"))
            ]
            if not rows:
                continue
            header = [h or f"Column {i + 1}" for i, h in enumerate(rows[0])]
            for number, row in enumerate(rows[1:], start=2):
                cells = [
                    f"{header[i] if i < len(header) else f'Column {i + 1}'}: {v}"
                    for i, v in enumerate(row)
                    if v
                ]
                if cells:
                    add(f"Row {number} — " + "; ".join(cells), f"table {tables} row {number}")
    text = "\n".join(parts)
    if not text.strip():
        raise ValueError("The document contains no text.")
    return Extraction(text, [0], "docx", segments, LINEARIZED)


def _cell(value) -> str:
    """把单元格值转成文本：整数去 .0，日期用 ISO，其余保持原表示。"""
    import datetime as _dt

    if value is None:
        return ""
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    if isinstance(value, _dt.datetime | _dt.date | _dt.time):
        return value.isoformat()
    return str(value).strip()


def _extract_spreadsheet(data: bytes, ext: str) -> Extraction:
    """按工作簿顺序提取工作表与数据行，见 §10.2。

    仅读缓存值，不索引公式；隐藏表仍读取并标记隐藏。合并单元格使用左上角值；行号与表格软件一致。"""
    if ext != ".xls":  # .xls 本来就是 OLE，其余这些格式为 ZIP。
        _refuse_encrypted_office(data)
        check_archive(data)
    from python_calamine import CalamineWorkbook

    workbook = CalamineWorkbook.from_filelike(io.BytesIO(data))
    visibility = {
        m.name: str(getattr(m, "visible", "visible")).lower() for m in workbook.sheets_metadata
    }
    parts: list[str] = []
    segments: list[tuple[int, str]] = []
    offset = 0
    cells_seen = 0
    for name in workbook.sheet_names:
        rows = workbook.get_sheet_by_name(name).to_python(skip_empty_area=False)
        numbered = [(i + 1, [_cell(v) for v in row]) for i, row in enumerate(rows)]
        numbered = [(n, row) for n, row in numbered if any(row)]
        if not numbered:
            continue
        width = max(len(row) for _, row in numbered)
        cells_seen += len(numbered) * width
        if (
            len(numbered) > MAX_TABLE_ROWS
            or width > MAX_TABLE_COLUMNS
            or cells_seen > MAX_TABLE_CELLS
        ):
            raise ValueError(
                f"Sheet {name!r} is {len(numbered)} rows by {width} columns; the corpus reads "
                f"at most {MAX_TABLE_ROWS} rows, {MAX_TABLE_COLUMNS} columns and "
                f"{MAX_TABLE_CELLS} cells per workbook."
            )
        hidden = "" if "visible" == visibility.get(name, "visible").split(".")[-1] else " (hidden)"
        heading = f"## Sheet: {name}{hidden}"
        segments.append((offset, f"Sheet {name}"))
        parts.append(heading)
        offset += len(heading) + 1
        header_row, header = numbered[0]
        header = [h or f"Column {i + 1}" for i, h in enumerate(header)]
        for number, row in numbered[1:]:
            cells = [
                f"{header[i] if i < len(header) else f'Column {i + 1}'}: {v}"
                for i, v in enumerate(row)
                if v
            ]
            line = f"Row {number} — " + "; ".join(cells)
            segments.append((offset, f"Sheet {name}, row {number}"))
            parts.append(line)
            offset += len(line) + 1
    text = "\n".join(parts)
    if not text.strip():
        raise ValueError("The workbook has no data.")
    return Extraction(text, [0], "spreadsheet", segments, LINEARIZED)


def _xml(archive: zipfile.ZipFile, name: str):
    """解析压缩包内单个 XML 部件，关闭实体解析和网络访问。"""
    from lxml import etree

    parser = etree.XMLParser(resolve_entities=False, no_network=True, load_dtd=False)
    return etree.fromstring(archive.read(name), parser)


_A = "{http://schemas.openxmlformats.org/drawingml/2006/main}"
_P = "{http://schemas.openxmlformats.org/presentationml/2006/main}"
_R = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"
_PKG_REL = "{http://schemas.openxmlformats.org/package/2006/relationships}"


def _rels(archive: zipfile.ZipFile, part: str) -> dict[str, str]:
    """读取 OOXML 部件的关系 ID 到目标路径映射。"""
    folder, _, name = part.rpartition("/")
    rels_name = f"{folder}/_rels/{name}.rels"
    if rels_name not in archive.namelist():
        return {}
    out = {}
    for rel in _xml(archive, rels_name).iter(f"{_PKG_REL}Relationship"):
        target = rel.get("Target") or ""
        out[rel.get("Id")] = (
            target.lstrip("/")
            if target.startswith("/")
            else posixpath.normpath(f"{folder}/{target}")
        )
    return out


def _paragraphs(element) -> list[str]:
    """按顺序提取 element 下每个 DrawingML 段落。"""
    out = []
    for para in element.iter(f"{_A}p"):
        text = "".join(t.text or "" for t in para.iter(f"{_A}t")).strip()
        if text:
            out.append(text)
    return out


def _extract_pptx(data: bytes) -> Extraction:
    """按演示文稿定义的顺序输出幻灯片标题、正文与备注。

    不按文件名排序，否则 slide10 会排在 slide2 前；文本按形状顺序，表格按带表头的行输出。"""
    _refuse_encrypted_office(data)
    check_archive(data)
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        presentation = _xml(archive, "ppt/presentation.xml")
        rels = _rels(archive, "ppt/presentation.xml")
        slide_parts = [
            rels[s.get(f"{_R}id")]
            for s in presentation.iter(f"{_P}sldId")
            if s.get(f"{_R}id") in rels
        ]
        parts: list[str] = []
        segments: list[tuple[int, str]] = []
        offset = 0
        for number, part in enumerate(slide_parts, start=1):
            slide = _xml(archive, part)
            title, lines = "", []
            for shape in slide.iter(f"{_P}sp"):
                placeholder = shape.find(f".//{_P}ph")
                texts = _paragraphs(shape)
                if not texts:
                    continue
                if (
                    placeholder is not None
                    and placeholder.get("type") in ("title", "ctrTitle")
                    and not title
                ):
                    title = " ".join(texts)
                else:
                    lines.extend(texts)
            for table in slide.iter(f"{_A}tbl"):
                rows = [
                    [" ".join(_paragraphs(cell)) for cell in row.iter(f"{_A}tc")]
                    for row in table.iter(f"{_A}tr")
                ]
                if rows:
                    header = [h or f"Column {i + 1}" for i, h in enumerate(rows[0])]
                    for n, row in enumerate(rows[1:], start=2):
                        cells = [
                            f"{header[i] if i < len(header) else f'Column {i + 1}'}: {v}"
                            for i, v in enumerate(row)
                            if v
                        ]
                        if cells:
                            lines.append(f"Row {n} — " + "; ".join(cells))
            notes = []
            for target in _rels(archive, part).values():
                if "notesSlide" in target and target in archive.namelist():
                    notes_xml = _xml(archive, target)
                    for shape in notes_xml.iter(f"{_P}sp"):
                        placeholder = shape.find(f".//{_P}ph")
                        if placeholder is not None and placeholder.get("type") == "body":
                            notes.extend(_paragraphs(shape))
            block = f"## Slide {number}" + (f": {title}" if title else "")
            if lines:
                block += "\n" + "\n".join(lines)
            if notes:
                block += "\nNotes:\n" + "\n".join(notes)
            segments.append((offset, f"slide {number}"))
            parts.append(block)
            offset += len(block) + 2
    text = "\n\n".join(parts)
    if not re.search(r"^(?!## Slide \d+$).+", text, re.M):
        raise ValueError("The presentation contains no text.")
    return Extraction(text, [0], "pptx", segments, LINEARIZED)


_ODF_TEXT = "{urn:oasis:names:tc:opendocument:xmlns:text:1.0}"
_ODF_TABLE = "{urn:oasis:names:tc:opendocument:xmlns:table:1.0}"


def _extract_odt(data: bytes) -> Extraction:
    """按顺序提取 OpenDocument 文本的标题、段落和表格行。"""
    check_archive(data)
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        if "content.xml" not in archive.namelist():
            raise ValueError("Not a readable OpenDocument file: no content.xml.")
        content = _xml(archive, "content.xml")
    body = content.find(".//{urn:oasis:names:tc:opendocument:xmlns:office:1.0}text")
    if body is None:
        raise ValueError("Not a readable OpenDocument text file.")
    parts: list[str] = []
    segments: list[tuple[int, str]] = []
    offset = 0
    path: list[str] = []
    tables = 0

    def add(text: str, label: str) -> None:
        nonlocal offset
        segments.append((offset, label))
        parts.append(text)
        offset += len(text) + 1

    def text_of(el) -> str:
        return "".join(el.itertext()).strip()

    for block in body.iter(f"{_ODF_TEXT}h", f"{_ODF_TEXT}p", f"{_ODF_TABLE}table"):
        if block.tag == f"{_ODF_TABLE}table":
            tables += 1
            rows = [
                [text_of(c) for c in r.iter(f"{_ODF_TABLE}table-cell")]
                for r in block.iter(f"{_ODF_TABLE}table-row")
            ]
            if rows:
                header = [h or f"Column {i + 1}" for i, h in enumerate(rows[0])]
                for n, row in enumerate(rows[1:], start=2):
                    cells = [
                        f"{header[i] if i < len(header) else f'Column {i + 1}'}: {v}"
                        for i, v in enumerate(row)
                        if v
                    ]
                    if cells:
                        add(f"Row {n} — " + "; ".join(cells), f"table {tables} row {n}")
            continue
        if any(a.tag == f"{_ODF_TABLE}table-cell" for a in block.iterancestors()):
            continue  # 随所属表格一起读取。
        text = text_of(block)
        if not text:
            continue
        if block.tag == f"{_ODF_TEXT}h":
            level = int(block.get(f"{_ODF_TEXT}outline-level") or 1)
            path = path[: level - 1] + [text]
            add("#" * level + " " + text, " › ".join(path))
        else:
            add(text, " › ".join(path) or "body")
    text = "\n".join(parts)
    if not text.strip():
        raise ValueError("The document contains no text.")
    return Extraction(text, [0], "odt", segments, LINEARIZED)


def _extract_epub(data: bytes) -> Extraction:
    """按 EPUB 的 spine 阅读顺序提取章节，正文使用 HTML 提取器。"""
    check_archive(data)
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        names = set(archive.namelist())
        if "META-INF/container.xml" not in names:
            raise ValueError("Not a readable EPUB: no META-INF/container.xml.")
        container = _xml(archive, "META-INF/container.xml")
        rootfile = next(
            (el.get("full-path") for el in container.iter() if el.tag.endswith("rootfile")), None
        )
        if not rootfile or rootfile not in names:
            raise ValueError("Not a readable EPUB: the package document is missing.")
        opf = _xml(archive, rootfile)
        base = rootfile.rpartition("/")[0]
        manifest = {
            el.get("id"): (f"{base}/{el.get('href')}" if base else el.get("href"))
            for el in opf.iter()
            if el.tag.endswith("}item")
        }
        spine = [el.get("idref") for el in opf.iter() if el.tag.endswith("}itemref")]
        parts: list[str] = []
        segments: list[tuple[int, str]] = []
        offset = 0
        for number, idref in enumerate(spine, start=1):
            href = manifest.get(idref)
            if not href or href not in names:
                continue
            try:
                chapter, _, _ = _extract_html(archive.read(href))
            except ValueError:
                continue  # 没有文本的章节（如纯图片封面）不贡献正文。
            first = chapter.split("\n", 1)[0][:80]
            segments.append((offset, f"chapter {number}: {first}"))
            parts.append(chapter)
            offset += len(chapter) + 2
    text = "\n\n".join(parts)
    if not text.strip():
        raise ValueError("The book contains no text.")
    return Extraction(text, [0], "epub", segments, VERBATIM)


def _extract_rtf(data: bytes) -> Extraction:
    """去掉 RTF 控制字提取文字；格式已丢失，因此标记为结构转写文本。"""
    from striprtf.striprtf import rtf_to_text

    raw = data.decode("latin-1")
    if not raw.lstrip().startswith("{\\rtf"):
        raise ValueError("Not a readable RTF file.")
    text = rtf_to_text(raw, errors="replace").replace("\r\n", "\n").strip()
    if not text:
        raise ValueError("The document contains no text.")
    return Extraction(text, [0], "rtf", [], LINEARIZED)
