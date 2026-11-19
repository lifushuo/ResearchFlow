"""报告导出流程：Markdown 转安全样式 HTML，再转 PDF（docs/05 §3，docs/29）。

WeasyPrint 需要原生 Pango/Cairo，放在 render_pdf 内延迟导入；开发环境缺库仍可启动 API 和导出 Markdown，仅 PDF 入口明确失败。"""

from __future__ import annotations

import html
import re

import markdown as md

# 模型署名实现统一放在 research_engine.bundle，供桌面直接使用。
# 这里仅重新导出以兼容 research.py；桌面不应导入会间接涉及 WeasyPrint 的本模块。
from research_engine.bundle import render_model_attribution_md  # noqa: F401

# 打印样式完全自包含，不加载外部资源，渲染时可离线工作。
_PDF_CSS = """
@page { size: A4; margin: 2cm; }
body { font-family: 'Helvetica Neue', Arial, sans-serif; font-size: 11pt;
       line-height: 1.5; color: #1a1a1a; }
h1 { font-size: 20pt; margin: 0 0 4pt; }
h2 { font-size: 14pt; margin: 18pt 0 6pt; border-bottom: 1px solid #ddd; padding-bottom: 2pt; }
h3 { font-size: 12pt; margin: 12pt 0 4pt; }
p, li { orphans: 2; widows: 2; }
a { color: #4338ca; text-decoration: none; }
sup { font-size: 0.7em; color: #4338ca; }
blockquote { border-left: 3px solid #ddd; margin: 8pt 0; padding-left: 10pt; color: #555; }
code { font-family: 'SFMono-Regular', Consolas, monospace; font-size: 0.9em;
       background: #f3f3f5; padding: 1pt 3pt; border-radius: 3px; }
pre { background: #f3f3f5; padding: 8pt; border-radius: 4px; overflow-x: auto; }
.meta { color: #777; font-size: 9pt; margin-bottom: 14pt; }
.sources { margin-top: 20pt; border-top: 1px solid #ddd; padding-top: 8pt; font-size: 10pt; }
.sources ol { padding-left: 18pt; }
.sources li { margin-bottom: 5pt; }
.sources .domain { color: #777; }
.sources .snippet { color: #555; font-style: italic; }
.models { margin-top: 14pt; padding-top: 8pt; font-size: 9pt; color: #777; }
.models ul { padding-left: 18pt; list-style: none; }
.models code { background: none; padding: 0; }
"""

# 同时匹配单个与成组引用，例如 [1] 和 [1, 3]。
#
# 成组引用要逐个转上标，与前端引用标记一致；仅匹配单编号会使导出样式悄悄丢失（docs/12 M5，D1）。
_CITATION_RE = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")


def _superscript_citations(match: re.Match[str]) -> str:
    """把 [1, 3] 拆成两个来源各自的上标引用标记。"""
    return "".join(f"<sup>[{part.strip()}]</sup>" for part in match.group(1).split(","))


def _domain(url: str) -> str:
    m = re.match(r"https?://(?:www\.)?([^/]+)", url or "")
    return m.group(1) if m else (url or "")


def _sources_html(sources: list[dict]) -> str:
    if not sources:
        return ""
    items = []
    for s in sources:
        idx = html.escape(str(s.get("index", "")))
        url = html.escape(s.get("url", ""), quote=True)
        title = html.escape(s.get("title") or _domain(s.get("url", "")))
        domain = html.escape(_domain(s.get("url", "")))
        snippet = html.escape(s.get("snippet", ""))
        snippet_html = f'<div class="snippet">&ldquo;{snippet}&rdquo;</div>' if snippet else ""
        items.append(
            f'<li id="src-{idx}"><a href="{url}">{title}</a> '
            f'<span class="domain">— {domain}</span>{snippet_html}</li>'
        )
    return '<div class="sources"><h2>Sources</h2><ol>' + "".join(items) + "</ol></div>"


def _models_html(model_routing: dict[str, str] | None, checker_accuracy: dict | None) -> str:
    """披露角色到实际模型路由的对应关系。

    尚未解析过路由的失败运行或旧报告返回空值，不猜默认模型。"""
    if not model_routing:
        return ""
    items = "".join(
        f"<li><strong>{html.escape(role)}</strong> — <code>{html.escape(route)}</code></li>"
        for role, route in sorted(model_routing.items())
    )
    accuracy = ""
    if checker_accuracy and checker_accuracy.get("label"):
        label = checker_accuracy["label"]
        accuracy = f"<p>{html.escape(label[0].upper() + label[1:])}.</p>"
    return f'<div class="models"><h2>Models used</h2><ul>{items}</ul>{accuracy}</div>'


def render_html(
    report_markdown: str,
    sources: list[dict],
    *,
    title: str = "Research Report",
    model_routing: dict[str, str] | None = None,
    checker_accuracy: dict | None = None,
) -> str:
    """生成报告的独立完整 HTML，供 PDF 渲染器使用。"""
    # 先把正文 [n] 转成上标，再执行 Markdown 转换，确保输出保留 sup 标记。
    body_md = _CITATION_RE.sub(_superscript_citations, report_markdown or "")
    body_html = md.markdown(body_md, extensions=["extra", "sane_lists"])
    safe_title = html.escape(title)
    return (
        "<!DOCTYPE html><html><head><meta charset='utf-8'>"
        f"<title>{safe_title}</title><style>{_PDF_CSS}</style></head><body>"
        f"{body_html}{_sources_html(sources)}{_models_html(model_routing, checker_accuracy)}"
        "</body></html>"
    )


def render_pdf(
    report_markdown: str,
    sources: list[dict],
    *,
    title: str = "Research Report",
    model_routing: dict[str, str] | None = None,
    checker_accuracy: dict | None = None,
) -> bytes:
    """将报告渲染为 PDF；缺少原生库时抛 RuntimeError，由调用方转换成清楚的 501。"""
    try:
        from weasyprint import HTML
    except (OSError, ImportError) as e:  # 缺少 Pango/Cairo 原生库。
        raise RuntimeError(f"PDF rendering is unavailable in this environment: {e}") from e

    document = render_html(
        report_markdown,
        sources,
        title=title,
        model_routing=model_routing,
        checker_accuracy=checker_accuracy,
    )
    return HTML(string=document).write_pdf()
