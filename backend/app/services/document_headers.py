"""两种主机共用的文档下载响应策略。

仅依赖标准库，不能放在会导入 app.config、app.db 的路由模块，否则桌面缺少服务端配置时无法启动。安全头策略集中维护，避免两端出现不同防护。"""

from __future__ import annotations

# 根据存储类型确定 Content-Type，不能原样相信用户上传的类型，以免反射恶意类型或触发嗅探。
_MEDIA_TYPES = {
    "pdf": "application/pdf",
    "md": "text/markdown; charset=utf-8",
    "txt": "text/plain; charset=utf-8",
    # 如实声明字节类型以便正确打开；安全前提是此类文档只作为附件下载，不能同源直接执行。
    "html": "text/html; charset=utf-8",
    "code": "text/plain; charset=utf-8",
    "csv": "text/csv; charset=utf-8",
    "ipynb": "application/x-ipynb+json",
    "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    # 同一类型覆盖 xlsx、xls、ods，无法仅靠 kind 精确区分，因此使用通用二进制类型，文件名仍保留扩展名。
    "spreadsheet": "application/octet-stream",
    "pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    "odt": "application/vnd.oasis.opendocument.text",
    "epub": "application/epub+zip",
    "rtf": "application/rtf",
}


def media_type_for(kind: str) -> str:
    return _MEDIA_TYPES.get(kind, "application/octet-stream")


def download_headers(kind: str, filename: str) -> dict[str, str]:
    """构造存储文档的安全响应头（internal/07 Phase 6）。

    只有 PDF 允许 inline，由浏览器独立查看器渲染，配合 nosniff。其他文件保持 attachment；文本和 Markdown 预览由客户端读取后安全渲染，不需要放宽下载策略。
    PDF 允许本站预览框架嵌入，显式设置 frame-ancestors self 和 X-Frame-Options；其他类型禁止嵌入并加 sandbox，避免上传 HTML 在本站执行。"""
    # 上传文件名不可包含可注入响应头的换行或引号，必须先清理。
    safe_name = filename.replace("\\", "_").replace('"', "_").replace("\n", "_").replace("\r", "_")
    if kind == "pdf":
        return {
            "Content-Disposition": f'inline; filename="{safe_name}"',
            "X-Content-Type-Options": "nosniff",
            "Content-Security-Policy": (
                "default-src 'none'; frame-ancestors 'self'; base-uri 'none'"
            ),
            "X-Frame-Options": "SAMEORIGIN",
        }
    return {
        "Content-Disposition": f'attachment; filename="{safe_name}"',
        "X-Content-Type-Options": "nosniff",
        "Content-Security-Policy": (
            "default-src 'none'; frame-ancestors 'none'; base-uri 'none'; sandbox"
        ),
    }
