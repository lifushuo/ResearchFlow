"""共用 SSE 响应头策略。

压缩代理会缓冲事件流，连接看似正常却迟迟没事件。Cache-Control 的 no-transform 禁止中间层压缩变换，无需关闭整个站点压缩。
X-Accel-Buffering: no 另外禁止 nginx 代理缓冲（docs/09 §5）。"""

SSE_HEADERS: dict[str, str] = {
    # no-transform 阻止代理压缩，从而避免事件流被缓冲。
    "Cache-Control": "no-cache, no-transform",
    "X-Accel-Buffering": "no",
    "Connection": "keep-alive",
}
