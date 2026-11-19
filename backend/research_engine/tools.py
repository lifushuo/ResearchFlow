"""各司其职的代理工具，见 docs/architecture/04-agent-architecture.md §4。

web_search 走缓存检索链，read_webpage 做 SSRF 校验与正文提取，calculate 做受限 AST 运算。
仅资料库模式下，读取只解析已安装资料库的 corpus:// 位置并拒绝其他 URL。
这限制工具检索，不限制代理模型路由。"""

from __future__ import annotations

import ast
import operator
from typing import Any
from urllib.parse import urlsplit

import httpx
from bs4 import BeautifulSoup
from langchain_core.tools import tool

from research_engine.corpus import get_corpus
from research_engine.net_guard import SSRFBlocked, validate_url
from research_engine.retrievers import search
from research_engine.runconfig import get_run_config

MAX_PAGE_CHARS = 8000
MAX_BODY_BYTES = 2 * 1024 * 1024
_MAX_REDIRECTS = 3

#: 搜索结果页的域名与路径清单；空路径列表表示任意携带查询的路径。
#: 匹配域名及子域，google、yandex 允许各国顶级域。
_SEARCH_RESULTS_PAGES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("google", ("/search", "/scholar")),
    ("yandex", ("/search",)),
    ("bing.com", ("/search",)),
    ("duckduckgo.com", ("/html", "/lite", "")),
    ("search.yahoo.com", ("/search",)),
    ("search.brave.com", ("/search",)),
    ("ecosia.org", ("/search",)),
    ("qwant.com", ("",)),
    ("mojeek.com", ("/search",)),
    ("baidu.com", ("/s",)),
    ("startpage.com", ("/do/search", "/sp/search")),
)

SEARCH_RESULTS_PAGE_REFUSAL = (
    "not fetched: this is a search engine's results page, which returns bot checks or "
    "ranked snippets rather than a source. Search with the web_search tool — it returns "
    "titles, URLs and snippets — then read_webpage the specific pages it finds."
)


def is_search_results_page(url: str) -> bool:
    """判断地址是否指向搜索结果页而非可引用来源。

    模型绕过 web_search 时曾大量抓取验证码或脚本墙。read_webpage 应拒绝并提示使用搜索工具，
    不能浪费抓取预算返回几乎无法引用的结果页。"""
    parts = urlsplit(url)
    host = (parts.hostname or "").lower()
    path = parts.path.rstrip("/")
    labels = host.split(".")
    for domain, paths in _SEARCH_RESULTS_PAGES:
        if "." in domain:
            on_engine = host == domain or host.endswith("." + domain)
        else:
            on_engine = domain in labels[:-1]  # `google.co.uk`, `scholar.google.com`
        if not on_engine:
            continue
        for results_path in paths:
            if results_path and (path == results_path or path.startswith(results_path + "/")):
                return True
            if not results_path and parts.query:
                return True
    return False


# 搜索入口：返回标题、URL 和摘要，必要时按供应商回退。
# 下方英文文档字符串会作为工具指令发送给模型，保留原文以免改变模型行为。
@tool
async def web_search(query: str, max_results: int | None = None) -> list[dict]:
    """Search the web and return results as a list of {title, url, snippet}.

    Use this first to find relevant pages for a research task. It is the only way to
    search: a search engine's results page opened with read_webpage is refused. Include
    date ranges in the query when the topic is time-sensitive.
    """
    # 模型可指定结果数；省略时用 retrieval_k，而不是在工具数据结构里固定写死，见 internal/07 Phase 3。
    k = max_results if max_results is not None else get_run_config().retrieval_k
    try:
        return await search(query, max_results=k)
    except Exception as e:  # noqa: BLE001 — 将可用错误提示返回给代理
        return [{"title": "Search unavailable", "url": "", "snippet": str(e)}]


CORPUS_READ_REFUSAL = "blocked: corpus documents can be read only by corpus-only research"


# 读取具体来源正文；拒绝搜索结果页和不安全网络地址，资料库模式只接受 corpus://。
# 下方英文文档字符串供模型选择工具使用，不是普通开发注释。
@tool
async def read_webpage(url: str) -> dict:
    """Fetch one specific source page and return {url, title, text, error}.

    Use after web_search to read a promising result. Not a search tool: a search
    engine's results page (Google, Bing, DuckDuckGo, …) is refused — call web_search to
    search. SSRF-guarded: internal, loopback, and cloud-metadata addresses are refused.
    Not for PDFs/videos. In corpus-only mode, only corpus:// locations can be read.
    """
    if url.startswith("corpus://"):
        # 资料库位置表示文件偏移，不是网络抓取；在 fake 分支前解析，让脚本也走真实存储。
        if not get_run_config().corpus_mode:
            # 只有资料库模式可读 corpus://；网页运行即使得到模型编造或页面注入的资料库 URL 也不能读取。
            return {"url": url, "title": "", "text": "", "error": CORPUS_READ_REFUSAL}
        try:
            return await get_corpus().read(url)
        except Exception as e:  # noqa: BLE001 — 将可用错误提示返回给代理
            return {"url": url, "title": "", "text": "", "error": str(e)}

    if get_run_config().corpus_mode:
        # 仅资料库模式不能抓取任何网页；返回错误字典让执行器改用资料库证据，避免直接抛错阻断任务。
        return {
            "url": url,
            "title": "",
            "text": "",
            "error": "blocked: corpus-only mode — network access is disabled",
        }

    if is_search_results_page(url):
        # 搜索结果页拒绝规则也先于 fake；规则与实际抓取内容无关。
        # 错误返回空正文，避免错误提示被拿来当证据。
        return {"url": url, "title": "", "text": "", "error": SEARCH_RESULTS_PAGE_REFUSAL}

    cfg = get_run_config()
    if cfg.llm_mode == "fake":
        if cfg.demo:
            from research_engine.demo_fixtures import demo_read_webpage

            return demo_read_webpage(url)
        from research_engine.fakes import fake_read_webpage

        return fake_read_webpage(url)

    current = url
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(10.0), follow_redirects=False) as client:
            for _ in range(_MAX_REDIRECTS + 1):
                validate_url(current)  # 每次重定向都重新校验，见 docs/06 §3。
                resp = await client.get(
                    current, headers={"User-Agent": "ResearchBot/1.0 (+research-assistant)"}
                )
                if resp.is_redirect and "location" in resp.headers:
                    current = str(resp.url.join(resp.headers["location"]))
                    continue
                break
            else:
                return {"url": url, "title": "", "text": "", "error": "too many redirects"}

        resp.raise_for_status()
        ctype = resp.headers.get("content-type", "")
        if not ("text/html" in ctype or "text/plain" in ctype):
            return {
                "url": url,
                "title": "",
                "text": "",
                "error": f"unsupported content-type: {ctype}",
            }
        body = resp.content[:MAX_BODY_BYTES]

        soup = BeautifulSoup(body, "lxml")
        for tag in soup(["script", "style", "nav", "footer", "header", "aside", "iframe"]):
            tag.decompose()
        title = soup.title.string.strip() if soup.title and soup.title.string else "No title"
        text = " ".join(soup.get_text(separator=" ").split())[:MAX_PAGE_CHARS]
        return {"url": url, "title": title, "text": text, "error": None}

    except SSRFBlocked as e:
        return {"url": url, "title": "", "text": "", "error": f"blocked: {e}"}
    except httpx.TimeoutException:
        return {"url": url, "title": "", "text": "", "error": "timeout (>10s)"}
    except httpx.HTTPStatusError as e:
        return {"url": url, "title": "", "text": "", "error": f"HTTP {e.response.status_code}"}
    except Exception as e:  # noqa: BLE001
        return {"url": url, "title": "", "text": "", "error": str(e)}


_SAFE_OPERATORS: dict[type, Any] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
}


# 仅计算允许的算术语法，不执行任意 Python 代码；英文说明保留给模型读取。
@tool
def calculate(expression: str) -> float:
    """Safely evaluate an arithmetic expression (+, -, *, /, **, parentheses)."""

    def _eval(node: ast.expr) -> float:
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.BinOp):
            fn = _SAFE_OPERATORS.get(type(node.op))
            if not fn:
                raise ValueError(f"unsupported operator: {type(node.op).__name__}")
            return fn(_eval(node.left), _eval(node.right))
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            return -_eval(node.operand)
        raise ValueError(f"unsupported expression node: {type(node).__name__}")

    try:
        tree = ast.parse(expression.strip(), mode="eval")
        return _eval(tree.body)
    except ZeroDivisionError as e:
        raise ValueError("division by zero") from e
    except Exception as e:  # noqa: BLE001
        raise ValueError(f"calculation error: {e}") from e


EXECUTOR_TOOLS = [web_search, read_webpage, calculate]
