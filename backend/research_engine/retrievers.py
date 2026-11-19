"""搜索检索链，见 docs/architecture/02-system-architecture.md、docs/04 §4。

按 Tavily → Brave → DuckDuckGo 回退，首个成功即返回；DuckDuckGo 仅作无密钥兜底。
统一为 title、url、snippet，并缓存 24 小时；fake 模式使用固定检索结果。
仅资料库模式直接替换整条链，不允许回退外网；此判断先于 fake，以便脚本也走真实资料库。"""

from __future__ import annotations

import asyncio
import json

import httpx
import structlog

from research_engine.cache import get_cache
from research_engine.corpus import get_corpus
from research_engine.runconfig import get_run_config

logger = structlog.get_logger()

SearchResult = dict  # {"title": str, "url": str, "snippet": str}
_CACHE_TTL = 86_400


async def _tavily(query: str, max_results: int) -> list[SearchResult]:
    api_key = get_run_config().tavily_api_key
    if not api_key:
        return []
    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.post(
            "https://api.tavily.com/search",
            json={
                "api_key": api_key,
                "query": query,
                "max_results": max_results,
            },
        )
        resp.raise_for_status()
        data = resp.json()
    return [
        {"title": r.get("title", ""), "url": r.get("url", ""), "snippet": r.get("content", "")}
        for r in data.get("results", [])
    ]


async def _brave(query: str, max_results: int) -> list[SearchResult]:
    api_key = get_run_config().brave_api_key
    if not api_key:
        return []
    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.get(
            "https://api.search.brave.com/res/v1/web/search",
            params={"q": query, "count": max_results},
            headers={"X-Subscription-Token": api_key},
        )
        resp.raise_for_status()
        data = resp.json()
    return [
        {"title": r.get("title", ""), "url": r.get("url", ""), "snippet": r.get("description", "")}
        for r in data.get("web", {}).get("results", [])
    ]


async def _duckduckgo(query: str, max_results: int) -> list[SearchResult]:
    from ddgs import DDGS

    def _run() -> list[dict]:
        with DDGS() as ddgs:
            return list(ddgs.text(query, max_results=max_results))

    raw = await asyncio.to_thread(_run)
    return [
        {"title": r.get("title", ""), "url": r.get("href", ""), "snippet": r.get("body", "")}
        for r in raw
    ]


_CHAIN = (("tavily", _tavily), ("brave", _brave), ("duckduckgo", _duckduckgo))


async def search(query: str, max_results: int = 5) -> list[SearchResult]:
    """通过缓存端口运行检索链，所有检索器耗尽时抛错。"""
    if get_run_config().corpus_mode:
        # 仅资料库模式只能由端口回答；未安装资料库时明确失败，不能回退网页链路而泄露查询。
        return await get_corpus().search(query, max_results)

    cfg = get_run_config()
    if cfg.llm_mode == "fake":
        if cfg.demo:
            from research_engine.demo_fixtures import demo_search

            return demo_search(query, max_results)
        from research_engine.fakes import fake_search

        return fake_search(query, max_results)

    # 尽力读取宿主缓存：服务端 Redis、桌面 SQLite，默认空缓存。缓存失败不能影响搜索。
    cache = get_cache()
    cache_key = f"search:{max_results}:{query.strip().lower()}"
    try:
        cached = await cache.get(cache_key)
        if cached:
            return json.loads(cached)
    except Exception:  # noqa: BLE001 — 缓存不可用不等于搜索失败
        pass

    errors: list[str] = []
    for name, fn in _CHAIN:
        try:
            results = await fn(query, max_results)
            if results:
                try:
                    await cache.set(cache_key, json.dumps(results), _CACHE_TTL)
                except Exception:  # noqa: BLE001 — 同上，缓存故障不影响搜索
                    pass
                logger.info("retriever_hit", retriever=name, count=len(results))
                return results
        except Exception as e:  # noqa: BLE001 — 失败后尝试下一个检索器
            errors.append(f"{name}: {e}")
            logger.warning("retriever_failed", retriever=name, error=str(e))

    raise RuntimeError(
        f"All retrievers failed or returned nothing: {'; '.join(errors) or 'no keys'}"
    )
