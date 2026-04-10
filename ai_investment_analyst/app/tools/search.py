"""
Web search tool for the Research Agent.

Uses DuckDuckGo via the `ddgs` package (free, no API key required).
DDGS is synchronous only — we wrap every call in asyncio.to_thread()
so it plays nicely with FastAPI's async event loop.

Two search types:
  web_search_tool  — general text search (news, analysis, filings)
  news_search_tool — recent news only (past week), higher signal for stocks
"""
import asyncio
import time
from typing import List
from ddgs import DDGS
from app.core.logging import logger


SEARCH_TIMEOUT_SECONDS = 15


# ── Internal helpers ──────────────────────────────────────────────────────────

def _run_text_search(query: str, max_results: int) -> List[dict]:
    """Sync DDGS text search — runs inside asyncio.to_thread."""
    with DDGS() as ddgs:
        return ddgs.text(
            query,
            max_results=max_results,
            safesearch="moderate",
        )


def _run_news_search(query: str, max_results: int) -> List[dict]:
    """Sync DDGS news search — runs inside asyncio.to_thread."""
    with DDGS() as ddgs:
        return ddgs.news(
            query,
            max_results=max_results,
            safesearch="moderate",
            timelimit="w",  # past week — most relevant for stock research
        )


def _format_text_results(results: List[dict]) -> str:
    """Format text search results into a clean string for the LLM."""
    if not results:
        return "No results found."
    parts = []
    for i, r in enumerate(results, 1):
        parts.append(
            f"[{i}] {r.get('title', 'No title')}\n"
            f"    URL: {r.get('href', '')}\n"
            f"    {r.get('body', 'No snippet')}"
        )
    return "\n\n".join(parts)


def _format_news_results(results: List[dict]) -> str:
    """Format news search results into a clean string for the LLM."""
    if not results:
        return "No recent news found."
    parts = []
    for i, r in enumerate(results, 1):
        parts.append(
            f"[{i}] {r.get('title', 'No title')}\n"
            f"    Source: {r.get('source', '')} | Date: {r.get('date', '')}\n"
            f"    URL: {r.get('url', '')}\n"
            f"    {r.get('body', 'No summary')}"
        )
    return "\n\n".join(parts)


# ── Public tool functions ─────────────────────────────────────────────────────

async def web_search_tool(
    query: str,
    max_results: int = 5,
    finance_context: bool = True,
) -> str:
    """
    General web search using DuckDuckGo.

    Args:
        query:           Search query string.
        max_results:     Max number of results to return (default 5).
        finance_context: If True, appends "investment finance" to the query
                         to bias results toward financial content.

    Returns:
        Formatted string of search results ready for LLM consumption.
    """
    search_query = f"{query} investment finance" if finance_context else query
    logger.info(f"[WebSearch] text | query='{search_query[:80]}'")
    start = time.perf_counter()

    try:
        results = await asyncio.wait_for(
            asyncio.to_thread(_run_text_search, search_query, max_results),
            timeout=SEARCH_TIMEOUT_SECONDS,
        )
        formatted = _format_text_results(results)
        elapsed = time.perf_counter() - start
        logger.info(
            f"[WebSearch] Got {len(results)} text results in {elapsed:.2f}s"
        )
        return formatted

    except asyncio.TimeoutError:
        elapsed = time.perf_counter() - start
        logger.error(
            f"[WebSearch] text search timed out after {elapsed:.2f}s "
            f"(limit={SEARCH_TIMEOUT_SECONDS}s)"
        )
        return "[WebSearch unavailable: timed out]"
    except Exception as e:
        elapsed = time.perf_counter() - start
        logger.error(f"[WebSearch] text search failed after {elapsed:.2f}s: {e}")
        return f"[WebSearch unavailable: {e}]"


async def news_search_tool(
    query: str,
    max_results: int = 5,
) -> str:
    """
    Recent news search (past week) using DuckDuckGo News.
    Higher signal than general web search for stock-related queries.

    Args:
        query:       Search query (ticker, company name, topic).
        max_results: Max number of news items (default 5).

    Returns:
        Formatted string of news items ready for LLM consumption.
    """
    logger.info(f"[WebSearch] news | query='{query[:80]}'")
    start = time.perf_counter()

    try:
        results = await asyncio.wait_for(
            asyncio.to_thread(_run_news_search, query, max_results),
            timeout=SEARCH_TIMEOUT_SECONDS,
        )
        formatted = _format_news_results(results)
        elapsed = time.perf_counter() - start
        logger.info(
            f"[WebSearch] Got {len(results)} news results in {elapsed:.2f}s"
        )
        return formatted

    except asyncio.TimeoutError:
        elapsed = time.perf_counter() - start
        logger.error(
            f"[WebSearch] news search timed out after {elapsed:.2f}s "
            f"(limit={SEARCH_TIMEOUT_SECONDS}s)"
        )
        return "[NewsSearch unavailable: timed out]"
    except Exception as e:
        elapsed = time.perf_counter() - start
        logger.error(f"[WebSearch] news search failed after {elapsed:.2f}s: {e}")
        return f"[NewsSearch unavailable: {e}]"


async def combined_search_tool(
    query: str,
    max_web: int = 3,
    max_news: int = 3,
) -> str:
    """
    Run web + news searches concurrently and combine results.
    Used by the Research Agent for maximum coverage.

    Returns:
        Combined formatted string with both web and news sections.
    """
    logger.info(f"[WebSearch] combined | query='{query[:80]}'")

    web_task = web_search_tool(query, max_results=max_web)
    news_task = news_search_tool(query, max_results=max_news)

    web_results, news_results = await asyncio.gather(web_task, news_task)
    logger.info("[WebSearch] combined search complete")

    return (
        f"=== WEB RESULTS ===\n{web_results}\n\n"
        f"=== RECENT NEWS ===\n{news_results}"
    )
