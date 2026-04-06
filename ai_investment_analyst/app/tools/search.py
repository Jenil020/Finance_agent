"""
Web search tool for the Research Agent.
Uses DuckDuckGo (free, no API key needed) via duckduckgo-search.
"""
from duckduckgo_search import DDGS
from app.core.logging import logger


async def web_search_tool(query: str, max_results: int = 5) -> str:
    """
    Search the web using DuckDuckGo.
    Returns formatted string of results.
    """
    logger.info(f"[Web Search] Query: {query[:80]}")
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(
                query + " investment finance stock",
                max_results=max_results,
            ))
        formatted = []
        for r in results:
            formatted.append(f"Title: {r.get('title', '')}\n"
                             f"URL: {r.get('href', '')}\n"
                             f"Snippet: {r.get('body', '')}\n")
        return "\n---\n".join(formatted)
    except Exception as e:
        logger.error(f"[Web Search] Failed: {e}")
        return f"Web search unavailable: {e}"
