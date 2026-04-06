"""
Research Agent — ReAct (Reason + Act) pattern.

Responsibilities:
  1. Extract ticker / topic from the query
  2. Query the RAG knowledge base (hybrid search + CRAG)
  3. Run web + news search if RAG confidence is low OR to supplement
  4. Fetch live ticker summary from yfinance (if ticker detected)
  5. Synthesise all evidence into a structured research brief

The agent does NOT make a recommendation — that is the Report Agent's job.
It purely gathers and summarises evidence for the Analysis Agent to use.
"""
import re
from app.orchestration.state import AgentState
from app.agents.base import llm_call
from app.tools.rag_query import rag_query_tool
from app.tools.search import combined_search_tool
from app.tools.portfolio import get_ticker_summary
from app.core.logging import logger

# ── System prompt (persona) ───────────────────────────────────────────────────
SYSTEM_PROMPT = """You are a senior equity research analyst at a top-tier investment bank.
Your job is to gather, verify, and synthesise financial evidence — not to give opinions.
Be precise, cite data points, and flag uncertainty explicitly.
Never fabricate numbers. If data is unavailable, say so."""

# ── Task prompt ───────────────────────────────────────────────────────────────
SYNTHESIS_PROMPT = """RESEARCH TASK
=============
Query: {query}

EVIDENCE GATHERED
-----------------
[1] Internal Knowledge Base (RAG, confidence={rag_confidence:.2f}):
{rag_context}

[2] Live Market Data (yfinance):
{market_data}

[3] Web + News Search:
{web_context}

INSTRUCTIONS
------------
Synthesise the above evidence into a concise research brief (300-500 words).
Structure your response as:
  OVERVIEW:     One paragraph on the subject.
  KEY FACTS:    Bullet list of verifiable data points with numbers.
  RECENT NEWS:  2-3 most relevant recent developments.
  DATA GAPS:    Any important information that was unavailable.
  SOURCES:      List the source types used (RAG / yfinance / web search).

Do not make a BUY/SELL/HOLD recommendation — that comes later."""

# ── Ticker extraction ─────────────────────────────────────────────────────────
_TICKER_RE = re.compile(r'\b([A-Z]{1,5})\b')
_COMMON_WORDS = {
    "I", "A", "AN", "THE", "FOR", "IN", "IS", "AT", "BY", "TO", "DO",
    "AI", "US", "UK", "ETF", "IPO", "GDP", "CEO", "CFO", "Q1", "Q2",
    "Q3", "Q4", "PE", "EPS", "YTD", "ROE", "ROI", "NPV", "IRR",
}


def _extract_ticker(query: str) -> str | None:
    """
    Heuristically extract a stock ticker from the query.
    Returns the first uppercase word that looks like a ticker and isn't
    a common acronym. Returns None if nothing found.
    """
    candidates = _TICKER_RE.findall(query)
    for c in candidates:
        if c not in _COMMON_WORDS and 1 < len(c) <= 5:
            return c
    return None


# ── Agent node ────────────────────────────────────────────────────────────────

async def research_agent_node(state: AgentState) -> dict:
    """
    LangGraph node: Research Agent.

    Returns a partial state dict — LangGraph merges it into the full state.
    Keys updated: research_results, messages, error (on failure).
    """
    query = state["query"]
    logger.info(f"[Research] Starting | query='{query[:80]}'")

    try:
        # ── Step 1: RAG retrieval (hybrid BM25 + vector + CRAG) ─────────────
        logger.info("[Research] Step 1/4 — RAG retrieval")
        rag_result = await rag_query_tool(query)
        rag_context = rag_result["context"]
        rag_confidence = rag_result["confidence"]
        rag_sources = rag_result["sources"]
        needs_web = rag_result["needs_web_search"]

        logger.info(
            f"[Research] RAG done | "
            f"confidence={rag_confidence:.3f} | needs_web={needs_web}"
        )

        # ── Step 2: Live market data (yfinance) ──────────────────────────────
        logger.info("[Research] Step 2/4 — Live market data")
        market_data = "No ticker identified in query."
        ticker = _extract_ticker(query)
        if ticker:
            summary = await get_ticker_summary(ticker)
            if "error" not in summary:
                market_data = (
                    f"Ticker: {summary.get('ticker')}\n"
                    f"Last price:          ${summary.get('last_price', 'N/A')}\n"
                    f"Previous close:      ${summary.get('previous_close', 'N/A')}\n"
                    f"52-week high/low:    ${summary.get('year_high', 'N/A')} / ${summary.get('year_low', 'N/A')}\n"
                    f"Market cap:          ${summary.get('market_cap', 'N/A'):,}\n"
                    f"YTD change:          {summary.get('year_change', 'N/A')}\n"
                    f"50-day avg:          ${summary.get('fifty_day_average', 'N/A')}\n"
                    f"200-day avg:         ${summary.get('two_hundred_day_average', 'N/A')}"
                )
            else:
                market_data = f"yfinance error for {ticker}: {summary['error']}"

        # ── Step 3: Web + news search ────────────────────────────────────────
        # Always run if RAG confidence is low; also always get recent news
        logger.info("[Research] Step 3/4 — Web + news search")
        web_context = await combined_search_tool(
            query,
            max_web=3 if needs_web else 2,
            max_news=3,
        )

        # ── Step 4: LLM synthesis ────────────────────────────────────────────
        logger.info("[Research] Step 4/4 — LLM synthesis")
        prompt = SYNTHESIS_PROMPT.format(
            query=query,
            rag_confidence=rag_confidence,
            rag_context=rag_context,
            market_data=market_data,
            web_context=web_context,
        )

        synthesis = await llm_call(
            prompt=prompt,
            system_instruction=SYSTEM_PROMPT,
            temperature=0.1,        # low temp — we want factual, not creative
            max_output_tokens=1024,
        )

        # Combine all sources
        all_sources = list(set(rag_sources + (["yfinance"] if ticker else []) + ["web_search"]))

        logger.info(f"[Research] Complete | sources={all_sources}")

        return {
            "research_results": [synthesis],
            "messages": state["messages"] + [
                {
                    "role": "research_agent",
                    "content": synthesis,
                    "sources": all_sources,
                    "rag_confidence": rag_confidence,
                }
            ],
        }

    except Exception as e:
        logger.error(f"[Research] Failed: {e}", exc_info=True)
        return {
            "research_results": [f"[Research error: {e}]"],
            "error": str(e),
            "messages": state["messages"] + [
                {"role": "research_agent", "content": f"Error: {e}"}
            ],
        }