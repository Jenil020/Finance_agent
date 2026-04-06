"""Unit tests for agent nodes."""
import pytest
from unittest.mock import AsyncMock, patch


@pytest.mark.asyncio
async def test_research_agent_returns_results():
    """Research agent should populate research_results."""
    from app.agents.research import research_agent_node
    state = {
        "query": "Should I buy Apple stock?",
        "session_id": "test-123",
        "portfolio": [],
        "messages": [],
        "research_results": [],
        "analysis_results": {},
        "final_report": None,
        "error": None,
        "stream_token": None,
    }
    with patch("app.agents.research.llm_call_with_retry", new_callable=AsyncMock) as mock_llm, \
         patch("app.agents.research.rag_query_tool", new_callable=AsyncMock) as mock_rag, \
         patch("app.agents.research.web_search_tool", new_callable=AsyncMock) as mock_web:
        mock_llm.return_value = "Apple looks strong with solid fundamentals."
        mock_rag.return_value = "RAG result: AAPL P/E ratio is 28."
        mock_web.return_value = "Apple Q4 earnings beat estimates."
        result = await research_agent_node(state)
    assert len(result["research_results"]) > 0


@pytest.mark.asyncio
async def test_report_agent_validates_schema():
    """Report agent must produce Pydantic-valid InvestmentReport."""
    import json
    from app.agents.report import report_agent_node
    valid_json = json.dumps({
        "ticker": "AAPL",
        "recommendation": "BUY",
        "confidence": 0.78,
        "summary": "Apple is a strong buy.",
        "key_risks": ["macro headwinds"],
        "key_catalysts": ["AI integration"],
        "target_price": 220.0,
        "sources": ["Reuters"],
    })
    state = {
        "query": "Analyse AAPL",
        "session_id": "test-456",
        "portfolio": [],
        "messages": [],
        "research_results": ["Apple is innovating rapidly."],
        "analysis_results": {"summary": "Metrics look good."},
        "final_report": None,
        "error": None,
        "stream_token": None,
    }
    with patch("app.agents.report.llm_call_with_retry", new_callable=AsyncMock) as mock_llm:
        mock_llm.return_value = valid_json
        result = await report_agent_node(state)
    assert result["final_report"] is not None
    assert result["final_report"].recommendation in ("BUY", "SELL", "HOLD")
