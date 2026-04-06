"""LangGraph shared state definition."""
from typing import TypedDict, List, Optional, Dict, Any
from app.api.schemas import PortfolioItem, InvestmentReport


class AgentState(TypedDict):
    query: str
    session_id: str
    portfolio: List[PortfolioItem]
    messages: List[Dict[str, Any]]       # Conversation history
    research_results: List[str]           # Web search / RAG results
    analysis_results: Dict[str, Any]      # Portfolio metrics
    final_report: Optional[InvestmentReport]
    error: Optional[str]
    stream_token: Optional[str]           # For SSE streaming
