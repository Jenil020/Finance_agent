"""
LangGraph shared state definition.

Design decisions:
  - messages + research_results use Annotated[List, operator.add] reducers.
    This means each agent node returns ONLY its new items — LangGraph
    appends them automatically. No need for state['x'] + [new] in agents.

  - All other fields (query, session_id, portfolio, analysis_results,
    final_report, error) are plain types — last-write-wins replacement.

  - stream_token removed: SSE streaming is handled via astream_events()
    in graph.py, not by passing tokens through state.
"""
import operator
from typing import TypedDict, List, Annotated, Optional, Dict, Any
from app.api.schemas import PortfolioItem, InvestmentReport


class AgentState(TypedDict):
    # ── Immutable inputs (set once at graph entry, never changed) ────────────
    query: str
    session_id: str
    portfolio: List[PortfolioItem]

    # ── Append-only lists (Annotated reducer — agents return new items only) ─
    messages: Annotated[List[Dict[str, Any]], operator.add]
    research_results: Annotated[List[str], operator.add]

    # ── Replace fields (last agent to write wins) ────────────────────────────
    analysis_results: Dict[str, Any]        # set by Analysis Agent
    final_report: Optional[InvestmentReport] # set by Report Agent
    error: Optional[str]                    # set by any agent on failure