"""Routing logic between LangGraph nodes."""
from app.orchestration.state import AgentState


def route_after_research(state: AgentState) -> str:
    """Decide next node after research agent completes."""
    if state.get("error"):
        return "end"
    # If portfolio provided, run full analysis; else go straight to report
    if state.get("portfolio"):
        return "analysis"
    return "report"
