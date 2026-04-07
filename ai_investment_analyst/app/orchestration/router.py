"""
LangGraph routing logic — decides which node to visit next.

Graph edges:
  research  →  route_after_research()  →  "analysis" | "report" | "end"
  analysis  →  report   (always, fixed edge)
  report    →  END      (always, fixed edge)

Routing rules for route_after_research:
  - error set by Research Agent  → "end"   (abort early, surface the error)
  - portfolio present            → "analysis" (full quant path)
  - no portfolio                 → "report"   (skip straight to recommendation)
"""
from app.orchestration.state import AgentState
from app.core.logging import logger


def route_after_research(state: AgentState) -> str:
    """
    Called by LangGraph after the research node completes.
    Returns the name of the next node to execute.
    """
    error = state.get("error")
    portfolio = state.get("portfolio")

    if error:
        logger.warning(f"[Router] Research error detected — aborting graph. error='{error[:80]}'")
        return "end"

    if portfolio:
        logger.info(f"[Router] Portfolio with {len(portfolio)} positions — routing to Analysis Agent")
        return "analysis"

    logger.info("[Router] No portfolio — routing directly to Report Agent")
    return "report"