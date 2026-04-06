"""
Analysis Agent — Portfolio calculations and financial analysis.
Uses tool calling for quantitative computations.
"""
from app.orchestration.state import AgentState
from app.agents.base import get_gemini_model, llm_call_with_retry
from app.tools.portfolio import calculate_portfolio_metrics
from app.core.logging import logger

ANALYSIS_PROMPT = """You are a quantitative investment analyst.

Query: {query}
Research findings: {research}
Portfolio: {portfolio}
Portfolio Metrics: {metrics}

Provide a detailed analysis including:
- Risk assessment (volatility, beta, correlation)
- Return analysis (current vs benchmark)
- Position sizing recommendations
- Entry/exit considerations

Be specific with numbers."""


async def analysis_agent_node(state: AgentState) -> AgentState:
    """LangGraph node: Analysis Agent."""
    logger.info("[Analysis Agent] Running portfolio analysis")

    model = get_gemini_model()

    # Calculate portfolio metrics using tool
    metrics = {}
    if state.get("portfolio"):
        metrics = await calculate_portfolio_metrics(state["portfolio"])

    prompt = ANALYSIS_PROMPT.format(
        query=state["query"],
        research="\n".join(state.get("research_results", [])),
        portfolio=state.get("portfolio", []),
        metrics=metrics,
    )

    analysis = await llm_call_with_retry(model, prompt)

    return {
        **state,
        "analysis_results": {"summary": analysis, "metrics": metrics},
        "messages": state["messages"] + [
            {"role": "analysis_agent", "content": analysis}
        ],
    }
