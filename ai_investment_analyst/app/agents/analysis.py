"""
Analysis Agent — Quantitative analysis layer.

Responsibilities:
  1. Run portfolio metrics tool (beta, Sharpe, drawdown, weights)
  2. Interpret the numbers in the context of the research brief
  3. Produce a structured quantitative analysis for the Report Agent

If no portfolio is provided the agent performs a general market/stock
analysis using only the research findings.
"""
import json
from app.orchestration.state import AgentState
from app.agents.base import llm_call
from app.tools.portfolio import calculate_portfolio_metrics
from app.core.logging import logger

# ── System prompt ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are a quantitative analyst (quant) at a hedge fund.
You specialise in risk-adjusted return analysis, position sizing, and portfolio construction.
Always ground your analysis in the numbers provided. Flag data quality issues.
Express risk metrics precisely (e.g. "beta of 1.34 implies 34% more volatility than SPY")."""

# ── Prompt: with portfolio ────────────────────────────────────────────────────
PORTFOLIO_ANALYSIS_PROMPT = """QUANTITATIVE ANALYSIS TASK
===========================
Original Query: {query}

RESEARCH BRIEF (from Research Agent):
{research}

PORTFOLIO SNAPSHOT:
{portfolio_summary}

COMPUTED METRICS:
{metrics_json}

INSTRUCTIONS
------------
Write a quantitative analysis (400-600 words) covering:

  RISK PROFILE:
    - Interpret beta for each position (vs SPY benchmark)
    - Max drawdown and what it means in dollar terms
    - Concentration risk (flag any position > 40% of portfolio)

  RETURN ANALYSIS:
    - Sharpe ratio interpretation (< 1 = poor, 1-2 = good, > 2 = excellent)
    - YTD return vs benchmark
    - Annualised return trajectory

  POSITION SIZING:
    - Are current weights optimal given the risk profile?
    - Which positions are oversized / undersized?
    - Suggest rebalancing targets (keep as percentages, no specific dollar advice)

  KEY SIGNALS:
    - 2-3 quantitative signals that should drive the final recommendation

Be specific with numbers. Reference the metrics provided above."""

# ── Prompt: no portfolio ──────────────────────────────────────────────────────
STOCK_ANALYSIS_PROMPT = """STOCK ANALYSIS TASK
====================
Original Query: {query}

RESEARCH BRIEF (from Research Agent):
{research}

INSTRUCTIONS
------------
Write a quantitative assessment (300-400 words) covering:

  VALUATION SIGNALS:
    - Any P/E, P/S, EV/EBITDA data mentioned in the research
    - How does this compare to sector averages?

  MOMENTUM SIGNALS:
    - Price vs 50-day and 200-day moving averages (golden/death cross)
    - YTD performance vs benchmark

  RISK FACTORS:
    - Key quantitative risks (leverage, liquidity, earnings volatility)

  INVESTMENT THESIS STRENGTH:
    - Rate the bull case 1-5 based solely on quantitative evidence
    - Rate the bear case 1-5 based solely on quantitative evidence

Be precise. Use numbers from the research brief where available."""


def _format_portfolio_summary(portfolio) -> str:
    """Format portfolio items into a readable table string."""
    if not portfolio:
        return "No portfolio provided."
    lines = ["Ticker  |  Qty   |  Avg Cost  |  Cost Basis"]
    lines.append("-" * 44)
    for item in portfolio:
        cost = item.quantity * item.avg_cost
        lines.append(
            f"{item.ticker:<8}|{item.quantity:>7.1f} |${item.avg_cost:>9.2f} |${cost:>10.2f}"
        )
    return "\n".join(lines)


# ── Agent node ────────────────────────────────────────────────────────────────

async def analysis_agent_node(state: AgentState) -> dict:
    """
    LangGraph node: Analysis Agent.

    Returns partial state — updates analysis_results and messages.
    """
    query = state["query"]
    portfolio = state.get("portfolio") or []
    research = "\n".join(state.get("research_results", ["No research available."]))

    logger.info(f"[Analysis] Starting | has_portfolio={bool(portfolio)}")

    try:
        if portfolio:
            # ── Portfolio path: compute full metrics ─────────────────────────
            logger.info(f"[Analysis] Computing metrics for {len(portfolio)} positions")
            raw_metrics = await calculate_portfolio_metrics(portfolio)

            # Pretty-print metrics for LLM consumption
            metrics_json = json.dumps(raw_metrics, indent=2, default=str)
            portfolio_summary = _format_portfolio_summary(portfolio)

            prompt = PORTFOLIO_ANALYSIS_PROMPT.format(
                query=query,
                research=research,
                portfolio_summary=portfolio_summary,
                metrics_json=metrics_json,
            )

            analysis_text = await llm_call(
                prompt=prompt,
                system_instruction=SYSTEM_PROMPT,
                temperature=0.15,
                max_output_tokens=1500,
            )

            return {
                "analysis_results": {
                    "summary": analysis_text,
                    "metrics": raw_metrics,
                    "has_portfolio": True,
                },
                "messages": state["messages"] + [
                    {
                        "role": "analysis_agent",
                        "content": analysis_text,
                        "metrics_computed": True,
                    }
                ],
            }

        else:
            # ── No-portfolio path: qualitative quant assessment ──────────────
            logger.info("[Analysis] No portfolio — running stock-only analysis")

            prompt = STOCK_ANALYSIS_PROMPT.format(
                query=query,
                research=research,
            )

            analysis_text = await llm_call(
                prompt=prompt,
                system_instruction=SYSTEM_PROMPT,
                temperature=0.15,
                max_output_tokens=1000,
            )

            return {
                "analysis_results": {
                    "summary": analysis_text,
                    "metrics": {},
                    "has_portfolio": False,
                },
                "messages": state["messages"] + [
                    {
                        "role": "analysis_agent",
                        "content": analysis_text,
                        "metrics_computed": False,
                    }
                ],
            }

    except Exception as e:
        logger.error(f"[Analysis] Failed: {e}", exc_info=True)
        return {
            "analysis_results": {"summary": f"[Analysis error: {e}]", "metrics": {}},
            "error": str(e),
            "messages": state["messages"] + [
                {"role": "analysis_agent", "content": f"Error: {e}"}
            ],
        }