"""
Portfolio calculation tools for the Analysis Agent.
Pure Python — no external API needed.
"""
from typing import List, Dict, Any
from app.api.schemas import PortfolioItem


async def calculate_portfolio_metrics(
    portfolio: List[PortfolioItem],
) -> Dict[str, Any]:
    """
    Calculate basic portfolio metrics.
    In production: integrate with yfinance for live prices.
    """
    if not portfolio:
        return {}

    total_cost = sum(item.quantity * item.avg_cost for item in portfolio)
    tickers = [item.ticker for item in portfolio]
    weights = {
        item.ticker: (item.quantity * item.avg_cost) / total_cost
        for item in portfolio
    }

    return {
        "total_cost_basis": round(total_cost, 2),
        "tickers": tickers,
        "weights": weights,
        "num_positions": len(portfolio),
        # TODO: Fetch live prices via yfinance and compute P&L, beta, etc.
    }
