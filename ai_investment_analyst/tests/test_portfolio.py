"""Tests for portfolio calculation tools."""
import pytest
from app.tools.portfolio import calculate_portfolio_metrics
from app.api.schemas import PortfolioItem


@pytest.mark.asyncio
async def test_portfolio_metrics_basic():
    portfolio = [
        PortfolioItem(ticker="AAPL", quantity=10, avg_cost=150.0),
        PortfolioItem(ticker="GOOGL", quantity=5, avg_cost=140.0),
    ]
    metrics = await calculate_portfolio_metrics(portfolio)
    assert metrics["total_cost_basis"] == 2200.0
    assert metrics["num_positions"] == 2
    assert abs(metrics["weights"]["AAPL"] - 0.6818) < 0.01


@pytest.mark.asyncio
async def test_empty_portfolio():
    metrics = await calculate_portfolio_metrics([])
    assert metrics == {}
