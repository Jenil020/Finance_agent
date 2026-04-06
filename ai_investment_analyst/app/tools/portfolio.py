"""
Portfolio calculation tools for the Analysis Agent.

Two tiers of metrics:
  1. Static metrics  — cost basis, weights, concentration (pure Python, instant)
  2. Live metrics    — P&L, beta, Sharpe, drawdown (yfinance, needs network)

yfinance is synchronous — all calls are wrapped in asyncio.to_thread()
so they don't block FastAPI's event loop.

Live metrics gracefully degrade if yfinance is unavailable (network error,
bad ticker, etc.) — static metrics are always returned.
"""
import asyncio
from typing import List, Dict, Any
import numpy as np
import pandas as pd
import yfinance as yf

from app.api.schemas import PortfolioItem
from app.core.logging import logger

# ── Constants ─────────────────────────────────────────────────────────────────
BENCHMARK_TICKER = "SPY"          # S&P 500 ETF as market proxy
RISK_FREE_RATE = 0.05             # 5% annualised (approx current T-bill rate)
HISTORY_PERIOD = "1y"             # 1 year of daily data for beta / Sharpe
CONCENTRATION_WARNING = 0.40      # flag if any single position > 40% of portfolio


# ── Sync helpers (run inside asyncio.to_thread) ───────────────────────────────

def _fetch_prices(tickers: List[str]) -> pd.DataFrame:
    """
    Download daily Close prices for tickers + SPY benchmark.
    Returns a DataFrame with tickers as columns, dates as index.
    """
    all_tickers = list(set(tickers + [BENCHMARK_TICKER]))
    data = yf.download(
        all_tickers,
        period=HISTORY_PERIOD,
        interval="1d",
        auto_adjust=True,
        progress=False,
        multi_level_index=True,
    )
    if data.empty:
        return pd.DataFrame()
    # With multi_level_index=True, columns are (metric, ticker)
    closes = data["Close"] if "Close" in data else data
    return closes.dropna(how="all")


def _compute_live_metrics(
    tickers: List[str],
    weights: Dict[str, float],
    closes: pd.DataFrame,
) -> Dict[str, Any]:
    """
    Compute beta, Sharpe ratio, max drawdown, YTD P&L from price history.
    All metrics are per-ticker AND portfolio-level.
    """
    metrics: Dict[str, Any] = {}
    risk_free_daily = RISK_FREE_RATE / 252

    spy_closes = closes.get(BENCHMARK_TICKER)
    if spy_closes is None or len(spy_closes) < 20:
        return {"error": "Insufficient benchmark data"}

    spy_returns = spy_closes.pct_change().dropna()

    per_ticker: Dict[str, Dict] = {}
    portfolio_return_series = pd.Series(0.0, index=spy_returns.index)

    for ticker in tickers:
        if ticker not in closes.columns:
            per_ticker[ticker] = {"error": "Price data unavailable"}
            continue

        price_series = closes[ticker].dropna()
        if len(price_series) < 20:
            per_ticker[ticker] = {"error": "Too few data points"}
            continue

        returns = price_series.pct_change().dropna()
        aligned = returns.align(spy_returns, join="inner")
        stock_ret, bench_ret = aligned[0], aligned[1]

        # ── Beta ────────────────────────────────────────────────────────────
        cov_matrix = np.cov(stock_ret, bench_ret)
        beta = float(cov_matrix[0, 1] / cov_matrix[1, 1]) if cov_matrix[1, 1] != 0 else None

        # ── Sharpe Ratio ────────────────────────────────────────────────────
        excess = stock_ret - risk_free_daily
        sharpe = float(excess.mean() / excess.std() * np.sqrt(252)) if excess.std() != 0 else None

        # ── Max Drawdown ────────────────────────────────────────────────────
        cumulative = (1 + stock_ret).cumprod()
        rolling_max = cumulative.cummax()
        drawdown = (cumulative - rolling_max) / rolling_max
        max_drawdown = float(drawdown.min())

        # ── Annualised Return ───────────────────────────────────────────────
        ann_return = float((1 + stock_ret.mean()) ** 252 - 1)

        # ── Current price + YTD change ──────────────────────────────────────
        current_price = float(price_series.iloc[-1])
        ytd_change = float(
            (price_series.iloc[-1] - price_series.iloc[0]) / price_series.iloc[0]
        )

        per_ticker[ticker] = {
            "current_price": round(current_price, 2),
            "ytd_change_pct": round(ytd_change * 100, 2),
            "beta": round(beta, 3) if beta is not None else None,
            "sharpe_ratio": round(sharpe, 3) if sharpe is not None else None,
            "max_drawdown_pct": round(max_drawdown * 100, 2),
            "annualised_return_pct": round(ann_return * 100, 2),
        }

        # Accumulate weighted portfolio returns
        w = weights.get(ticker, 0.0)
        portfolio_return_series = portfolio_return_series.add(
            stock_ret * w, fill_value=0
        )

    # ── Portfolio-level metrics ──────────────────────────────────────────────
    port_ret = portfolio_return_series.dropna()
    if len(port_ret) > 20:
        port_excess = port_ret - risk_free_daily
        port_sharpe = float(port_excess.mean() / port_excess.std() * np.sqrt(252)) \
            if port_excess.std() != 0 else None

        port_cumulative = (1 + port_ret).cumprod()
        port_rolling_max = port_cumulative.cummax()
        port_drawdown = (port_cumulative - port_rolling_max) / port_rolling_max
        port_max_drawdown = float(port_drawdown.min())
        port_ytd = float(port_cumulative.iloc[-1] - 1)

        metrics["portfolio_level"] = {
            "ytd_return_pct": round(port_ytd * 100, 2),
            "sharpe_ratio": round(port_sharpe, 3) if port_sharpe is not None else None,
            "max_drawdown_pct": round(port_max_drawdown * 100, 2),
        }

    metrics["per_ticker"] = per_ticker
    return metrics


# ── Public tool ───────────────────────────────────────────────────────────────

async def calculate_portfolio_metrics(
    portfolio: List[PortfolioItem],
) -> Dict[str, Any]:
    """
    Full portfolio analysis tool for the Analysis Agent.

    Always returns static metrics (instant).
    Attempts live metrics via yfinance; gracefully degrades on network failure.

    Args:
        portfolio: List of PortfolioItem(ticker, quantity, avg_cost)

    Returns:
        Dict with static_metrics, live_metrics, and risk_flags sections.
    """
    if not portfolio:
        return {}

    # ── 1. Static metrics (always available) ──────────────────────────────
    total_cost = sum(item.quantity * item.avg_cost for item in portfolio)
    tickers = [item.ticker.upper() for item in portfolio]

    weights = {
        item.ticker.upper(): round((item.quantity * item.avg_cost) / total_cost, 4)
        for item in portfolio
    }

    positions = {
        item.ticker.upper(): {
            "quantity": item.quantity,
            "avg_cost": item.avg_cost,
            "cost_basis": round(item.quantity * item.avg_cost, 2),
            "weight_pct": round(weights[item.ticker.upper()] * 100, 2),
        }
        for item in portfolio
    }

    # ── Concentration risk flag ────────────────────────────────────────────
    max_weight_ticker = max(weights, key=lambda t: weights[t])
    concentration_risk = weights[max_weight_ticker] > CONCENTRATION_WARNING

    static = {
        "total_cost_basis": round(total_cost, 2),
        "num_positions": len(portfolio),
        "tickers": tickers,
        "weights": weights,
        "positions": positions,
    }

    risk_flags = {
        "concentration_risk": concentration_risk,
        "concentrated_ticker": max_weight_ticker if concentration_risk else None,
        "max_weight_pct": round(weights[max_weight_ticker] * 100, 2),
    }

    # ── 2. Live metrics via yfinance ──────────────────────────────────────
    live: Dict[str, Any] = {}
    try:
        logger.info(f"[Portfolio] Fetching price history for {tickers}")
        closes = await asyncio.to_thread(_fetch_prices, tickers)

        if closes.empty:
            logger.warning("[Portfolio] yfinance returned no data")
            live = {"error": "Price data unavailable (check tickers or network)"}
        else:
            live = _compute_live_metrics(tickers, weights, closes)
            logger.info(f"[Portfolio] Live metrics computed for {len(tickers)} tickers")

    except Exception as e:
        logger.error(f"[Portfolio] yfinance failed: {e}")
        live = {"error": str(e)}

    return {
        "static_metrics": static,
        "live_metrics": live,
        "risk_flags": risk_flags,
    }


async def get_ticker_summary(ticker: str) -> Dict[str, Any]:
    """
    Quick single-ticker summary using yfinance fast_info.
    Used by the Research Agent for context when analysing a specific stock.
    """
    ticker = ticker.upper()
    logger.info(f"[Portfolio] Fetching summary for {ticker}")

    def _fetch():
        t = yf.Ticker(ticker)
        fi = t.fast_info
        info = {}
        for field in [
            "last_price", "previous_close", "day_high", "day_low",
            "market_cap", "fifty_day_average", "two_hundred_day_average",
            "year_high", "year_low", "year_change",
        ]:
            try:
                val = getattr(fi, field, None)
                if val is not None:
                    info[field] = round(float(val), 2) if isinstance(val, float) else val
            except Exception:
                pass
        return info

    try:
        summary = await asyncio.to_thread(_fetch)
        summary["ticker"] = ticker
        return summary
    except Exception as e:
        logger.error(f"[Portfolio] Ticker summary failed for {ticker}: {e}")
        return {"ticker": ticker, "error": str(e)}