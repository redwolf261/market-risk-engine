"""
Portfolio Construction Module
=============================
Handles data ingestion, log return computation, weight definition,
and portfolio return aggregation.

Mathematical Foundation:
    Log return:  r_t = ln(P_t / P_{t-1})
    Portfolio return:  R_p = w^T * R
"""

import numpy as np
import pandas as pd
import yfinance as yf
from typing import Dict, List, Optional, Tuple


# ─────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────
DEFAULT_TICKERS: List[str] = ["SPY", "QQQ", "JPM", "TLT", "GLD"]

DEFAULT_WEIGHTS: Dict[str, float] = {
    "SPY": 0.30,   # Broad equity index
    "QQQ": 0.20,   # High-beta tech
    "JPM": 0.15,   # Financial sector
    "TLT": 0.20,   # Long-duration bonds
    "GLD": 0.15,   # Gold hedge
}

TRADING_DAYS_PER_YEAR: int = 252


def fetch_data(
    tickers: List[str] = DEFAULT_TICKERS,
    start: str = "2021-01-01",
    end: str = "2026-01-01",
    save_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Download adjusted close prices from Yahoo Finance.

    Parameters
    ----------
    tickers : list of str
        Ticker symbols to download.
    start : str
        Start date in YYYY-MM-DD format.
    end : str
        End date in YYYY-MM-DD format.
    save_path : str, optional
        If provided, saves the DataFrame as CSV.

    Returns
    -------
    pd.DataFrame
        Adjusted close prices indexed by date.
    """
    raw = yf.download(tickers, start=start, end=end, auto_adjust=True)

    # Handle multi-level columns from yfinance
    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"].copy()
    else:
        prices = raw[["Close"]].copy()
        prices.columns = tickers

    prices.dropna(inplace=True)

    if save_path:
        prices.to_csv(save_path)

    return prices


def load_data(path: str) -> pd.DataFrame:
    """
    Load price data from CSV file.

    Parameters
    ----------
    path : str
        Path to the CSV file with Date index and asset columns.

    Returns
    -------
    pd.DataFrame
        Price DataFrame indexed by date.
    """
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df.dropna(inplace=True)
    return df


def compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Compute logarithmic returns from price series.

    Mathematical Definition:
        r_t = ln(P_t / P_{t-1})

    Parameters
    ----------
    prices : pd.DataFrame
        DataFrame of asset prices.

    Returns
    -------
    pd.DataFrame
        DataFrame of log returns (first row dropped).
    """
    log_returns = np.log(prices / prices.shift(1)).dropna()
    return log_returns


def define_weights(
    custom_weights: Optional[Dict[str, float]] = None,
    tickers: Optional[List[str]] = None,
) -> np.ndarray:
    """
    Define and validate portfolio weights.

    Parameters
    ----------
    custom_weights : dict, optional
        Dictionary mapping ticker -> weight. Must sum to 1.
    tickers : list of str, optional
        Ordered list of tickers (used with default weights).

    Returns
    -------
    np.ndarray
        Weight vector aligned with asset order.

    Raises
    ------
    ValueError
        If weights do not sum to approximately 1.
    """
    if custom_weights is None:
        custom_weights = DEFAULT_WEIGHTS

    if tickers is None:
        tickers = list(custom_weights.keys())

    weights = np.array([custom_weights[t] for t in tickers])

    if not np.isclose(weights.sum(), 1.0, atol=1e-6):
        raise ValueError(
            f"Weights must sum to 1.0, got {weights.sum():.6f}"
        )

    return weights


def compute_portfolio_returns(
    returns: pd.DataFrame, weights: np.ndarray
) -> pd.Series:
    """
    Compute portfolio returns as weighted sum of asset returns.

    Mathematical Definition:
        R_p = w^T * R

    Parameters
    ----------
    returns : pd.DataFrame
        Asset log returns (T x N).
    weights : np.ndarray
        Weight vector (N,).

    Returns
    -------
    pd.Series
        Portfolio return series.
    """
    portfolio_returns = returns.values @ weights
    return pd.Series(
        portfolio_returns, index=returns.index, name="portfolio_return"
    )


def get_portfolio_summary(
    prices: pd.DataFrame, weights: np.ndarray
) -> Dict[str, float]:
    """
    Compute summary statistics for the portfolio.

    Parameters
    ----------
    prices : pd.DataFrame
        Asset prices.
    weights : np.ndarray
        Portfolio weights.

    Returns
    -------
    dict
        Dictionary with annualized return, volatility, and Sharpe ratio.
    """
    returns = compute_log_returns(prices)
    port_ret = compute_portfolio_returns(returns, weights)

    ann_return = port_ret.mean() * TRADING_DAYS_PER_YEAR
    ann_vol = port_ret.std() * np.sqrt(TRADING_DAYS_PER_YEAR)
    sharpe = ann_return / ann_vol if ann_vol > 0 else 0.0

    return {
        "annualized_return": ann_return,
        "annualized_volatility": ann_vol,
        "sharpe_ratio": sharpe,
        "num_observations": len(port_ret),
    }
