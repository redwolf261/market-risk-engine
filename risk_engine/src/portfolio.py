"""
Portfolio Construction Module
=============================
Handles data ingestion, return computation, weight definition,
and portfolio return aggregation.

Mathematical Foundation:
    Log return:    r_t = ln(P_t / P_{t-1})
    Simple return: r_t = (P_t - P_{t-1}) / P_{t-1}
    Portfolio return:  R_p = w^T * R

Design note:
    Log returns are used for simulation and volatility estimation
    (their additive property makes Cholesky decomposition valid).
    Simple returns are used for P&L attribution and tail-shape
    diagnostics (skewness, kurtosis) that are distorted by the
    log transformation on large moves.
"""

import warnings
import numpy as np
import pandas as pd
import yfinance as yf
from scipy import stats as scipy_stats
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
    window: Optional[int] = None,
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
    window : int, optional
        If given, return only the *last* `window` trading days before
        `end`.  Useful for rolling backtests and custom horizons without
        changing the start date globally.

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

    if window is not None:
        prices = prices.iloc[-window:]

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

    Used for: simulation inputs, volatility/covariance estimation,
    Cholesky decomposition (additive property).

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


def compute_simple_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Compute simple (arithmetic) returns from price series.

    Mathematical Definition:
        r_t = (P_t - P_{t-1}) / P_{t-1}

    Used for: P&L attribution, tail-shape diagnostics (skewness,
    kurtosis), and dollar P&L computations where log approximation
    breaks down for large moves.

    Parameters
    ----------
    prices : pd.DataFrame
        DataFrame of asset prices.

    Returns
    -------
    pd.DataFrame
        DataFrame of simple returns (first row dropped).
    """
    return prices.pct_change().dropna()


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
        If weights do not sum to approximately 1, or if any ticker in
    `tickers` has no corresponding weight defined.
    """
    if custom_weights is None:
        custom_weights = DEFAULT_WEIGHTS

    if tickers is None:
        # Sort alphabetically so the order matches yfinance column order
        # (yfinance downloads always return columns in A→Z order)
        tickers = sorted(custom_weights.keys())

    # pd.Series.reindex guarantees element-by-element alignment regardless
    # of the insertion order of the source dict.  This is the ONLY safe way
    # to build a weight vector for a DataFrame whose column order may differ
    # from the order the dict was written in.
    weight_series = pd.Series(custom_weights).reindex(tickers)

    if weight_series.isna().any():
        missing = weight_series[weight_series.isna()].index.tolist()
        raise ValueError(
            f"Weight alignment failed — no weight defined for tickers: {missing}"
        )

    weights = weight_series.values

    weight_sum = weights.sum()

    # Tight tolerance consistent with numerical precision requirements
    # of multi-asset covariance operations.
    if not np.isclose(weight_sum, 1.0, rtol=1e-8, atol=1e-8):
        if abs(weight_sum - 1.0) < 1e-4:
            # Small floating-point drift (e.g., 0.9999999) — silently
            # normalize so downstream operations remain numerically stable.
            warnings.warn(
                f"Weights sum to {weight_sum:.10f}; auto-normalizing.",
                UserWarning,
                stacklevel=2,
            )
            weights = weights / weight_sum
        else:
            raise ValueError(
                f"Weights must sum to 1.0, got {weight_sum:.6f}"
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

    Includes both performance metrics (return, Sharpe) and risk-oriented
    diagnostics (downside volatility, skewness, excess kurtosis) that
    are essential for understanding tail behaviour in a risk engine.

    Skewness < 0 (left skew) and excess kurtosis > 0 (leptokurtic) both
    indicate heavier-than-Gaussian left tails — precisely the regime
    where Gaussian VaR underestimates true risk.

    Parameters
    ----------
    prices : pd.DataFrame
        Asset prices.
    weights : np.ndarray
        Portfolio weights.

    Returns
    -------
    dict
        Performance metrics, downside risk metrics, and distribution
        shape statistics.
    """
    log_ret = compute_log_returns(prices)
    simple_ret = compute_simple_returns(prices)

    port_ret = compute_portfolio_returns(log_ret, weights)

    # ── Performance metrics ───────────────────────────────────
    ann_return = port_ret.mean() * TRADING_DAYS_PER_YEAR
    ann_vol = port_ret.std() * np.sqrt(TRADING_DAYS_PER_YEAR)
    sharpe = ann_return / ann_vol if ann_vol > 0 else 0.0

    # ── Downside risk metrics ─────────────────────────────────
    # Downside deviation: volatility of *negative* returns only.
    # Aligns with CVaR/ES semantics — penalises losses, not gains.
    losses = port_ret[port_ret < 0]
    downside_vol_daily = losses.std() if len(losses) > 1 else 0.0
    ann_downside_vol = downside_vol_daily * np.sqrt(TRADING_DAYS_PER_YEAR)
    sortino = ann_return / ann_downside_vol if ann_downside_vol > 0 else 0.0

    # ── Distribution shape (on simple returns for P&L accuracy) ──
    port_simple = compute_portfolio_returns(simple_ret, weights)
    skewness = float(scipy_stats.skew(port_simple.values))
    # excess_kurtosis = 0 for a Gaussian; > 0 means heavier tails
    excess_kurtosis = float(scipy_stats.kurtosis(port_simple.values))

    return {
        "annualized_return": ann_return,
        "annualized_volatility": ann_vol,
        "sharpe_ratio": sharpe,
        "annualized_downside_vol": ann_downside_vol,
        "sortino_ratio": sortino,
        "skewness": skewness,
        "excess_kurtosis": excess_kurtosis,
        "num_observations": len(port_ret),
    }
