"""
Backtesting Module
==================
Implements rolling-window VaR backtesting with breach analysis
and the Kupiec Proportion of Failures (POF) test.

Framework:
    1. Use 250-day rolling window to estimate μ, Σ
    2. Predict next-day 99% VaR (Parametric)
    3. Compare predicted VaR vs actual realized loss
    4. Count breaches and compute breach ratio
    5. Kupiec likelihood ratio test for model adequacy
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, Tuple

from src.risk_metrics import compute_parametric_var
from src.statistics import (
    compute_mean_vector,
    compute_covariance_matrix,
)


# ─────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────
DEFAULT_WINDOW: int = 250
DEFAULT_CONFIDENCE: float = 0.99


def rolling_var_backtest(
    returns: pd.DataFrame,
    weights: np.ndarray,
    window: int = DEFAULT_WINDOW,
    confidence_level: float = DEFAULT_CONFIDENCE,
) -> pd.DataFrame:
    """
    Perform rolling-window VaR backtesting.

    Algorithm:
        For each day t (starting from index `window`):
            1. Estimate μ and Σ from days [t - window, t)
            2. Compute portfolio σ_p from w^T Σ w
            3. Compute 1-day Parametric VaR at given confidence
            4. Record actual portfolio return at day t
            5. Flag if actual loss exceeds VaR (breach)

    Parameters
    ----------
    returns : pd.DataFrame
        Daily log returns of all assets (T x N).
    weights : np.ndarray
        Portfolio weight vector (N,).
    window : int
        Rolling estimation window size (default: 250 trading days).
    confidence_level : float
        VaR confidence level (default: 0.99).

    Returns
    -------
    pd.DataFrame
        Columns: date, predicted_var, actual_return, actual_loss, breach
    """
    n_obs = len(returns)
    results = []

    for t in range(window, n_obs):
        # Estimation window
        window_returns = returns.iloc[t - window : t]

        # Estimate parameters
        mu = compute_mean_vector(window_returns)
        cov = compute_covariance_matrix(window_returns)

        # Portfolio statistics
        port_mean = float(weights @ mu)
        port_var = float(weights @ cov @ weights)
        port_std = np.sqrt(port_var)

        # Predicted VaR
        predicted_var = compute_parametric_var(
            port_mean, port_std, confidence_level
        )

        # Actual portfolio return on day t
        actual_return = float(returns.iloc[t].values @ weights)
        actual_loss = -actual_return

        # Breach detection
        breach = actual_loss > predicted_var

        results.append({
            "date": returns.index[t],
            "predicted_var": predicted_var,
            "actual_return": actual_return,
            "actual_loss": actual_loss,
            "breach": breach,
        })

    return pd.DataFrame(results)


def compute_breach_statistics(
    backtest_results: pd.DataFrame,
    confidence_level: float = DEFAULT_CONFIDENCE,
) -> Dict[str, float]:
    """
    Compute breach analysis statistics.

    Parameters
    ----------
    backtest_results : pd.DataFrame
        Output from rolling_var_backtest.
    confidence_level : float
        VaR confidence level used.

    Returns
    -------
    dict
        Contains: total_observations, num_breaches, breach_rate,
        expected_breach_rate, breach_ratio.
    """
    total = len(backtest_results)
    breaches = int(backtest_results["breach"].sum())
    breach_rate = breaches / total if total > 0 else 0.0
    expected_rate = 1 - confidence_level

    return {
        "total_observations": total,
        "num_breaches": breaches,
        "breach_rate": breach_rate,
        "expected_breach_rate": expected_rate,
        "breach_ratio": breach_rate / expected_rate if expected_rate > 0 else 0.0,
    }


def kupiec_test(
    backtest_results: pd.DataFrame,
    confidence_level: float = DEFAULT_CONFIDENCE,
) -> Dict[str, float]:
    """
    Kupiec Proportion of Failures (POF) test.

    Null hypothesis: actual breach rate = expected breach rate.

    Likelihood Ratio Statistic:
        LR = -2 ln[(1-p)^(T-x) · p^x] + 2 ln[(1-x/T)^(T-x) · (x/T)^x]

    Where:
        p = 1 - confidence_level (expected failure rate)
        x = number of breaches
        T = total observations

    Under H0, LR ~ χ²(1).

    Parameters
    ----------
    backtest_results : pd.DataFrame
        Output from rolling_var_backtest.
    confidence_level : float
        VaR confidence level.

    Returns
    -------
    dict
        Contains: lr_statistic, p_value, reject_h0 (at 5% significance).
    """
    T = len(backtest_results)
    x = int(backtest_results["breach"].sum())
    p = 1 - confidence_level  # Expected failure rate

    if x == 0 or x == T:
        # Edge case: no breaches or all breaches
        return {
            "lr_statistic": 0.0,
            "p_value": 1.0,
            "reject_h0": False,
            "interpretation": "Insufficient breaches for test",
        }

    # Observed failure rate
    p_hat = x / T

    # Log-likelihood ratio
    lr = -2 * (
        (T - x) * np.log(1 - p) + x * np.log(p)
        - (T - x) * np.log(1 - p_hat) - x * np.log(p_hat)
    )

    # p-value from chi-squared distribution (1 degree of freedom)
    p_value = 1 - stats.chi2.cdf(lr, df=1)

    reject = p_value < 0.05

    if reject:
        interpretation = "Model rejected — VaR estimates may be inadequate"
    else:
        interpretation = "Model not rejected — VaR estimates appear adequate"

    return {
        "lr_statistic": float(lr),
        "p_value": float(p_value),
        "reject_h0": reject,
        "interpretation": interpretation,
    }


def run_full_backtest(
    returns: pd.DataFrame,
    weights: np.ndarray,
    window: int = DEFAULT_WINDOW,
    confidence_level: float = DEFAULT_CONFIDENCE,
) -> Tuple[pd.DataFrame, Dict[str, float], Dict[str, float]]:
    """
    Execute complete backtesting pipeline.

    Parameters
    ----------
    returns : pd.DataFrame
        Daily log returns.
    weights : np.ndarray
        Portfolio weights.
    window : int
        Rolling estimation window.
    confidence_level : float
        VaR confidence level.

    Returns
    -------
    tuple
        (backtest_results_df, breach_stats, kupiec_results)
    """
    bt_results = rolling_var_backtest(
        returns, weights, window, confidence_level
    )
    breach_stats = compute_breach_statistics(bt_results, confidence_level)
    kupiec_results = kupiec_test(bt_results, confidence_level)

    return bt_results, breach_stats, kupiec_results
