"""
Risk Metrics Module
====================
Implements Historical VaR and Parametric (Variance-Covariance) VaR
with Expected Shortfall for each method.

Mathematical Foundation:
    Historical VaR:   Quantile of empirical loss distribution
    Parametric VaR:   VaR_α = z_α · σ_p - μ_p
    Expected Shortfall: ES = E[L | L > VaR]
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict


# ─────────────────────────────────────────────────────────────
# Historical VaR
# ─────────────────────────────────────────────────────────────

def compute_historical_var(
    portfolio_returns: pd.Series,
    confidence_level: float = 0.99,
) -> float:
    """
    Compute Value-at-Risk using Historical Simulation.

    Algorithm:
        1. Convert returns to losses: L = -R_p
        2. Extract the quantile at confidence_level

    Parameters
    ----------
    portfolio_returns : pd.Series
        Historical portfolio returns.
    confidence_level : float
        Confidence level (default: 0.99).

    Returns
    -------
    float
        Historical VaR (positive = loss magnitude).
    """
    losses = -portfolio_returns.values
    var = np.percentile(losses, confidence_level * 100)
    return float(var)


def compute_historical_es(
    portfolio_returns: pd.Series,
    confidence_level: float = 0.99,
) -> float:
    """
    Compute Expected Shortfall using Historical Simulation.

    ES = E[L | L > VaR]

    Parameters
    ----------
    portfolio_returns : pd.Series
        Historical portfolio returns.
    confidence_level : float
        Confidence level (default: 0.99).

    Returns
    -------
    float
        Historical ES (positive = loss magnitude).
    """
    losses = -portfolio_returns.values
    var = np.percentile(losses, confidence_level * 100)
    tail_losses = losses[losses >= var]
    return float(np.mean(tail_losses))


def historical_risk_metrics(
    portfolio_returns: pd.Series,
) -> Dict[str, float]:
    """
    Compute full suite of historical risk metrics.

    Parameters
    ----------
    portfolio_returns : pd.Series
        Portfolio return series.

    Returns
    -------
    dict
        VaR and ES at 95% and 99% confidence levels.
    """
    return {
        "hist_var_95": compute_historical_var(portfolio_returns, 0.95),
        "hist_var_99": compute_historical_var(portfolio_returns, 0.99),
        "hist_es_95": compute_historical_es(portfolio_returns, 0.95),
        "hist_es_99": compute_historical_es(portfolio_returns, 0.99),
    }


# ─────────────────────────────────────────────────────────────
# Parametric (Variance-Covariance) VaR
# ─────────────────────────────────────────────────────────────

def compute_parametric_var(
    portfolio_mean: float,
    portfolio_std: float,
    confidence_level: float = 0.99,
) -> float:
    """
    Compute Parametric VaR assuming Gaussian returns.

    Mathematical Definition:
        VaR_α = z_α · σ_p - μ_p

    Where z_α is the standard normal quantile.

    Parameters
    ----------
    portfolio_mean : float
        Daily portfolio mean return.
    portfolio_std : float
        Daily portfolio standard deviation.
    confidence_level : float
        Confidence level (default: 0.99).

    Returns
    -------
    float
        Parametric VaR (positive = loss magnitude).
    """
    z_alpha = stats.norm.ppf(confidence_level)
    var = z_alpha * portfolio_std - portfolio_mean
    return float(var)


def compute_parametric_es(
    portfolio_mean: float,
    portfolio_std: float,
    confidence_level: float = 0.99,
) -> float:
    """
    Compute Parametric Expected Shortfall under Gaussian assumption.

    Mathematical Definition:
        ES_α = μ_p + σ_p · φ(z_α) / (1 - α)

    Where φ is the standard normal PDF.

    Parameters
    ----------
    portfolio_mean : float
        Daily portfolio mean return.
    portfolio_std : float
        Daily portfolio standard deviation.
    confidence_level : float
        Confidence level.

    Returns
    -------
    float
        Parametric ES (positive = loss magnitude).
    """
    z_alpha = stats.norm.ppf(confidence_level)
    phi_z = stats.norm.pdf(z_alpha)
    es = -portfolio_mean + portfolio_std * phi_z / (1 - confidence_level)
    return float(es)


def parametric_risk_metrics(
    portfolio_mean: float,
    portfolio_std: float,
) -> Dict[str, float]:
    """
    Compute full suite of parametric risk metrics.

    Parameters
    ----------
    portfolio_mean : float
        Daily portfolio mean.
    portfolio_std : float
        Daily portfolio std dev.

    Returns
    -------
    dict
        VaR and ES at 95% and 99% confidence levels.
    """
    return {
        "param_var_95": compute_parametric_var(portfolio_mean, portfolio_std, 0.95),
        "param_var_99": compute_parametric_var(portfolio_mean, portfolio_std, 0.99),
        "param_es_95": compute_parametric_es(portfolio_mean, portfolio_std, 0.95),
        "param_es_99": compute_parametric_es(portfolio_mean, portfolio_std, 0.99),
    }
