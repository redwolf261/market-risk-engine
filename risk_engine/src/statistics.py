"""
Statistical Estimation Module
==============================
Computes mean return vector, covariance matrix, correlation matrix,
and portfolio-level statistics using NumPy linear algebra.

Mathematical Foundation:
    Mean:        μ = E[r]
    Covariance:  Σ = E[(r - μ)(r - μ)^T]
    Portfolio σ²: σ_p² = w^T Σ w
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple


def compute_mean_vector(returns: pd.DataFrame) -> np.ndarray:
    """
    Compute the annualized mean return vector.

    Parameters
    ----------
    returns : pd.DataFrame
        Daily log returns (T x N).

    Returns
    -------
    np.ndarray
        Daily mean return vector (N,).
    """
    return returns.mean().values


def compute_covariance_matrix(returns: pd.DataFrame) -> np.ndarray:
    """
    Compute the sample covariance matrix of daily returns.

    Uses unbiased estimator (ddof=1).
    No loops — pure matrix computation via NumPy.

    Parameters
    ----------
    returns : pd.DataFrame
        Daily log returns (T x N).

    Returns
    -------
    np.ndarray
        Covariance matrix (N x N).
    """
    return np.cov(returns.values, rowvar=False, ddof=1)


def compute_correlation_matrix(returns: pd.DataFrame) -> np.ndarray:
    """
    Compute the Pearson correlation matrix.

    Parameters
    ----------
    returns : pd.DataFrame
        Daily log returns (T x N).

    Returns
    -------
    np.ndarray
        Correlation matrix (N x N).
    """
    return np.corrcoef(returns.values, rowvar=False)


def compute_portfolio_statistics(
    returns: pd.DataFrame,
    weights: np.ndarray,
    cov_matrix: np.ndarray,
) -> Dict[str, float]:
    """
    Compute portfolio-level risk statistics.

    Mathematical Definitions:
        Portfolio mean:      μ_p = w^T μ
        Portfolio variance:  σ_p² = w^T Σ w
        Portfolio std dev:   σ_p  = sqrt(σ_p²)

    Parameters
    ----------
    returns : pd.DataFrame
        Daily log returns.
    weights : np.ndarray
        Portfolio weight vector.
    cov_matrix : np.ndarray
        Covariance matrix.

    Returns
    -------
    dict
        Dictionary with portfolio_mean, portfolio_variance, portfolio_std.
    """
    mu = compute_mean_vector(returns)

    portfolio_mean: float = float(weights @ mu)
    portfolio_variance: float = float(weights @ cov_matrix @ weights)
    portfolio_std: float = float(np.sqrt(portfolio_variance))

    return {
        "portfolio_mean": portfolio_mean,
        "portfolio_variance": portfolio_variance,
        "portfolio_std": portfolio_std,
    }


def compute_rolling_volatility(
    portfolio_returns: pd.Series, window: int = 21
) -> pd.Series:
    """
    Compute rolling annualized volatility of portfolio returns.

    Parameters
    ----------
    portfolio_returns : pd.Series
        Daily portfolio returns.
    window : int
        Rolling window size (default: 21 = ~1 month).

    Returns
    -------
    pd.Series
        Rolling annualized volatility.
    """
    return portfolio_returns.rolling(window=window).std() * np.sqrt(252)


def validate_covariance_matrix(cov_matrix: np.ndarray) -> bool:
    """
    Check if covariance matrix is symmetric and positive semi-definite.

    Parameters
    ----------
    cov_matrix : np.ndarray
        Covariance matrix to validate.

    Returns
    -------
    bool
        True if valid, False otherwise.
    """
    # Symmetry check
    if not np.allclose(cov_matrix, cov_matrix.T, atol=1e-10):
        return False

    # Positive semi-definiteness: all eigenvalues >= 0
    eigenvalues = np.linalg.eigvalsh(cov_matrix)
    return bool(np.all(eigenvalues >= -1e-10))


def regularize_covariance(
    cov_matrix: np.ndarray, epsilon: float = 1e-8
) -> np.ndarray:
    """
    Regularize covariance matrix to ensure positive definiteness.

    Adds a small value to the diagonal (Tikhonov regularization).

    Parameters
    ----------
    cov_matrix : np.ndarray
        Original covariance matrix.
    epsilon : float
        Regularization parameter.

    Returns
    -------
    np.ndarray
        Regularized covariance matrix.
    """
    n = cov_matrix.shape[0]
    return cov_matrix + epsilon * np.eye(n)


def get_all_statistics(
    returns: pd.DataFrame, weights: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, float]]:
    """
    Compute all statistical estimates in one call.

    Parameters
    ----------
    returns : pd.DataFrame
        Daily log returns.
    weights : np.ndarray
        Portfolio weight vector.

    Returns
    -------
    tuple
        (mean_vector, cov_matrix, corr_matrix, portfolio_stats)
    """
    mu = compute_mean_vector(returns)
    cov = compute_covariance_matrix(returns)
    corr = compute_correlation_matrix(returns)
    port_stats = compute_portfolio_statistics(returns, weights, cov)

    return mu, cov, corr, port_stats
