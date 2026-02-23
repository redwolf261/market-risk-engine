"""
Monte Carlo Simulation Engine (Flagship Module)
================================================
Generates correlated asset return simulations using Cholesky decomposition
and computes portfolio-level loss distributions.

Mathematical Foundation:
    Cholesky:     Σ = L L^T
    Simulation:   R = μ + L Z,  where Z ~ N(0, I)
    Portfolio:    R_p = w^T R
"""

import numpy as np
from typing import Dict, Optional, Tuple

from src.statistics import (
    regularize_covariance,
    validate_covariance_matrix,
)


# ─────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────
DEFAULT_NUM_SIMULATIONS: int = 100_000
DEFAULT_SEED: int = 42


def cholesky_decomposition(cov_matrix: np.ndarray) -> np.ndarray:
    """
    Perform Cholesky decomposition of the covariance matrix.

    Decomposes Σ into lower triangular L such that Σ = L L^T.
    If matrix is not positive definite, applies regularization.

    Parameters
    ----------
    cov_matrix : np.ndarray
        Covariance matrix (N x N). Must be symmetric.

    Returns
    -------
    np.ndarray
        Lower triangular Cholesky factor L (N x N).

    Raises
    ------
    np.linalg.LinAlgError
        If decomposition fails even after regularization.
    """
    if not validate_covariance_matrix(cov_matrix):
        cov_matrix = regularize_covariance(cov_matrix)

    try:
        L = np.linalg.cholesky(cov_matrix)
    except np.linalg.LinAlgError:
        # Aggressive regularization fallback
        cov_matrix = regularize_covariance(cov_matrix, epsilon=1e-6)
        L = np.linalg.cholesky(cov_matrix)

    return L


def simulate_correlated_returns(
    mean_vector: np.ndarray,
    cov_matrix: np.ndarray,
    num_simulations: int = DEFAULT_NUM_SIMULATIONS,
    seed: int = DEFAULT_SEED,
) -> np.ndarray:
    """
    Generate correlated asset return simulations via Cholesky decomposition.

    Algorithm:
        1. Decompose Σ = L L^T
        2. Draw Z ~ N(0, I) of shape (N, num_simulations)
        3. Transform: R = μ + L Z

    Parameters
    ----------
    mean_vector : np.ndarray
        Daily mean return vector (N,).
    cov_matrix : np.ndarray
        Covariance matrix (N x N).
    num_simulations : int
        Number of Monte Carlo paths (default: 100,000).
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    np.ndarray
        Simulated returns matrix (num_simulations x N).
    """
    rng = np.random.default_rng(seed)
    n_assets = len(mean_vector)

    # Step 1: Cholesky decomposition
    L = cholesky_decomposition(cov_matrix)

    # Step 2: Generate standard normal draws
    Z = rng.standard_normal(size=(num_simulations, n_assets))

    # Step 3: Transform to correlated returns
    # R_i = μ + (L Z_i^T)^T  →  vectorized as  R = Z @ L^T + μ
    simulated_returns = Z @ L.T + mean_vector

    return simulated_returns


def simulate_portfolio_pnl(
    mean_vector: np.ndarray,
    cov_matrix: np.ndarray,
    weights: np.ndarray,
    num_simulations: int = DEFAULT_NUM_SIMULATIONS,
    seed: int = DEFAULT_SEED,
) -> np.ndarray:
    """
    Simulate portfolio P&L distribution.

    Mathematical Definition:
        R_p = w^T R  for each simulation path.

    Parameters
    ----------
    mean_vector : np.ndarray
        Daily mean return vector (N,).
    cov_matrix : np.ndarray
        Covariance matrix (N x N).
    weights : np.ndarray
        Portfolio weight vector (N,).
    num_simulations : int
        Number of Monte Carlo paths.
    seed : int
        Random seed.

    Returns
    -------
    np.ndarray
        Simulated portfolio returns (num_simulations,).
    """
    simulated_returns = simulate_correlated_returns(
        mean_vector, cov_matrix, num_simulations, seed
    )

    # Portfolio aggregation: R_p = w^T R
    portfolio_pnl = simulated_returns @ weights

    return portfolio_pnl


def compute_mc_var(
    portfolio_pnl: np.ndarray,
    confidence_level: float = 0.99,
) -> float:
    """
    Compute Value-at-Risk from simulated P&L distribution.

    VaR is the loss at the (1 - confidence_level) quantile of P&L.

    Parameters
    ----------
    portfolio_pnl : np.ndarray
        Simulated portfolio returns.
    confidence_level : float
        Confidence level (default: 0.99).

    Returns
    -------
    float
        VaR as a positive number (loss magnitude).
    """
    var = -np.percentile(portfolio_pnl, (1 - confidence_level) * 100)
    return float(var)


def compute_mc_expected_shortfall(
    portfolio_pnl: np.ndarray,
    confidence_level: float = 0.99,
) -> float:
    """
    Compute Expected Shortfall (CVaR) from simulated P&L.

    ES = E[Loss | Loss > VaR]
    Mean of losses exceeding VaR threshold.

    Parameters
    ----------
    portfolio_pnl : np.ndarray
        Simulated portfolio returns.
    confidence_level : float
        Confidence level (default: 0.99).

    Returns
    -------
    float
        Expected Shortfall as a positive number.
    """
    threshold = np.percentile(portfolio_pnl, (1 - confidence_level) * 100)
    tail_losses = portfolio_pnl[portfolio_pnl <= threshold]
    es = -np.mean(tail_losses)
    return float(es)


def run_monte_carlo_engine(
    mean_vector: np.ndarray,
    cov_matrix: np.ndarray,
    weights: np.ndarray,
    num_simulations: int = DEFAULT_NUM_SIMULATIONS,
    seed: int = DEFAULT_SEED,
) -> Dict[str, object]:
    """
    Full Monte Carlo risk engine execution.

    Runs simulation and computes all risk metrics.

    Parameters
    ----------
    mean_vector : np.ndarray
        Daily mean return vector.
    cov_matrix : np.ndarray
        Covariance matrix.
    weights : np.ndarray
        Portfolio weights.
    num_simulations : int
        Number of simulations.
    seed : int
        Random seed.

    Returns
    -------
    dict
        Contains: portfolio_pnl, var_95, var_99, es_95, es_99,
        simulated_returns, num_simulations.
    """
    simulated_returns = simulate_correlated_returns(
        mean_vector, cov_matrix, num_simulations, seed
    )
    portfolio_pnl = simulated_returns @ weights

    results = {
        "portfolio_pnl": portfolio_pnl,
        "simulated_returns": simulated_returns,
        "var_95": compute_mc_var(portfolio_pnl, 0.95),
        "var_99": compute_mc_var(portfolio_pnl, 0.99),
        "es_95": compute_mc_expected_shortfall(portfolio_pnl, 0.95),
        "es_99": compute_mc_expected_shortfall(portfolio_pnl, 0.99),
        "num_simulations": num_simulations,
    }

    return results
