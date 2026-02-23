"""
Student-t Monte Carlo Simulation Engine
========================================
Extends the Gaussian MC engine with heavy-tailed Student-t innovations.

Why Student-t?
    Real market returns exhibit excess kurtosis (fat tails) and more
    frequent extreme moves than Gaussian models predict.  The Student-t
    distribution captures this by introducing a degrees-of-freedom
    parameter ν — lower ν → heavier tails.

    When ν → ∞ the Student-t converges to Normal, so this module
    nests the Gaussian MC as a special case.

Mathematical Foundation:
    1. Fit ν via MLE on historical portfolio returns
    2. Cholesky:  Σ = L L^T
    3. Draw  Z ~ t(ν) i.i.d., shape (N_sim, N_assets)
    4. Scale:  Z̃ = Z · √((ν-2)/ν)   so that Var(Z̃) = 1
    5. Correlate: R = μ + L Z̃^T
    6. Extract VaR / ES from the simulated loss distribution
"""

import numpy as np
from scipy import stats
from typing import Dict, Optional, Tuple

from src.monte_carlo import (
    cholesky_decomposition,
    compute_mc_var,
    compute_mc_expected_shortfall,
    DEFAULT_NUM_SIMULATIONS,
    DEFAULT_SEED,
)


def fit_degrees_of_freedom(
    portfolio_returns: np.ndarray,
    method: str = "mle",
) -> Tuple[float, float, float]:
    """
    Estimate Student-t degrees of freedom from historical data via MLE.

    Parameters
    ----------
    portfolio_returns : np.ndarray
        Historical portfolio return series.
    method : str
        Fitting method (default: MLE via scipy).

    Returns
    -------
    tuple[float, float, float]
        (degrees_of_freedom, loc, scale) — fitted t-distribution params.
    """
    df, loc, scale = stats.t.fit(portfolio_returns)
    return float(df), float(loc), float(scale)


def simulate_student_t_returns(
    mean_vector: np.ndarray,
    cov_matrix: np.ndarray,
    df: float,
    num_simulations: int = DEFAULT_NUM_SIMULATIONS,
    seed: int = DEFAULT_SEED,
) -> np.ndarray:
    """
    Generate correlated asset returns using Student-t innovations.

    Algorithm:
        1. Cholesky decompose Σ = L L^T
        2. Draw Z ~ t(ν), shape (N_sim, N_assets)
        3. Scale Z so that Var(Z) = 1:  Z̃ = Z · √((ν−2)/ν)
        4. Correlate:  R = μ + Z̃ @ L^T

    Parameters
    ----------
    mean_vector : np.ndarray
        Daily mean return vector (N,).
    cov_matrix : np.ndarray
        Covariance matrix (N × N).
    df : float
        Degrees of freedom (ν > 2 required for finite variance).
    num_simulations : int
        Number of Monte Carlo paths.
    seed : int
        Random seed.

    Returns
    -------
    np.ndarray
        Simulated returns matrix (num_simulations × N).
    """
    rng = np.random.default_rng(seed)
    n_assets = len(mean_vector)

    # Step 1: Cholesky factorisation
    L = cholesky_decomposition(cov_matrix)

    # Step 2: Draw i.i.d. t(ν) innovations
    Z = rng.standard_t(df=df, size=(num_simulations, n_assets))

    # Step 3: Rescale so that marginal variance = 1
    #         (Student-t with ν dof has Var = ν/(ν−2) for ν>2)
    if df > 2:
        Z = Z * np.sqrt((df - 2) / df)

    # Step 4: Correlate and shift to mean
    simulated_returns = Z @ L.T + mean_vector

    return simulated_returns


def run_student_t_mc_engine(
    mean_vector: np.ndarray,
    cov_matrix: np.ndarray,
    weights: np.ndarray,
    df: float,
    num_simulations: int = DEFAULT_NUM_SIMULATIONS,
    seed: int = DEFAULT_SEED,
) -> Dict[str, object]:
    """
    Full Student-t Monte Carlo risk engine execution.

    Parameters
    ----------
    mean_vector : np.ndarray
        Daily mean return vector.
    cov_matrix : np.ndarray
        Covariance matrix.
    weights : np.ndarray
        Portfolio weight vector.
    df : float
        Student-t degrees of freedom.
    num_simulations : int
        Number of simulation paths.
    seed : int
        Random seed.

    Returns
    -------
    dict
        Contains: portfolio_pnl, var_95, var_99, es_95, es_99,
        simulated_returns, num_simulations, degrees_of_freedom.
    """
    simulated_returns = simulate_student_t_returns(
        mean_vector, cov_matrix, df, num_simulations, seed
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
        "degrees_of_freedom": df,
    }

    return results
