"""
Stress Testing Module
=====================
Implements volatility shock analysis and correlation stress scenarios
to measure portfolio sensitivity under extreme market conditions.

Stress Scenarios:
    1. Volatility Doubling:  Σ_shock = k · Σ  (k=2)
    2. Correlation Stress:   ρ_ij → 0.9  (diversification collapse)
"""

import numpy as np
from typing import Dict, Optional

from src.monte_carlo import run_monte_carlo_engine


def apply_volatility_shock(
    cov_matrix: np.ndarray,
    shock_factor: float = 2.0,
) -> np.ndarray:
    """
    Apply multiplicative volatility shock to covariance matrix.

    Mathematical Definition:
        Σ_shock = k · Σ

    This scales all variances and covariances uniformly,
    equivalent to multiplying all volatilities by √k.

    Parameters
    ----------
    cov_matrix : np.ndarray
        Original covariance matrix (N x N).
    shock_factor : float
        Multiplicative shock factor (default: 2.0 = vol doubling).

    Returns
    -------
    np.ndarray
        Shocked covariance matrix.
    """
    return shock_factor * cov_matrix


def apply_correlation_stress(
    cov_matrix: np.ndarray,
    target_correlation: float = 0.9,
) -> np.ndarray:
    """
    Stress correlations to a high uniform value (diversification collapse).

    Algorithm:
        1. Extract volatilities from diagonal of Σ
        2. Build new correlation matrix with ρ_ij = target for i ≠ j
        3. Reconstruct covariance: Σ_stress = D · ρ_stress · D
           where D = diag(σ_1, ..., σ_N)

    Parameters
    ----------
    cov_matrix : np.ndarray
        Original covariance matrix.
    target_correlation : float
        Stressed off-diagonal correlation (default: 0.9).

    Returns
    -------
    np.ndarray
        Stressed covariance matrix.
    """
    n = cov_matrix.shape[0]

    # Extract volatilities
    vols = np.sqrt(np.diag(cov_matrix))
    D = np.diag(vols)

    # Build stressed correlation matrix
    corr_stress = np.full((n, n), target_correlation)
    np.fill_diagonal(corr_stress, 1.0)

    # Reconstruct covariance
    cov_stressed = D @ corr_stress @ D

    return cov_stressed


def run_stress_scenario(
    mean_vector: np.ndarray,
    cov_matrix: np.ndarray,
    weights: np.ndarray,
    shock_factor: float = 2.0,
    num_simulations: int = 100_000,
    seed: int = 42,
) -> Dict[str, object]:
    """
    Run Monte Carlo under volatility-shocked covariance matrix.

    Parameters
    ----------
    mean_vector : np.ndarray
        Daily mean return vector.
    cov_matrix : np.ndarray
        Original covariance matrix.
    weights : np.ndarray
        Portfolio weights.
    shock_factor : float
        Volatility shock multiplier.
    num_simulations : int
        Number of MC simulations.
    seed : int
        Random seed.

    Returns
    -------
    dict
        Stressed MC results (var_95, var_99, es_95, es_99, portfolio_pnl).
    """
    cov_shocked = apply_volatility_shock(cov_matrix, shock_factor)

    results = run_monte_carlo_engine(
        mean_vector, cov_shocked, weights, num_simulations, seed
    )

    return results


def run_correlation_stress(
    mean_vector: np.ndarray,
    cov_matrix: np.ndarray,
    weights: np.ndarray,
    target_correlation: float = 0.9,
    num_simulations: int = 100_000,
    seed: int = 42,
) -> Dict[str, object]:
    """
    Run Monte Carlo under correlation-stressed covariance matrix.

    Parameters
    ----------
    mean_vector : np.ndarray
        Daily mean return vector.
    cov_matrix : np.ndarray
        Original covariance matrix.
    weights : np.ndarray
        Portfolio weights.
    target_correlation : float
        Stressed uniform correlation.
    num_simulations : int
        Number of MC simulations.
    seed : int
        Random seed.

    Returns
    -------
    dict
        Stressed MC results.
    """
    cov_stressed = apply_correlation_stress(cov_matrix, target_correlation)

    results = run_monte_carlo_engine(
        mean_vector, cov_stressed, weights, num_simulations, seed
    )

    return results


def compute_stress_impact(
    base_results: Dict[str, object],
    stressed_results: Dict[str, object],
) -> Dict[str, float]:
    """
    Compare base and stressed risk metrics.

    Parameters
    ----------
    base_results : dict
        Baseline Monte Carlo results.
    stressed_results : dict
        Stressed Monte Carlo results.

    Returns
    -------
    dict
        Percentage changes in VaR and ES metrics.
    """
    metrics = ["var_95", "var_99", "es_95", "es_99"]
    impact = {}

    for m in metrics:
        base_val = base_results[m]
        stress_val = stressed_results[m]
        pct_change = ((stress_val - base_val) / base_val) * 100 if base_val != 0 else 0
        impact[f"{m}_base"] = base_val
        impact[f"{m}_stressed"] = stress_val
        impact[f"{m}_pct_change"] = pct_change

    return impact


def full_stress_analysis(
    mean_vector: np.ndarray,
    cov_matrix: np.ndarray,
    weights: np.ndarray,
    base_results: Dict[str, object],
    num_simulations: int = 100_000,
    seed: int = 42,
) -> Dict[str, Dict]:
    """
    Execute complete stress testing suite.

    Runs both volatility shock and correlation stress scenarios.

    Parameters
    ----------
    mean_vector : np.ndarray
        Daily mean return vector.
    cov_matrix : np.ndarray
        Original covariance matrix.
    weights : np.ndarray
        Portfolio weights.
    base_results : dict
        Baseline Monte Carlo results for comparison.
    num_simulations : int
        Number of MC simulations.
    seed : int
        Random seed.

    Returns
    -------
    dict
        Contains 'vol_shock' and 'corr_stress' sub-dicts with
        results and impact analysis.
    """
    # Volatility doubling scenario
    vol_results = run_stress_scenario(
        mean_vector, cov_matrix, weights,
        shock_factor=2.0, num_simulations=num_simulations, seed=seed,
    )
    vol_impact = compute_stress_impact(base_results, vol_results)

    # Correlation stress scenario
    corr_results = run_correlation_stress(
        mean_vector, cov_matrix, weights,
        target_correlation=0.9, num_simulations=num_simulations, seed=seed,
    )
    corr_impact = compute_stress_impact(base_results, corr_results)

    return {
        "vol_shock": {
            "results": vol_results,
            "impact": vol_impact,
        },
        "corr_stress": {
            "results": corr_results,
            "impact": corr_impact,
        },
    }
