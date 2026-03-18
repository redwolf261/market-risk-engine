import numpy as np
import pandas as pd

from src.risk_metrics import (
    compute_historical_es,
    compute_historical_var,
    compute_parametric_es,
    compute_parametric_var,
)


def test_historical_var_es_relationships_hold():
    returns = pd.Series([-0.05, -0.02, -0.01, 0.00, 0.01, 0.02])

    var_95 = compute_historical_var(returns, 0.95)
    var_99 = compute_historical_var(returns, 0.99)
    es_95 = compute_historical_es(returns, 0.95)
    es_99 = compute_historical_es(returns, 0.99)

    # Tail metrics should become more conservative at higher confidence.
    assert var_99 >= var_95
    assert es_99 >= es_95

    # Expected shortfall should be at least as large as VaR at same alpha.
    assert es_95 >= var_95
    assert es_99 >= var_99


def test_parametric_var_es_match_closed_form():
    mu = 0.0004
    sigma = 0.012
    alpha = 0.99

    # Precomputed standard normal constants at alpha=0.99.
    z_alpha = 2.3263478740408408
    phi = 0.02665214220345808

    expected_var = z_alpha * sigma - mu
    expected_es = -mu + sigma * phi / (1 - alpha)

    calc_var = compute_parametric_var(mu, sigma, alpha)
    calc_es = compute_parametric_es(mu, sigma, alpha)

    assert np.isclose(calc_var, expected_var, rtol=0, atol=1e-12)
    assert np.isclose(calc_es, expected_es, rtol=0, atol=1e-12)
