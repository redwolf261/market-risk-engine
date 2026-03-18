import numpy as np
import pandas as pd
from typing import cast

from src.backtesting import (
    compute_breach_statistics,
    kupiec_test,
    rolling_var_backtest,
)
from src.monte_carlo import run_monte_carlo_engine


def test_monte_carlo_reproducible_with_fixed_seed():
    mu = np.array([0.0002, -0.0001])
    cov = np.array([[1.0e-4, 2.0e-5], [2.0e-5, 9.0e-5]])
    weights = np.array([0.6, 0.4])

    first = run_monte_carlo_engine(mu, cov, weights, num_simulations=5000, seed=7)
    second = run_monte_carlo_engine(mu, cov, weights, num_simulations=5000, seed=7)
    third = run_monte_carlo_engine(mu, cov, weights, num_simulations=5000, seed=8)

    first_pnl = cast(np.ndarray, first["portfolio_pnl"])
    second_pnl = cast(np.ndarray, second["portfolio_pnl"])
    third_pnl = cast(np.ndarray, third["portfolio_pnl"])
    first_var99 = cast(float, first["var_99"])
    first_es99 = cast(float, first["es_99"])

    np.testing.assert_allclose(first_pnl, second_pnl)
    assert not np.array_equal(first_pnl, third_pnl)

    assert first_var99 > 0
    assert first_es99 >= first_var99


def test_breach_statistics_are_computed_correctly():
    backtest_results = pd.DataFrame({"breach": [False, True, False, True]})
    stats = compute_breach_statistics(backtest_results, confidence_level=0.99)

    assert stats["total_observations"] == 4
    assert stats["num_breaches"] == 2
    assert np.isclose(stats["breach_rate"], 0.5)
    assert np.isclose(stats["expected_breach_rate"], 0.01)
    assert np.isclose(stats["breach_ratio"], 50.0)


def test_kupiec_handles_no_breach_edge_case():
    no_breach_results = pd.DataFrame({"breach": [False] * 10})
    kupiec = cast(dict[str, object], kupiec_test(no_breach_results, confidence_level=0.99))

    assert cast(float, kupiec["lr_statistic"]) == 0.0
    assert cast(float, kupiec["p_value"]) == 1.0
    assert cast(bool, kupiec["reject_h0"]) is False
    assert "Insufficient breaches" in cast(str, kupiec["interpretation"])


def test_rolling_var_backtest_returns_expected_shape_and_columns():
    rng = np.random.default_rng(0)
    data = rng.normal(loc=0.0001, scale=0.01, size=(40, 3))
    returns = pd.DataFrame(
        data,
        index=pd.date_range("2024-01-01", periods=40, freq="B"),
        columns=["A", "B", "C"],
    )
    weights = np.array([0.5, 0.3, 0.2])

    out = rolling_var_backtest(returns, weights, window=20, confidence_level=0.99)

    assert len(out) == 20
    assert {"date", "predicted_var", "actual_return", "actual_loss", "breach"}.issubset(out.columns)
    assert (out["predicted_var"] > 0).all()
