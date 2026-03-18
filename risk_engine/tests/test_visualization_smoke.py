from pathlib import Path

import numpy as np
import pandas as pd

from src.visualization import (
    plot_correlation_heatmap,
    plot_pnl_distribution,
    plot_rolling_var_vs_losses,
)


def test_visualization_functions_create_output_files(tmp_path: Path):
    output_dir: Path = tmp_path / "figures"

    rng = np.random.default_rng(123)
    portfolio_pnl = rng.normal(loc=0.0002, scale=0.01, size=2500)

    var_95 = float(-np.percentile(portfolio_pnl, 5))
    var_99 = float(-np.percentile(portfolio_pnl, 1))
    es_99 = float(-portfolio_pnl[portfolio_pnl <= np.percentile(portfolio_pnl, 1)].mean())

    dist_path = plot_pnl_distribution(
        portfolio_pnl,
        var_95=var_95,
        var_99=var_99,
        es_99=es_99,
        output_dir=str(output_dir),
    )

    bt_df = pd.DataFrame(
        {
            "date": pd.date_range("2025-01-01", periods=30, freq="B"),
            "predicted_var": np.full(30, 0.015),
            "actual_loss": rng.uniform(0.0, 0.03, size=30),
        }
    )
    bt_df["breach"] = bt_df["actual_loss"] > bt_df["predicted_var"]

    backtest_path = plot_rolling_var_vs_losses(bt_df, output_dir=str(output_dir))

    corr = np.array(
        [
            [1.0, 0.3, -0.1],
            [0.3, 1.0, 0.2],
            [-0.1, 0.2, 1.0],
        ]
    )
    labels: list[str] = ["SPY", "QQQ", "TLT"]
    heatmap_path = plot_correlation_heatmap(corr, labels, output_dir=str(output_dir))

    assert Path(dist_path).exists()
    assert Path(backtest_path).exists()
    assert Path(heatmap_path).exists()
