"""
Monte Carlo Market Risk Engine — Main Orchestrator
===================================================
Entry point for the complete risk analysis pipeline.

Execution Flow:
    1. Fetch / load price data
    2. Compute log returns and portfolio construction
    3. Statistical estimation (μ, Σ, ρ)
    4. Historical VaR & ES
    5. Parametric VaR & ES
    6. Monte Carlo VaR & ES (flagship)
    7. Rolling-window backtesting with Kupiec test
    8. Stress testing (volatility shock + correlation stress)
    9. Visualization
   10. Results export

Author: Rivan Avinash Shetty
"""

import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

# ─────────────────────────────────────────────────────────────
# Add project root to path
# ─────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.portfolio import (
    fetch_data,
    load_data,
    compute_log_returns,
    define_weights,
    compute_portfolio_returns,
    get_portfolio_summary,
    DEFAULT_TICKERS,
)
from src.statistics import (
    get_all_statistics,
    compute_rolling_volatility,
)
from src.risk_metrics import (
    historical_risk_metrics,
    parametric_risk_metrics,
)
from src.monte_carlo import run_monte_carlo_engine
from src.backtesting import run_full_backtest
from src.stress_testing import full_stress_analysis
from src.student_t_mc import (
    fit_degrees_of_freedom,
    run_student_t_mc_engine,
)
from src.visualization import (
    plot_pnl_distribution,
    plot_left_tail_zoom,
    plot_rolling_var_vs_losses,
    plot_correlation_heatmap,
    plot_stress_comparison,
    plot_gaussian_vs_student_t,
)

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────
DATA_PATH = PROJECT_ROOT / "data" / "raw_prices.csv"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
TABLES_DIR = RESULTS_DIR / "tables"

NUM_SIMULATIONS = 100_000
RANDOM_SEED = 42
BACKTEST_WINDOW = 250
CONFIDENCE_LEVELS = [0.95, 0.99]


def print_header(text: str) -> None:
    """Print formatted section header."""
    width = 60
    print("\n" + "=" * width)
    print(f"  {text}")
    print("=" * width)


def print_metrics(metrics: dict, indent: int = 4) -> None:
    """Print dictionary of metrics with formatting."""
    prefix = " " * indent
    for key, val in metrics.items():
        if isinstance(val, float):
            print(f"{prefix}{key:.<35} {val:>12.6f}")
        else:
            print(f"{prefix}{key:.<35} {str(val):>12}")


def main() -> None:
    """Execute the complete risk engine pipeline."""

    print("\n" + "╔" + "═" * 58 + "╗")
    print("║   MONTE CARLO MARKET RISK ENGINE                        ║")
    print("║   1-Day Trading Desk Risk Model                         ║")
    print("╚" + "═" * 58 + "╝")

    # ── PHASE 1: Data & Portfolio ──────────────────────────────
    print_header("PHASE 1 — DATA ACQUISITION & PORTFOLIO CONSTRUCTION")

    if DATA_PATH.exists():
        print(f"  Loading cached data from {DATA_PATH}")
        prices = load_data(str(DATA_PATH))
    else:
        print(f"  Fetching price data for: {DEFAULT_TICKERS}")
        prices = fetch_data(save_path=str(DATA_PATH))

    weights = define_weights()
    log_returns = compute_log_returns(prices)
    port_returns = compute_portfolio_returns(log_returns, weights)

    print(f"\n  Assets:        {list(prices.columns)}")
    print(f"  Weights:       {weights}")
    print(f"  Period:        {prices.index[0].date()} → {prices.index[-1].date()}")
    print(f"  Observations:  {len(log_returns)}")

    summary = get_portfolio_summary(prices, weights)
    print("\n  Portfolio Summary:")
    print_metrics(summary)

    # ── PHASE 2: Statistical Estimation ────────────────────────
    print_header("PHASE 2 — STATISTICAL ESTIMATION")

    mu, cov, corr, port_stats = get_all_statistics(log_returns, weights)

    print("\n  Mean Return Vector (daily):")
    for i, ticker in enumerate(prices.columns):
        print(f"    {ticker}: {mu[i]:.6f}")

    print("\n  Portfolio Statistics:")
    print_metrics(port_stats)

    print("\n  Covariance Matrix:")
    cov_df = pd.DataFrame(cov, index=prices.columns, columns=prices.columns)
    print(cov_df.to_string(float_format=lambda x: f"{x:.8f}"))

    # ── PHASE 3: Risk Models ──────────────────────────────────
    print_header("PHASE 3 — RISK MODEL ESTIMATION")

    # 3.1 Historical VaR
    print("\n  ┌─ Historical VaR ─────────────────────────────┐")
    hist_metrics = historical_risk_metrics(port_returns)
    print_metrics(hist_metrics)

    # 3.2 Parametric VaR
    print("\n  ┌─ Parametric VaR (Variance-Covariance) ───────┐")
    param_metrics = parametric_risk_metrics(
        port_stats["portfolio_mean"],
        port_stats["portfolio_std"],
    )
    print_metrics(param_metrics)

    # 3.3 Monte Carlo VaR (Flagship)
    print("\n  ┌─ Monte Carlo VaR (Flagship Engine) ──────────┐")
    print(f"    Running {NUM_SIMULATIONS:,} simulations...")
    mc_results = run_monte_carlo_engine(
        mu, cov, weights, NUM_SIMULATIONS, RANDOM_SEED
    )
    mc_metrics = {
        "mc_var_95": mc_results["var_95"],
        "mc_var_99": mc_results["var_99"],
        "mc_es_95": mc_results["es_95"],
        "mc_es_99": mc_results["es_99"],
    }
    print_metrics(mc_metrics)

    # ── PHASE 3b: Student-t Monte Carlo ───────────────────────
    print_header("PHASE 3b — STUDENT-t MONTE CARLO (HEAVY TAILS)")

    print("  Fitting Student-t degrees of freedom via MLE...")
    df_t, loc_t, scale_t = fit_degrees_of_freedom(port_returns.values)
    print(f"    Fitted ν = {df_t:.2f}  (lower ν → heavier tails)")
    print(f"    Location = {loc_t:.6f}, Scale = {scale_t:.6f}")

    print(f"    Running {NUM_SIMULATIONS:,} Student-t simulations...")
    t_results = run_student_t_mc_engine(
        mu, cov, weights, df_t, NUM_SIMULATIONS, RANDOM_SEED
    )
    t_metrics = {
        "t_var_95": t_results["var_95"],
        "t_var_99": t_results["var_99"],
        "t_es_95": t_results["es_95"],
        "t_es_99": t_results["es_99"],
        "t_dof": df_t,
    }
    print("\n  ┌─ Student-t Monte Carlo Results ─────────────┐")
    print_metrics(t_metrics)

    # ── PHASE 4: Backtesting ──────────────────────────────────
    print_header("PHASE 4 — ROLLING-WINDOW BACKTESTING")

    print(f"  Window: {BACKTEST_WINDOW} days | Confidence: 99%")
    bt_results, breach_stats, kupiec = run_full_backtest(
        log_returns, weights, BACKTEST_WINDOW, 0.99
    )

    print("\n  Breach Statistics:")
    print_metrics(breach_stats)

    print("\n  Kupiec POF Test:")
    print_metrics(kupiec)

    # ── PHASE 5: Stress Testing ───────────────────────────────
    print_header("PHASE 5 — STRESS TESTING")

    stress_results = full_stress_analysis(
        mu, cov, weights, mc_results, NUM_SIMULATIONS, RANDOM_SEED
    )

    print("\n  ┌─ Volatility Shock (2× Σ) ──────────────────┐")
    print_metrics(stress_results["vol_shock"]["impact"])

    print("\n  ┌─ Correlation Stress (ρ = 0.9) ──────────────┐")
    print_metrics(stress_results["corr_stress"]["impact"])

    # ── PHASE 6: Visualization ────────────────────────────────
    print_header("PHASE 6 — GENERATING VISUALIZATIONS")

    fig_dir = str(FIGURES_DIR)

    p1 = plot_pnl_distribution(
        mc_results["portfolio_pnl"],
        mc_results["var_95"], mc_results["var_99"], mc_results["es_99"],
        output_dir=fig_dir,
    )
    print(f"  ✓ {p1}")

    p2 = plot_left_tail_zoom(
        mc_results["portfolio_pnl"],
        mc_results["var_95"], mc_results["var_99"], mc_results["es_99"],
        output_dir=fig_dir,
    )
    print(f"  ✓ {p2}")

    p3 = plot_rolling_var_vs_losses(bt_results, output_dir=fig_dir)
    print(f"  ✓ {p3}")

    p4 = plot_correlation_heatmap(
        corr, list(prices.columns), output_dir=fig_dir
    )
    print(f"  ✓ {p4}")

    base_for_chart = {
        "var_95": mc_results["var_95"],
        "var_99": mc_results["var_99"],
        "es_95": mc_results["es_95"],
        "es_99": mc_results["es_99"],
    }
    p5 = plot_stress_comparison(
        base_for_chart,
        {
            "var_95": stress_results["vol_shock"]["results"]["var_95"],
            "var_99": stress_results["vol_shock"]["results"]["var_99"],
            "es_95": stress_results["vol_shock"]["results"]["es_95"],
            "es_99": stress_results["vol_shock"]["results"]["es_99"],
        },
        {
            "var_95": stress_results["corr_stress"]["results"]["var_95"],
            "var_99": stress_results["corr_stress"]["results"]["var_99"],
            "es_95": stress_results["corr_stress"]["results"]["es_95"],
            "es_99": stress_results["corr_stress"]["results"]["es_99"],
        },
        output_dir=fig_dir,
    )
    print(f"  ✓ {p5}")

    p6 = plot_gaussian_vs_student_t(
        mc_results["portfolio_pnl"],
        t_results["portfolio_pnl"],
        mc_results["var_99"],
        t_results["var_99"],
        mc_results["es_99"],
        t_results["es_99"],
        df_t,
        output_dir=fig_dir,
    )
    print(f"  ✓ {p6}")

    # ── Results Summary Table ─────────────────────────────────
    print_header("RESULTS COMPARISON TABLE")

    comparison = pd.DataFrame({
        "Model": ["Historical", "Parametric", "Monte Carlo (Gaussian)",
                  f"Monte Carlo (Student-t, ν={df_t:.1f})"],
        "95% VaR": [
            hist_metrics["hist_var_95"],
            param_metrics["param_var_95"],
            mc_metrics["mc_var_95"],
            t_metrics["t_var_95"],
        ],
        "99% VaR": [
            hist_metrics["hist_var_99"],
            param_metrics["param_var_99"],
            mc_metrics["mc_var_99"],
            t_metrics["t_var_99"],
        ],
        "95% ES": [
            hist_metrics["hist_es_95"],
            param_metrics["param_es_95"],
            mc_metrics["mc_es_95"],
            t_metrics["t_es_95"],
        ],
        "99% ES": [
            hist_metrics["hist_es_99"],
            param_metrics["param_es_99"],
            mc_metrics["mc_es_99"],
            t_metrics["t_es_99"],
        ],
    })

    print("\n" + comparison.to_string(index=False, float_format=lambda x: f"{x:.6f}"))

    # Save comparison table
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(TABLES_DIR / "risk_metrics_comparison.csv", index=False)

    # ── Save all results as JSON ──────────────────────────────
    all_results = {
        "portfolio": {
            "assets": list(prices.columns),
            "weights": weights.tolist(),
            "summary": summary,
        },
        "statistics": port_stats,
        "historical": hist_metrics,
        "parametric": param_metrics,
        "monte_carlo": mc_metrics,
        "student_t_mc": t_metrics,
        "backtesting": {
            "breach_stats": breach_stats,
            "kupiec_test": {k: v for k, v in kupiec.items()},
        },
        "stress_testing": {
            "vol_shock": stress_results["vol_shock"]["impact"],
            "corr_stress": stress_results["corr_stress"]["impact"],
        },
    }

    results_path = TABLES_DIR / "full_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\n  Results saved to: {results_path}")

    print("\n" + "╔" + "═" * 58 + "╗")
    print("║   RISK ENGINE EXECUTION COMPLETE                        ║")
    print("╚" + "═" * 58 + "╝\n")


if __name__ == "__main__":
    main()
