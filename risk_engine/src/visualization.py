"""
Visualization Module
====================
Produces professional static charts for risk analysis reporting.

Generated Figures:
    1. Monte Carlo P&L Distribution (Histogram)
    2. Left-Tail Zoom with VaR/ES Lines
    3. Rolling VaR vs Actual Losses
    4. Correlation Heatmap
    5. Stress Test Comparison
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
from typing import Dict, Optional
from pathlib import Path


# ─────────────────────────────────────────────────────────────
# Style Configuration
# ─────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.figsize": (12, 7),
    "figure.dpi": 150,
    "font.size": 11,
    "font.family": "serif",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

COLORS = {
    "primary": "#1f77b4",
    "var_95": "#ff7f0e",
    "var_99": "#d62728",
    "es": "#9467bd",
    "breach": "#e74c3c",
    "safe": "#2ecc71",
}


def save_figure(fig: plt.Figure, name: str, output_dir: str = "results/figures") -> str:
    """Save figure to disk and return the path."""
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    filepath = path / f"{name}.png"
    fig.savefig(filepath, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return str(filepath)


def plot_pnl_distribution(
    portfolio_pnl: np.ndarray,
    var_95: float,
    var_99: float,
    es_99: float,
    title: str = "Monte Carlo Simulated 1-Day P&L Distribution",
    output_dir: str = "results/figures",
) -> str:
    """
    Plot histogram of simulated P&L with VaR and ES lines.

    Parameters
    ----------
    portfolio_pnl : np.ndarray
        Simulated portfolio returns.
    var_95 : float
        95% VaR (positive number).
    var_99 : float
        99% VaR (positive number).
    es_99 : float
        99% Expected Shortfall (positive number).
    title : str
        Plot title.
    output_dir : str
        Output directory for figure.

    Returns
    -------
    str
        Path to saved figure.
    """
    fig, ax = plt.subplots(figsize=(14, 7))

    ax.hist(
        portfolio_pnl, bins=300, density=True,
        color=COLORS["primary"], alpha=0.7, edgecolor="none",
        label="Simulated P&L",
    )

    # VaR and ES vertical lines
    ax.axvline(-var_95, color=COLORS["var_95"], linewidth=2,
               linestyle="--", label=f"95% VaR = {var_95:.4f}")
    ax.axvline(-var_99, color=COLORS["var_99"], linewidth=2,
               linestyle="--", label=f"99% VaR = {var_99:.4f}")
    ax.axvline(-es_99, color=COLORS["es"], linewidth=2,
               linestyle=":", label=f"99% ES = {es_99:.4f}")

    ax.set_xlabel("Portfolio Return", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(fontsize=11, loc="upper right")
    ax.xaxis.set_major_formatter(mtick.PercentFormatter(1.0))

    return save_figure(fig, "mc_pnl_distribution", output_dir)


def plot_left_tail_zoom(
    portfolio_pnl: np.ndarray,
    var_95: float,
    var_99: float,
    es_99: float,
    output_dir: str = "results/figures",
) -> str:
    """
    Zoom into the left tail of the P&L distribution.

    Parameters
    ----------
    portfolio_pnl : np.ndarray
        Simulated portfolio returns.
    var_95, var_99, es_99 : float
        Risk metrics (positive = loss).
    output_dir : str
        Output directory.

    Returns
    -------
    str
        Path to saved figure.
    """
    fig, ax = plt.subplots(figsize=(14, 7))

    # Filter to left tail only (worst 10%)
    cutoff = np.percentile(portfolio_pnl, 10)
    tail_data = portfolio_pnl[portfolio_pnl <= cutoff]

    ax.hist(
        tail_data, bins=150, density=True,
        color=COLORS["primary"], alpha=0.7, edgecolor="none",
        label="Left Tail P&L",
    )

    ax.axvline(-var_95, color=COLORS["var_95"], linewidth=2,
               linestyle="--", label=f"95% VaR = {var_95:.4f}")
    ax.axvline(-var_99, color=COLORS["var_99"], linewidth=2,
               linestyle="--", label=f"99% VaR = {var_99:.4f}")
    ax.axvline(-es_99, color=COLORS["es"], linewidth=2,
               linestyle=":", label=f"99% ES = {es_99:.4f}")

    # Shade the ES region
    ax.axvspan(tail_data.min(), -var_99, alpha=0.15, color=COLORS["var_99"],
               label="Tail Loss Region")

    ax.set_xlabel("Portfolio Return", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title("Left Tail Zoom — VaR & Expected Shortfall",
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.xaxis.set_major_formatter(mtick.PercentFormatter(1.0))

    return save_figure(fig, "left_tail_zoom", output_dir)


def plot_rolling_var_vs_losses(
    backtest_results: pd.DataFrame,
    output_dir: str = "results/figures",
) -> str:
    """
    Plot rolling VaR predictions vs actual portfolio losses.

    Parameters
    ----------
    backtest_results : pd.DataFrame
        Output from backtesting module with columns:
        date, predicted_var, actual_loss, breach.
    output_dir : str
        Output directory.

    Returns
    -------
    str
        Path to saved figure.
    """
    fig, ax = plt.subplots(figsize=(16, 7))

    dates = backtest_results["date"]
    var_line = backtest_results["predicted_var"]
    losses = backtest_results["actual_loss"]
    breaches = backtest_results["breach"]

    # Actual losses
    ax.plot(dates, losses, color=COLORS["primary"], alpha=0.5,
            linewidth=0.8, label="Actual Daily Loss")

    # VaR line
    ax.plot(dates, var_line, color=COLORS["var_99"], linewidth=1.5,
            label="99% VaR Forecast")

    # Highlight breaches
    breach_dates = dates[breaches]
    breach_losses = losses[breaches]
    ax.scatter(breach_dates, breach_losses, color=COLORS["breach"],
               s=30, zorder=5, label=f"Breaches (n={breaches.sum()})")

    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Loss (as fraction of portfolio)", fontsize=12)
    ax.set_title("Rolling 99% VaR Backtest — Predicted vs Actual Losses",
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    fig.autofmt_xdate()

    return save_figure(fig, "rolling_var_backtest", output_dir)


def plot_correlation_heatmap(
    corr_matrix: np.ndarray,
    labels: list,
    output_dir: str = "results/figures",
) -> str:
    """
    Plot correlation matrix as an annotated heatmap.

    Parameters
    ----------
    corr_matrix : np.ndarray
        Correlation matrix (N x N).
    labels : list
        Asset labels.
    output_dir : str
        Output directory.

    Returns
    -------
    str
        Path to saved figure.
    """
    fig, ax = plt.subplots(figsize=(9, 7))

    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)

    sns.heatmap(
        corr_matrix,
        mask=mask,
        annot=True,
        fmt=".3f",
        cmap="RdYlBu_r",
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=0.5,
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
        cbar_kws={"shrink": 0.8, "label": "Correlation"},
    )

    ax.set_title("Asset Correlation Matrix",
                 fontsize=14, fontweight="bold")

    return save_figure(fig, "correlation_heatmap", output_dir)


def plot_stress_comparison(
    base_metrics: Dict[str, float],
    vol_shock_metrics: Dict[str, float],
    corr_stress_metrics: Dict[str, float],
    output_dir: str = "results/figures",
) -> str:
    """
    Bar chart comparing VaR/ES across base and stress scenarios.

    Parameters
    ----------
    base_metrics : dict
        Baseline var_95, var_99, es_95, es_99.
    vol_shock_metrics : dict
        Volatility-shocked metrics.
    corr_stress_metrics : dict
        Correlation-stressed metrics.
    output_dir : str
        Output directory.

    Returns
    -------
    str
        Path to saved figure.
    """
    fig, ax = plt.subplots(figsize=(12, 7))

    metrics = ["var_95", "var_99", "es_95", "es_99"]
    labels = ["95% VaR", "99% VaR", "95% ES", "99% ES"]
    x = np.arange(len(metrics))
    width = 0.25

    base_vals = [base_metrics[m] for m in metrics]
    vol_vals = [vol_shock_metrics[m] for m in metrics]
    corr_vals = [corr_stress_metrics[m] for m in metrics]

    ax.bar(x - width, base_vals, width, label="Baseline",
           color=COLORS["primary"], alpha=0.8)
    ax.bar(x, vol_vals, width, label="Vol Shock (2×)",
           color=COLORS["var_99"], alpha=0.8)
    ax.bar(x + width, corr_vals, width, label="Corr Stress (ρ=0.9)",
           color=COLORS["es"], alpha=0.8)

    ax.set_xlabel("Risk Metric", fontsize=12)
    ax.set_ylabel("Value (fraction of portfolio)", fontsize=12)
    ax.set_title("Stress Test Comparison — Risk Metric Sensitivity",
                 fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.legend(fontsize=11)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))

    return save_figure(fig, "stress_comparison", output_dir)
