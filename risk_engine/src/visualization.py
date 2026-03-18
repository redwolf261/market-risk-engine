"""
Visualization Module
====================
Produces professional static charts for risk analysis reporting.

Generated Figures:
    1.  Monte Carlo P&L Distribution (Histogram)
    2.  Left-Tail Zoom with VaR/ES Lines
    3.  Rolling VaR vs Actual Losses (Backtest)
    4.  Correlation Heatmap
    5.  Stress Test Comparison
    6.  Gaussian vs Student-t Overlay
    7.  Portfolio Overview (weight pie + cumulative returns)
    8.  Rolling 30-Day Volatility with Regime Annotations
    9.  Breach Calendar Heatmap (month × year grid)
   10.  Four-Model VaR/ES Comparison (grouped bar chart)
   11.  VaR vs ES Side-by-Side (distribution + ES uplift bars)
   12.  Sharpe vs Sortino (vol breakdown + rolling ratio comparison)
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
    labels: list[str],
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


def plot_gaussian_vs_student_t(
    gaussian_pnl: np.ndarray,
    student_t_pnl: np.ndarray,
    gaussian_var99: float,
    student_t_var99: float,
    gaussian_es99: float,
    student_t_es99: float,
    df: float,
    output_dir: str = "results/figures",
) -> str:
    """
    Overlay Gaussian and Student-t MC P&L distributions.

    Highlights the heavier tails produced by Student-t innovations.

    Parameters
    ----------
    gaussian_pnl : np.ndarray
        Gaussian MC simulated portfolio P&L.
    student_t_pnl : np.ndarray
        Student-t MC simulated portfolio P&L.
    gaussian_var99 : float
        99% VaR from Gaussian MC (positive number).
    student_t_var99 : float
        99% VaR from Student-t MC (positive number).
    gaussian_es99 : float
        99% ES from Gaussian MC (positive number).
    student_t_es99 : float
        99% ES from Student-t MC (positive number).
    df : float
        Fitted degrees of freedom for the Student-t.
    output_dir : str
        Output directory for figure.

    Returns
    -------
    str
        Path to saved figure.
    """
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    # ── Left panel: Full distribution overlay ─────────────────
    ax = axes[0]
    ax.hist(gaussian_pnl, bins=250, density=True, alpha=0.55,
            color=COLORS["primary"], edgecolor="none", label="Gaussian MC")
    ax.hist(student_t_pnl, bins=250, density=True, alpha=0.55,
            color=COLORS["var_99"], edgecolor="none",
            label=f"Student-t MC (ν={df:.1f})")

    ax.axvline(-gaussian_var99, color=COLORS["primary"], linewidth=2,
               linestyle="--", label=f"Gauss 99% VaR = {gaussian_var99:.4f}")
    ax.axvline(-student_t_var99, color=COLORS["var_99"], linewidth=2,
               linestyle="--", label=f"t 99% VaR = {student_t_var99:.4f}")

    ax.set_xlabel("Portfolio Return", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title("Full P&L Distribution — Gaussian vs Student-t",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=9.5, loc="upper right")
    ax.xaxis.set_major_formatter(mtick.PercentFormatter(1.0))

    # ── Right panel: Left-tail zoom ───────────────────────────
    ax2 = axes[1]
    tail_cut = min(np.percentile(gaussian_pnl, 5),
                   np.percentile(student_t_pnl, 5))
    g_tail = gaussian_pnl[gaussian_pnl <= tail_cut]
    t_tail = student_t_pnl[student_t_pnl <= tail_cut]

    ax2.hist(g_tail, bins=120, density=True, alpha=0.55,
             color=COLORS["primary"], edgecolor="none", label="Gaussian MC")
    ax2.hist(t_tail, bins=120, density=True, alpha=0.55,
             color=COLORS["var_99"], edgecolor="none",
             label=f"Student-t MC (ν={df:.1f})")

    ax2.axvline(-gaussian_es99, color=COLORS["primary"], linewidth=2,
                linestyle=":", label=f"Gauss 99% ES = {gaussian_es99:.4f}")
    ax2.axvline(-student_t_es99, color=COLORS["var_99"], linewidth=2,
                linestyle=":", label=f"t 99% ES = {student_t_es99:.4f}")

    ax2.set_xlabel("Portfolio Return", fontsize=12)
    ax2.set_ylabel("Density", fontsize=12)
    ax2.set_title("Left-Tail Zoom — Fat-Tail Effect",
                  fontsize=13, fontweight="bold")
    ax2.legend(fontsize=9.5, loc="upper left")
    ax2.xaxis.set_major_formatter(mtick.PercentFormatter(1.0))

    fig.suptitle(
        "Gaussian vs Student-t Monte Carlo — Impact of Heavy Tails on Risk Estimates",
        fontsize=14, fontweight="bold", y=1.02,
    )
    fig.tight_layout()

    return save_figure(fig, "gaussian_vs_student_t", output_dir)


# ─────────────────────────────────────────────────────────────
# Chart 7: Portfolio Overview
# ─────────────────────────────────────────────────────────────

def plot_portfolio_overview(
    prices: pd.DataFrame,
    weights: np.ndarray,
    output_dir: str = "results/figures",
) -> str:
    """
    Two-panel overview: portfolio weights pie + cumulative return by asset.

    Shows what is inside the portfolio and how each component has
    contributed to performance over the full data window.

    Parameters
    ----------
    prices : pd.DataFrame
        Asset price history (T × N), columns = tickers.
    weights : np.ndarray
        Portfolio weight vector aligned with prices.columns.
    output_dir : str
        Output directory.

    Returns
    -------
    str
        Path to saved figure.
    """
    tickers = list(prices.columns)

    fig, (ax_pie, ax_ret) = plt.subplots(
        1, 2, figsize=(18, 7),
        gridspec_kw={"width_ratios": [1, 1.8]},
    )

    # ── Left: Pie chart of weights ───────────────────────────
    palette = plt.cm.tab10.colors
    wedge_colors = [palette[i] for i in range(len(tickers))]

    wedges, texts, autotexts = ax_pie.pie(
        weights,
        labels=tickers,
        autopct="%1.0f%%",
        startangle=90,
        colors=wedge_colors,
        wedgeprops={"linewidth": 1.5, "edgecolor": "white"},
        textprops={"fontsize": 12},
        pctdistance=0.75,
    )
    for at in autotexts:
        at.set_fontsize(11)
        at.set_fontweight("bold")

    ax_pie.set_title(
        "Portfolio Allocation",
        fontsize=13, fontweight="bold", pad=15,
    )

    # ── Right: Cumulative returns per asset + portfolio ──────
    cumret = (prices / prices.iloc[0] - 1) * 100  # in %

    for i, ticker in enumerate(tickers):
        ax_ret.plot(
            prices.index, cumret[ticker],
            color=wedge_colors[i], linewidth=1.5,
            alpha=0.85, label=ticker,
        )

    # Portfolio cumulative return
    log_ret = np.log(prices / prices.shift(1)).dropna()
    port_daily = log_ret.values @ weights
    port_cum = (np.exp(np.cumsum(port_daily)) - 1) * 100
    ax_ret.plot(
        prices.index[1:], port_cum,
        color="black", linewidth=2.5, linestyle="--",
        label="Portfolio", zorder=10,
    )

    ax_ret.axhline(0, color="gray", linewidth=0.8, linestyle=":")
    ax_ret.set_xlabel("Date", fontsize=12)
    ax_ret.set_ylabel("Cumulative Return (%)", fontsize=12)
    ax_ret.set_title(
        "Cumulative Returns — Individual Assets vs Portfolio",
        fontsize=13, fontweight="bold",
    )
    ax_ret.legend(fontsize=10, ncol=2)
    ax_ret.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
    fig.autofmt_xdate()
    fig.tight_layout()

    return save_figure(fig, "portfolio_overview", output_dir)


# ─────────────────────────────────────────────────────────────
# Chart 8: Rolling 30-Day Volatility
# ─────────────────────────────────────────────────────────────

def plot_rolling_volatility(
    portfolio_returns: pd.Series,
    window: int = 30,
    ann_vol: float = None,
    output_dir: str = "results/figures",
) -> str:
    """
    Rolling annualised volatility of portfolio returns with shading.

    Time-varying volatility is visible here — the clustering of high-vol
    periods (2022 rate shock, 2025 drawdown) motivates GARCH-style
    dynamic risk models over a constant-covariance assumption.

    Parameters
    ----------
    portfolio_returns : pd.Series
        Daily log portfolio returns.
    window : int
        Rolling window in trading days (default 30).
    ann_vol : float, optional
        Full-sample annualised volatility for reference line.
    output_dir : str
        Output directory.

    Returns
    -------
    str
        Path to saved figure.
    """
    roll_std = portfolio_returns.rolling(window).std() * np.sqrt(252) * 100
    roll_std = roll_std.dropna()

    fig, ax = plt.subplots(figsize=(16, 6))

    ax.fill_between(
        roll_std.index, roll_std.values,
        alpha=0.35, color=COLORS["primary"],
    )
    ax.plot(
        roll_std.index, roll_std.values,
        color=COLORS["primary"], linewidth=1.5,
        label=f"{window}-Day Rolling Volatility",
    )

    if ann_vol is not None:
        ax.axhline(
            ann_vol * 100, color=COLORS["var_99"],
            linewidth=1.5, linestyle="--",
            label=f"Full-Sample Vol = {ann_vol*100:.1f}%",
        )

    # Shade the top-quartile vol periods in red
    q75 = np.percentile(roll_std.values, 75)
    ax.fill_between(
        roll_std.index,
        roll_std.values,
        where=roll_std.values >= q75,
        alpha=0.45, color=COLORS["breach"],
        label="High-Volatility Regime (top quartile)",
    )

    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Annualised Volatility (%)", fontsize=12)
    ax.set_title(
        f"Portfolio Rolling {window}-Day Volatility — Volatility Clustering Visible",
        fontsize=14, fontweight="bold",
    )
    ax.legend(fontsize=11)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=1))
    fig.autofmt_xdate()
    fig.tight_layout()

    return save_figure(fig, "rolling_volatility", output_dir)


# ─────────────────────────────────────────────────────────────
# Chart 9: Breach Calendar Heatmap
# ─────────────────────────────────────────────────────────────

def plot_breach_calendar(
    backtest_results: pd.DataFrame,
    output_dir: str = "results/figures",
) -> str:
    """
    Calendar heatmap showing monthly VaR breach counts.

    Each cell = one calendar month.  Colour intensity = number of days
    that month where actual loss exceeded the 99% VaR forecast.
    Zero-breach months are white; high-breach months are deep red.

    This single chart makes it obvious that breaches cluster in
    crisis periods rather than being uniformly distributed —
    proving that the IID Gaussian assumption fails in practice.

    Parameters
    ----------
    backtest_results : pd.DataFrame
        Output from rolling_var_backtest / run_full_backtest.
        Must include columns: date, breach.
    output_dir : str
        Output directory.

    Returns
    -------
    str
        Path to saved figure.
    """
    df = backtest_results.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month

    pivot = (
        df.groupby(["year", "month"])["breach"]
        .sum()
        .unstack(fill_value=0)
    )

    # Ensure all 12 months are present
    for m in range(1, 13):
        if m not in pivot.columns:
            pivot[m] = 0
    pivot = pivot.sort_index(axis=1)

    month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    fig, ax = plt.subplots(figsize=(14, max(3, len(pivot) * 1.2)))

    sns.heatmap(
        pivot,
        annot=True,
        fmt="d",
        cmap="YlOrRd",
        linewidths=0.5,
        linecolor="white",
        xticklabels=month_labels,
        yticklabels=[str(y) for y in pivot.index],
        cbar_kws={"label": "Breach Count", "shrink": 0.6},
        ax=ax,
    )

    ax.set_xlabel("Month", fontsize=12)
    ax.set_ylabel("Year", fontsize=12)
    ax.set_title(
        "VaR Breach Calendar — Monthly Breach Counts at 99% Confidence\n"
        "Clustering = Gaussian IID Assumption Fails",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout()

    return save_figure(fig, "breach_calendar", output_dir)


# ─────────────────────────────────────────────────────────────
# Chart 10: Four-Model VaR/ES Comparison
# ─────────────────────────────────────────────────────────────

def plot_model_comparison(
    hist_metrics: dict,
    param_metrics: dict,
    mc_metrics: dict,
    t_metrics: dict,
    df_t: float,
    output_dir: str = "results/figures",
) -> str:
    """
    Grouped bar chart comparing all four risk models across all metrics.

    Lets the reader immediately see:
      - How close Parametric and MC (Gaussian) are (model consistency)
      - How much Student-t diverges at 99% vs 95% (fat-tail effect)
      - Where Historical simulation differs (non-parametric benchmark)

    Parameters
    ----------
    hist_metrics : dict
        Historical VaR/ES (keys: hist_var_95, hist_var_99, etc.)
    param_metrics : dict
        Parametric Gaussian metrics.
    mc_metrics : dict
        Monte Carlo Gaussian metrics.
    t_metrics : dict
        Student-t MC metrics.
    df_t : float
        Fitted degrees of freedom (for chart label).
    output_dir : str
        Output directory.

    Returns
    -------
    str
        Path to saved figure.
    """
    models = [
        "Historical",
        "Parametric\n(Gaussian)",
        "MC\n(Gaussian)",
        f"MC\n(Student-t\nν={df_t:.1f})",
    ]

    var95 = [
        hist_metrics["hist_var_95"],
        param_metrics["param_var_95"],
        mc_metrics["mc_var_95"],
        t_metrics["t_var_95"],
    ]
    var99 = [
        hist_metrics["hist_var_99"],
        param_metrics["param_var_99"],
        mc_metrics["mc_var_99"],
        t_metrics["t_var_99"],
    ]
    es95 = [
        hist_metrics["hist_es_95"],
        param_metrics["param_es_95"],
        mc_metrics["mc_es_95"],
        t_metrics["t_es_95"],
    ]
    es99 = [
        hist_metrics["hist_es_99"],
        param_metrics["param_es_99"],
        mc_metrics["mc_es_99"],
        t_metrics["t_es_99"],
    ]

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    x = np.arange(len(models))
    width = 0.38

    bar_colors = [COLORS["var_95"], COLORS["var_99"]]
    for ax, (vals_a, vals_b, label_a, label_b, title) in zip(
        axes,
        [
            (var95, var99, "95% VaR", "99% VaR", "Value at Risk — 1-Day Horizon"),
            (es95, es99, "95% ES (CVaR)", "99% ES (CVaR)", "Expected Shortfall — 1-Day Horizon"),
        ],
    ):
        bars_a = ax.bar(x - width / 2, vals_a, width,
                        label=label_a, color=bar_colors[0], alpha=0.85)
        bars_b = ax.bar(x + width / 2, vals_b, width,
                        label=label_b, color=bar_colors[1], alpha=0.85)

        # Value labels on top of each bar
        for bar in list(bars_a) + list(bars_b):
            h = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h + 0.0003,
                f"{h*100:.2f}%",
                ha="center", va="bottom", fontsize=8.5, fontweight="bold",
            )

        ax.set_xticks(x)
        ax.set_xticklabels(models, fontsize=10)
        ax.set_ylabel("Portfolio Loss (%)", fontsize=12)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.legend(fontsize=11)
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))

    fig.suptitle(
        "Risk Model Comparison — Four Approaches to Measuring Tail Risk",
        fontsize=14, fontweight="bold", y=1.02,
    )
    fig.tight_layout()

    return save_figure(fig, "model_comparison", output_dir)


# ─────────────────────────────────────────────────────────────
# Chart 11: VaR vs ES — What VaR Misses
# ─────────────────────────────────────────────────────────────

def plot_var_vs_es(
    mc_pnl: np.ndarray,
    mc_metrics: dict,
    t_metrics: dict,
    df_t: float,
    output_dir: str = "results/figures",
) -> str:
    """
    Two-panel chart showing the conceptual gap between VaR and ES.

    Left — P&L distribution with both VaR and ES lines marked and the
    tail region shaded; shows *what VaR ignores* (the shaded area)
    versus *what ES captures* (the average of that shaded area).

    Right — "ES Uplift" bar chart: ES − VaR for Gaussian and Student-t
    at 95% and 99% confidence.  This is the extra capital buffer that
    Basel III requires over Basel II.  The Student-t bars are visibly
    taller at 99%, showing how much fat tails inflate the uplift.

    Parameters
    ----------
    mc_pnl : np.ndarray
        100K simulated 1-day portfolio returns (Gaussian MC).
    mc_metrics : dict
        Keys: mc_var_95, mc_var_99, mc_es_95, mc_es_99.
    t_metrics : dict
        Keys: t_var_95, t_var_99, t_es_95, t_es_99.
    df_t : float
        Fitted Student-t degrees of freedom (for label).
    output_dir : str
        Output directory.

    Returns
    -------
    str
        Path to saved figure.
    """
    losses = -mc_pnl  # positive = loss

    fig, (ax_dist, ax_uplift) = plt.subplots(1, 2, figsize=(18, 7))

    # ── Left: distribution with VaR and ES markers ───────────
    ax_dist.hist(
        losses, bins=120, color=COLORS["primary"],
        alpha=0.55, edgecolor="none", density=True,
        label="MC P&L Distribution",
    )

    # Shade the region beyond 99% VaR (what VaR leaves uncovered)
    var99_g = mc_metrics["mc_var_99"]
    es99_g  = mc_metrics["mc_es_99"]
    tail_mask = losses >= var99_g
    if tail_mask.sum() > 0:
        hist_vals, bin_edges = np.histogram(
            losses[tail_mask], bins=60, density=True,
        )
        # Scale density back to match the main histogram
        scale = tail_mask.mean()
        for i in range(len(hist_vals)):
            ax_dist.fill_betweenx(
                [0, hist_vals[i] * scale],
                bin_edges[i], bin_edges[i + 1],
                color=COLORS["breach"], alpha=0.55,
            )
    # Invisible proxy patch for legend
    ax_dist.fill_between(
        [], [], color=COLORS["breach"], alpha=0.55,
        label="Tail VaR ignores (but ES captures)",
    )

    # VaR lines
    ax_dist.axvline(
        mc_metrics["mc_var_95"], color=COLORS["var_95"],
        linewidth=2, linestyle="--", label="95% VaR",
    )
    ax_dist.axvline(
        var99_g, color=COLORS["var_99"],
        linewidth=2, linestyle="--", label="99% VaR",
    )
    # ES lines (solid — represents the conditional average)
    ax_dist.axvline(
        mc_metrics["mc_es_95"], color=COLORS["var_95"],
        linewidth=2, linestyle="-", label="95% ES (CVaR)",
    )
    ax_dist.axvline(
        es99_g, color=COLORS["var_99"],
        linewidth=2, linestyle="-", label="99% ES (CVaR)",
    )

    # Annotations
    y_arrow = ax_dist.get_ylim()[1] * 0.6 if ax_dist.get_ylim()[1] != 1 else 30
    ax_dist.annotate(
        "← VaR sets\n   threshold",
        xy=(var99_g, 0), xytext=(var99_g - 0.008, y_arrow * 0.5),
        fontsize=9, color=COLORS["var_99"],
        arrowprops=dict(arrowstyle="->", color=COLORS["var_99"]),
    )
    ax_dist.annotate(
        "ES = average\nloss here →",
        xy=(es99_g, 0), xytext=(es99_g + 0.001, y_arrow * 0.5),
        fontsize=9, color=COLORS["var_99"],
        arrowprops=dict(arrowstyle="->", color=COLORS["var_99"]),
    )

    ax_dist.set_xlabel("1-Day Portfolio Loss", fontsize=12)
    ax_dist.set_ylabel("Density", fontsize=12)
    ax_dist.set_xlim(left=-0.04)
    ax_dist.xaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax_dist.set_title(
        "VaR = Loss Threshold  |  ES = Average Loss Beyond That Threshold\n"
        "Basel III switched from VaR to ES for exactly this reason",
        fontsize=11, fontweight="bold",
    )
    ax_dist.legend(fontsize=9.5)

    # ── Right: ES uplift (ES - VaR) per model and confidence ─
    labels = ["95%\nGaussian", "99%\nGaussian",
              f"95%\nStudent-t\nν={df_t:.1f}", f"99%\nStudent-t\nν={df_t:.1f}"]
    uplift = [
        mc_metrics["mc_es_95"] - mc_metrics["mc_var_95"],
        mc_metrics["mc_es_99"] - mc_metrics["mc_var_99"],
        t_metrics["t_es_95"]   - t_metrics["t_var_95"],
        t_metrics["t_es_99"]   - t_metrics["t_var_99"],
    ]
    bar_cols = [
        COLORS["var_95"], COLORS["var_99"],
        COLORS["var_95"], COLORS["var_99"],
    ]
    hatches = ["", "", "///", "///"]

    bars = ax_uplift.bar(
        labels, uplift, color=bar_cols, alpha=0.82,
        edgecolor="white", linewidth=1.2,
    )
    for bar, hatch in zip(bars, hatches):
        bar.set_hatch(hatch)

    # Value labels
    for bar in bars:
        h = bar.get_height()
        ax_uplift.text(
            bar.get_x() + bar.get_width() / 2,
            h + 0.0001,
            f"+{h*100:.2f}%",
            ha="center", va="bottom", fontsize=10, fontweight="bold",
        )

    ax_uplift.set_ylabel("ES − VaR  (Extra Capital Buffer, %)", fontsize=12)
    ax_uplift.set_title(
        "ES Uplift Over VaR — The Basel III Capital Premium\n"
        "Student-t uplift at 99% is larger because of fat tails",
        fontsize=11, fontweight="bold",
    )
    ax_uplift.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))

    # Legend patches for model type
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="grey", alpha=0.7, label="Gaussian MC"),
        Patch(facecolor="grey", alpha=0.7, hatch="///", label=f"Student-t MC (ν={df_t:.1f})"),
    ]
    ax_uplift.legend(handles=legend_elements, fontsize=10)

    fig.suptitle(
        "VaR vs Expected Shortfall — What the Threshold Misses",
        fontsize=14, fontweight="bold", y=1.02,
    )
    fig.tight_layout()

    return save_figure(fig, "var_vs_es", output_dir)


# ─────────────────────────────────────────────────────────────
# Chart 12: Sharpe vs Sortino — Upside vs Downside Volatility
# ─────────────────────────────────────────────────────────────

def plot_sharpe_vs_sortino(
    portfolio_returns: pd.Series,
    summary: dict,
    window: int = 90,
    output_dir: str = "results/figures",
) -> str:
    """
    Two-panel comparison of Sharpe and Sortino ratios.

    Left — Static breakdown: annualised total volatility vs downside
    volatility side by side, with the resulting ratio printed on each
    bar.  Makes immediately clear why Sortino > Sharpe: this portfolio's
    volatility is disproportionately on the *upside*, so penalising all
    volatility (Sharpe) is unfair to the portfolio.

    Right — Rolling {window}-day Sharpe vs Sortino through time.
    The gap between the two lines widens during rallies (upside vol
    dominates) and narrows during drawdowns (downside vol dominates).
    When Sortino >> Sharpe, the portfolio is rewarding investors well
    *without* generating harmful downside risk.

    Parameters
    ----------
    portfolio_returns : pd.Series
        Daily log portfolio returns.
    summary : dict
        Output of get_portfolio_summary() — provides full-sample metrics.
    window : int
        Rolling window in trading days (default 90).
    output_dir : str
        Output directory.

    Returns
    -------
    str
        Path to saved figure.
    """
    ann_factor = np.sqrt(252)
    ret = portfolio_returns.dropna()

    # ── Rolling Sharpe & Sortino ─────────────────────────────
    roll_mean  = ret.rolling(window).mean() * 252
    roll_std   = ret.rolling(window).std()  * ann_factor

    downside   = ret.copy()
    downside[downside > 0] = 0.0
    roll_d_std = downside.rolling(window).apply(
        lambda x: np.std(x[x < 0]) * ann_factor if (x < 0).sum() > 1 else np.nan,
        raw=True,
    )

    roll_sharpe  = roll_mean / roll_std
    roll_sortino = roll_mean / roll_d_std

    fig, (ax_bar, ax_roll) = plt.subplots(1, 2, figsize=(18, 7))

    # ── Left: Static vol breakdown + ratio labels ─────────────
    full_vol     = summary["annualized_volatility"]   * 100
    down_vol     = summary["annualized_downside_vol"] * 100
    upside_vol   = full_vol - down_vol                        # implied

    categories   = ["Total\nVolatility", "Downside\nVolatility"]
    vol_vals     = [full_vol, down_vol]
    bar_c        = [COLORS["primary"], COLORS["breach"]]
    ratios       = [summary["sharpe_ratio"], summary["sortino_ratio"]]
    ratio_labels = ["Sharpe", "Sortino"]

    bars = ax_bar.bar(
        categories, vol_vals, color=bar_c,
        alpha=0.80, width=0.45, edgecolor="white", linewidth=1.5,
    )

    for bar, vol, ratio, rl in zip(bars, vol_vals, ratios, ratio_labels):
        ax_bar.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.2,
            f"{vol:.1f}%",
            ha="center", va="bottom", fontsize=12, fontweight="bold",
        )
        ax_bar.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() / 2,
            f"{rl} = {ratio:.2f}",
            ha="center", va="center", fontsize=13,
            fontweight="bold", color="white",
        )

    # Annotate the upside component
    ax_bar.annotate(
        f"Upside vol\n= {upside_vol:.1f}%\n(Sortino ignores this)",
        xy=(0, down_vol), xytext=(0.55, full_vol * 0.85),
        fontsize=9.5, color="dimgrey",
        arrowprops=dict(arrowstyle="->", color="dimgrey", lw=1.2),
    )

    ax_bar.set_ylabel("Annualised Volatility (%)", fontsize=12)
    ax_bar.set_title(
        "Total vs Downside Volatility\nSortino ignores upside vol → higher ratio",
        fontsize=12, fontweight="bold",
    )
    ax_bar.set_ylim(0, full_vol * 1.35)
    ax_bar.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=1))

    # ── Right: Rolling Sharpe vs Sortino ─────────────────────
    idx = ret.index

    ax_roll.plot(
        idx, roll_sharpe,
        color=COLORS["primary"], linewidth=1.8,
        label=f"{window}-Day Rolling Sharpe",
    )
    ax_roll.plot(
        idx, roll_sortino,
        color=COLORS["safe"], linewidth=1.8,
        label=f"{window}-Day Rolling Sortino",
    )

    ax_roll.fill_between(
        idx,
        roll_sharpe.values, roll_sortino.values,
        where=(roll_sortino.values > roll_sharpe.values),
        alpha=0.18, color=COLORS["safe"],
        label="Gap = upside vol (Sortino advantage)",
    )
    ax_roll.fill_between(
        idx,
        roll_sharpe.values, roll_sortino.values,
        where=(roll_sortino.values <= roll_sharpe.values),
        alpha=0.25, color=COLORS["breach"],
        label="Gap inverted = elevated downside risk",
    )

    ax_roll.axhline(0, color="black", linewidth=0.8, linestyle=":")
    ax_roll.axhline(
        summary["sharpe_ratio"], color=COLORS["primary"],
        linewidth=1.2, linestyle="--", alpha=0.6,
        label=f"Full-sample Sharpe = {summary['sharpe_ratio']:.2f}",
    )
    ax_roll.axhline(
        summary["sortino_ratio"], color=COLORS["safe"],
        linewidth=1.2, linestyle="--", alpha=0.6,
        label=f"Full-sample Sortino = {summary['sortino_ratio']:.2f}",
    )

    ax_roll.set_xlabel("Date", fontsize=12)
    ax_roll.set_ylabel("Risk-Adjusted Return Ratio", fontsize=12)
    ax_roll.set_title(
        f"Rolling {window}-Day Sharpe vs Sortino\n"
        "Green gap = volatility is mostly gains; red gap = drawdown stress",
        fontsize=12, fontweight="bold",
    )
    ax_roll.legend(fontsize=9.5, ncol=2)
    fig.autofmt_xdate()

    fig.suptitle(
        "Sharpe vs Sortino Ratio — Why Penalising All Volatility Understates Performance",
        fontsize=14, fontweight="bold", y=1.02,
    )
    fig.tight_layout()

    return save_figure(fig, "sharpe_vs_sortino", output_dir)
