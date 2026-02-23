# Monte Carlo Market Risk Engine

**Production-grade 1-day trading desk risk model** implementing correlated Monte Carlo simulation, multi-model VaR estimation, Expected Shortfall, rolling-window backtesting, and volatility shock stress testing.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Mathematical Framework](#mathematical-framework)
3. [Risk Models Implemented](#risk-models-implemented)
4. [Backtesting Framework](#backtesting-framework)
5. [Stress Testing](#stress-testing)
6. [Project Structure](#project-structure)
7. [Installation & Usage](#installation--usage)
8. [Key Results](#key-results)
9. [Limitations](#limitations)
10. [Future Improvements](#future-improvements)

---

## Project Overview

This engine estimates the **1-day loss distribution** of a multi-asset portfolio using three complementary VaR methodologies, validated through rolling-window backtesting and stress-tested under extreme market scenarios.

### Portfolio Composition

| Asset | Ticker | Weight | Role |
|-------|--------|--------|------|
| S&P 500 ETF | SPY | 30% | Broad equity exposure |
| Nasdaq 100 ETF | QQQ | 20% | High-beta technology |
| JPMorgan Chase | JPM | 15% | Financial sector |
| 20+ Year Treasury | TLT | 20% | Duration / rate exposure |
| Gold ETF | GLD | 15% | Safe-haven hedge |

Cross-asset covariance structure is central to accurate risk estimation. The portfolio spans equity, fixed income, and commodities to capture meaningful correlation dynamics.

---

## Mathematical Framework

### Log Returns

$$r_t = \ln\left(\frac{P_t}{P_{t-1}}\right)$$

### Portfolio Return

$$R_p = \mathbf{w}^T \mathbf{R}$$

### Covariance Matrix

$$\Sigma = E\left[(\mathbf{r} - \boldsymbol{\mu})(\mathbf{r} - \boldsymbol{\mu})^T\right]$$

### Portfolio Variance

$$\sigma_p^2 = \mathbf{w}^T \Sigma \mathbf{w}$$

### Cholesky Decomposition (Monte Carlo Core)

$$\Sigma = \mathbf{L}\mathbf{L}^T$$

Correlated simulation:

$$\mathbf{R} = \boldsymbol{\mu} + \mathbf{L}\mathbf{Z}, \quad \mathbf{Z} \sim N(\mathbf{0}, \mathbf{I})$$

### Value-at-Risk

**Parametric:**
$$\text{VaR}_\alpha = z_\alpha \cdot \sigma_p - \mu_p$$

**Monte Carlo / Historical:**
$$\text{VaR}_\alpha = -Q_\alpha(R_p)$$

### Expected Shortfall

$$\text{ES}_\alpha = E\left[L \mid L > \text{VaR}_\alpha\right]$$

---

## Risk Models Implemented

### 1. Historical Simulation
- Distribution-free approach
- Directly uses empirical quantiles of realized portfolio losses
- **Strength:** No distributional assumptions
- **Weakness:** Backward-looking, limited by sample size

### 2. Parametric (Variance-Covariance)
- Assumes Gaussian returns
- Analytically computes VaR from portfolio mean and standard deviation
- **Strength:** Computationally efficient
- **Weakness:** Underestimates tail risk due to normality assumption

### 3. Monte Carlo Simulation (Flagship)
- 100,000 correlated return paths via Cholesky decomposition
- Full portfolio P&L distribution
- **Strength:** Flexible, captures non-linear dependencies
- **Weakness:** Computationally intensive, dependent on input distribution

---

## Backtesting Framework

Rolling-window approach with 250-day estimation window:

1. Estimate μ and Σ from trailing 250 trading days
2. Compute 1-day 99% Parametric VaR forecast
3. Compare prediction to actual realized loss at t+1
4. Record breach (actual loss > predicted VaR)

### Kupiec Proportion of Failures Test

Likelihood ratio statistic:

$$LR = -2\ln\left[\frac{(1-p)^{T-x} \cdot p^x}{(1-\hat{p})^{T-x} \cdot \hat{p}^x}\right]$$

Under $H_0$: $LR \sim \chi^2(1)$

Where $p = 1 - \alpha$ is the expected failure rate and $\hat{p} = x/T$ is the observed rate.

---

## Stress Testing

### Volatility Shock
$$\Sigma_{\text{shock}} = 2\Sigma$$

Equivalent to doubling all asset volatilities. Re-runs full Monte Carlo engine to measure VaR/ES sensitivity.

### Correlation Stress
$$\rho_{ij} \rightarrow 0.9 \quad \forall \, i \neq j$$

Simulates diversification collapse under systemic crisis. Reconstructs covariance from stressed correlation matrix.

---

## Project Structure

```
risk_engine/
│
├── data/
│   └── raw_prices.csv          # Historical price data
│
├── src/
│   ├── __init__.py
│   ├── portfolio.py             # Data loading, returns, weights
│   ├── statistics.py            # μ, Σ, ρ estimation
│   ├── monte_carlo.py           # Cholesky MC simulation engine
│   ├── risk_metrics.py          # Historical & Parametric VaR/ES
│   ├── backtesting.py           # Rolling VaR backtest + Kupiec
│   ├── stress_testing.py        # Vol shock & correlation stress
│   └── visualization.py         # Professional chart generation
│
├── notebooks/
│   └── report.ipynb             # Full analysis report
│
├── results/
│   ├── figures/                 # Generated visualizations
│   └── tables/                  # CSV/JSON output
│
├── main.py                      # Pipeline orchestrator
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Installation & Usage

### Prerequisites
- Python 3.9+

### Setup

```bash
git clone https://github.com/redwolf261/market-risk-engine.git
cd market-risk-engine/risk_engine
pip install -r requirements.txt
```

### Run

```bash
python main.py
```

The engine will:
1. Fetch historical data (or load from cache)
2. Run all three VaR models
3. Execute backtesting pipeline
4. Perform stress analysis
5. Generate visualizations in `results/figures/`
6. Export results to `results/tables/`

---

## Key Results

Results are generated dynamically. After running `main.py`, see:

- **Comparison table:** `results/tables/risk_metrics_comparison.csv`
- **Full output:** `results/tables/full_results.json`
- **Visualizations:** `results/figures/`

### Output Figures

| Figure | Description |
|--------|-------------|
| `mc_pnl_distribution.png` | Full Monte Carlo P&L histogram with VaR/ES lines |
| `left_tail_zoom.png` | Zoomed view of the loss tail |
| `rolling_var_backtest.png` | 99% VaR forecast vs actual losses with breach markers |
| `correlation_heatmap.png` | Cross-asset correlation structure |
| `stress_comparison.png` | Bar chart comparing base vs stressed risk metrics |

---

## Limitations

1. **Gaussian assumption** — Parametric and MC models assume normal returns, which underestimates tail risk. Real asset returns exhibit fat tails and skewness.

2. **Static correlation** — Covariance is estimated from a fixed historical window. In reality, correlations spike during market stress (correlation breakdown).

3. **No liquidity effects** — VaR assumes positions can be liquidated at market prices. During crises, bid-ask spreads widen and market depth evaporates.

4. **No regime switching** — Volatility clustering and mean reversion are not modeled. A GARCH or regime-switching framework would better capture time-varying dynamics.

5. **Linear portfolio** — No options, derivatives, or non-linear instruments. VaR for non-linear portfolios requires delta-gamma or full revaluation approaches.

6. **Single-period horizon** — Only considers 1-day risk. Multi-day VaR scaling by √t assumes i.i.d. returns, which is violated in practice.

---

## Future Improvements

- [ ] **Student-t Monte Carlo** — Replace Gaussian with heavy-tailed distribution to better capture extreme losses
- [ ] **GARCH volatility** — Time-varying conditional volatility for more responsive risk estimates
- [ ] **Regime-switching model** — Hidden Markov Model for bull/bear market detection
- [ ] **Cornish-Fisher VaR** — Adjust parametric VaR for skewness and kurtosis
- [ ] **Incremental VaR** — Decompose risk contributions by asset
- [ ] **Liquidity-adjusted VaR** — Incorporate bid-ask spread and market impact

---

## License

MIT License

---

*Built as a quantitative risk analysis project demonstrating proficiency in probability theory, linear algebra, statistical estimation, and model validation — aligned with trading desk market risk methodologies.*
