# Monte Carlo Market Risk Engine

> A production-grade **1-day market risk model** that estimates portfolio loss distributions using Monte Carlo simulation, validates predictions through backtesting, and stress-tests under extreme scenarios.

Built from scratch in Python — no black-box libraries. Every risk metric is derived from first principles.

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Simulations](https://img.shields.io/badge/Monte%20Carlo-100%2C000%20paths-orange)

---

## What This Project Does

This engine answers the fundamental question in market risk:

> **"What is the maximum loss this portfolio could suffer tomorrow, with 99% confidence?"**

It calculates this using three independent methods, cross-validates them via backtesting, and measures how risk escalates under market stress.

### Quick Results

| Model | 95% VaR | 99% VaR | 99% Expected Shortfall |
|-------|---------|---------|------------------------|
| Historical Simulation | 1.17% | 1.99% | 2.63% |
| Parametric (Gaussian) | 1.20% | 1.72% | 1.98% |
| **Monte Carlo** | **1.20%** | **1.73%** | **1.96%** |

**Translation:** Under normal conditions, the portfolio's worst daily loss should stay below **1.73%** on 99 out of 100 trading days. When losses do exceed that threshold, the average loss is roughly **1.96%** (Expected Shortfall).

---

## Portfolio

Five assets chosen to span asset classes — equities, fixed income, and commodities — so that cross-asset correlation effects are captured:

| Asset | Ticker | Weight | Why |
|-------|--------|--------|-----|
| S&P 500 ETF | SPY | 30% | Broad U.S. equity market |
| Nasdaq 100 ETF | QQQ | 20% | High-beta technology exposure |
| JPMorgan Chase | JPM | 15% | Financial sector, rate-sensitive |
| 20+ Year Treasury ETF | TLT | 20% | Long-duration bonds, equity hedge |
| Gold ETF | GLD | 15% | Safe-haven, inflation hedge |

**Data:** 5 years of daily prices (2021–2025), ~1,254 trading days.

---

## How It Works

### Step 1 — Statistical Estimation

From historical daily prices, compute:

- **Log returns** for each asset: $r_t = \ln(P_t / P_{t-1})$
- **Mean return vector** $\boldsymbol{\mu}$ — expected daily return per asset
- **Covariance matrix** $\Sigma$ — captures how assets move together
- **Portfolio variance** $\sigma_p^2 = \mathbf{w}^T \Sigma \mathbf{w}$

### Step 2 — Three VaR Models

**1. Historical Simulation** — Sort actual past portfolio losses, pick the 1st percentile. No assumptions about the shape of the distribution.

**2. Parametric VaR** — Assume returns follow a normal (Gaussian) distribution:

$$\text{VaR}_\alpha = z_\alpha \cdot \sigma_p - \mu_p$$

Fast to compute, but underestimates extreme events because real markets have fat tails.

**3. Monte Carlo Simulation** (flagship) — The most flexible approach:
1. **Cholesky decomposition:** Factor the covariance matrix as $\Sigma = \mathbf{L}\mathbf{L}^T$
2. **Generate random draws:** $\mathbf{Z} \sim N(\mathbf{0}, \mathbf{I})$ — 100,000 independent standard normal vectors
3. **Correlate them:** $\mathbf{R} = \boldsymbol{\mu} + \mathbf{L}\mathbf{Z}$ — now the simulated returns have the same correlation structure as the real data
4. **Aggregate to portfolio:** $R_p = \mathbf{w}^T \mathbf{R}$ — weighted sum gives 100,000 possible portfolio outcomes
5. **Extract risk metrics:** VaR = 1st percentile loss; ES = average of losses beyond VaR

### Step 3 — Backtesting

How do we know if the VaR model actually works?

- Use a **rolling 250-day window** (≈ 1 year)
- At each day, estimate VaR using only past data, then check if the next day's actual loss exceeded the prediction
- Count **breaches** — days where actual loss > predicted VaR
- At 99% confidence, expect ~1% breach rate

**Our results:** 26 breaches out of 1,004 test days (2.6% breach rate vs 1% expected). The **Kupiec test** formally rejects the model (p < 0.001), suggesting the parametric VaR underestimates tail risk — a known limitation of the Gaussian assumption.

### Step 4 — Stress Testing

What happens if markets go haywire?

| Scenario | What Changes | 99% VaR Impact | 99% ES Impact |
|----------|-------------|----------------|---------------|
| **Volatility Doubling** | $\Sigma_{\text{shock}} = 2\Sigma$ | +43% | +42% |
| **Correlation Collapse** | All $\rho_{ij} \to 0.9$ | +50% | +52% |

The correlation stress scenario is more severe — when all assets start moving together, diversification evaporates and tail risk amplifies by over 50%.

---

## Visualizations

The engine generates 5 publication-quality charts:

| Figure | What It Shows |
|--------|---------------|
| `mc_pnl_distribution.png` | Full histogram of 100K simulated daily returns with VaR/ES lines |
| `left_tail_zoom.png` | Close-up of the loss tail — where risk lives |
| `rolling_var_backtest.png` | 99% VaR forecast overlaid on actual daily losses, breaches highlighted in red |
| `correlation_heatmap.png` | Cross-asset correlation structure |
| `stress_comparison.png` | Side-by-side bar chart: baseline vs vol-shock vs correlation-stress |

All saved to `results/figures/`.

---

## Project Structure

```
risk_engine/
│
├── data/
│   └── raw_prices.csv              # Downloaded price data (auto-generated)
│
├── src/
│   ├── portfolio.py                # Data loading, log returns, weight management
│   ├── statistics.py               # Mean, covariance, correlation estimation
│   ├── monte_carlo.py              # Cholesky MC engine (100K simulations)
│   ├── risk_metrics.py             # Historical & Parametric VaR/ES
│   ├── backtesting.py              # Rolling-window backtest + Kupiec test
│   ├── stress_testing.py           # Vol shock + correlation stress
│   └── visualization.py            # All chart generation
│
├── notebooks/
│   └── report.ipynb                # Interactive analysis report
│
├── results/
│   ├── figures/                    # Generated charts (PNG)
│   └── tables/                     # CSV & JSON outputs
│
├── main.py                         # Run everything in one command
├── requirements.txt
└── README.md
```

All logic lives in `src/`. The notebook only imports and visualizes. `main.py` orchestrates the full pipeline.

---

## Getting Started

### Prerequisites

- Python 3.9 or higher
- Internet connection (first run downloads price data from Yahoo Finance)

### Install & Run

```bash
# Clone the repository
git clone https://github.com/redwolf261/market-risk-engine.git
cd market-risk-engine/risk_engine

# Install dependencies
pip install -r requirements.txt

# Run the full pipeline
python main.py
```

**What happens when you run it:**
1. Downloads 5 years of price data (or loads from cache)
2. Computes statistical estimates (mean, covariance, correlation)
3. Runs Historical, Parametric, and Monte Carlo VaR
4. Executes rolling 250-day backtest with Kupiec test
5. Performs volatility shock and correlation stress tests
6. Generates all visualizations → `results/figures/`
7. Exports numerical results → `results/tables/`

### Interactive Report

Open the Jupyter notebook for a step-by-step walkthrough:

```bash
cd notebooks
jupyter notebook report.ipynb
```

---

## Mathematical Framework

<details>
<summary>Click to expand full mathematical details</summary>

### Log Returns
$$r_t = \ln\left(\frac{P_t}{P_{t-1}}\right)$$

### Portfolio Return
$$R_p = \mathbf{w}^T \mathbf{R}$$

### Covariance Matrix
$$\Sigma = E\left[(\mathbf{r} - \boldsymbol{\mu})(\mathbf{r} - \boldsymbol{\mu})^T\right]$$

### Portfolio Variance
$$\sigma_p^2 = \mathbf{w}^T \Sigma \mathbf{w}$$

### Cholesky Decomposition
$$\Sigma = \mathbf{L}\mathbf{L}^T$$

### Correlated Simulation
$$\mathbf{R} = \boldsymbol{\mu} + \mathbf{L}\mathbf{Z}, \quad \mathbf{Z} \sim N(\mathbf{0}, \mathbf{I})$$

### Parametric VaR
$$\text{VaR}_\alpha = z_\alpha \cdot \sigma_p - \mu_p$$

### Expected Shortfall
$$\text{ES}_\alpha = E\left[L \mid L > \text{VaR}_\alpha\right]$$

### Kupiec Test Statistic
$$LR = -2\ln\left[\frac{(1-p)^{T-x} \cdot p^x}{(1-\hat{p})^{T-x} \cdot \hat{p}^x}\right] \sim \chi^2(1)$$

</details>

---

## Known Limitations

| Limitation | Why It Matters |
|-----------|----------------|
| **Gaussian returns** | Real markets have fat tails and skewness — VaR underestimates extreme losses |
| **Static covariance** | Correlations change over time, especially spiking during crises |
| **No liquidity modeling** | Assumes you can sell everything at current market prices instantly |
| **No regime switching** | Doesn't distinguish between calm and volatile market regimes |
| **Single-day horizon** | Only models 1-day risk; multi-day requires different assumptions |

These are acknowledged limitations, not flaws — they represent the standard trade-offs in desk-level risk models. The backtesting results confirm that the Gaussian assumption specifically leads to VaR underestimation.

---

## Future Improvements

- **Student-t Monte Carlo** — Heavy-tailed distributions for better tail risk capture
- **GARCH volatility** — Time-varying conditional volatility
- **Regime-switching model** — Bull/bear market detection via Hidden Markov Models
- **Incremental VaR** — Decompose risk contribution by asset
- **Cornish-Fisher adjustment** — Correct parametric VaR for skewness/kurtosis

---

## What This Demonstrates

| Skill | Evidence |
|-------|----------|
| Linear algebra | Cholesky decomposition, portfolio variance via matrix multiplication |
| Probability theory | VaR quantiles, Expected Shortfall, distributional assumptions |
| Statistical estimation | Covariance matrices, mean estimation, rolling windows |
| Model validation | Backtesting framework, Kupiec test, breach analysis |
| Stress testing | Volatility shocks, correlation stress, sensitivity interpretation |
| Software engineering | Modular architecture, type hints, documented functions |

---

## License

MIT License — use freely.

---

*Built by [Rivan Avinash Shetty](https://github.com/redwolf261) as a quantitative risk analysis project aligned with trading desk market risk methodologies.*
