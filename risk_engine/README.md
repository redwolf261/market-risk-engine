# Monte Carlo Market Risk Engine

> A production-grade **1-day market risk model** that estimates portfolio loss distributions using Monte Carlo simulation, validates predictions through backtesting, and stress-tests under extreme scenarios — similar to professional risk analytics used on bank trading desks.

Built from scratch in Python — no black-box libraries. Every risk metric is derived from first principles.

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Simulations](https://img.shields.io/badge/Monte%20Carlo-100%2C000%20paths-orange)
![Student-t](https://img.shields.io/badge/Student--t-Heavy%20Tails-red)

---

## Introduction & Motivation

Market risk — the risk of losses from movements in asset prices — is one of the most critical concerns for any financial institution. Every trading desk, hedge fund, and asset manager needs a quantitative answer to the question:

> **"What is the worst-case loss this portfolio could suffer tomorrow?"**

This project implements a complete **1-day market risk engine** that answers that question using multiple methods, comparing their strengths and weaknesses. The engine follows the same analytical approach used by professional risk teams in investment banks.

### Regulatory Context — From VaR to ES (Basel III)

Historically, banks relied on **Value-at-Risk (VaR)** as the primary risk measure. However, VaR has a fundamental flaw: it tells you *the threshold* of the worst 1% of days, but says nothing about *how bad* that worst 1% actually is.

The **Basel III / Fundamental Review of the Trading Book (FRTB)** framework addressed this by shifting the regulatory standard from VaR to **Expected Shortfall (ES)**, which captures the *average* loss in the tail. This engine computes both:

| Measure | What It Answers | Regulatory Role |
|---------|----------------|-----------------|
| **VaR** | "What is the maximum loss at 99% confidence?" | Basel II standard (legacy) |
| **ES** | "When losses exceed VaR, how bad are they on average?" | Basel III / FRTB standard |

This project implements both metrics across four model families—Historical, Parametric, Gaussian MC, and **Student-t MC**—giving a comprehensive view of tail risk under different distributional assumptions.

---

## Quick Results

| Model | 95% VaR | 99% VaR | 99% Expected Shortfall |
|-------|---------|---------|------------------------|
| Historical Simulation | 1.29% | 2.17% | 2.90% |
| Parametric (Gaussian) | 1.30% | 1.86% | 2.14% |
| Monte Carlo (Gaussian) | 1.31% | 1.86% | 2.13% |
| **Monte Carlo (Student-t, ν=4.6)** | **1.26%** | **2.02%** | **2.59%** |

> *The Student-t MC produces higher 99% VaR (+9%) and dramatically higher 99% ES (+22%) compared to Gaussian MC — because it models the fat tails observed in real market data.*

> **Why is Student-t 95% VaR (1.26%) slightly *lower* than Gaussian (1.31%)?**  
> This is a known and expected property of fat-tailed distributions at moderate confidence levels.  
> A Student-t with ν ≈ 4.6 has heavier tails than the Gaussian, which means probability mass is redistributed *away* from the centre into the extreme regions.  
> At the 95th percentile — still in the near-tail — this redistribution leaves slightly *less* mass beyond the threshold compared to the Gaussian, causing a marginally lower VaR.  
> At the 99th percentile, where fat-tail effects fully dominate, the Student-t **correctly** produces higher VaR (+9%) and substantially higher ES (+22%).  
> The crossover behaviour (lower at 95%, higher at 99%) is therefore a *feature* confirming the model is working as intended — not a bug.

**Translation:** Under normal conditions, the portfolio's worst daily loss should stay below **~1.9–2.0%** on 99 out of 100 trading days. When losses do exceed that threshold, the Student-t model predicts an average tail loss of **2.59%** — significantly worse than the Gaussian estimate of 2.13%.

---

## Portfolio Construction

Five assets chosen to span asset classes — equities, fixed income, and commodities — so cross-asset correlation effects are captured:

| Asset | Ticker | Weight | Rationale |
|-------|--------|--------|-----------|
| S&P 500 ETF | SPY | 30% | Broad U.S. equity market |
| Nasdaq 100 ETF | QQQ | 20% | High-beta technology exposure |
| JPMorgan Chase | JPM | 15% | Financial sector, rate-sensitive |
| 20+ Year Treasury ETF | TLT | 20% | Long-duration bonds, equity hedge |
| Gold ETF | GLD | 15% | Safe-haven, inflation hedge |

**Data:** 5 years of daily prices (2021–2025), ~1,254 trading days via Yahoo Finance.

---

## How It Works

### Step 1 — Statistical Estimation

From historical daily prices, compute:

- **Log returns** for each asset: $r_t = \ln(P_t / P_{t-1})$
- **Mean return vector** $\boldsymbol{\mu}$ — expected daily return per asset
- **Covariance matrix** $\Sigma$ — captures how assets move together
- **Portfolio variance** $\sigma_p^2 = \mathbf{w}^T \Sigma \mathbf{w}$

### Step 2 — Risk Metrics (VaR & ES)

Four independent methods, each with different assumptions:

**1. Historical Simulation** — Sort actual past portfolio losses; pick the 1st percentile. No distributional assumptions.

**2. Parametric VaR** — Assume returns follow a normal (Gaussian) distribution:

$$\text{VaR}_\alpha = z_\alpha \cdot \sigma_p - \mu_p$$

Fast to compute, but underestimates extreme events because real markets have fat tails.

**3. Gaussian Monte Carlo** (flagship) — The most flexible approach:
1. **Cholesky decomposition:** Factor the covariance matrix as $\Sigma = \mathbf{L}\mathbf{L}^T$
2. **Generate random draws:** $\mathbf{Z} \sim N(\mathbf{0}, \mathbf{I})$ — 100,000 independent standard normal vectors
3. **Correlate them:** $\mathbf{R} = \boldsymbol{\mu} + \mathbf{L}\mathbf{Z}$ — now the simulated returns have the same correlation structure as the real data
4. **Aggregate to portfolio:** $R_p = \mathbf{w}^T \mathbf{R}$ — weighted sum gives 100,000 possible portfolio outcomes
5. **Extract risk metrics:** VaR = 1st percentile loss; ES = average of losses beyond VaR

**4. Student-t Monte Carlo** — Addresses the Gaussian MC's main weakness: fat tails.
1. **Fit ν via MLE** on historical portfolio returns — lower ν means heavier tails
2. **Draw innovations** $Z \sim t(\nu)$ instead of $Z \sim N(0,1)$
3. **Rescale** so that $\text{Var}(Z) = 1$: $\tilde{Z} = Z \cdot \sqrt{(\nu-2)/\nu}$
4. **Correlate** via the same Cholesky factor: $\mathbf{R} = \boldsymbol{\mu} + \mathbf{L}\tilde{\mathbf{Z}}$

The Student-t captures the empirical observation that extreme market moves (e.g., 3–5σ events) happen far more often than a Gaussian predicts. When $\nu \to \infty$ the Student-t converges to Normal, so this nests the Gaussian MC as a special case.

### Step 3 — Backtesting Framework

How do we know if the VaR model actually works?

- Use a **rolling 250-day window** (≈ 1 year of trading)
- At each day, estimate VaR using only past data, then check if the next day's actual loss exceeded the prediction
- Count **breaches** — days where actual loss > predicted VaR
- At 99% confidence, expect ~1% breach rate

**Our results (parametric backtest):** 30 breaches out of 1,004 test days (3.0% breach rate vs 1% expected). A parallel **Monte Carlo backtest** (10,000 simulations per step) validates the MC engine directly and produces consistent results. The **Kupiec POF test** formally rejects the Gaussian VaR model (p < 0.001), confirming that Gaussian assumptions underestimate tail risk — precisely the motivation for the Student-t extension.

### Step 4 — Stress Scenarios

What happens when markets go haywire?

| Scenario | What Changes | 99% VaR Impact | 99% ES Impact |
|----------|-------------|----------------|---------------|
| **Volatility Doubling** | $\Sigma_{\text{shock}} = 2\Sigma$ | +42% | +42% |
| **Correlation Collapse** | All $\rho_{ij} \to 0.9$ | +40% | +41% |

The correlation stress scenario is severe — when all assets start moving together, diversification evaporates and tail risk amplifies by ~40%.

---

## Visualizations

The engine generates 6 publication-quality charts:

| Figure | What It Shows |
|--------|---------------|
| `mc_pnl_distribution.png` | Full histogram of 100K simulated daily returns with VaR/ES lines |
| `left_tail_zoom.png` | Close-up of the loss tail — where risk lives |
| `rolling_var_backtest.png` | 99% VaR forecast overlaid on actual daily losses, breaches highlighted in red |
| `correlation_heatmap.png` | Cross-asset correlation structure |
| `stress_comparison.png` | Side-by-side bar chart: baseline vs vol-shock vs correlation-stress |
| `gaussian_vs_student_t.png` | **New:** Gaussian vs Student-t P&L overlay with tail zoom |

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
│   ├── monte_carlo.py              # Cholesky Gaussian MC engine (100K sims)
│   ├── student_t_mc.py             # Student-t MC engine (heavy-tailed extension)
│   ├── risk_metrics.py             # Historical & Parametric VaR/ES
│   ├── backtesting.py              # Rolling-window backtest + Kupiec test
│   ├── stress_testing.py           # Vol shock + correlation stress
│   └── visualization.py            # All chart generation (6 figures)
│
├── notebooks/
│   └── report.ipynb                # Interactive analysis report (executable docs)
│
├── results/
│   ├── figures/                    # Generated charts (PNG)
│   └── tables/                     # CSV & JSON outputs
│
├── main.py                         # Run everything in one command
├── requirements.txt                # Pinned dependencies
├── LICENSE                         # MIT License
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

# Create virtual environment
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
.venv\Scripts\activate      # Windows

# Install dependencies
pip install -r requirements.txt

# Run the full pipeline
python main.py
```

**Expected output:**

```
99% Monte Carlo VaR (Gaussian):  ~1.73%
99% Monte Carlo VaR (Student-t): ~1.8–2.1% (depends on fitted ν)
99% Expected Shortfall:           ~1.96–2.5%
Backtest breaches:                26 / 1,004 days
```

### Interactive Report

Open the Jupyter notebook for a step-by-step walkthrough with explanations:

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

### Gaussian Simulation
$$\mathbf{R} = \boldsymbol{\mu} + \mathbf{L}\mathbf{Z}, \quad \mathbf{Z} \sim N(\mathbf{0}, \mathbf{I})$$

### Student-t Simulation
$$\mathbf{Z} \sim t(\nu), \quad \tilde{\mathbf{Z}} = \mathbf{Z} \cdot \sqrt{\frac{\nu-2}{\nu}}, \quad \mathbf{R} = \boldsymbol{\mu} + \mathbf{L}\tilde{\mathbf{Z}}$$

### Parametric VaR
$$\text{VaR}_\alpha = z_\alpha \cdot \sigma_p - \mu_p$$

### Expected Shortfall
$$\text{ES}_\alpha = E\left[L \mid L > \text{VaR}_\alpha\right]$$

### Kupiec Test Statistic
$$LR = -2\ln\left[\frac{(1-p)^{T-x} \cdot p^x}{(1-\hat{p})^{T-x} \cdot \hat{p}^x}\right] \sim \chi^2(1)$$

### Student-t PDF
$$f(x; \nu) = \frac{\Gamma\left(\frac{\nu+1}{2}\right)}{\sqrt{\nu\pi}\;\Gamma\left(\frac{\nu}{2}\right)}\left(1 + \frac{x^2}{\nu}\right)^{-\frac{\nu+1}{2}}$$

</details>

---

## Known Limitations

| Limitation | Why It Matters |
|-----------|----------------|
| **Static covariance** | Correlations change over time, especially spiking during crises |
| **No liquidity modeling** | Assumes you can sell everything at current market prices instantly |
| **No regime switching** | Doesn't distinguish between calm and volatile market regimes |
| **Single-day horizon** | Only models 1-day risk; multi-day requires different scaling |
| **MLE ν estimation** | Student-t dof is point-estimated; uncertainty in ν is not propagated |

These are acknowledged limitations, not flaws — they represent the standard trade-offs in desk-level risk models. The backtesting results confirm that Gaussian assumptions lead to VaR underestimation, which the Student-t extension helps address.

---

## Future Improvements

- **GARCH volatility** — Time-varying conditional volatility for dynamic risk
- **Regime-switching model** — Bull/bear market detection via Hidden Markov Models
- **Incremental VaR** — Decompose risk contribution by individual asset
- **Cornish-Fisher adjustment** — Correct parametric VaR for skewness/kurtosis
- **Multi-day VaR** — Extend horizon with appropriate scaling assumptions
- **Filtered Historical Simulation** — Combine GARCH dynamics with non-parametric tails

---

## What This Demonstrates

| Skill | Evidence |
|-------|----------|
| Linear algebra | Cholesky decomposition, portfolio variance via matrix multiplication |
| Probability & statistics | VaR quantiles, Expected Shortfall, Student-t fitting (MLE) |
| Distributional modeling | Gaussian vs heavy-tailed distributions, tail risk comparison |
| Model validation | Backtesting framework, Kupiec test, breach analysis |
| Stress testing | Volatility shocks, correlation stress, sensitivity interpretation |
| Regulatory awareness | Basel III VaR → ES transition, FRTB context |
| Software engineering | Modular architecture, type hints, documented functions |

---

## Dependencies

```
numpy==2.4.2
pandas==2.3.0
scipy==1.17.1
matplotlib==3.10.3
seaborn==0.13.2
yfinance==0.2.55
jupyter==1.1.1
nbformat==5.10.4
```

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

*Built by [Rivan Avinash Shetty](https://github.com/redwolf261) as a quantitative risk analysis project demonstrating trading desk market risk methodologies aligned with Basel III / FRTB standards.*
