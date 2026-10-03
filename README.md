# Statistical Arbitrage Lab

A compact research framework for **pairs trading** with Engle–Granger cointegration screening, OLS hedge ratios, rolling spread normalization, regime-aware signals, next-session execution, leg-level costs, walk-forward evaluation, and quantitative performance reporting.

## Live Demo

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://statistical-arbitrage-lab.streamlit.app/)

Explore the interactive dashboard: [Live Demo](https://statistical-arbitrage-lab.streamlit.app/)

> Research software, not investment advice. Backtests are sensitive to data quality, parameter choices, execution assumptions, and market impact. Validate independently before using capital.
>
> ## Pair Quality Score

The lab includes an interpretable **Pair Quality Score** designed to rank
candidate pairs for statistical-arbitrage research.

The score combines multiple research diagnostics:

| Metric | Purpose |
|---|---|
| Engle-Granger p-value | Measures evidence of cointegration |
| ADF p-value | Tests spread stationarity |
| Mean-reversion half-life | Measures how quickly the spread tends to revert |
| Sharpe ratio | Measures risk-adjusted performance |
| Maximum drawdown | Measures downside risk |

The resulting score is normalized to **0–100** and classified as:

- **0–39:** Weak Candidate
- **40–69:** Moderate Candidate
- **70–100:** Strong Candidate

The scoring system is intentionally transparent and rule-based rather than
machine-learning based, making the research decision process easier to
interpret and audit.

> **Important:** The Pair Quality Score is a research-ranking tool, not a
> trading recommendation. Statistical significance and historical
> performance do not guarantee future returns.

## What it does

- Screens price series with the Engle–Granger two-step procedure (`statsmodels.tsa.stattools.coint`) and reports residual ADF diagnostics.
- Estimates the hedge ratio by OLS on log prices using a causal rolling formation window.
- Computes a rolling spread z-score and mean-reversion half-life diagnostic.
- Optionally gates entries using a causal rolling-volatility regime filter.
- Enters and exits at the next bar after a close-based signal. Positions are shifted one period before returns are earned.
- Charges costs independently on each leg whenever that leg's notional changes.
- Reports Sharpe ratio, CAGR, maximum drawdown, volatility, and turnover.
- Supports walk-forward folds that refit model parameters using training data only.

## Install

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install -e .
```

This installs the research package and the Streamlit and Plotly dashboard dependencies. To install dependencies without installing the package, use `python -m pip install -r requirements.txt`.

## Interactive dashboard

Start the demo from the project root:

```bash
streamlit run app.py
```

The app opens in your browser. Choose **Synthetic Demo** to generate a seeded cointegrated pair and control its random seed and observation count, or choose **CSV Upload** and provide historical prices. Set the walk-forward and strategy controls in the sidebar, then select **Run Backtest**. Performance cards and equity/drawdown charts report accepted walk-forward out-of-sample periods; the signal charts show the causal full-history signal trace.

### Demo screenshot

> Screenshot placeholder: capture the dashboard after launch and add the image here.

### CSV format

CSV uploads must contain exactly these required fields (additional columns are ignored):

```csv
Date,Asset_Y,Asset_X
2022-01-03,100.25,54.10
2022-01-04,101.10,54.32
```

Dates must parse as unique dates. Both asset columns must contain finite, numeric, positive prices. Rows are sorted chronologically by date. Use adjusted prices when possible and align both instruments to the same trading calendar before upload.

### Dashboard workflow and methodology

1. Choose synthetic data or upload the required CSV format.
2. Set training and test lengths, rolling Z-score lookback, entry/exit thresholds, ADF significance, maximum half-life, and per-leg transaction cost.
3. Select **Run Backtest**. Each walk-forward fold estimates the Engle–Granger relationship and residual diagnostics on its training window; only folds that meet the cointegration, ADF, and half-life filters contribute out-of-sample returns.
4. Review diagnostics, signal charts, accepted-fold table, and out-of-sample performance cards.

The spread is `log(Asset_Y) - (alpha + beta * log(Asset_X))`, with alpha and beta estimated by trailing OLS. The rolling spread Z-score creates mean-reversion entries and exits. The causal volatility regime gate is evaluated on historical spread volatility. Close signals are shifted one bar before position returns are credited. Transaction costs are charged per leg according to changes in leg notional. Total Return, CAGR, Sharpe, volatility, drawdown, win rate, trading days, and costs are calculated from accepted walk-forward test periods only.

### Pair Quality Score

The optional 0–100 Pair Quality Score is an explainable, fixed-tier rubric rather than a fitted model. It awards up to 25 points for Engle–Granger cointegration p-value, 20 for the residual ADF p-value, 20 for half-life, 20 for aggregate walk-forward out-of-sample Sharpe, and 15 for maximum drawdown. Smaller p-values, shorter half-life, higher Sharpe, and shallower drawdown earn more points. Scores of 0–49 are **Weak Candidates**, 50–74 are **Moderate Candidates**, and 75–100 are **Strong Candidates**. The dashboard shows component points; the diagnostics come from the latest accepted training fold, while Sharpe and drawdown come from the combined accepted test folds.

### Limitations

This is a research/backtesting tool. Synthetic data is only a software demonstration and is not a market forecast. Close-to-close execution approximates fills; the framework does not model borrow availability, financing, market impact, taxes, corporate actions, delisting, or data survivorship. Cointegration and hedge ratios can change over time. Skipped validation folds are excluded from the performance sample, so inspect fold diagnostics and report the accepted-period coverage when interpreting results. Historical or synthetic results do not guarantee future performance.

## Quick start

Run a deterministic synthetic example (no market-data API or credentials required):

```bash
statarb-demo
# or: python -m statarb.demo
```

Use your own aligned, positive adjusted-close prices as a `DataFrame` indexed by date, with one column per asset:

```python
import pandas as pd
from statarb import PairConfig, backtest_pair, engle_granger_test, performance_report

prices = pd.read_csv("prices.csv", index_col=0, parse_dates=True)
diagnostics = engle_granger_test(prices["ASSET_A"], prices["ASSET_B"])
print(diagnostics)

result = backtest_pair(
    prices["ASSET_A"], prices["ASSET_B"],
    config=PairConfig(
        formation_window=252,
        z_window=60,
        entry_z=2.0,
        exit_z=0.5,
        cost_bps=5.0,
        regime_vol_quantile=0.9,
    ),
)
print(performance_report(result.strategy_returns, turnover=result.turnover))
```

`backtest_pair` returns a DataFrame with prices, fitted hedge ratio/intercept, spread, z-score, regime flag, signal, executed position, gross/net returns, and per-leg costs. Positions are in units of the dependent asset; the hedge leg is scaled by the estimated beta.

## Walk-forward evaluation

The helper uses each fold's training interval to estimate the Engle–Granger relationship and test eligibility. It then trades only the following validation interval and concatenates out-of-sample returns:

```python
from statarb import PairConfig, walk_forward

wf = walk_forward(
    prices["ASSET_A"], prices["ASSET_B"],
    train_size=756,
    test_size=126,
    config=PairConfig(formation_window=252, z_window=60),
)
print(wf.fold_diagnostics)
print(wf.out_of_sample_returns.describe())
```

Folds with a non-stationary training residual are skipped by default. Fold diagnostics include training/test boundaries and cointegration statistics. The walk-forward result is a research scaffold: do additional sensitivity analysis, use survivorship-bias-free data, and account for borrow, financing, corporate actions, and realistic fills.

## Signal and execution conventions

The spread is `log(A) - (intercept + beta * log(B))`. A positive z-score opens a short spread (short A, long beta units of B); a negative z-score opens a long spread. Positions are held until the z-score crosses the exit threshold or flips through the opposite entry threshold. Signals are computed at close and shifted one bar for execution. The return credited on a bar uses the position held over that bar's price move; transaction costs apply at the execution bar to the absolute change in each leg's notional.

The optional regime filter compares current rolling spread volatility to a trailing quantile threshold computed with a one-bar lag, avoiding use of the current observation in the threshold. Set `regime_vol_quantile=None` to disable it.

## Project structure

```text
src/statarb/
  backtest.py       signal generation and next-bar execution
  cointegration.py  Engle–Granger, OLS and ADF diagnostics
  demo.py           deterministic synthetic end-to-end example
  metrics.py        equity curve and performance statistics
  walkforward.py    train-only formation and out-of-sample folds
```

## Limitations

OLS hedge ratios are not guaranteed to remain stable; cointegration is not permanent; the volatility gate is a simple illustrative regime filter; and close-to-close execution is an approximation. This package intentionally does not fetch data or model borrow availability, market impact, taxes, or intraday execution.
