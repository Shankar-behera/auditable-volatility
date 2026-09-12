# Methodology

How the volatility forecasts in this repository are produced, and how their accuracy is measured.

## What is being forecast

For each ticker, once per trading day after market close, this system produces:

* a **volatility forecast** for the next `H` trading days, where `H` defaults to 10
* a **screen flag**, indicating whether the most recent trailing window's variance is in the top 5% of its own historical distribution

Both are recorded before the outcome is known. The forecast is scored against realized volatility once `H` trading days have elapsed.

## The forecast

The forecast is a **GARCH(1,1)** conditional volatility estimate:

```text
σ²_t = ω + α · ε²_{t-1} + β · σ²_{t-1}
```

Parameters are re-estimated periodically, not every day. Between re-estimations, the conditional variance is filtered forward through each new return using the cached parameters. This is the standard "update without re-estimating" approach used in deployed GARCH systems.

The reported number is the **root-mean-square of the forecasted variance path over the next `H` days**, on the original return scale:

```text
forecast_vol = sqrt(mean(σ²_hat(1..H)))
```

For `H = 1`, this is exactly the one-step conditional volatility. For `H > 1`, it uses GARCH(1,1)'s closed-form mean-reverting forecast path.

Why GARCH and not a novel model: this project originally developed a spectral-entropy volatility engine, and the evaluation showed it reduces to a plain variance calculation and underperforms GARCH by a wide margin on the causal test. That negative result is documented in [RESULTS.md](RESULTS.md). The production system uses GARCH because it works.

## The target

For a prediction made on date `D` with horizon `H`:

```text
target_vol = std(returns[D+1 : D+1+H])
```

where `D+1` is the first trading day **after** the origin, and returns are log returns.

The `+1` is deliberate. The origin day is the last day whose return was visible to the forecast; including it in the target would score the model against a return it already saw. This matches the convention used in `scripts/causal_evaluation.py`, so the live track record is comparable to the historical evaluation.

Trading days, not calendar days. The target block is the next `H` observed returns, whatever dates they fall on. The `H` in the prediction log is the same `H` used in the target.

## The screen

The screen flag answers a different question: is the current trailing window's variance unusual relative to its own history?

```text
energy = 0.5 × mean(|FFT_entropy_weighted(returns[-W:])|²)
```

where `W` defaults to 64 trading days.

By Parseval's theorem, this is proportional to `np.var(returns[-W:])`. The entropy weighting is a near-constant rescale and does not change the ranking of windows.

The flag fires when `energy` is above the 95th percentile of the historical distribution of trailing-window energies.

Tested causally — flag from a trailing window, target is a **FUTURE** block the flag never sees — the flag has:

* **precision 1.000** at the 95th-percentile cutoff
* **recall 0.214** against future blocks containing a top-5% absolute return

"Precision 1.000" means that, on this 7-year sample, every window on which the flag fired was followed by a block containing an extreme-move day. The sample is dominated by one event (March 2020); this should be read as "no observed false positives on one sample," not as a stable estimate.

See [RESULTS.md](RESULTS.md) for the full operating curve and caveats.

## How accuracy is measured

Three things are logged per resolved prediction:

* **mean absolute error**: `mean(|forecast - target|)`
* **RMSE**: `sqrt(mean((forecast - target)^2))`
* **bias**: `mean(forecast - target)`, signed
* **Pearson correlation**: correlation between forecasts and targets across the resolved sample

All four are reported on every track-record page, over both the full history and a rolling 30-prediction window.

**A single correlation number is not enough.**

With few predictions, correlation is dominated by whether one or two large events happened to fall inside the sample. The track-record pages report `n_resolved` alongside every statistic so a reader can judge the sample size for themselves.

Correlations from fewer than approximately 30 resolved predictions should be read as **insufficient data**, regardless of the numerical value.

### What is not measured

Statistical significance is not computed. No confidence intervals are shown. No model comparison is presented on the track-record pages.

These are available in the research artifacts (`scripts/causal_evaluation.py`, `docs/RESULTS.md`) for anyone who wants them, but the public record is deliberately just the numbers and the sample size.

## What the log is

Every prediction is written to:

```text
data/predictions/<TICKER>.jsonl
```

at the moment it is made, before the outcome is known. Every resolution is appended to the same file later.

The log is:

* **append-only**: predictions are never edited or deleted
* **plain text**: one JSON object per line, readable and diffable
* **committed to git**: the git history is the public record

The property that matters: **a prediction cannot be revised after the fact.**

```bash
git log -p data/predictions/<TICKER>.jsonl
```

shows every prediction ever made, in the order it was made, unmodified. Anyone can verify the track record independently.

## What this system does NOT do

* **It does not trade.** No orders, no positions, no execution.
* **It does not size risk.** The forecast is a volatility estimate, not a position size or a limit.
* **It does not handle multiple horizons.** One `H` is used per prediction, configured in the workflow.
* **It does not adjust for corporate actions or index composition changes.** It uses the price series as returned by the data source.
* **It is not a recommendation.** It is a measurement, published transparently so that anyone reading it can judge for themselves whether the measurement has been accurate.

## Reproducing the record

Everything in the track record is reproducible from the repository.

View the full validation write-up, including the negative result:

```bash
cat docs/RESULTS.md
```

View the prediction log:

```bash
cat data/predictions/GSPC.jsonl
```

Re-run the accuracy summary:

```bash
python -c "from src.predictions import accuracy_summary; print(accuracy_summary('GSPC'))"
```

Re-generate the reports:

```bash
python scripts/publish_report.py cde
```
