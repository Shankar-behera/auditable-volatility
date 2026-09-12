# Validation Results

## Summary

CETE began as a hand-designed spectral-entropy volatility engine. It
does not work. After two real bugs were found and fixed, and after the
remaining dynamical component was removed, the engine reduces to a
percentile screen on rolling-window variance. That screen has one
measured property — high-precision, moderate-recall flagging of future
extreme-move blocks — and no demonstrated forecasting skill.

This document reports the negative result in full. The honest version
of the project is: an engine that turned out to be `np.var`, a
screening property that survived contact with a leakage-free
evaluation, and a bug-fix log that is itself the contribution.

## Data

Real daily S&P 500 closes, 2018-01-01 to 2024-12-31 (1,759 log
returns), spanning the Feb 2018 "Volmageddon" spike, the Q4 2018
rate-driven selloff, the Feb–Mar 2020 COVID crash, the 2022 bear
market, and into 2024. Fetched via `yfinance` (`^GSPC`); a FRED-sourced
equivalent was used during initial development.

The 7-year sample is dominated by one shock event (March 2020). This
matters for every precision/recall figure below: the effective sample
for "distinct volatility events" is closer to 3 or 4 than to the 1,759
observations the raw count suggests. Caveats are stated per-claim.

## What the engine actually computes

After the two bug fixes documented in `src/cete.py`, CETE's reported
energy is:

    energy = 0.5 * mean(|FFT_entropy_weighted(x_centered)|²)

where the entropy weighting is `(1 + H(p))` per frequency bin, `H(p)`
being the per-bin entropy contribution of the power distribution. By
Parseval's theorem, `mean(|FFT(x)|²)` is proportional to `var(x)`. The
entropy weighting is a near-constant rescale of the spectrum.

Measured directly on this data: CETE's score and plain
`np.var(window)` correlate at **Pearson 0.9996, Spearman 0.9996** over
1,695 windows. The engine is a variance calculation with an ornate
wrapper around it.

The relaxation dynamics — momentum, per-dimension metric, four-operator
least-squares blend, 20 steps — were removed entirely (see
`src/cete.py`). They contributed a bounded ±10% modulation to the
reported score and, when tested causally, did not improve any
downstream metric.

## The bug-fix log

Two bugs, both found during validation against real data, both real.

**Bug 1 — divergent gradient term.** `compute_gradient` contained
`-0.1/(|state|+1e-6)`, a log-barrier that diverges as the encoded
state's magnitude shrinks — which happens exactly when the input
window is calm. This injected large synthetic momentum into calm
windows, inflating their reported energy. Fixing this alone raised the
real-data correlation with realized volatility from **-0.24 to +0.15** —
right direction, far too weak, meaning a second bug dominated.

**Bug 2 — scale-erasing operator.** The dynamics included
`op3 = state * tanh(|state|) / |state|`, a saturating normalizer.
Traced directly: over 20 relaxation steps, a calm window's mean state
magnitude *grew* (0.036 → 0.052) while a volatile window's *shrank*
(0.35 → 0.21). The dynamics were compressing away the amplitude
difference that made the signal useful. The confirming test: encoded
spectral power measured immediately after `encode()`, before any
dynamics ran, already correlated **0.956** with real realized
volatility. The dynamics were destroying a signal that was already
good at the input stage.

**The fix that wasn't.** The first response to Bug 2 was to anchor the
*reported score* to pre-dynamics spectral power, so the dynamics could
only modulate it by ±10%. This made the score correct. It did not fix
the dynamics — they were still compressing state magnitudes, still in
the code. A later test that measured scale preservation in the
dynamics directly found a 10× input std ratio compressed to a **3.99×**
output ratio, i.e. Bug 2 was never fixed, only bypassed. The dynamics
were then removed entirely. This is the most useful thing in the whole
bug-fix log: a workaround that made the reported number right while
leaving the underlying defect in place, caught months later by a test
that measured the thing the workaround had bypassed.

## Retrospective correlation (the result that doesn't mean what it looks like)

| Estimator | Correlation with realized volatility (same window) |
|---|---|
| CETE energy (post-fix) | **0.956** |
| GARCH(1,1) conditional volatility | 0.729 |
| Plain rolling variance | ~1.000 (by construction) |

The 0.956 is expected, not impressive. Both CETE's score and the
realized-volatility ground truth are computed from the same centered
window, and by Parseval they are proportional. A high correlation here
re-confirms "windows with big moves have high variance." It does not
demonstrate forecasting.

GARCH's 0.729 is not GARCH performing worse. GARCH's conditional
volatility at time *t* is a forecast built only from returns before
*t*, while both CETE's score and the realized-vol target use data
through *t*. It's the difference between measuring a window and
predicting one.

## Causal forecasting evaluation

The test that actually matters for deployability: at each origin *i*,
use only returns through *i* to forecast the realized volatility of
returns *i* through *i+10* (unseen at forecast time). GARCH refits on
expanding windows using only pre-origin data, horizon-matched to the
target block.

| Forecaster | Pearson vs future realized vol |
|---|---|
| CETE (trailing window) | 0.374 |
| Naive persistence (last window's vol) | 0.429 |
| GARCH(1,1), walk-forward | **0.660** |

**CETE loses to the simplest possible baseline.** Persistence — "assume
nothing changes" — beats it. Volatility clustering is well known and
easy to exploit; a forecaster that can't even match naive persistence
isn't adding value.

One attempted rescue: letting CETE's momentum/metric state persist
across windows instead of resetting each call, on the theory that its
dynamical variables could accumulate recursive memory like GARCH's.
Result: correlation unchanged (0.374 → 0.374). CETE's momentum doesn't
function as a memory mechanism.

## Transition detection

The last open question after the dynamics were removed: does the
entropy weighting help detect regime transitions, even if it doesn't
help level tracking or forecasting? A score that tracks variance at
0.999 correlation could in principle fire earlier on changepoints if
the weighting redistributes spectral power near transitions.

Test: identify volatility changepoints (20-day rolling std ratio ≥ 2,
minimum 30 observations apart), then evaluate both detectors at the
90th-percentile flag threshold over a 5-window lookahead.

| Detector | Precision | Recall | F1 | Median lag | Fires |
|---|---|---|---|---|---|
| CETE (entropy-weighted) | 0.012 | 0.087 | 0.022 | 1.0 | 170 |
| Plain variance | 0.012 | 0.087 | 0.022 | 1.0 | 170 |

Head-to-head across 23 detected transitions: 0 CETE-earlier, 0
variance-earlier, 2 tied, 21 caught by neither detector.

**Two findings.** First, the entropy weighting adds nothing to
transition detection either. Second, and more useful: the flag does not
detect volatility changepoints at all. It fired 170 times at the 90th
percentile over 1,695 windows and caught 2 of 23 changepoints. The
"100% precision" screening claim below is specific to a *different*
event definition — future blocks containing a top-5% absolute
single-day return — and does not generalize to volatility
transitions. Users should be clear about which event they're targeting.

## The one surviving property: a high-precision screen

Tested causally: at each origin *i*, compute the score from only the
trailing window returns[:i], and ask whether it flags the *future*
block returns[i:i+10] as containing a crisis day (top 5% of |return|).

| Flag percentile | Precision | Recall | Alert rate |
|---|---|---|---|
| 75 | ~0.54 | higher | higher |
| 90 | intermediate | intermediate | intermediate |
| **95** | **1.000** | **0.422** | **~5%** |

At the 95th-percentile cutoff: precision 1.000, recall 0.422 over 169
forecast origins with 88 crisis days in the sample.

**Caveats that scope this claim, stated explicitly:**

1. The crisis-day threshold is computed on the full sample. The flag is
   causal; the event definition is not. This is defensible as a fixed
   ex-post event definition but should be read as "the screen would
   have flagged these windows if we knew then what a crisis day would
   look like in this sample," not as a fully deployable evaluation.

2. 88 crisis days in a 7-year sample is dominated by a single episode
   (March 2020). Precision 1.000 should be read as "one confirmed
   true-positive event with no observed false positives," not as a
   stable estimate.

3. The recall figure (0.422) is for the 95th-percentile flag against
   the *top-5% absolute return* event definition. It is 0.087 against
   the volatility-changepoint definition. These are different tasks.

## What the honest conclusion is

Three claims are now supported by evidence:

1. **CETE-as-spectral-engine adds nothing over `np.var(window)`** for
   level tracking (correlation >0.999), for causal forecasting (loses to
   persistence), or for transition detection (identical to variance).

2. **CETE-as-causal-forecaster is worse than persistence** and much
   worse than GARCH.

3. **CETE-as-crisis-screen has one measured property**: high precision
   at a strict percentile cutoff, against one specific event
   definition, on a sample dominated by one shock. This survives the
   leakage-free evaluation; nothing else does.

The right use of this codebase is therefore: the data layer, the causal
evaluation harness, the walk-forward GARCH baseline, and the
percentile screen. The engine is a research artifact whose value is the
documented negative result.

## Limitations and future work

- **The bug-fix log is the contribution.** A working scientist finding
  two real bugs in their own model and following the evidence to "the
  model is `np.var`" is a genuinely useful thing to publish. The
  detailed trace — particularly the Bug 2 workaround that made the
  reported number right while leaving the defect in place, caught
  months later by a test that measured the thing the workaround had
  bypassed — is the single most instructive part of this project.

- **The screening property needs more events to be trusted.** One
  confirmed shock in one sample is not a precision estimate. A longer
  history (2000–2024, if data is available), or a cross-sectional
  test (does the flag generalize across assets), would give a real
  number.

- **The event definition should be made causal.** An expanding-window
  crisis-day threshold would give an honest, deployable precision/recall
  figure without the full-sample lookahead in the event definition.

- **The window and step parameters were inherited, not tuned.** A
  parameter sweep over window size vs detection lag is natural
  follow-up work.

## Files

- `src/data.py` — versioned, validated, cached data layer with Stooq
  fallback. Never returns synthetic data silently; raises on failure.
- `src/cete.py` — the score. Reduced to entropy-weighted spectral power
  of a mean-centered window; no dynamics.
- `src/regime.py` — scoring, retrospective and causal evaluation,
  operating curves.
- `src/garch_filter.py` — recursive GARCH state update between refits.
- `src/metrics.py` — shared correlation and error metrics.
- `scripts/live_monitor.py` — one-shot entrypoint: fetch, score, GARCH
  forecast, decide, exit.
- `scripts/baseline_comparison.py` — retrospective estimator comparison.
- `scripts/causal_evaluation.py` — leakage-free forecasting and
  screening evaluation.
- `scripts/transition_evaluation.py` — changepoint detection
  comparison.
- `tests/test_data.py`, `tests/test_cete.py` — regression tests,
  including a canary (`test_cete_base_power_tracks_plain_variance`) that
  fails if the null result stops holding.