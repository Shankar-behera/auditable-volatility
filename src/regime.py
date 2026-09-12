"""
regime.py
=========

Slides CETE's score over a return series to build a volatility timeline,
and classifies each window relative to the distribution of scores seen
in that run.

------------------------------------------------------------------
WHAT THE SCORE ACTUALLY IS
------------------------------------------------------------------
CETE's reported "energy" is 0.5 * mean(|encoded|**2) where `encoded` is
the entropy-weighted FFT of a mean-centered window. By Parseval's
theorem this is proportional to np.var(window). The relaxation
dynamics that once modulated this value were removed -- see
src/cete.py's docstring and docs/RESULTS.md.

So the honest description of what this module computes is: a percentile
screen on rolling-window variance. The "CETE energy" naming is retained
for continuity with the rest of the repo.

------------------------------------------------------------------
WHY PERCENTILE-RELATIVE AND NOT A FIXED CUTOFF
------------------------------------------------------------------
The score's absolute scale depends on the asset and timeframe (an FX
pair's daily return variance is not a small-cap stock's). A hardcoded
threshold tuned on one series silently breaks on another.
Percentile-relative classification is scale-invariant by construction --
but note the limitation that follows from it: every run reports ~25%
LOW, ~50% MID, ~25% HIGH regardless of the actual market regime,
because those are the percentiles. It ranks windows *within* a run; it
does not say "this period is objectively elevated".

------------------------------------------------------------------
REVISION NOTE (external review response)
------------------------------------------------------------------
An external review identified two methodology bugs, both fixed here:

1. The original GARCH baseline compared its ONE-STEP conditional
   volatility against a `step`-day future target -- an apples-to-oranges
   horizon mismatch. `garch_causal_forecast_eval()` below uses `arch`'s
   proper `.forecast(horizon=step)`. NOTE: this function fits GARCH
   ONCE on the full series and is therefore NOT truly walk-forward; the
   genuinely leakage-free version lives in
   scripts/causal_evaluation.py's `garch_walk_forward_forecast()`. This
   one is kept for continuity and its docstring says so.

2. The original crisis-flag evaluation computed CETE's score FROM THE
   SAME WINDOW used to define the "contains a crisis" ground truth --
   near-tautological given energy's identity with variance.
   `causal_crisis_flag_evaluation()` below is the corrected, genuinely
   predictive version: trailing window flags a future block that the
   flag never sees.
------------------------------------------------------------------
"""

import numpy as np

from .cete import CETE


# ----------------------------------------------------------------------
# Core scoring
# ----------------------------------------------------------------------

def fast_score(history: np.ndarray, window_size: int = 64) -> float:
    """CETE's score: entropy-weighted FFT spectral power of a
    mean-centered window. By Parseval, proportional to the window's
    variance.

    Reproduces exactly what CETE.run() returns as final_energy_score.

    The transformation order matches CETE.encode() precisely:

        slice trailing window   (fast_score's job -- encode() receives
                                 an already-sliced array from run())
        -> pad/truncate to window_size
        -> mean-center
        -> FFT
        -> entropy-weight
        -> 0.5 * mean(|encoded|**2)

    There is a test (test_fast_score_matches_cete_run) that enforces
    this invariant. If you change one of these two functions, change
    the other to match, or the evaluation suite will silently measure
    a different quantity than live_monitor.py reports.
    """
    history = np.asarray(history, dtype=float)

    # Take the trailing window. fast_score receives a full history and
    # selects the last window_size observations; encode() receives an
    # already-sliced array. After this step the two functions perform
    # the identical sequence.
    if len(history) > window_size:
        history = history[-window_size:]

    # Pad/truncate to window_size (matches encode()).
    if len(history) > window_size:
        data = history[:window_size]
    else:
        data = np.pad(history, (0, window_size - len(history)))

    # Mean-center AFTER padding (matches encode()).
    data = data - np.mean(data)

    spectrum = np.fft.fft(data)
    power = np.abs(spectrum) ** 2
    total_power = np.sum(power) + 1e-12

    prob = power / total_power
    entropy_weights = -prob * np.log(prob + 1e-12)
    encoded = spectrum * (1.0 + entropy_weights)

    return float(0.5 * np.mean(np.abs(encoded) ** 2))

def forecast_from_history(history: np.ndarray,
                          window_size: int = 64) -> float:
    """Compute CETE's score from ONLY the array passed in.

    Structural leakage guard: callers must slice `returns[:origin]` (or
    a trailing window of it) themselves before calling this, so it is
    impossible for this function to see data past the forecast origin --
    there is no index into a larger array here for a future bug to
    misuse.

    Delegates to fast_score. There is no dynamics path anymore.
    """
    return fast_score(history, window_size=window_size)


# ----------------------------------------------------------------------
# Retrospective (centered-window) analysis
# ----------------------------------------------------------------------

def detect_regimes(returns, dates, window_size: int = 64, step: int = 10,
                   low_pct: float = 25, high_pct: float = 75):
    """Run CETE's score over `returns` in overlapping (centered) windows.

    Returns a dict with timestamps, energy_scores, metric_means,
    verdicts, and the percentile cutoffs used.

    NOTE: this uses centered windows (returns[i:i+window_size], labeled
    at the window's center) for RETROSPECTIVE labeling of historical
    data -- it is not a live/causal forecast. For live, causal
    computation see scripts/live_monitor.py, which uses a trailing
    window ending "today".

    `metric_means` is retained for API stability but is filled with NaN
    -- the CETE metric array no longer exists (it was part of the
    removed dynamics). Nothing reads it.
    """
    timestamps, energy_scores = [], []
    for i in range(0, len(returns) - window_size, step):
        window = returns[i : i + window_size]
        timestamps.append(dates[i + window_size // 2])
        energy_scores.append(fast_score(window, window_size=window_size))

    energy_scores = np.array(energy_scores)
    lo_cut = float(np.percentile(energy_scores, low_pct))
    hi_cut = float(np.percentile(energy_scores, high_pct))
    verdicts = [
        "VERIFIED_CONVERGENCE" if e <= lo_cut else
        "HIGH_UNCERTAINTY_FLAGGED" if e >= hi_cut else
        "PARTIAL_CONVERGENCE"
        for e in energy_scores
    ]

    return {
        "timestamps": np.array(timestamps),
        "energy_scores": energy_scores,
        "metric_means": np.full(len(energy_scores), np.nan),
        "verdicts": verdicts,
        "low_cutoff": lo_cut,
        "high_cutoff": hi_cut,
    }


def realized_volatility(returns, window_size: int = 64, step: int = 10):
    """Ground-truth rolling realized volatility (std dev) on centered
    windows, for retrospective validation of the energy score. Not a
    causal/forecasting quantity -- see causal_forecast_eval for that."""
    return np.array([
        np.std(returns[i : i + window_size])
        for i in range(0, len(returns) - window_size, step)
    ])


# ----------------------------------------------------------------------
# Causal (trailing-window) forecasting evaluation
# ----------------------------------------------------------------------

def causal_forecast_eval(returns, window_size: int = 64, step: int = 10):
    """Genuine causal (deployable) forecasting evaluation: for each origin
    i, use ONLY returns[:i] (the past) to forecast the realized
    volatility of returns[i:i+step] (the future, unseen at forecast
    time).

    Returns a dict with the CETE forecast, a naive persistence baseline
    (next vol = last window's vol), and the ground-truth future realized
    vol, all aligned to the same forecast origins.
    """
    idxs = list(range(window_size, len(returns) - step, step))
    cete_fc, persistence_fc, target = [], [], []
    for i in idxs:
        past = returns[:i]
        future = returns[i : i + step]

        cete_fc.append(fast_score(past, window_size=window_size))
        persistence_fc.append(float(np.std(past[-window_size:])))
        target.append(float(np.std(future)))

    return {
        "origins": np.array(idxs),
        "cete_forecast": np.array(cete_fc),
        "persistence_forecast": np.array(persistence_fc),
        "target_future_vol": np.array(target),
    }


def garch_causal_forecast_eval(returns, window_size: int = 64, step: int = 10):
    """GARCH(1,1) counterpart to causal_forecast_eval, using a proper
    `step`-day-ahead forecast.

    WARNING -- NOT TRULY CAUSAL: fits GARCH once on the full series. The
    fitted PARAMETERS therefore see future data, even though each
    forecast's variance path is conditioned on the training sample. This
    is retained for continuity only. The genuinely leakage-free
    implementation is `garch_walk_forward_forecast` in
    scripts/causal_evaluation.py, which refits on expanding windows and
    is what docs/RESULTS.md reports against.

    IMPLEMENTATION NOTE: `arch`'s `.forecast(start=k, horizon=h)`
    computes forecasts for EVERY origin from k to the end of the sample
    in one call, returned as a DataFrame with one row per origin (row 0
    = origin k). Calling `.forecast()` separately inside a per-origin
    loop and reading `.values[-1]` is a bug -- that always reads the
    LAST row of the whole series regardless of which origin you asked
    for, silently producing a near-constant "forecast". Fixed by calling
    `.forecast()` once and indexing the correct row per origin.
    """
    from arch import arch_model

    am = arch_model(returns * 100, vol="Garch", p=1, q=1, dist="normal")
    fit_result = am.fit(disp="off")

    fc_all = fit_result.forecast(start=window_size - 1, horizon=step,
                                 reindex=False)
    variance_matrix = fc_all.variance.values

    idxs = list(range(window_size, len(returns) - step, step))
    garch_fc = []
    for i in idxs:
        row = i - window_size
        garch_fc.append(float(np.sqrt(np.mean(variance_matrix[row])) / 100))
    return np.array(garch_fc)


# ----------------------------------------------------------------------
# Crisis-flag evaluation
# ----------------------------------------------------------------------

def crisis_flag_evaluation(returns, verdicts, window_starts,
                           window_size: int = 64,
                           crisis_percentile: float = 95):
    """RETROSPECTIVE/CONTEMPORANEOUS classifier evaluation (NOT
    predictive): evaluates CETE's HIGH_UNCERTAINTY_FLAGGED verdict
    against whether the SAME window contains a crisis day.

    Important caveat: because the flag is computed from the same window
    used to define ground truth, and CETE's energy is near-identical to
    that window's variance, a high score here mostly re-confirms
    "windows containing big moves have high variance" rather than
    demonstrating any detection capability. Kept for continuity; for an
    honest test of predictive value, use
    causal_crisis_flag_evaluation().
    """
    threshold = np.percentile(np.abs(returns), crisis_percentile)
    crisis_days = np.abs(returns) > threshold

    tp = fp = fn = tn = 0
    for idx, i in enumerate(window_starts):
        contains_crisis = crisis_days[i : i + window_size].any()
        flagged = verdicts[idx] == "HIGH_UNCERTAINTY_FLAGGED"
        if flagged and contains_crisis:
            tp += 1
        elif flagged and not contains_crisis:
            fp += 1
        elif not flagged and contains_crisis:
            fn += 1
        else:
            tn += 1

    return _prf(tp, fp, fn, tn, threshold, crisis_days)


def causal_crisis_flag_evaluation(returns, window_size: int = 64,
                                  step: int = 10,
                                  crisis_percentile: float = 95,
                                  flag_percentile: float = 75):
    """PREDICTIVE (causal) crisis-flag evaluation: at each origin i,
    compute CETE's score from ONLY the trailing window returns[:i], and
    ask whether that flags the FUTURE block returns[i:i+step] as
    containing a crisis day. The future slice is never touched by
    anything that produces the flag.

    Caveat on the EVENT DEFINITION: the crisis-day threshold is computed
    on the full sample (np.percentile over all returns). The FLAG is
    causal; the event definition is not. A stricter version would use an
    expanding-window threshold.
    """
    threshold = np.percentile(np.abs(returns), crisis_percentile)
    crisis_days = np.abs(returns) > threshold

    idxs = list(range(window_size, len(returns) - step, step))
    energies = np.array([fast_score(returns[:i], window_size=window_size)
                         for i in idxs])
    flag_cut = float(np.percentile(energies, flag_percentile))

    tp = fp = fn = tn = 0
    for e, i in zip(energies, idxs):
        will_have_crisis = crisis_days[i : i + step].any()
        flagged = e >= flag_cut
        if flagged and will_have_crisis:
            tp += 1
        elif flagged and not will_have_crisis:
            fp += 1
        elif not flagged and will_have_crisis:
            fn += 1
        else:
            tn += 1

    return _prf(tp, fp, fn, tn, threshold, crisis_days)


def multi_threshold_operating_curve(returns, energies_or_verdicts_fn=None,
                                    window_size: int = 64,
                                    step: int = 10,
                                    crisis_percentile: float = 95,
                                    flag_percentiles=(75, 90, 95, 97.5, 99)):
    """Sweeps the flagging threshold and reports precision/recall/F1/
    false-positive-rate/alert-rate at each, using the CAUSAL evaluation.

    The second positional arg is retained for backwards compatibility
    with callers that passed a scoring function; it is ignored.

    Caveat: same as causal_crisis_flag_evaluation -- the crisis-day
    threshold is computed on the full sample.
    """
    threshold = np.percentile(np.abs(returns), crisis_percentile)
    crisis_days = np.abs(returns) > threshold
    idxs = list(range(window_size, len(returns) - step, step))
    energies = np.array([fast_score(returns[:i], window_size=window_size)
                         for i in idxs])

    rows = []
    for pct in flag_percentiles:
        cut = float(np.percentile(energies, pct))
        tp = fp = fn = tn = 0
        for e, i in zip(energies, idxs):
            will_have_crisis = crisis_days[i : i + step].any()
            flagged = e >= cut
            if flagged and will_have_crisis:
                tp += 1
            elif flagged and not will_have_crisis:
                fp += 1
            elif not flagged and will_have_crisis:
                fn += 1
            else:
                tn += 1
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * precision * recall / (precision + recall)
              if (precision + recall) else 0.0)
        fpr = fp / (fp + tn) if (fp + tn) else 0.0
        alert_rate = (tp + fp) / len(idxs) if idxs else 0.0
        rows.append({
            "flag_percentile": float(pct),
            "cutoff": cut,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "false_positive_rate": fpr,
            "alert_rate": alert_rate,
        })
    return rows


def _prf(tp, fp, fn, tn, threshold, crisis_days):
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) else 0.0)
    return {
        "threshold": float(threshold),
        "n_crisis_days": int(crisis_days.sum()),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }