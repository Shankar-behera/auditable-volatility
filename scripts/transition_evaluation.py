"""
transition_evaluation.py
========================

The last open question from docs/RESULTS.md's limitations section.

With the relaxation dynamics removed (see src/cete.py), the project's
score is entropy-weighted FFT spectral power of a mean-centered window.
By Parseval's theorem this is proportional to the window's variance.
The canary test (test_cete_base_power_tracks_plain_variance) confirms
the correlation is >0.99.

But level-tracking correlation and transition detection are different
questions. A score that tracks variance at 0.999 correlation could
still, in principle, fire EARLIER on regime transitions if the entropy
weighting redistributes spectral power near changepoints in a way that
plain variance doesn't. That's the specific hypothesis this script
tests.

WHAT IT DOES
------------
1. Identifies regime transitions in the return series using a
   changepoint detector on rolling variance (ratio of post/pre std
   over a window > threshold).

2. Runs two detectors on the same trailing windows:
     - fast_score (entropy-weighted spectral power)
     - plain np.var of the same window

3. Both are evaluated as: "does the detector fire in the N days
   BEFORE a transition?" Both fire at the same PERCENTILE threshold,
   so they fire the same NUMBER of times on average, and we compare
   WHEN they fire, not whether one is simply more sensitive.

4. Reports precision, recall, median lag-to-detection, and false-
   positive rate for each detector. Also reports the fraction of
   transitions where one detector fired strictly earlier than the
   other.

INTERPRETATION
--------------
- If the two detectors fire at the same time within the window
  resolution, entropy weighting adds nothing for transition detection
  either, and the project's finding is complete: the engine is
  variance, full stop.

- If fast_score fires meaningfully earlier (say, on 30%+ of
  transitions, by 2+ windows), that's a real finding and the entropy
  weighting has a use after all.

Usage:
    python scripts/transition_evaluation.py --ticker ^GSPC
    python scripts/transition_evaluation.py --csv path/to/prices.csv
    python scripts/transition_evaluation.py --csv sp500.csv --outdir outputs
"""

import argparse
import os
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data import get_market_returns, load_local_csv
from src.metrics import safe_corr, safe_spearman
from src.regime import fast_score


# ============================================================
# TRANSITION DETECTION
# ============================================================

def find_transitions(returns, window=20, ratio_threshold=2.0,
                     min_spacing=30):
    """Find regime transitions: points where the std of the NEXT
    `window` returns is at least `ratio_threshold` times the std of the
    PREVIOUS `window` returns (or vice versa).

    A transition is labeled at its changepoint index (the first index
    of the "after" window).

    `min_spacing` merges transitions that are within this many
    observations of each other (keeping the first), so a single
    prolonged high-vol regime doesn't register as dozens of separate
    transitions.

    Returns a sorted array of transition indices.
    """
    returns = np.asarray(returns, dtype=float)
    n = len(returns)

    if n < 2 * window:
        return np.array([], dtype=int)

    transitions = []
    for i in range(window, n - window):
        before = np.std(returns[i - window:i])
        after = np.std(returns[i:i + window])

        if before <= 0:
            continue

        ratio = after / before
        if ratio >= ratio_threshold or ratio <= 1.0 / ratio_threshold:
            transitions.append(i)

    # Merge transitions that are close together (keep first).
    merged = []
    for t in transitions:
        if not merged or t - merged[-1] >= min_spacing:
            merged.append(t)

    return np.array(merged, dtype=int)


# ============================================================
# DETECTOR SCORING
# ============================================================

def compute_scores(returns, window_size, step=1):
    """Compute both detectors on identical trailing windows.

    For each origin i (window_size <= i < len(returns)), score uses
    returns[i-window_size:i] only -- trailing, causal.

    Returns dict:
        origins    : array of origin indices
        cete       : entropy-weighted spectral score at each origin
        variance   : plain np.var at each origin
    """
    returns = np.asarray(returns, dtype=float)
    origins = np.arange(window_size, len(returns), step)

    cete = np.empty(len(origins))
    variance = np.empty(len(origins))

    for j, i in enumerate(origins):
        past = returns[:i]
        window = past[-window_size:]
        cete[j] = fast_score(past, window_size=window_size)
        variance[j] = float(np.var(window))

    return {"origins": origins, "cete": cete, "variance": variance}


# ============================================================
# DETECTION EVALUATION
# ============================================================

def evaluate_detector(origins, scores, transitions,
                      flag_percentile=90, lookahead_windows=5):
    """For a given detector, at the given percentile threshold, count:

      - TRUE POSITIVE : the detector fires in the `lookahead_windows`
        windows immediately before a transition. "Fires" means the
        score exceeds the threshold.
      - FALSE POSITIVE: the detector fires but no transition occurs in
        the next `lookahead_windows` windows.
      - FALSE NEGATIVE: a transition occurs but the detector did not
        fire in the preceding `lookahead_windows`.

    Returns a dict with tp, fp, fn, precision, recall, plus the lag
    (in windows) between the last fire before each caught transition
    and the transition itself.
    """
    origins = np.asarray(origins, dtype=int)
    scores = np.asarray(scores, dtype=float)
    transitions = np.asarray(transitions, dtype=int)

    cutoff = float(np.percentile(scores, flag_percentile))
    fires = scores >= cutoff

    # Map each origin to its position in the origins array
    origin_to_idx = {int(o): i for i, o in enumerate(origins)}

    tp = fp = fn = 0
    lags = []

    # Track which origins are "near a transition" (within lookahead)
    # so we don't double-count them as false positives.
    near_transition = np.zeros(len(origins), dtype=bool)
    for t in transitions:
        for k in range(lookahead_windows):
            idx = origin_to_idx.get(int(t) - k - 1)
            # origin index is the observation index; transition at t
            # means we look at origins t-lookahead .. t-1
            if idx is not None and origins[idx] < t:
                near_transition[idx] = True

    for t in transitions:
        # Which origins fired in the lookahead window before t?
        fire_positions = []
        for k in range(lookahead_windows):
            origin_idx = origin_to_idx.get(int(t) - k - 1)
            if origin_idx is None:
                continue
            if fires[origin_idx]:
                fire_positions.append((k + 1, origin_idx))

        if fire_positions:
            tp += 1
            # lag = smallest k (closest fire to the transition)
            lags.append(min(k for k, _ in fire_positions))
        else:
            fn += 1

    # False positives: fires that are NOT within lookahead of a transition
    fp = int(np.sum(fires & ~near_transition))

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) else 0.0)

    return {
        "cutoff": cutoff,
        "tp": tp, "fp": fp, "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "median_lag_windows": (float(np.median(lags)) if lags else float("nan")),
        "mean_lag_windows": (float(np.mean(lags)) if lags else float("nan")),
        "n_transitions": len(transitions),
        "n_fires": int(np.sum(fires)),
    }


def first_fire_before(origins, fires, transition_idx, lookahead):
    """Return the lag (in origin-steps) of the FIRST fire before a
    transition, or None if no fire in the lookahead window.

    'First' = furthest from the transition (earliest detection).
    """
    origins = np.asarray(origins, dtype=int)
    transition_idx = int(transition_idx)

    best = None
    for j, o in enumerate(origins):
        if o < transition_idx and transition_idx - o <= lookahead:
            if fires[j]:
                lag = transition_idx - o
                if best is None or lag > best:
                    best = lag
    return best


def compare_detectors(origins, cete_scores, var_scores, transitions,
                      flag_percentile=90, lookahead_windows=5):
    """For each transition, determine which detector fired first (or
    neither, or both at the same lag). Returns counts."""
    cete_cut = float(np.percentile(cete_scores, flag_percentile))
    var_cut = float(np.percentile(var_scores, flag_percentile))

    cete_fires = cete_scores >= cete_cut
    var_fires = var_scores >= var_cut

    cete_earlier = 0
    var_earlier = 0
    tied = 0
    cete_only = 0
    var_only = 0
    neither = 0

    for t in transitions:
        c_lag = first_fire_before(origins, cete_fires, t, lookahead_windows)
        v_lag = first_fire_before(origins, var_fires, t, lookahead_windows)

        if c_lag is None and v_lag is None:
            neither += 1
        elif c_lag is None:
            var_only += 1
        elif v_lag is None:
            cete_only += 1
        elif c_lag > v_lag:
            cete_earlier += 1
        elif v_lag > c_lag:
            var_earlier += 1
        else:
            tied += 1

    return {
        "cete_earlier": cete_earlier,
        "var_earlier": var_earlier,
        "tied": tied,
        "cete_only": cete_only,
        "var_only": var_only,
        "neither": neither,
        "n_transitions": len(transitions),
    }


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Does the entropy-weighted score detect regime "
                    "transitions better than plain variance?")
    parser.add_argument("--ticker", default="^GSPC")
    parser.add_argument("--start", default="2018-01-01")
    parser.add_argument("--end", default="2024-12-31")
    parser.add_argument("--csv", default=None)
    parser.add_argument("--window", type=int, default=64,
                        help="Scoring window size (observations)")
    parser.add_argument("--change-window", type=int, default=20,
                        help="Window for the changepoint detector")
    parser.add_argument("--change-ratio", type=float, default=2.0,
                        help="std ratio threshold for a transition")
    parser.add_argument("--min-spacing", type=int, default=30,
                        help="Min observations between distinct transitions")
    parser.add_argument("--flag-percentile", type=float, default=90,
                        help="Both detectors fire at this percentile")
    parser.add_argument("--lookahead", type=int, default=5,
                        help="How many windows before a transition counts "
                             "as 'fired in advance'")
    parser.add_argument("--outdir", default="outputs")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    if args.csv:
        print(f"Loading local CSV: {args.csv}")
        returns, dates = load_local_csv(args.csv)
        source = "local_csv"
    else:
        returns, dates, source = get_market_returns(
            args.ticker, args.start, args.end)
        returns = np.asarray(returns, dtype=float)
    print(f"Loaded {len(returns)} returns (source={source}).")

    if source == "synthetic":
        print("WARNING: synthetic data. Do not treat as real validation.")

    # ------------------------------------------------------------------
    # Transitions (ground truth)
    # ------------------------------------------------------------------
    transitions = find_transitions(
        returns,
        window=args.change_window,
        ratio_threshold=args.change_ratio,
        min_spacing=args.min_spacing,
    )
    print(f"Found {len(transitions)} regime transitions "
          f"(window={args.change_window}, ratio>={args.change_ratio}).")
    if len(transitions) == 0:
        print("FATAL: no transitions detected; tune --change-ratio down "
              "or --change-window.", file=sys.stderr)
        sys.exit(2)

    # ------------------------------------------------------------------
    # Scores
    # ------------------------------------------------------------------
    print(f"Scoring {len(returns) - args.window} trailing windows...")
    scores = compute_scores(returns, args.window, step=1)
    origins = scores["origins"]

    corr_level = safe_corr(scores["cete"], scores["variance"])
    corr_rank = safe_spearman(scores["cete"], scores["variance"])
    print(f"Level correlation (CETE vs variance): "
          f"Pearson={corr_level:.6f}  Spearman={corr_rank:.6f}")

    # ------------------------------------------------------------------
    # Evaluate both detectors
    # ------------------------------------------------------------------
    cete_eval = evaluate_detector(
        origins, scores["cete"], transitions,
        flag_percentile=args.flag_percentile,
        lookahead_windows=args.lookahead,
    )
    var_eval = evaluate_detector(
        origins, scores["variance"], transitions,
        flag_percentile=args.flag_percentile,
        lookahead_windows=args.lookahead,
    )

    head_to_head = compare_detectors(
        origins, scores["cete"], scores["variance"], transitions,
        flag_percentile=args.flag_percentile,
        lookahead_windows=args.lookahead,
    )

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------
    lines = [
        "",
        "TRANSITION DETECTION EVALUATION",
        "(does entropy weighting help beyond plain variance?)",
        "=" * 70,
        f"Data source: {source}",
        f"Observations: {len(returns)}",
        f"Scoring window: {args.window}",
        f"Changepoint window: {args.change_window}",
        f"Changepoint std ratio threshold: {args.change_ratio}",
        f"Min spacing between transitions: {args.min_spacing}",
        f"Detector flag percentile: {args.flag_percentile}",
        f"Lookahead (windows): {args.lookahead}",
        f"Transitions found: {len(transitions)}",
        "",
        "LEVEL ASSOCIATION (as expected from Parseval)",
        "-" * 70,
        f"  Pearson  CETE vs variance: {corr_level:.6f}",
        f"  Spearman CETE vs variance: {corr_rank:.6f}",
        "",
        "TRANSITION DETECTION PERFORMANCE",
        "-" * 70,
        f"{'Detector':<20}"
        f"{'Precision':>11}"
        f"{'Recall':>9}"
        f"{'F1':>7}"
        f"{'MedLag':>8}"
        f"{'Fires':>8}",
    ]

    for name, ev in (("CETE (entropy)", cete_eval),
                     ("plain variance", var_eval)):
        lines.append(
            f"{name:<20}"
            f"{ev['precision']:>11.3f}"
            f"{ev['recall']:>9.3f}"
            f"{ev['f1']:>7.3f}"
            f"{ev['median_lag_windows']:>8.1f}"
            f"{ev['n_fires']:>8d}"
        )

    lines += [
        "",
        "HEAD-TO-HEAD (which fired first, per transition)",
        "-" * 70,
        f"  CETE fired strictly earlier:   {head_to_head['cete_earlier']}",
        f"  Variance fired strictly earlier: {head_to_head['var_earlier']}",
        f"  Tied (both fired same lag):    {head_to_head['tied']}",
        f"  CETE fired, variance did not:  {head_to_head['cete_only']}",
        f"  Variance fired, CETE did not:  {head_to_head['var_only']}",
        f"  Neither fired:                 {head_to_head['neither']}",
        "",
        "INTERPRETATION",
        "-" * 70,
    ]

    cete_win = head_to_head["cete_earlier"] + head_to_head["cete_only"]
    var_win = head_to_head["var_earlier"] + head_to_head["var_only"]
    n_trans = head_to_head["n_transitions"]

    if cete_win > var_win * 1.5 and cete_win >= 5:
        lines.append(
            f"CETE fires meaningfully earlier than plain variance on "
            f"{cete_win}/{n_trans} transitions (vs {var_win} for "
            f"variance). This is a real finding: the entropy weighting "
            f"carries transition information that plain variance does "
            f"not. Investigate before discarding."
        )
    elif var_win > cete_win * 1.5 and var_win >= 5:
        lines.append(
            f"Plain variance fires earlier than CETE on {var_win}/"
            f"{n_trans} transitions (vs {cete_win} for CETE). The "
            f"entropy weighting is not just unhelpful -- it is "
            f"actively worse than the simpler baseline for this task."
        )
    else:
        lines.append(
            f"Neither detector fires meaningfully earlier than the "
            f"other ({cete_win} vs {var_win} of {n_trans} transitions). "
            f"Entropy weighting adds nothing for transition detection "
            f"either. The project's finding is complete: the score is "
            f"variance, full stop."
        )

    report = "\n".join(lines)
    print(report)

    report_path = os.path.join(args.outdir, "transition_evaluation.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report + "\n")
    print(f"\nReport saved to {report_path}")

    # ------------------------------------------------------------------
    # Plot: scores + transitions, on the same time axis
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    x = origins

    axes[0].plot(x, scores["cete"] / scores["cete"].max(),
                 label="CETE (normalized)", linewidth=1.0)
    axes[0].plot(x, scores["variance"] / scores["variance"].max(),
                 label="plain variance (normalized)",
                 linewidth=1.0, alpha=0.75)
    for t in transitions:
        axes[0].axvline(t, color="red", alpha=0.25, linewidth=0.8)
    axes[0].set_title(
        f"CETE vs variance over the same trailing windows "
        f"(level corr={corr_level:.4f})")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(x, returns[x] if len(returns) > x[-1] else returns,
                 linewidth=0.5, color="black")
    for t in transitions:
        axes[1].axvline(t, color="red", alpha=0.25, linewidth=0.8)
    axes[1].set_title("Returns (red lines = detected transitions)")
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(args.outdir, "transition_evaluation.png")
    plt.savefig(plot_path, dpi=130)
    plt.close()
    print(f"Plot saved to {plot_path}")

    # ------------------------------------------------------------------
    # Exit code: 0 if the null holds (no difference), 1 if CETE wins,
    # 2 if variance wins. Useful for CI.
    # ------------------------------------------------------------------
    if cete_win > var_win * 1.5 and cete_win >= 5:
        sys.exit(1)
    elif var_win > cete_win * 1.5 and var_win >= 5:
        sys.exit(2)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()