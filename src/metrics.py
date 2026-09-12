"""
metrics.py
==========

Shared evaluation metrics.

These functions were previously duplicated (byte-for-byte identical) in
scripts/baseline_comparison.py and scripts/causal_evaluation.py. Two
copies of the same Spearman implementation is a class of bug waiting to
happen -- one copy gets fixed, the other doesn't, and the two scripts
report slightly different numbers for the same input with no obvious
reason why.

Why no scipy: scipy is not a declared dependency of this project, and
rankdata_simple + corrcoef gives identical results to scipy.stats.
spearmanr for the inputs this project actually uses (no ties beyond
exact duplicates, which rankdata_simple handles by average-rank). If
scipy is added as a dependency later, these can delegate to it; until
then, the local implementations are the single source of truth.
"""

import numpy as np


# ----------------------------------------------------------------------
# Correlation
# ----------------------------------------------------------------------

def safe_corr(x, y) -> float:
    """Pearson correlation, NaN-safe and constant-safe.

    Returns np.nan (not 0.0) if either array has < 2 finite pairs or zero
    variance. Returning nan rather than 0 matters: 0.0 would be silently
    averaged into downstream summaries as "no correlation observed",
    while nan propagates and forces the caller to decide what to do.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    if len(x) < 2:
        return np.nan
    if np.std(x) == 0 or np.std(y) == 0:
        return np.nan

    return float(np.corrcoef(x, y)[0, 1])


def rankdata_simple(x) -> np.ndarray:
    """Average ranks, no scipy dependency.

    Ties get the average of the ranks they would occupy. Matches
    scipy.stats.rankdata(method='average') for all inputs.
    """
    x = np.asarray(x, dtype=float)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=float)
    sorted_x = x[order]

    i = 0
    while i < len(x):
        j = i + 1
        while j < len(x) and sorted_x[j] == sorted_x[i]:
            j += 1
        avg_rank = 0.5 * (i + j - 1) + 1.0
        ranks[order[i:j]] = avg_rank
        i = j

    return ranks


def safe_spearman(x, y) -> float:
    """Spearman rank correlation, computed as Pearson-on-ranks."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    if len(x) < 2:
        return np.nan

    return safe_corr(rankdata_simple(x), rankdata_simple(y))


# ----------------------------------------------------------------------
# Error metrics
# ----------------------------------------------------------------------

def mae(pred, target) -> float:
    """Mean absolute error over finite pairs."""
    pred = np.asarray(pred, dtype=float)
    target = np.asarray(target, dtype=float)
    mask = np.isfinite(pred) & np.isfinite(target)
    return float(np.mean(np.abs(pred[mask] - target[mask])))


def rmse(pred, target) -> float:
    """Root mean squared error over finite pairs."""
    pred = np.asarray(pred, dtype=float)
    target = np.asarray(target, dtype=float)
    mask = np.isfinite(pred) & np.isfinite(target)
    return float(np.sqrt(np.mean((pred[mask] - target[mask]) ** 2)))