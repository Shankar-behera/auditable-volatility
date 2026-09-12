"""
test_cete.py
============

Regression tests for CETE's scoring.

These encode specific bugs found during validation -- both the original
two (see src/cete.py docstring) and issues raised by an external review
(see docs/RESULTS.md) -- so a future refactor can't silently
reintroduce any of them.

The relaxation dynamics were removed (src/cete.py). Tests that only
made sense with the dynamics -- fast-vs-dynamics agreement, and a
step-by-step scale-compression check -- are gone. The property those
tests were checking (scale preservation) is restated for the version
without dynamics in test_energy_preserves_input_scale, which can still
fail if someone reintroduces a compressing mechanism.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.cete import CETE
from src.data import get_synthetic_regime_switch
from src.regime import (
    detect_regimes,
    realized_volatility,
    fast_score,
    forecast_from_history,
    causal_forecast_eval,
    crisis_flag_evaluation,
    causal_crisis_flag_evaluation,
)


# ----------------------------------------------------------------------
# Bug 1: energy must scale with input volatility
# ----------------------------------------------------------------------

def test_energy_scales_monotonically_with_volatility():
    """Regression test for Bug 1 (divergent gradient term). Before the
    fix, low-std windows scored HIGHER energy than high-std windows.

    Asserts true non-decreasing order on a fixed seed verified to be
    monotonic. The correlation check is kept as a secondary signal."""
    rng = np.random.RandomState(0)
    stds = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2]
    energies = []
    for std in stds:
        window = std * rng.randn(64)
        window = window - np.mean(window)
        engine = CETE(state_dim=64)
        result = engine.run(window)
        energies.append(result["final_energy_score"])

    assert all(energies[i] <= energies[i + 1]
               for i in range(len(energies) - 1)), (
        f"Energy is not monotonically non-decreasing with volatility: "
        f"{energies}"
    )
    assert energies[-1] > energies[0] * 10
    assert np.corrcoef(stds, energies)[0, 1] > 0.8


# ----------------------------------------------------------------------
# Bug 2 restated: score must preserve input scale
# ----------------------------------------------------------------------

def test_energy_preserves_input_scale():
    """The score must scale with the input's VARIANCE: a 10x std
    difference is a 100x variance difference, and without any dynamics
    to compress the scale, the score ratio should reflect that, up to
    the entropy weighting's per-window variation.

    This is the property Bug 2 was about. An earlier version with
    dynamics compressed a 10x std ratio to a 4x score ratio. With the
    dynamics removed, the compression mechanism is gone; this test
    asserts the score scales proportionally with variance instead.

    Band is generous (30x-300x for a 100x variance ratio) because the
    entropy weighting is not exactly constant across windows.
    """
    rng = np.random.RandomState(1)
    calm = 0.005 * rng.randn(64)
    calm -= np.mean(calm)
    volatile = 0.05 * rng.randn(64)
    volatile -= np.mean(volatile)

    calm_score = CETE(state_dim=64).run(calm)["final_energy_score"]
    vol_score = CETE(state_dim=64).run(volatile)["final_energy_score"]

    ratio = vol_score / calm_score
    assert 30.0 < ratio < 300.0, (
        f"Score ratio {ratio:.2f} for a 100x variance difference. "
        f"calm={calm_score:.6f} vol={vol_score:.6f}. The score should "
        f"scale with variance; a ratio near 4x would mean scale "
        f"compression is back."
    )


# ----------------------------------------------------------------------
# End-to-end synthetic regime detection
# ----------------------------------------------------------------------

def test_synthetic_regime_switch_detected():
    """End-to-end: low-vol -> high-vol -> low-vol should show up as
    low -> high -> low in the energy trajectory."""
    returns, dates = get_synthetic_regime_switch()
    result = detect_regimes(returns, dates, window_size=64, step=10)
    energy = result["energy_scores"]

    n = len(energy)
    low_start = np.mean(energy[:20])
    high_mid = np.mean(energy[n // 2 - 10 : n // 2 + 10])
    low_end = np.mean(energy[-20:])

    assert high_mid > low_start
    assert high_mid > low_end


def test_correlation_with_realized_volatility_is_strong():
    """Retrospective validation: energy score should track a real
    volatility proxy closely on the SAME windows. NOT a predictive
    claim -- see the leakage tests below."""
    returns, dates = get_synthetic_regime_switch()
    result = detect_regimes(returns, dates, window_size=64, step=10)
    rv = realized_volatility(returns, window_size=64, step=10)

    corr = np.corrcoef(result["energy_scores"], rv)[0, 1]
    assert corr > 0.7, f"Correlation too weak: {corr:.3f}"


# ----------------------------------------------------------------------
# The canary: base power must track plain variance
# ----------------------------------------------------------------------

def test_cete_base_power_tracks_plain_variance():
    """Documents the baseline-comparison finding in docs/RESULTS.md: by
    Parseval's theorem, CETE's score should be (near) proportional to
    plain np.var() of the same window.

    This test asserts the NULL RESULT holds. If it ever fails -- i.e.
    correlation drops meaningfully below 0.99 -- something in encode()'s
    entropy reweighting has changed enough to actually add information
    beyond a one-line variance calculation. That would be a real
    finding worth knowing about, either way.

    Uses fast_score (the actual scoring path) rather than reconstructing
    base_power by hand, so this tests what is actually used."""
    rng = np.random.RandomState(2)
    plain_vars, base_powers = [], []
    for _ in range(50):
        w = 0.01 * rng.randn(64)
        w = w - np.mean(w)
        plain_vars.append(np.var(w))
        base_powers.append(fast_score(w, window_size=64))

    corr = np.corrcoef(plain_vars, base_powers)[0, 1]
    assert corr > 0.99, (
        f"CETE base power no longer tracks plain variance as closely "
        f"(corr={corr:.4f}) -- re-check docs/RESULTS.md's baseline "
        f"comparison conclusion, it may need updating."
    )


# ----------------------------------------------------------------------
# CETE.run and fast_score must agree
# ----------------------------------------------------------------------
def test_fast_score_matches_cete_run():
    """CETE.run() and fast_score() must compute the same quantity.

    They are two implementations of the same transformation. If they
    ever diverge, the evaluation suite is measuring something
    different from what live_monitor.py reports, and every number in
    docs/RESULTS.md is suspect.

    Checks ALL THREE input-length paths -- shorter than state_dim
    (padding), equal to state_dim (no-op), and longer than state_dim
    (truncation). An earlier version of this test used only n=64,
    which cannot catch a truncation-direction mismatch: encode() kept
    the FIRST state_dim observations while fast_score() kept the LAST,
    producing different scores for n=100 and identical scores for
    n=64. Verified by the n=100 case below.
    """
    rng = np.random.RandomState(11)

    for n in (10, 20, 32, 64, 100):
        for _ in range(20):
            x = 0.01 * rng.randn(n)

            a = CETE(state_dim=64).run(x)["final_energy_score"]
            b = fast_score(x, window_size=64)

            rel = abs(a - b) / max(abs(a), abs(b), 1e-12)

            assert rel < 1e-10, (
                f"CETE.run and fast_score disagree at n={n}: "
                f"run={a}, fast_score={b}, rel={rel:.2e}"
            )

# ----------------------------------------------------------------------
# Retrospective crisis flag: kept, but explicitly labeled tautological
# ----------------------------------------------------------------------


# ----------------------------------------------------------------------
# Retrospective crisis flag: kept, but explicitly labeled tautological
# ----------------------------------------------------------------------

def test_retrospective_crisis_flag_is_tautological_by_construction():
    """NOT a validation test. This documents that the retrospective
    (same-window) crisis evaluation is near-tautological: the flag is
    computed from the same window used to define the ground-truth
    crisis day, and energy is proportional to that window's variance.

    The assertion (precision >= 0.5) is deliberately weak -- it exists
    to catch a version that made the flag ANTI-correlated with the
    window it's computed from. For the honest, causal version, see
    causal_crisis_flag_evaluation() and scripts/causal_evaluation.py.
    """
    rng = np.random.RandomState(3)
    calm = 0.005 * rng.randn(700)
    spike = np.concatenate([calm[:400], 0.08 * rng.randn(20), calm[400:]])

    dates = np.arange(len(spike))
    result = detect_regimes(spike, dates, window_size=64, step=10)
    window_starts = list(range(0, len(spike) - 64, 10))
    ev = crisis_flag_evaluation(
        spike, result["verdicts"], window_starts, window_size=64)

    assert ev["precision"] >= 0.5, (
        f"Same-window crisis flag precision is below chance "
        f"({ev['precision']:.2f}) -- energy is now ANTI-correlated with "
        f"the window it's computed from, which contradicts Bug 1's fix."
    )


# ----------------------------------------------------------------------
# Percentile threshold: a property that could actually fail
# ----------------------------------------------------------------------

def test_percentile_cutoff_is_monotone_in_threshold():
    """A stricter (higher) percentile cutoff must not flag MORE windows
    than a looser one.

    This looks tautological but is not: it would fail if the energy
    array had NaNs (np.percentile skips them, np.mean doesn't), or if
    the flag comparison used `>` instead of `>=`, or if the energies
    were recomputed between the two evaluations with a different
    window. It's a real integration check on the two evaluation paths
    agreeing about what "flag at percentile p" means."""
    rng = np.random.RandomState(4)
    calm = 0.005 * rng.randn(900)
    spike = np.concatenate([calm[:500], 0.08 * rng.randn(15), calm[500:]])

    results = {
        pct: causal_crisis_flag_evaluation(
            spike, window_size=64, step=10,
            crisis_percentile=95, flag_percentile=pct)
        for pct in (50, 75, 90, 95)
    }

    alerts = {pct: r["tp"] + r["fp"] for pct, r in results.items()}
    thresholds = sorted(alerts)
    for lo, hi in zip(thresholds, thresholds[1:]):
        assert alerts[hi] <= alerts[lo], (
            f"Flagging at percentile {hi} produced {alerts[hi]} alerts, "
            f"more than percentile {lo} produced ({alerts[lo]}). The "
            f"percentile threshold logic is not monotone."
        )


# ----------------------------------------------------------------------
# Structural leakage guard
# ----------------------------------------------------------------------

def test_forecast_from_history_does_not_leak_future():
    """forecast_from_history's output must be IDENTICAL regardless of
    what data exists outside the slice it was given."""
    rng = np.random.RandomState(5)
    history = 0.01 * rng.randn(200)

    forecast_a = forecast_from_history(history, window_size=64)

    future_a = np.concatenate([history, 0.5 * rng.randn(50)])
    future_b = np.concatenate([history, 0.001 * rng.randn(50)])

    forecast_from_a = forecast_from_history(future_a[:len(history)],
                                            window_size=64)
    forecast_from_b = forecast_from_history(future_b[:len(history)],
                                            window_size=64)

    assert forecast_a == forecast_from_a == forecast_from_b, (
        "forecast_from_history's output changed based on data outside "
        "the slice it was given -- this should be structurally "
        "impossible."
    )


def test_causal_forecast_eval_ignores_post_origin_data():
    """Integration-level leakage guard: the CAUSAL evaluation must
    produce IDENTICAL forecasts from returns[:origin] regardless of
    what comes after `origin`.

    Constructs two series that share a common prefix but diverge after
    it, runs the causal evaluation on both, and checks the forecasts at
    every shared origin are identical."""
    rng = np.random.RandomState(6)
    prefix = 0.01 * rng.randn(300)

    series_a = np.concatenate([prefix, 0.5 * rng.randn(100)])
    series_b = np.concatenate([prefix, 0.001 * rng.randn(100)])

    result_a = causal_forecast_eval(series_a, window_size=64, step=10)
    result_b = causal_forecast_eval(series_b, window_size=64, step=10)

    for i, (oa, ob) in enumerate(zip(result_a["origins"],
                                     result_b["origins"])):
        if oa > len(prefix) or oa != ob:
            continue
        fa = result_a["cete_forecast"][i]
        fb = result_b["cete_forecast"][i]
        assert fa == fb, (
            f"CETE forecast at origin {oa} changed based on data after "
            f"{oa}: {fa} vs {fb}. This is a leakage bug."
        )


# ----------------------------------------------------------------------
# API stability for live_monitor.py
# ----------------------------------------------------------------------

def test_fast_score_window_handling():
    """fast_score must behave identically whether it is given exactly
    `window_size` observations or a longer history ending in the same
    window. This is the property live_monitor.py relies on.

    Also checks the padding case: a history shorter than window_size
    must not raise (it pads, matching CETE.encode)."""
    rng = np.random.RandomState(8)
    history = 0.01 * rng.randn(200)
    ws = 64

    a = fast_score(history[:150], window_size=ws)
    b = fast_score(history[150 - ws:150], window_size=ws)
    assert abs(a - b) < 1e-12, (
        f"fast_score depends on more than the trailing window: "
        f"long-history={a}, short-slice={b}"
    )

    c = fast_score(history[:20], window_size=ws)
    assert np.isfinite(c), (
        "fast_score raised or returned non-finite for short input"
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))