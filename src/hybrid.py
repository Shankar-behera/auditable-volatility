"""
hybrid.py
=========

Why this file exists: after fixing CETE's two bugs (see cete.py), the
project's own baseline comparison (docs/RESULTS.md) found the fixed
energy score correlates >0.999 with plain np.var(window) — i.e. it adds
no information over a one-line variance calculation for the RETROSPECTIVE
volatility-level task, and when forced into a genuine CAUSAL FORECASTING
setup (predicting future, unseen volatility from only past data), it
underperforms naive persistence and clearly loses to a properly
horizon-matched GARCH(1,1) forecast. So CETE is not a competitive
volatility FORECASTER.

It does have one property worth keeping, but it needs to be stated
carefully: evaluated the HONEST (causal, non-leaky) way — flag from a
trailing window, target is a future block the flag never sees — CETE's
percentile-based flag reaches 100% precision only at a 95th-percentile
cutoff (at ~20% recall); a looser 75th-percentile cutoff (this project's
first draft) only reaches ~54% precision once same-window leakage is
removed from the evaluation. See docs/RESULTS.md's operating-curve table
for the full trade-off across thresholds.

HybridVolatilityMonitor combines both components honestly:
  - GARCH(1,1), properly horizon-matched, for the actual volatility
    forecast (the number to act on)
  - CETE's percentile flag (95th-percentile default) as a fast,
    always-available anomaly screen that can run continuously without
    model refitting

This is a legitimate design pattern precisely because the two components
are NOT redundant in their failure modes: CETE's flag is conservative
(high precision, low recall) and needs no fitting; GARCH is the better
point forecaster but requires periodic re-estimation. Neither replaces
the other.
"""

import numpy as np

from .regime import forecast_from_history, causal_crisis_flag_evaluation


class HybridVolatilityMonitor:
    """GARCH(1,1), properly horizon-matched, for the volatility forecast;
    CETE's causal (non-leaky) percentile flag as a cheap anomaly screen."""

    def __init__(self, window_size: int = 64, step: int = 10, flag_percentile: float = 95):
        self.window_size = window_size
        self.step = step
        self.flag_percentile = flag_percentile
        self._garch_fitted = None

    def fit_garch(self, returns: np.ndarray):
        """Fit GARCH(1,1) once on the full available history. In a real
        deployment this would be re-fit periodically (e.g. weekly) via
        src/garch_filter.py's recursive update between refits, not
        refit on every call — see scripts/live_monitor.py."""
        from arch import arch_model

        am = arch_model(returns * 100, vol="Garch", p=1, q=1, dist="normal")
        self._garch_fitted = am.fit(disp="off")
        return self

    def analyze(self, returns: np.ndarray, dates=None):
        """Run both components and return a combined, honestly-labeled
        report: a properly horizon-matched GARCH forecast series, and
        CETE's causal (non-leaky) screening flag over the same history."""
        if self._garch_fitted is None:
            self.fit_garch(returns)

        if dates is None:
            dates = np.arange(len(returns))

        # GARCH: one .forecast() call for the whole series (see
        # regime.garch_causal_forecast_eval's docstring for why calling
        # this per-origin in a loop is a bug, not just slow).
        fc_all = self._garch_fitted.forecast(start=self.window_size - 1, horizon=self.step, reindex=False)
        variance_matrix = fc_all.variance.values
        garch_forecast_series = np.sqrt(variance_matrix.mean(axis=1)) / 100

        # CETE: causal trailing-window energies (same construction the
        # flag is evaluated with — see causal_crisis_flag_evaluation).
        idxs = list(range(self.window_size, len(returns) - self.step, self.step))
        cete_energies = np.array([forecast_from_history(returns[:i], window_size=self.window_size) for i in idxs])
        flag_cut = np.percentile(cete_energies, self.flag_percentile)
        cete_flags = cete_energies >= flag_cut

        screen_eval = causal_crisis_flag_evaluation(
            returns, window_size=self.window_size, step=self.step,
            flag_percentile=self.flag_percentile,
        )

        return {
            "garch_forecast_series": garch_forecast_series,
            "cete_energy_scores": cete_energies,
            "cete_flags": cete_flags,
            "timestamps": np.array([dates[i] for i in idxs]),
            "screen_precision": screen_eval["precision"],
            "screen_recall": screen_eval["recall"],
            "flag_percentile_used": self.flag_percentile,
            "note": (
                "garch_forecast_series is a proper {}-day-ahead volatility "
                "FORECAST (the number to act on). cete_flags is a cheap, "
                "causal SCREEN evaluated honestly (no same-window leakage): "
                "at the {:.0f}th-percentile cutoff it has high precision "
                "({:.0%}) but conservative recall ({:.0%}) against future "
                "extreme-move blocks. Loosening the cutoff trades precision "
                "for recall -- see docs/RESULTS.md's operating-curve table."
            ).format(self.step, self.flag_percentile, screen_eval["precision"], screen_eval["recall"]),
        }