"""
cete.py
=======

CETE's scoring, after the two bugs documented below were addressed and
the relaxation dynamics were removed.

------------------------------------------------------------------
WHAT THIS MODULE IS NOW
------------------------------------------------------------------
This is a scalar volatility proxy: entropy-weighted FFT spectral power
of a mean-centered window, which by Parseval's theorem is proportional
to the window's variance. It is computed in one function (`CETE.run`)
with no relaxation dynamics.

An earlier version had a hand-designed dynamical system (momentum, a
per-dimension metric, operator blending, 20 relaxation steps) layered
on top. Three independent lines of evidence showed the dynamics
contribute nothing:

  1. The reported score was anchored to pre-dynamics spectral power
     (`base_power`); the dynamics modulated it by at most ~10%.
  2. Measured causally (predicting FUTURE realized vol from PAST
     returns), CETE underperformed naive persistence and GARCH(1,1).
     See docs/RESULTS.md for the reproducible numbers; the exact
     values are not duplicated here so there is exactly one source
     of truth.
  3. The dynamics' state magnitudes converged toward similar values
     regardless of input scale. Measured directly: a 10x input std
     separation compressed to a 3.99x output ratio. Anchoring the
     SCORE to base_power bypassed the compression for the reported
     value, but the dynamics themselves were never fixed -- they were
     still in the code, still compressing.

So the dynamics were removed rather than kept as dead weight. See
docs/RESULTS.md for the full validation and the negative result.

------------------------------------------------------------------
BUG-FIX LOG (kept for context; see docs/RESULTS.md for detail)
------------------------------------------------------------------
BUG 1 -- divergent gradient term. The original compute_gradient had a
term -0.1/(|state|+1e-6) that diverges as |state| -> 0, i.e. exactly
when the input window is CALM (low vol -> low FFT power -> small
state magnitude). It injected large synthetic momentum into calm
windows, inflating their reported energy -- backwards. Removing it
moved the real-data correlation from -0.24 to +0.15.

BUG 2 -- scale-erasing operator. The original dynamics included
op3 = state * tanh(|state|) / |state|, a saturating normalizer that
pulled every state toward a similar magnitude regardless of input
volatility. Measured: a calm window's mean magnitude grew
0.036 -> 0.052 over 20 steps while a volatile window's SHRANK
0.35 -> 0.21. The dynamics were compressing the dynamic range that
made the signal useful.

The first response to Bug 2 was to anchor the REPORTED SCORE to
pre-dynamics spectral power, so the dynamics could only modulate it
by +-10%. This made the reported number correct without fixing the
defect -- the dynamics were still in the code, still compressing.
A later test measured scale preservation directly and found a 10x
input std ratio compressed to a 3.99x output ratio. The dynamics
were then removed entirely.

------------------------------------------------------------------
CONSISTENCY WITH regime.fast_score()
------------------------------------------------------------------
The transformation performed by CETE.encode() -- slice/pad to
state_dim, mean-center, FFT, entropy-weight -- is reproduced exactly
in src/regime.py's fast_score(). The two must produce identical
numbers for the same input, or CETE.run() and the evaluation suite
compute different quantities. There is a test
(test_fast_score_matches_cete_run in tests/test_cete.py) that
enforces this invariant.

If you change encode()'s transformation, change fast_score() to match.

------------------------------------------------------------------
THRESHOLDS
------------------------------------------------------------------
CETE.run() returns a placeholder verdict ("HIGH_ENERGY" / "LOW_ENERGY")
using an absolute cutoff that is NOT portable across assets or
timeframes -- raw spectral power has no universal "high". For the
classification that matters, use regime.detect_regimes(), which
classifies relative to the percentile distribution of scores observed
in a given run.
------------------------------------------------------------------
"""

import numpy as np


# Absolute cutoff for the standalone verdict. NOT portable across
# assets or timeframes -- raw spectral power has no universal "high".
# Kept as a named constant so it is obvious that it is a placeholder
# and not a calibrated threshold. See module docstring.
PLACEHOLDER_ENERGY_THRESHOLD = 1e-3


class CETE:
    """Entropy-weighted FFT spectral power of a return window.

    By Parseval, proportional to the window's variance. Named CETE for
    continuity with the rest of the project; see module docstring for
    why the dynamical system was removed.
    """

    def __init__(self, state_dim: int = 64):
        self.state_dim = state_dim

    def encode(self, raw_data: np.ndarray) -> np.ndarray:
        """FFT-encode a mean-centered return window.

        The window is truncated or zero-padded to `state_dim`, then
        mean-centered before the FFT. The resulting spectrum is
        weighted by each frequency bin's spectral entropy contribution.

        The reported energy is 0.5 * mean(|encoded|**2).

        IMPORTANT: this exact transformation is reproduced in
        src/regime.py's fast_score(). The two must agree to within
        floating-point precision, or CETE.run() and the evaluation
        suite compute different quantities. See
        test_fast_score_matches_cete_run in tests/test_cete.py.
        """
        raw_data = np.asarray(raw_data, dtype=float)

        if len(raw_data) > self.state_dim:
            # Keep the MOST RECENT state_dim observations, matching
            # fast_score() in regime.py. The previous version kept the
            # FIRST state_dim, which silently produced a different
            # window (and therefore a different score) whenever
            # len(raw_data) > state_dim. That branch was never hit by
            # run()'s callers, which always passed exactly state_dim
            # observations -- but it is hit by test_fast_score_matches_
            # cete_run's n=100 case, and would be hit by any future
            # caller that passes a longer history.
            data = raw_data[-self.state_dim:]
        else:
            data = np.pad(raw_data, (0, self.state_dim - len(raw_data)))
        # Mean-center AFTER padding/truncation, so the DC component is
        # removed from the final state_dim-length window. fast_score()
        # in regime.py does the same.
        data = data - np.mean(data)

        spectrum = np.fft.fft(data)
        power = np.abs(spectrum) ** 2
        total_power = np.sum(power) + 1e-12

        prob = power / total_power
        entropy_weights = -prob * np.log(prob + 1e-12)
        encoded = spectrum * (1.0 + entropy_weights)
        return encoded.astype(np.complex128)

    def run(self, raw_data: np.ndarray, max_steps: int = 20) -> dict:
        """Compute the CETE score for one window.

        `max_steps` is accepted but ignored, kept only for call-site
        compatibility with the version that ran relaxation dynamics.
        There are no dynamics to step through now.

        The returned dict has the same shape as the previous version's,
        with `steps_taken` always 0 and `energy_trajectory` a singleton.
        Any caller that read `final_energy_score` continues to work.
        """
        state = self.encode(raw_data)
        energy = 0.5 * float(np.mean(np.abs(state) ** 2))

        verdict = (
            "HIGH_ENERGY"
            if energy >= PLACEHOLDER_ENERGY_THRESHOLD
            else "LOW_ENERGY"
        )

        return {
            "status": "success",
            "audit_verdict": verdict,
            "final_energy_score": energy,
            "steps_taken": 0,
            "energy_trajectory": [energy],
        }