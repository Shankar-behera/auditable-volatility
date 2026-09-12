"""
garch_filter.py
================

Fixes the GARCH live-cache staleness bug identified in external review:
the original live_monitor cached a full GARCH fit (including its
conditional_volatility array) and, on non-refit days, read
`cond_vol[-1]` — the LAST value from the OLD fit, which does not reflect
any return that arrived after that fit. A forecast that ignores today's
actual price action isn't "today's forecast".

The fix does not require refitting the model (the expensive step —
re-estimating omega/alpha/beta by optimization) every day. Instead it
recursively FILTERS the conditional variance forward through new
returns using the cached parameters — this is the standard "update
without re-estimating" approach used in real deployed GARCH systems:

    sigma2_t = omega + alpha * eps_{t-1}^2 + beta * sigma2_{t-1}

Given the last (sigma2, eps) pair from the fit and any new returns since
then, this recursion gives the correct one-step-ahead conditional
variance as of "today" cheaply. A multi-step-ahead (h-day) forecast from
there uses GARCH(1,1)'s closed-form mean-reversion:

    sigma2_forecast(1) = sigma2_next
    sigma2_forecast(h) = long_run_var + (alpha+beta) * (sigma2_forecast(h-1) - long_run_var)
    long_run_var = omega / (1 - alpha - beta)

Parameters should still be re-estimated periodically (the `refit_every_days`
schedule in live_monitor.py) since alpha/beta/omega do drift over time —
this module only replaces "stale array lookup" with "correct recursive
update between refits", not refitting itself.
"""

import numpy as np


def extract_garch_state(fit_result, returns_used_for_fit: np.ndarray):
    """Pulls the (mu, omega, alpha, beta) parameters and the terminal
    (sigma2, eps) pair out of a freshly-fit arch GARCH(1,1) result, in a
    plain-dict form that's cheap to cache (avoids pickling the whole
    fitted model object, which the external review also flagged as
    fragile/non-portable)."""
    params = fit_result.params
    mu = float(params.get("mu", 0.0))
    omega = float(params["omega"])
    alpha = float(params["alpha[1]"])
    beta = float(params["beta[1]"])

    cond_var = fit_result.conditional_volatility ** 2  # already in *100-scale units
    sigma2_last = float(cond_var[-1])
    eps_last = float(returns_used_for_fit[-1] * 100 - mu)

    return {
        "mu": mu, "omega": omega, "alpha": alpha, "beta": beta,
        "sigma2_last": sigma2_last, "eps_last": eps_last,
        "n_returns_at_fit": len(returns_used_for_fit),
    }


def filter_forward(state: dict, new_returns: np.ndarray):
    """Recursively rolls the cached GARCH state forward through any
    `new_returns` observed since the state was fit, WITHOUT
    re-estimating parameters. Returns an updated state dict whose
    sigma2_last/eps_last reflect the most recent observation.

    If `new_returns` is empty, returns `state` unchanged — this is what
    makes repeated same-day calls a no-op instead of double-counting.
    """
    mu, omega, alpha, beta = state["mu"], state["omega"], state["alpha"], state["beta"]
    sigma2, eps = state["sigma2_last"], state["eps_last"]

    for r in new_returns:
        sigma2 = omega + alpha * eps ** 2 + beta * sigma2
        eps = r * 100 - mu

    return {
        **state,
        "sigma2_last": sigma2,
        "eps_last": eps,
        "n_returns_at_fit": state["n_returns_at_fit"] + len(new_returns),
    }


def multistep_forecast(state: dict, horizon: int) -> float:
    """h-day-ahead volatility forecast (std dev, original return scale)
    from the current filtered state, using GARCH(1,1)'s closed-form
    mean-reverting forecast path — matches what `arch`'s `.forecast()`
    would give from an up-to-date fit, without needing to refit."""
    mu, omega, alpha, beta = state["mu"], state["omega"], state["alpha"], state["beta"]
    sigma2, eps = state["sigma2_last"], state["eps_last"]

    sigma2_h1 = omega + alpha * eps ** 2 + beta * sigma2
    long_run_var = omega / (1 - alpha - beta) if (alpha + beta) < 1 else sigma2_h1

    path = [sigma2_h1]
    for _ in range(horizon - 1):
        path.append(long_run_var + (alpha + beta) * (path[-1] - long_run_var))

    return float(np.sqrt(np.mean(path)) / 100)