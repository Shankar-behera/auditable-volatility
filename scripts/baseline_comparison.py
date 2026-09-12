"""
baseline_comparison.py
======================

Retrospective baseline comparison for CETE.

This script answers:

    "Is CETE's energy score actually different from a simple
     variance calculation when both are computed from the SAME window?"

It compares:

    1. CETE energy score
    2. Plain rolling variance
    3. Plain rolling standard deviation
    4. Full-sample GARCH conditional volatility
       (DESCRIPTIVE / IN-SAMPLE ONLY)

IMPORTANT:

The CETE and variance comparison is RETROSPECTIVE.

Both are computed from the same historical window, so a high
correlation does NOT demonstrate forecasting ability.

The causal forecasting question is handled separately by:

    scripts/causal_evaluation.py

That script uses:
    - CETE from past data
    - persistence from past data
    - walk-forward GARCH
    - future unseen volatility as the target

Usage:

    python scripts/baseline_comparison.py \
        --csv path/to/prices.csv

or:

    python scripts/baseline_comparison.py \
        --ticker ^GSPC
"""

import argparse
import os
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(__file__),
        "..",
    ),
)

from src.data import (
    get_market_returns,
    load_local_csv,
)

from src.regime import (
    detect_regimes,
    realized_volatility,
)


# ============================================================
# METRICS
# ============================================================

def safe_corr(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    mask = np.isfinite(x) & np.isfinite(y)

    x = x[mask]
    y = y[mask]

    if len(x) < 2:
        return np.nan

    if np.std(x) == 0 or np.std(y) == 0:
        return np.nan

    return float(
        np.corrcoef(x, y)[0, 1]
    )


def rankdata_simple(x):
    """Average ranks without requiring scipy."""
    x = np.asarray(x, dtype=float)

    order = np.argsort(
        x,
        kind="mergesort",
    )

    ranks = np.empty(
        len(x),
        dtype=float,
    )

    sorted_x = x[order]

    i = 0

    while i < len(x):

        j = i + 1

        while (
            j < len(x)
            and sorted_x[j] == sorted_x[i]
        ):
            j += 1

        avg_rank = (
            0.5 * (i + j - 1)
            + 1.0
        )

        ranks[
            order[i:j]
        ] = avg_rank

        i = j

    return ranks


def safe_spearman(x, y):
    return safe_corr(
        rankdata_simple(x),
        rankdata_simple(y),
    )


# ============================================================
# ROLLING VARIANCE
# ============================================================

def rolling_window_variance(
    returns,
    window_size,
    step,
):
    """
    Compute plain np.var(window) on exactly the same windows
    used by CETE.
    """

    returns = np.asarray(
        returns,
        dtype=float,
    )

    values = []

    for start in range(
        0,
        len(returns) - window_size,
        step,
    ):

        window = returns[
            start : start + window_size
        ]

        values.append(
            np.var(window)
        )

    return np.asarray(
        values,
        dtype=float,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Retrospective CETE vs simple volatility "
            "baseline comparison."
        )
    )

    parser.add_argument(
        "--ticker",
        default="^GSPC",
    )

    parser.add_argument(
        "--start",
        default="2018-01-01",
    )

    parser.add_argument(
        "--end",
        default="2024-12-31",
    )

    parser.add_argument(
        "--csv",
        default=None,
        help="Local CSV with date,value columns.",
    )

    parser.add_argument(
        "--window",
        type=int,
        default=64,
    )

    parser.add_argument(
        "--step",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--outdir",
        default="outputs",
    )

    args = parser.parse_args()

    os.makedirs(
        args.outdir,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # DATA
    # --------------------------------------------------------

    if args.csv:

        print(
            f"Loading local CSV: {args.csv}"
        )

        returns, dates = load_local_csv(
            args.csv
        )

        source = "local_csv"

    else:

        returns, dates, source = (
            get_market_returns(
                args.ticker,
                args.start,
                args.end,
            )
        )

    returns = np.asarray(
        returns,
        dtype=float,
    )

    print(
        f"Loaded {len(returns)} return observations "
        f"(source={source})."
    )

    if len(returns) < args.window:

        raise ValueError(
            f"Not enough observations for window={args.window}."
        )

    window_size = args.window
    step = args.step

    # --------------------------------------------------------
    # CETE
    # --------------------------------------------------------

    result = detect_regimes(
        returns,
        dates,
        window_size=window_size,
        step=step,
    )

    cete_energy = np.asarray(
        result["energy_scores"],
        dtype=float,
    )

    # --------------------------------------------------------
    # REALIZED VOLATILITY
    # --------------------------------------------------------

    rv = np.asarray(
        realized_volatility(
            returns,
            window_size=window_size,
            step=step,
        ),
        dtype=float,
    )

    # --------------------------------------------------------
    # PLAIN VARIANCE
    # --------------------------------------------------------

    rolling_var = (
        rolling_window_variance(
            returns,
            window_size,
            step,
        )
    )

    # --------------------------------------------------------
    # ALIGN
    # --------------------------------------------------------

    n = min(
        len(cete_energy),
        len(rv),
        len(rolling_var),
    )

    cete_energy = cete_energy[:n]
    rv = rv[:n]
    rolling_var = rolling_var[:n]

    # --------------------------------------------------------
    # CETE vs VARIANCE
    # --------------------------------------------------------

    corr_cete_var = safe_corr(
        cete_energy,
        rolling_var,
    )

    corr_cete_rv = safe_corr(
        cete_energy,
        rv,
    )

    corr_var_rv = safe_corr(
        rolling_var,
        rv,
    )

    spearman_cete_var = safe_spearman(
        cete_energy,
        rolling_var,
    )

    spearman_cete_rv = safe_spearman(
        cete_energy,
        rv,
    )

    spearman_var_rv = safe_spearman(
        rolling_var,
        rv,
    )

    # --------------------------------------------------------
    # GARCH DESCRIPTIVE COMPARISON
    # --------------------------------------------------------

    have_garch = False
    garch_at_centers = None

    try:

        from arch import arch_model

        am = arch_model(
            returns * 100.0,
            mean="Constant",
            vol="Garch",
            p=1,
            q=1,
            dist="normal",
        )

        garch_res = am.fit(
            disp="off"
        )

        cond_vol = (
            np.asarray(
                garch_res.conditional_volatility,
                dtype=float,
            )
            / 100.0
        )

        starts = list(
            range(
                0,
                len(returns) - window_size,
                step,
            )
        )

        garch_at_centers = np.asarray(
            [
                cond_vol[
                    start
                    + window_size // 2
                ]
                for start in starts
            ],
            dtype=float,
        )

        garch_at_centers = (
            garch_at_centers[:n]
        )

        have_garch = True

    except ImportError:

        print(
            "`arch` package not installed. "
            "Skipping descriptive GARCH comparison."
        )

    # --------------------------------------------------------
    # REPORT
    # --------------------------------------------------------

    lines = [
        "",
        "RETROSPECTIVE BASELINE COMPARISON",
        "=" * 70,
        f"Data source: {source}",
        f"Observations: {len(returns)}",
        f"Windows: {n}",
        f"Window size: {window_size}",
        f"Step: {step}",
        "",
        "CETE vs SIMPLE VOLATILITY BASELINES",
        "-" * 70,
        f"CETE vs rolling variance:",
        f"    Pearson  = {corr_cete_var:.6f}",
        f"    Spearman = {spearman_cete_var:.6f}",
        "",
        f"CETE vs rolling standard deviation:",
        f"    Pearson  = {corr_cete_rv:.6f}",
        f"    Spearman = {spearman_cete_rv:.6f}",
        "",
        f"Rolling variance vs rolling standard deviation:",
        f"    Pearson  = {corr_var_rv:.6f}",
        f"    Spearman = {spearman_var_rv:.6f}",
    ]

    if have_garch:

        corr_garch_rv = safe_corr(
            garch_at_centers,
            rv,
        )

        spearman_garch_rv = safe_spearman(
            garch_at_centers,
            rv,
        )

        corr_cete_garch = safe_corr(
            cete_energy,
            garch_at_centers,
        )

        lines += [
            "",
            "FULL-SAMPLE GARCH DESCRIPTIVE COMPARISON",
            "-" * 70,
            f"GARCH conditional volatility vs realized volatility:",
            f"    Pearson  = {corr_garch_rv:.6f}",
            f"    Spearman = {spearman_garch_rv:.6f}",
            "",
            f"CETE vs GARCH conditional volatility:",
            f"    Pearson  = {corr_cete_garch:.6f}",
        ]

    lines += [
        "",
        "INTERPRETATION",
        "-" * 70,
        "1. CETE energy and rolling variance are computed from the",
        "   same historical window. Their association therefore",
        "   measures retrospective redundancy, NOT forecasting skill.",
        "",
        "2. A very high CETE-vs-variance correlation is expected",
        "   because FFT spectral power is mathematically related to",
        "   time-domain squared magnitude through Parseval's theorem.",
        "",
        "3. CETE's entropy weighting modifies the spectral components,",
        "   but the current final score remains explicitly anchored to",
        "   pre-dynamics spectral power.",
        "",
        "4. The GARCH result in this script is DESCRIPTIVE/IN-SAMPLE.",
        "   The model is fitted using the full dataset and its",
        "   conditional volatility is compared with historical windows.",
        "",
        "5. Therefore this script MUST NOT be used to claim that",
        "   GARCH is or is not a superior causal forecaster.",
        "",
        "6. The genuine forecasting comparison is implemented in:",
        "",
        "       scripts/causal_evaluation.py",
        "",
        "   where GARCH parameters are estimated only from data",
        "   available before each forecast origin and the forecast",
        "   horizon is matched to the future volatility target.",
    ]

    report = "\n".join(
        lines
    )

    print(report)

    report_path = os.path.join(
        args.outdir,
        "baseline_comparison.txt",
    )

    with open(
        report_path,
        "w",
        encoding="utf-8",
    ) as f:

        f.write(
            report + "\n"
        )

    print(
        f"\nReport saved to {report_path}"
    )

    # ========================================================
    # PLOT 1: CETE VS VARIANCE
    # ========================================================

    x = np.arange(n)

    fig, ax = plt.subplots(
        figsize=(14, 5)
    )

    cete_norm = (
        cete_energy
        / np.max(cete_energy)
        if np.max(cete_energy) > 0
        else cete_energy
    )

    var_norm = (
        rolling_var
        / np.max(rolling_var)
        if np.max(rolling_var) > 0
        else rolling_var
    )

    ax.plot(
        x,
        cete_norm,
        label="CETE energy (normalized)",
    )

    ax.plot(
        x,
        var_norm,
        label="Plain rolling variance (normalized)",
        alpha=0.75,
    )

    ax.set_title(
        "CETE vs plain rolling variance "
        f"(Pearson r={corr_cete_var:.4f})"
    )

    ax.set_xlabel(
        "Window"
    )

    ax.set_ylabel(
        "Normalized value"
    )

    ax.legend()

    ax.grid(
        alpha=0.3
    )

    plt.tight_layout()

    cete_var_path = os.path.join(
        args.outdir,
        "cete_vs_variance.png",
    )

    plt.savefig(
        cete_var_path,
        dpi=130,
    )

    plt.close()

    print(
        f"Plot saved to {cete_var_path}"
    )

    # ========================================================
    # PLOT 2: CETE VS REALIZED VOL
    # ========================================================

    fig, ax = plt.subplots(
        figsize=(14, 5)
    )

    rv_norm = (
        rv / np.max(rv)
        if np.max(rv) > 0
        else rv
    )

    ax.plot(
        x,
        cete_norm,
        label="CETE energy (normalized)",
    )

    ax.plot(
        x,
        rv_norm,
        label="Rolling realized volatility (normalized)",
        alpha=0.75,
    )

    ax.set_title(
        "CETE vs retrospective realized volatility "
        f"(Pearson r={corr_cete_rv:.4f})"
    )

    ax.set_xlabel(
        "Window"
    )

    ax.set_ylabel(
        "Normalized value"
    )

    ax.legend()

    ax.grid(
        alpha=0.3
    )

    plt.tight_layout()

    cete_rv_path = os.path.join(
        args.outdir,
        "cete_vs_realized_volatility.png",
    )

    plt.savefig(
        cete_rv_path,
        dpi=130,
    )

    plt.close()

    print(
        f"Plot saved to {cete_rv_path}"
    )

    # ========================================================
    # PLOT 3: ALL DESCRIPTIVE ESTIMATORS
    # ========================================================

    fig, ax = plt.subplots(
        figsize=(14, 6)
    )

    ax.plot(
        x,
        cete_norm,
        label="CETE",
    )

    ax.plot(
        x,
        var_norm,
        label="Rolling variance",
        alpha=0.75,
    )

    ax.plot(
        x,
        rv_norm,
        label="Realized volatility",
        alpha=0.75,
    )

    if have_garch:

        garch_norm = (
            garch_at_centers
            / np.max(garch_at_centers)
            if np.max(garch_at_centers) > 0
            else garch_at_centers
        )

        ax.plot(
            x,
            garch_norm,
            label="GARCH conditional volatility",
            alpha=0.75,
        )

    ax.set_title(
        "Retrospective/descriptive volatility estimators"
    )

    ax.set_xlabel(
        "Window"
    )

    ax.set_ylabel(
        "Normalized value"
    )

    ax.legend()

    ax.grid(
        alpha=0.3
    )

    plt.tight_layout()

    all_path = os.path.join(
        args.outdir,
        "baseline_comparison.png",
    )

    plt.savefig(
        all_path,
        dpi=130,
    )

    plt.close()

    print(
        f"Plot saved to {all_path}"
    )


if __name__ == "__main__":
    main()