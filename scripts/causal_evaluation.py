"""
causal_evaluation.py
====================

Leakage-free evaluation of CETE as a volatility forecaster and anomaly
screen.

This script separates two fundamentally different tasks:

1. CAUSAL VOLATILITY FORECASTING
   --------------------------------
   At forecast origin t, only returns available through t are allowed.

   Compare:
       - CETE trailing-window forecast
       - naive persistence
       - expanding-window, horizon-matched GARCH(1,1)

   GARCH parameters are re-estimated using ONLY data available before
   each forecast origin. The GARCH forecast horizon matches `--step`.

2. CAUSAL ANOMALY SCREENING
   --------------------------
   CETE computes a score from a trailing window ending at t.

   The target is a FUTURE block:
       returns[t : t + step]

   Therefore the flag never sees the block used to determine whether
   the future event occurred.

   Multiple percentile thresholds are evaluated to show the precision /
   recall / alert-rate trade-off.

IMPORTANT:
    Correlation is not sufficient to establish forecast quality.
    This script therefore also reports MAE and RMSE.

Usage:
    python scripts/causal_evaluation.py --csv path/to/prices.csv

    python scripts/causal_evaluation.py --ticker ^GSPC

Example:
    python scripts/causal_evaluation.py \
        --csv data/sp500.csv \
        --window 64 \
        --step 10 \
        --min-train 504
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
from src.regime import (
    forecast_from_history,
    causal_crisis_flag_evaluation,
)


# ============================================================
# METRICS
# ============================================================

def safe_corr(x, y):
    """Pearson correlation with protection against constant arrays."""
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


def rankdata_simple(x):
    """
    Lightweight average-rank implementation.

    Avoids requiring scipy just for Spearman correlation.
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


def safe_spearman(x, y):
    """Spearman rank correlation."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    mask = np.isfinite(x) & np.isfinite(y)

    x = x[mask]
    y = y[mask]

    if len(x) < 2:
        return np.nan

    return safe_corr(rankdata_simple(x), rankdata_simple(y))


def mae(pred, target):
    """Mean absolute error."""
    pred = np.asarray(pred, dtype=float)
    target = np.asarray(target, dtype=float)

    mask = np.isfinite(pred) & np.isfinite(target)

    return float(np.mean(np.abs(pred[mask] - target[mask])))


def rmse(pred, target):
    """Root mean squared error."""
    pred = np.asarray(pred, dtype=float)
    target = np.asarray(target, dtype=float)

    mask = np.isfinite(pred) & np.isfinite(target)

    return float(np.sqrt(np.mean((pred[mask] - target[mask]) ** 2)))


# ============================================================
# GARCH WALK-FORWARD
# ============================================================

def garch_walk_forward_forecast(
    returns,
    origins,
    horizon,
    min_train=504,
    refit_every=10,
):
    """
    Leakage-free expanding-window GARCH forecast.

    At origin i:

        training data = returns[:i]

    Therefore NOTHING after i is used to estimate GARCH parameters.

    The model predicts the next `horizon` observations.

    Forecast volatility is calculated as:

        sqrt(mean(predicted_variance[1:horizon]))

    on the original return scale.

    Parameters
    ----------
    returns : np.ndarray
        Full return series.

    origins : iterable[int]
        Forecast origins.

    horizon : int
        Number of future observations predicted.

    min_train : int
        Minimum history required before the first GARCH fit.

    refit_every : int
        Re-estimate GARCH parameters every N origins.

        Between refits the previous parameters are reused, but the model
        is filtered on the expanding history by fitting on the available
        training data at each refit point.

    Returns
    -------
    np.ndarray
        Forecast volatility for each valid origin.
    """

    try:
        from arch import arch_model
    except ImportError:
        raise ImportError(
            "The `arch` package is required for GARCH evaluation.\n"
            "Install it with:\n"
            "    pip install arch"
        )

    returns = np.asarray(returns, dtype=float)

    origins = list(origins)

    forecasts = np.full(len(origins), np.nan)

    cached_fit = None
    last_fit_origin = None

    for j, origin in enumerate(origins):

        if origin < min_train:
            continue

        if origin + horizon > len(returns):
            continue

        needs_refit = (
            cached_fit is None
            or last_fit_origin is None
            or origin - last_fit_origin >= refit_every
        )

        if needs_refit:

            train = returns[:origin]

            # IMPORTANT:
            # Only returns[:origin] are used.
            # No future observations enter the fit.
            am = arch_model(
                train * 100.0,
                mean="Constant",
                vol="Garch",
                p=1,
                q=1,
                dist="normal",
            )

            cached_fit = am.fit(disp="off")

            last_fit_origin = origin

        # Forecast from the fitted model.
        #
        # `horizon` is explicitly matched to the future target block.
        fc = cached_fit.forecast(
            horizon=horizon,
            reindex=False,
        )

        variance = np.asarray(
            fc.variance.iloc[-1].values,
            dtype=float,
        )

        # Aggregate horizon variance into RMS volatility.
        forecast_vol = np.sqrt(np.mean(variance)) / 100.0

        forecasts[j] = forecast_vol

    return forecasts


# ============================================================
# CAUSAL CETE + PERSISTENCE
# ============================================================

def build_causal_forecasts(
    returns,
    window_size,
    step,
    min_train,
):
    """
    Build CETE and persistence forecasts.

    At origin i:

        history = returns[:i]

    CETE sees only the trailing window from that history.

    Target:

        future = returns[i : i + step]

    Returns a dictionary containing all aligned arrays.
    """

    returns = np.asarray(returns, dtype=float)

    origins = list(
        range(
            min_train,
            len(returns) - step + 1,
            step,
        )
    )

    cete = []
    persistence = []
    target = []
    valid_origins = []

    for i in origins:

        history = returns[:i]

        if len(history) < window_size:
            continue

        trailing = history[-window_size:]

        # CETE forecast from history only.
        cete_score = forecast_from_history(
            history,
            window_size=window_size,
        )

        # Naive persistence:
        # use volatility observed in the latest trailing window.
        persistence_score = float(np.std(trailing))

        # Future block NEVER enters either forecast.
        future = returns[i : i + step]

        if len(future) != step:
            continue

        future_vol = float(np.std(future))

        valid_origins.append(i)
        cete.append(float(cete_score))
        persistence.append(persistence_score)
        target.append(future_vol)

    return {
        "origins": np.asarray(valid_origins, dtype=int),
        "cete_forecast": np.asarray(cete, dtype=float),
        "persistence_forecast": np.asarray(
            persistence,
            dtype=float,
        ),
        "target_future_vol": np.asarray(target, dtype=float),
    }


# ============================================================
# CRISIS SCREEN OPERATING CURVE
# ============================================================

def evaluate_screen_curve(
    returns,
    window_size,
    step,
    crisis_percentile=95.0,
    thresholds=(75, 90, 95, 97.5, 99),
):
    """
    Evaluate CETE as a genuinely causal future-event screen.

    For every origin:

        score = CETE(history up to t)

        target = whether a future block contains an extreme return.

    The crisis threshold is computed from the entire return series.
    This is a fixed event-definition threshold rather than a learned
    model threshold.

    The CETE alert threshold is estimated from historical causal scores.

    Returns a list of operating points.
    """

    returns = np.asarray(returns, dtype=float)

    origins = list(
        range(
            window_size,
            len(returns) - step + 1,
            step,
        )
    )

    scores = []
    future_blocks = []

    for i in origins:

        history = returns[:i]

        score = forecast_from_history(
            history,
            window_size=window_size,
        )

        future = returns[i : i + step]

        if len(future) != step:
            continue

        scores.append(score)
        future_blocks.append(future)

    scores = np.asarray(scores, dtype=float)

    if len(scores) == 0:
        return []

    # Event definition.
    crisis_threshold = np.percentile(
        np.abs(returns),
        crisis_percentile,
    )

    labels = np.array(
        [
            np.any(np.abs(block) >= crisis_threshold)
            for block in future_blocks
        ],
        dtype=bool,
    )

    results = []

    for threshold in thresholds:

        cutoff = np.percentile(scores, threshold)

        flags = scores >= cutoff

        tp = int(np.sum(flags & labels))
        fp = int(np.sum(flags & ~labels))
        fn = int(np.sum(~flags & labels))
        tn = int(np.sum(~flags & ~labels))

        precision = (
            tp / (tp + fp)
            if (tp + fp) > 0
            else 0.0
        )

        recall = (
            tp / (tp + fn)
            if (tp + fn) > 0
            else 0.0
        )

        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        fpr = (
            fp / (fp + tn)
            if (fp + tn) > 0
            else 0.0
        )

        alert_rate = float(np.mean(flags))

        results.append(
            {
                "flag_percentile": float(threshold),
                "cutoff": float(cutoff),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "fpr": float(fpr),
                "alert_rate": alert_rate,
                "n_events": int(np.sum(labels)),
            }
        )

    return results


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Leakage-free causal evaluation of CETE, "
            "persistence and GARCH."
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
        help="Forecast horizon / future block size.",
    )

    parser.add_argument(
        "--min-train",
        type=int,
        default=504,
        help=(
            "Minimum number of observations before the first "
            "walk-forward GARCH fit."
        ),
    )

    parser.add_argument(
        "--garch-refit-every",
        type=int,
        default=10,
        help=(
            "Refit GARCH every N forecast origins."
        ),
    )

    parser.add_argument(
        "--outdir",
        default="outputs",
    )

    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # --------------------------------------------------------
    # DATA
    # --------------------------------------------------------

    if args.csv:

        print(f"Loading local CSV: {args.csv}")

        returns, dates = load_local_csv(
            args.csv
        )

        source = "local_csv"

    else:

        returns, dates, source = get_market_returns(
            args.ticker,
            args.start,
            args.end,
        )

    returns = np.asarray(
        returns,
        dtype=float,
    )

    print(
        f"Loaded {len(returns)} return observations "
        f"(source={source})."
    )

    if source == "synthetic":

        print(
            "WARNING: synthetic data is being used. "
            "Do not treat this as real-market validation."
        )

    if len(returns) < args.min_train + args.step:

        raise ValueError(
            f"Not enough observations: {len(returns)}. "
            f"Need at least {args.min_train + args.step}."
        )

    # --------------------------------------------------------
    # CAUSAL CETE / PERSISTENCE
    # --------------------------------------------------------

    fc = build_causal_forecasts(
        returns,
        window_size=args.window,
        step=args.step,
        min_train=args.min_train,
    )

    origins = fc["origins"]

    target = fc["target_future_vol"]

    cete_forecast = fc["cete_forecast"]

    persistence_forecast = fc["persistence_forecast"]

    print(
        f"\nCausal forecast points: {len(origins)}"
    )

    # --------------------------------------------------------
    # WALK-FORWARD GARCH
    # --------------------------------------------------------

    print(
        "\nRunning leakage-free expanding-window "
        "GARCH evaluation..."
    )

    garch_forecast = garch_walk_forward_forecast(
        returns,
        origins,
        horizon=args.step,
        min_train=args.min_train,
        refit_every=args.garch_refit_every,
    )

    # --------------------------------------------------------
    # ALIGN VALID GARCH RESULTS
    # --------------------------------------------------------

    valid = np.isfinite(garch_forecast)

    aligned_target = target[valid]
    aligned_cete = cete_forecast[valid]
    aligned_persistence = persistence_forecast[valid]
    aligned_garch = garch_forecast[valid]
    aligned_origins = origins[valid]

    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    metrics = {}

    models = {
        "CETE": aligned_cete,
        "Persistence": aligned_persistence,
        "GARCH(1,1)": aligned_garch,
    }

    for name, prediction in models.items():

        metrics[name] = {
            "pearson": safe_corr(
                prediction,
                aligned_target,
            ),
            "spearman": safe_spearman(
                prediction,
                aligned_target,
            ),
            "mae": mae(
                prediction,
                aligned_target,
            ),
            "rmse": rmse(
                prediction,
                aligned_target,
            ),
        }

    # --------------------------------------------------------
    # SCREEN CURVE
    # --------------------------------------------------------

    screen_curve = evaluate_screen_curve(
        returns,
        window_size=args.window,
        step=args.step,
        crisis_percentile=95.0,
        thresholds=(75, 90, 95, 97.5, 99),
    )

    # --------------------------------------------------------
    # REPORT
    # --------------------------------------------------------

    lines = [
        "",
        "CAUSAL FORECASTING EVALUATION",
        "(strict walk-forward / no future parameter leakage)",
        "=" * 70,
        f"Data source: {source}",
        f"Observations: {len(returns)}",
        f"Window: {args.window}",
        f"Forecast horizon: {args.step} observations",
        f"Minimum GARCH training observations: {args.min_train}",
        f"GARCH refit interval: {args.garch_refit_every} origins",
        f"Valid forecast points: {len(aligned_target)}",
        "",
        "FORECAST PERFORMANCE",
        "-" * 70,
        f"{'Model':<18}"
        f"{'Pearson':>12}"
        f"{'Spearman':>12}"
        f"{'MAE':>14}"
        f"{'RMSE':>14}",
    ]

    for name in (
        "CETE",
        "Persistence",
        "GARCH(1,1)",
    ):

        m = metrics[name]

        lines.append(
            f"{name:<18}"
            f"{m['pearson']:>12.4f}"
            f"{m['spearman']:>12.4f}"
            f"{m['mae']:>14.6f}"
            f"{m['rmse']:>14.6f}"
        )

    # Determine best model by RMSE.
    best_rmse_model = min(
        metrics,
        key=lambda name: metrics[name]["rmse"],
    )

    best_corr_model = max(
        metrics,
        key=lambda name: (
            -np.inf
            if not np.isfinite(metrics[name]["pearson"])
            else metrics[name]["pearson"]
        ),
    )

    lines += [
        "",
        f"Best RMSE model: {best_rmse_model}",
        f"Best Pearson-correlation model: {best_corr_model}",
        "",
        "INTERPRETATION",
        "-" * 70,
    ]

    cete_vs_persist = (
        metrics["CETE"]["pearson"]
        - metrics["Persistence"]["pearson"]
    )

    cete_vs_garch = (
        metrics["CETE"]["pearson"]
        - metrics["GARCH(1,1)"]["pearson"]
    )

    if cete_vs_persist > 0:
        lines.append(
            "CETE beats naive persistence on Pearson correlation "
            "in this walk-forward evaluation."
        )
    else:
        lines.append(
            "CETE does NOT beat naive persistence on Pearson "
            "correlation in this walk-forward evaluation."
        )

    if cete_vs_garch < 0:
        lines.append(
            "GARCH remains ahead of CETE on Pearson correlation."
        )
    else:
        lines.append(
            "CETE is not below GARCH on Pearson correlation; "
            "inspect MAE/RMSE before drawing a conclusion."
        )

    lines += [
        "",
        "IMPORTANT:",
        "The GARCH result above is genuinely causal:",
        "each refit uses only observations available before that",
        "forecast origin, and the forecast horizon matches the",
        "future realized-volatility target.",
        "",
        "CETE is evaluated from a trailing history only.",
        "The future volatility block is never passed into CETE.",
        "",
        "Correlation measures directional association, while MAE",
        "and RMSE measure forecast error magnitude. All three should",
        "be considered when judging forecasting usefulness.",
        "",
        "CAUSAL CETE SCREEN OPERATING CURVE",
        "(future-block event detection; no same-window target leakage)",
        "=" * 70,
        f"{'Threshold':>10}"
        f"{'Precision':>12}"
        f"{'Recall':>12}"
        f"{'F1':>10}"
        f"{'FPR':>10}"
        f"{'Alert %':>10}",
    ]

    for row in screen_curve:

        lines.append(
            f"{row['flag_percentile']:>9.1f}%"
            f"{row['precision']:>12.3f}"
            f"{row['recall']:>12.3f}"
            f"{row['f1']:>10.3f}"
            f"{row['fpr']:>10.3f}"
            f"{row['alert_rate'] * 100:>9.1f}%"
        )

    lines += [
        "",
        "The screening curve is intentionally reported across",
        "multiple thresholds. A high percentile cutoff produces",
        "fewer alerts and can improve precision, but generally",
        "reduces recall.",
    ]

    report = "\n".join(lines)

    print(report)

    report_path = os.path.join(
        args.outdir,
        "causal_evaluation.txt",
    )

    with open(
        report_path,
        "w",
        encoding="utf-8",
    ) as f:
        f.write(report + "\n")

    print(
        f"\nReport saved to {report_path}"
    )

    # --------------------------------------------------------
    # FORECAST PLOT
    # --------------------------------------------------------

    x = np.arange(len(aligned_target))

    fig, ax = plt.subplots(
        figsize=(14, 6)
    )

    ax.plot(
        x,
        aligned_target,
        label="Future realized volatility",
        linewidth=2,
    )

    # Standardize only for visual comparison.
    def zscore(v):
        std = np.std(v)

        if std == 0:
            return np.zeros_like(v)

        return (
            v - np.mean(v)
        ) / std

    ax.plot(
        x,
        zscore(aligned_cete),
        alpha=0.75,
        label=(
            f"CETE "
            f"(r={metrics['CETE']['pearson']:.2f})"
        ),
    )

    ax.plot(
        x,
        zscore(aligned_persistence),
        alpha=0.75,
        label=(
            f"Persistence "
            f"(r={metrics['Persistence']['pearson']:.2f})"
        ),
    )

    ax.plot(
        x,
        zscore(aligned_garch),
        alpha=0.75,
        label=(
            f"GARCH(1,1) "
            f"(r={metrics['GARCH(1,1)']['pearson']:.2f})"
        ),
    )

    ax.set_title(
        "Strict causal volatility forecasting comparison"
    )

    ax.set_xlabel(
        "Forecast origin"
    )

    ax.set_ylabel(
        "Volatility / standardized forecast"
    )

    ax.legend()

    ax.grid(
        alpha=0.3
    )

    plt.tight_layout()

    forecast_plot_path = os.path.join(
        args.outdir,
        "causal_forecast_comparison.png",
    )

    plt.savefig(
        forecast_plot_path,
        dpi=130,
    )

    plt.close()

    print(
        f"Forecast plot saved to {forecast_plot_path}"
    )

    # --------------------------------------------------------
    # SCREEN CURVE PLOT
    # --------------------------------------------------------

    if screen_curve:

        thresholds = [
            row["flag_percentile"]
            for row in screen_curve
        ]

        precision = [
            row["precision"]
            for row in screen_curve
        ]

        recall = [
            row["recall"]
            for row in screen_curve
        ]

        fig, ax = plt.subplots(
            figsize=(10, 6)
        )

        ax.plot(
            thresholds,
            precision,
            marker="o",
            label="Precision",
        )

        ax.plot(
            thresholds,
            recall,
            marker="o",
            label="Recall",
        )

        ax.set_xlabel(
            "CETE percentile threshold"
        )

        ax.set_ylabel(
            "Score"
        )

        ax.set_title(
            "CETE causal screening operating curve"
        )

        ax.set_ylim(
            0,
            1.05,
        )

        ax.grid(
            alpha=0.3
        )

        ax.legend()

        plt.tight_layout()

        screen_plot_path = os.path.join(
            args.outdir,
            "cete_screen_operating_curve.png",
        )

        plt.savefig(
            screen_plot_path,
            dpi=130,
        )

        plt.close()

        print(
            f"Screen curve saved to {screen_plot_path}"
        )


if __name__ == "__main__":
    main()