"""
publish_report.py
=================

Reads the prediction logs and writes human-readable Markdown reports
into docs/track_record/.

Two output files per ticker:

    docs/track_record/<TICKER>.md
        Latest prediction, latest resolution, rolling accuracy,
        and a full resolved-prediction table.

    docs/track_record/index.md
        One-line summary per ticker: latest forecast, latest resolution,
        and rolling correlation if enough points exist.

------------------------------------------------------------------
WHY MARKDOWN, NOT HTML
------------------------------------------------------------------
The public artifact of this project is the RECORD, not the model.
Markdown is plain text, diffable in git, readable on GitHub without
any rendering step, and archivable. If the track record is meant to
be inspectable and verifiable by a stranger, the format needs to be
one the stranger can read without a browser, a build step, or a
running server. Markdown is that format.

------------------------------------------------------------------
WHAT THE REPORT DOES NOT CLAIM
------------------------------------------------------------------
The report displays raw numbers. It does not compute statistical
significance, does not adjust for multiple comparisons, and does not
claim the model is "good" or "bad" at any sample size. A correlation
of 0.6 from 5 predictions and a correlation of 0.6 from 500
predictions look the same in the output. The `n_resolved` column is
there so a reader can judge that for themselves.

Reports are overwritten on every run. Git history preserves the
previous versions, so the record of "what the report said on date X"
is available via `git log -p`.

Usage:
    python scripts/publish_report.py
    python scripts/publish_report.py --outdir docs/track_record
    python scripts/publish_report.py --rolling 50
"""

import argparse
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.predictions import (
    _predictions_dir,
    _normalize_ticker,
    read_pending,
    read_resolved,
    latest_for_ticker,
    accuracy_summary,
)


# ----------------------------------------------------------------------
# Discovery: find every ticker that has a log file
# ----------------------------------------------------------------------

def find_tickers() -> list[str]:
    """Every ticker with a non-empty prediction log, sorted."""
    d = _predictions_dir()
    if not d.exists():
        return []
    tickers = []
    for p in sorted(d.glob("*.jsonl")):
        if p.stat().st_size > 0:
            tickers.append(p.stem)
    return tickers


# ----------------------------------------------------------------------
# Rolling metrics
# ----------------------------------------------------------------------

def rolling_accuracy(resolved: pd.DataFrame, window: int) -> dict:
    """Accuracy over the most recent `window` resolved predictions.

    Unlike accuracy_summary, which uses the full history, this
    restricts to a trailing slice. Reported alongside the full-history
    numbers so a reader can see whether recent performance differs
    from lifetime performance.
    """
    if resolved.empty:
        return {
            "n": 0, "mean_abs_error": float("nan"),
            "rmse": float("nan"), "mean_error": float("nan"),
            "pearson": float("nan"),
        }

    tail = resolved.tail(window)

    out = {
        "n": int(len(tail)),
        "mean_abs_error": float(tail["abs_error"].mean()),
        "rmse": float(np.sqrt(tail["squared_error"].mean())),
        "mean_error": float(tail["error"].mean()),
        "pearson": float("nan"),
    }

    if len(tail) >= 2:
        f = tail["forecast_vol"].to_numpy(dtype=float)
        t = tail["target_vol"].to_numpy(dtype=float)
        if np.std(f) > 0 and np.std(t) > 0:
            out["pearson"] = float(np.corrcoef(f, t)[0, 1])

    return out


def format_number(x, places=6) -> str:
    """Format a number for the report, handling nan/inf gracefully so
    the table stays aligned."""
    if x is None:
        return "--"
    try:
        f = float(x)
    except (TypeError, ValueError):
        return "--"
    if np.isnan(f):
        return "--"
    if np.isinf(f):
        return "inf"
    return f"{f:.{places}f}"


def format_corr(x) -> str:
    """Correlation formatted to 3 places, or '--' if undefined."""
    if x is None:
        return "--"
    try:
        f = float(x)
    except (TypeError, ValueError):
        return "--"
    if np.isnan(f):
        return "--"
    return f"{f:+.3f}"


# ----------------------------------------------------------------------
# Per-ticker report
# ----------------------------------------------------------------------

def write_ticker_report(ticker: str, outdir: Path, rolling_window: int) -> Path:
    """Write docs/track_record/<TICKER>.md. Returns the path written."""
    normalized = _normalize_ticker(ticker)

    pending = read_pending(ticker)
    resolved = read_resolved(ticker)
    summary = accuracy_summary(ticker)
    rolling = rolling_accuracy(resolved, rolling_window)
    latest = latest_for_ticker(ticker)

    now = datetime.now(timezone.utc).isoformat(timespec="seconds")

    lines = [
        f"# {normalized} — volatility forecast track record",
        "",
        f"_Last updated: {now}_",
        "",
        "This is an automatically generated record of every volatility "
        "forecast made by this system for this ticker, and every "
        "subsequent realized outcome. The model is GARCH(1,1). The "
        "forecast is a `horizon`-day-ahead volatility estimate; the "
        "target is the realized standard deviation over the next "
        "`horizon` observed returns. See "
        "[METHODOLOGY.md](../METHODOLOGY.md) for the full description.",
        "",
        "## Current status",
        "",
    ]

    if latest is not None:
        lines += [
            f"- **Latest prediction**: `{latest.get('prediction_id', '--')}`",
            f"  - Origin date: {latest.get('origin_date', '--')}",
            f"  - Forecast volatility ({latest.get('horizon_days', '--')}-day): "
            f"`{format_number(latest.get('forecast_vol'))}`",
            f"  - Screen flagged: `{latest.get('screen_flagged', '--')}`",
            f"  - Recorded at: {latest.get('recorded_at', '--')}",
            "",
        ]
    else:
        lines += ["- No predictions recorded yet.", ""]

    lines += [
        f"- **Pending predictions**: {summary['n_pending']}",
        f"- **Resolved predictions**: {summary['n_resolved']}",
        "",
    ]

    if summary["n_resolved"] > 0:
        lines += [
            "## Accuracy (all resolved predictions)",
            "",
            f"- Mean absolute error: `{format_number(summary['mean_abs_error'])}`",
            f"- RMSE: `{format_number(summary['rmse'])}`",
            f"- Mean error (bias): `{format_number(summary['mean_error'])}`",
            f"- Pearson correlation (forecast vs target): "
            f"`{format_corr(summary['pearson'])}`",
            f"- First origin: {summary['first_origin']}",
            f"- Last origin: {summary['last_origin']}",
            "",
            f"## Accuracy (last {rolling_window} resolved)",
            "",
            f"- N: {rolling['n']}",
            f"- Mean absolute error: `{format_number(rolling['mean_abs_error'])}`",
            f"- RMSE: `{format_number(rolling['rmse'])}`",
            f"- Mean error (bias): `{format_number(rolling['mean_error'])}`",
            f"- Pearson correlation: `{format_corr(rolling['pearson'])}`",
            "",
        ]
    else:
        lines += [
            "## Accuracy",
            "",
            "No resolved predictions yet. Accuracy statistics will appear "
            "here once predictions mature past their forecast horizon.",
            "",
        ]

    # Full resolved table
    if not resolved.empty:
        lines += [
            "## Resolved predictions",
            "",
            "| Origin | Horizon | Forecast | Target | Abs error |",
            "|---|---|---|---|---|",
        ]
        for _, r in resolved.iterrows():
            lines.append(
                f"| {r['origin_date']} "
                f"| {int(r['horizon_days'])} "
                f"| `{format_number(r['forecast_vol'])}` "
                f"| `{format_number(r['target_vol'])}` "
                f"| `{format_number(r['abs_error'])}` |"
            )
        lines += [""]

    # Pending table
    if not pending.empty:
        lines += [
            "## Pending predictions",
            "",
            "| Origin | Horizon | Forecast | Screen |",
            "|---|---|---|---|",
        ]
        for _, r in pending.iterrows():
            lines.append(
                f"| {r['origin_date']} "
                f"| {int(r['horizon_days'])} "
                f"| `{format_number(r['forecast_vol'])}` "
                f"| {r.get('screen_flagged', '--')} |"
            )
        lines += [""]

    lines += [
        "---",
        "",
        "_This report is generated by `scripts/publish_report.py` from "
        "`data/predictions/`. Predictions are append-only and cannot be "
        "revised after the fact._",
        "",
    ]

    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / f"{normalized}.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


# ----------------------------------------------------------------------
# Index report
# ----------------------------------------------------------------------

def write_index(tickers: list[str], outdir: Path) -> Path:
    """Write docs/track_record/index.md — one row per ticker."""
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")

    lines = [
        "# Volatility forecast track record",
        "",
        f"_Last updated: {now}_",
        "",
        "Automated record of volatility forecasts and their realized "
        "outcomes. Every forecast is logged at prediction time and "
        "cannot be revised; every outcome is filled in when the "
        "forecast horizon elapses. See [METHODOLOGY.md](../METHODOLOGY.md) "
        "for the model and evaluation description.",
        "",
    ]

    if not tickers:
        lines += [
            "No predictions recorded yet. Once `scripts/live_monitor.py` "
            "has run and written a prediction, this index will list "
            "each ticker with its current state.",
            "",
        ]
    else:
        lines += [
            "| Ticker | Pending | Resolved | Latest forecast | "
            "Latest origin | All-time corr |",
            "|---|---|---|---|---|---|",
        ]
        for t in tickers:
            s = accuracy_summary(t)
            latest = latest_for_ticker(t)
            latest_fc = (format_number(latest.get("forecast_vol"))
                         if latest else "--")
            latest_origin = (latest.get("origin_date") if latest else "--")
            corr = format_corr(s["pearson"]) if s["n_resolved"] >= 2 else "--"
            link = f"[{_normalize_ticker(t)}]({_normalize_ticker(t)}.md)"
            lines.append(
                f"| {link} "
                f"| {s['n_pending']} "
                f"| {s['n_resolved']} "
                f"| `{latest_fc}` "
                f"| {latest_origin} "
                f"| {corr} |"
            )
        lines += [""]

    lines += [
        "---",
        "",
        "_Generated by `scripts/publish_report.py`._",
        "",
    ]

    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / "index.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate Markdown track-record reports.")
    parser.add_argument("--outdir", default="docs/track_record",
                        help="Where to write the reports")
    parser.add_argument("--rolling", type=int, default=30,
                        help="Window for the rolling accuracy section")
    parser.add_argument("--ticker", default=None,
                        help="Report on a single ticker instead of all")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if args.ticker:
        tickers = [_normalize_ticker(args.ticker)]
    else:
        tickers = find_tickers()

    if not tickers:
        print("No prediction logs found. Writing an empty index.")
        path = write_index([], outdir)
        print(f"  wrote {path}")
        sys.exit(0)

    print(f"Found {len(tickers)} ticker(s) with prediction logs.")
    for t in tickers:
        path = write_ticker_report(t, outdir, args.rolling)
        print(f"  wrote {path}")

    path = write_index(tickers, outdir)
    print(f"  wrote {path}")

    sys.exit(0)


if __name__ == "__main__":
    main()