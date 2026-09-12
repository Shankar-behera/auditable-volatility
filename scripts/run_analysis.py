"""
run_analysis.py
================

End-to-end entrypoint: fetch data -> run CETE -> validate against realized
volatility -> save plots + a text report.

Usage:
    python scripts/run_analysis.py                       # live ^GSPC via yfinance
    python scripts/run_analysis.py --ticker AAPL          # any ticker
    python scripts/run_analysis.py --csv my_prices.csv    # local CSV (date,value columns)
"""

import argparse
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.data import get_market_returns, load_local_csv
from src.regime import detect_regimes, realized_volatility


def main():
    parser = argparse.ArgumentParser(description="Run CETE regime detection.")
    parser.add_argument("--ticker", default="^GSPC", help="Ticker to fetch via yfinance")
    parser.add_argument("--start", default="2018-01-01")
    parser.add_argument("--end", default="2024-12-31")
    parser.add_argument("--csv", default=None, help="Local CSV with date,value columns instead of live fetch")
    parser.add_argument("--window", type=int, default=64)
    parser.add_argument("--step", type=int, default=10)
    parser.add_argument("--outdir", default="outputs")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    if args.csv:
        print(f"Loading local CSV: {args.csv}")
        returns, dates = load_local_csv(args.csv)
        source = "local_csv"
    else:
        returns, dates, source = get_market_returns(args.ticker, args.start, args.end)

    print(f"Loaded {len(returns)} return observations (source={source}).")

    result = detect_regimes(returns, dates, window_size=args.window, step=args.step)
    energy = result["energy_scores"]
    verdicts = result["verdicts"]
    timestamps = pd.to_datetime(result["timestamps"]) if source != "synthetic" else result["timestamps"]

    rv = realized_volatility(returns, window_size=args.window, step=args.step)
    corr = np.corrcoef(energy, rv)[0, 1]

    high_count = sum(1 for v in verdicts if v == "HIGH_UNCERTAINTY_FLAGGED")

    report_lines = [
        "CETE REGIME DETECTION — RUN REPORT",
        "=" * 50,
        f"Data source: {source} ({args.ticker if not args.csv else args.csv}, {args.start} to {args.end})",
        f"Windows analyzed: {len(energy)} (size={args.window}, step={args.step})",
        f"High-uncertainty windows: {high_count} ({high_count/len(energy)*100:.1f}%)",
        f"Energy score range: [{energy.min():.6f}, {energy.max():.6f}], mean={energy.mean():.6f}",
        f"Correlation vs realized volatility: {corr:.4f}",
    ]
    report = "\n".join(report_lines)
    print("\n" + report)

    with open(os.path.join(args.outdir, "report.txt"), "w") as f:
        f.write(report + "\n")

    # Plot
    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
    x = np.arange(len(energy))
    axes[0].plot(x, energy, color="blue", label="CETE energy score")
    high_idx = [i for i, v in enumerate(verdicts) if v == "HIGH_UNCERTAINTY_FLAGGED"]
    axes[0].scatter(high_idx, energy[high_idx], color="red", s=15, zorder=3, label="High uncertainty")
    axes[0].set_title(f"CETE energy — {args.ticker if not args.csv else args.csv} (corr with realized vol = {corr:.3f})")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(x, rv, color="darkorange", label="Realized volatility (rolling std)")
    axes[1].set_title("Ground truth realized volatility")
    if source != "synthetic":
        step_label = max(1, len(x) // 12)
        axes[1].set_xticks(x[::step_label])
        axes[1].set_xticklabels([d.strftime("%Y-%m") for d in timestamps[::step_label]], rotation=45, ha="right")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    outpath = os.path.join(args.outdir, "cete_energy_vs_volatility.png")
    plt.savefig(outpath, dpi=130)
    print(f"\nPlot saved to {outpath}")


if __name__ == "__main__":
    main()
