"""
score_outcomes.py
=================

Reads pending predictions from the log, checks which have matured past
their forecast horizon, computes realized volatility over the target
block, and appends a resolution for each one.

Run this on the same schedule as live_monitor.py (daily, after market
close). It is idempotent: a prediction that has already been resolved
will not be resolved again, because record_resolution raises on
double-resolve and this script catches and skips that case.

------------------------------------------------------------------
HOW "MATURED" IS DEFINED
------------------------------------------------------------------
A prediction with origin_date D and horizon_days H is scoreable once
the market has provided H trading days of returns AFTER D.

    target_vol = std(returns[D_idx + 1 : D_idx + 1 + H])

The +1 skips the origin day itself: the origin day is the last day
whose return was already visible to the forecast, so including it
would leak a day the model saw into the target.

This is the same convention used by scripts/causal_evaluation.py, so
the live track record will be comparable to the historical evaluation.

"Matured" therefore requires D_idx + 1 + H <= len(returns). If it
isn't, the prediction is skipped and will be retried on the next run.

Trading-day counting is implicit: the target block is the next H
OBSERVED returns in the series, whatever calendar days they happen to
fall on. This is the same convention the historical evaluation used.

------------------------------------------------------------------
USAGE
------------------------------------------------------------------
    python scripts/score_outcomes.py --ticker ^GSPC
    python scripts/score_outcomes.py --ticker ^GSPC --lookback-days 3650
    python scripts/score_outcomes.py --csv path/to/prices.csv

Exit codes:
    0  ran successfully (whether or not anything was resolved)
    2  data fetch failed; no resolutions written
"""

import argparse
import os
import sys
from datetime import datetime, timedelta

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data import get_market_returns, load_local_csv, DataValidationError
from src.predictions import read_pending, record_resolution


EXIT_OK = 0
EXIT_DATA_FAILED = 2


def _date_to_index(dates, target_date: str) -> int:
    """Find the position of `target_date` in the dates array.

    Prefers an exact match. If the exact date is not in the series
    (e.g. it was a weekend or market holiday), falls back to the most
    recent trading day STRICTLY BEFORE the target. This is the correct
    direction for a prediction whose origin is defined as "the last
    date the model saw": if the recorded origin is not a trading day,
    the model's actual last-seen date is the prior trading day.

    Returns the index, or -1 if no date at or before target_date is in
    the series (i.e. the series starts after the target).
    """
    import pandas as pd

    try:
        idx = pd.DatetimeIndex(dates)
        target = pd.Timestamp(target_date)

        # Exact match
        exact = np.where(idx == target)[0]
        if len(exact) > 0:
            return int(exact[0])

        # Most recent trading day strictly before target
        before = np.where(idx < target)[0]
        if len(before) > 0:
            return int(before[-1])

        # No date at or before target in the series
        return -1
    except Exception:
        pass

    # Fallback: string comparison
    best = -1
    for i, d in enumerate(dates):
        if str(d)[:10] <= target_date[:10]:
            best = i
        else:
            break
    return best


def score_one(returns, dates, row: dict) -> bool:
    """Attempt to resolve a single prediction row. Returns True if a
    resolution was appended, False if the prediction has not yet
    matured or its origin date is not in the series.

    `returns` is the full return series. `dates` is aligned to
    `returns` (each date is the date of that return observation).
    """
    origin_date = str(row["origin_date"])
    horizon = int(row["horizon_days"])
    ticker = row["ticker"]

    origin_idx = _date_to_index(dates, origin_date)
    if origin_idx < 0:
        print(f"  skip {row['prediction_id']}: origin {origin_date} "
              f"not found in the return series")
        return False

    target_start = origin_idx + 1
    target_end = target_start + horizon

    if target_end > len(returns):
        have = len(returns) - target_start
        print(f"  pending {row['prediction_id']}: only {have} of "
              f"{horizon} target observations available")
        return False

    future_block = returns[target_start:target_end]
    target_vol = float(np.std(future_block))

    try:
        record_resolution(
            ticker=ticker,
            origin_date=origin_date,
            horizon_days=horizon,
            target_vol=target_vol,
        )
        print(f"  resolved {row['prediction_id']}: "
              f"forecast={row['forecast_vol']:.6f} "
              f"target={target_vol:.6f}")
        return True
    except ValueError as e:
        # Most likely "already resolved" -- a race with another process
        # or a manual re-run. Not an error for this script.
        print(f"  skip {row['prediction_id']}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Resolve matured predictions with realized volatility.")
    parser.add_argument("--ticker", default="^GSPC")
    parser.add_argument("--csv", default=None,
                        help="Use a local CSV instead of live fetch")
    parser.add_argument("--lookback-days", type=int, default=365 * 5,
                        help="How much history to pull for scoring. "
                             "Must be long enough to cover the oldest "
                             "pending prediction's origin plus horizon.")
    args = parser.parse_args()

    # ----------------------------------------------------------------
    # Data
    # ----------------------------------------------------------------
    if args.csv:
        try:
            returns, dates = load_local_csv(args.csv)
        except (DataValidationError, FileNotFoundError, ValueError) as e:
            print(f"FATAL: could not load {args.csv}: {e}", file=sys.stderr)
            sys.exit(EXIT_DATA_FAILED)
        source = "local_csv"
        ticker_key = "csv"
    else:
        start = (datetime.now() - timedelta(days=args.lookback_days)
                 ).strftime("%Y-%m-%d")
        end = datetime.now().strftime("%Y-%m-%d")
        try:
            returns, dates, source = get_market_returns(
                args.ticker, start, end)
        except (RuntimeError, DataValidationError) as e:
            print(f"FATAL: {e}", file=sys.stderr)
            sys.exit(EXIT_DATA_FAILED)
        ticker_key = args.ticker

    returns = np.asarray(returns, dtype=float)
    print(f"Loaded {len(returns)} returns (source={source}) "
          f"for ticker={ticker_key}.")

    # ----------------------------------------------------------------
    # Pending predictions
    # ----------------------------------------------------------------
    pending = read_pending(ticker_key)
    if pending.empty:
        print("No pending predictions.")
        sys.exit(EXIT_OK)

    print(f"Found {len(pending)} pending prediction(s); "
          f"checking maturity...")

    resolved_count = 0
    for _, row in pending.iterrows():
        if score_one(returns, dates, row.to_dict()):
            resolved_count += 1

    print(f"\nResolved {resolved_count} of {len(pending)} pending "
          f"prediction(s).")

    sys.exit(EXIT_OK)


if __name__ == "__main__":
    main()