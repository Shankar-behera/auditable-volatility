"""
live_monitor.py
================

The "run this for real" entrypoint. Fetches validated data, filters a
cached GARCH(1,1) state forward through any new returns, computes
CETE's percentile screen on the latest trailing window, writes a
provenance manifest and a prediction to the append-only log, and
prints a decision.

Meant to be scheduled (cron / Task Scheduler / GitHub Actions), once
per trading day after market close.

------------------------------------------------------------------
WHAT THIS FILE WRITES
------------------------------------------------------------------
Per run, up to four artifacts:

    data/snapshots/<TICKER>/prices/<sha256>.parquet
        The returns series the model consumed. Content-addressed.

    data/snapshots/<TICKER>/garch/<sha256>.json
        The frozen fitted GARCH parameters. Content-addressed.

    data/manifests/<prediction_id>.json
        Provenance record referencing both snapshots by path + hash,
        plus the code SHA, environment versions, and model config.

    data/predictions/<TICKER>.jsonl
        One appended line: the prediction, plus manifest_path and
        manifest_sha256.

The order matters. Snapshots are written first, then the manifest,
then the log line. If the process crashes at any point before the
log line is written, no prediction is recorded -- which is correct,
because a prediction without provenance is not auditable.

------------------------------------------------------------------
EXIT CODES
------------------------------------------------------------------
  0  ran successfully (flagged or quiet -- a flag is not an error)
  2  data fetch failed
  3  GARCH forecast failed
  4  could not write manifest or prediction log

Usage:
    python scripts/live_monitor.py --ticker ^GSPC
    python scripts/live_monitor.py --csv path/to/prices.csv
    python scripts/live_monitor.py --ticker ^GSPC --out result.json
"""

import argparse
import json
import os
import sys
from datetime import datetime, timedelta

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data import get_market_returns, load_local_csv, DataValidationError
from src.regime import fast_score
from src.predictions import record_prediction
from src.manifest import (
    store_snapshot, build_manifest, write_manifest, hash_file,
)
from src.garch_filter import (
    extract_garch_state, filter_forward, multistep_forecast,
)


# ----------------------------------------------------------------------
# Exit codes
# ----------------------------------------------------------------------

EXIT_OK = 0
EXIT_DATA_FAILED = 2
EXIT_GARCH_FAILED = 3
EXIT_LOG_FAILED = 4


# ----------------------------------------------------------------------
# GARCH state persistence
# ----------------------------------------------------------------------

def _state_path(state_dir, ticker):
    safe = ticker.replace("^", "").replace("/", "_")
    return os.path.join(state_dir, f"garch_{safe}.json")


def get_or_update_garch_state(returns, ticker, state_dir, refit_every_days):
    """Load a cached GARCH state (JSON) and filter it forward through
    any new returns since it was cached. Fully re-fit parameters only
    if the cache is missing or older than `refit_every_days`.

    Returns (state, was_refit).
    """
    os.makedirs(state_dir, exist_ok=True)
    path = _state_path(state_dir, ticker)

    if os.path.exists(path):
        try:
            with open(path) as f:
                cache = json.load(f)
            fitted_at = datetime.fromisoformat(cache["fitted_at"])
            age = datetime.now() - fitted_at
            state = cache["state"]
            n_at_fit = state["n_returns_at_fit"]

            if age < timedelta(days=refit_every_days):
                new_returns = returns[n_at_fit:]
                if len(new_returns) > 0:
                    state = filter_forward(state, new_returns)
                    print(f"[garch] Using cached fit from {fitted_at.date()} "
                          f"({age.days}d old), filtered forward through "
                          f"{len(new_returns)} new returns.")
                else:
                    print(f"[garch] Using cached fit from {fitted_at.date()} "
                          f"({age.days}d old), no new data.")
                return state, False
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            print(f"[garch] Cache at {path} is unreadable ({e}); refitting.")
        except Exception as e:
            print(f"[garch] Unexpected error reading cache ({e}); refitting.")

    print("[garch] Refitting GARCH(1,1) parameters (cache missing or stale)...")
    from arch import arch_model

    am = arch_model(returns * 100, vol="Garch", p=1, q=1, dist="normal")
    fit_result = am.fit(disp="off")
    state = extract_garch_state(fit_result, returns)

    with open(path, "w") as f:
        json.dump({"fitted_at": datetime.now().isoformat(),
                   "state": state}, f, indent=2)
    print(f"[garch] Refit complete, cached to {path}")
    return state, True


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run the live volatility monitor once and exit.")
    parser.add_argument("--ticker", default="^GSPC")
    parser.add_argument("--csv", default=None,
                        help="Use a local CSV instead of live fetch (testing)")
    parser.add_argument("--lookback-days", type=int, default=365 * 3,
                        help="How much history to pull for fitting/screening")
    parser.add_argument("--window", type=int, default=64,
                        help="CETE scoring window in observations")
    parser.add_argument("--horizon", type=int, default=10,
                        help="GARCH forecast horizon, in days")
    parser.add_argument("--refit-every-days", type=int, default=5,
                        help="Refit GARCH parameters this often; filter "
                             "the cached state otherwise")
    parser.add_argument("--flag-percentile", type=float, default=95,
                        help="CETE flag cutoff")
    parser.add_argument("--state-dir", default=".cete_state")
    parser.add_argument("--out", default=None,
                        help="Optional path to write a JSON result")
    parser.add_argument("--no-record", action="store_true",
                        help="Skip writing manifest and prediction log. "
                             "For testing only.")
    args = parser.parse_args()

    # ----------------------------------------------------------------
    # 1. Data
    # ----------------------------------------------------------------
    if args.csv:
        try:
            returns, dates = load_local_csv(args.csv)
        except (DataValidationError, FileNotFoundError, ValueError) as e:
            print(f"FATAL: could not load {args.csv}: {e}", file=sys.stderr)
            sys.exit(EXIT_DATA_FAILED)
        source = "local_csv"
    else:
        start = (datetime.now() - timedelta(days=args.lookback_days)
                 ).strftime("%Y-%m-%d")
        end = datetime.now().strftime("%Y-%m-%d")
        try:
            returns, dates, source = get_market_returns(
                args.ticker, start, end)
        except RuntimeError as e:
            print(f"FATAL: {e}", file=sys.stderr)
            sys.exit(EXIT_DATA_FAILED)
        except DataValidationError as e:
            print(f"FATAL: data validation failed: {e}", file=sys.stderr)
            sys.exit(EXIT_DATA_FAILED)

    returns = np.asarray(returns, dtype=float)
    print(f"Loaded {len(returns)} return observations (source={source}).")

    if len(returns) < args.window + 1:
        print(f"FATAL: only {len(returns)} observations; "
              f"need more than window={args.window}.", file=sys.stderr)
        sys.exit(EXIT_DATA_FAILED)

    # ----------------------------------------------------------------
    # 2. CETE screen
    # ----------------------------------------------------------------
    cete_energy = fast_score(returns, window_size=args.window)

    stride = max(1, args.window // 8)
    hist_idxs = range(args.window, len(returns), stride)
    hist_energies = np.array(
        [fast_score(returns[:i], window_size=args.window)
         for i in hist_idxs]
    )
    flag_cut = float(np.percentile(hist_energies, args.flag_percentile))
    is_flagged = bool(cete_energy >= flag_cut)

    # ----------------------------------------------------------------
    # 3. GARCH forecast
    # ----------------------------------------------------------------
    ticker_key = args.ticker if not args.csv else "csv"
    try:
        state, was_refit = get_or_update_garch_state(
            returns, ticker_key, args.state_dir, args.refit_every_days)
        forecast_vol = multistep_forecast(state, horizon=args.horizon)
    except Exception as e:
        print(f"FATAL: GARCH forecast failed: {e}", file=sys.stderr)
        sys.exit(EXIT_GARCH_FAILED)

    # ----------------------------------------------------------------
    # 4. Provenance: snapshots + manifest + log line.
    # ----------------------------------------------------------------
    origin_date = str(dates[-1])[:10]
    prediction_id = None

    # Build the prediction ID up front so we can check whether it's
    # already logged BEFORE writing any snapshots or a manifest.
    # Re-running live_monitor.py on the same day must be a no-op, not
    # an error -- and it must not attempt to overwrite a manifest
    # that's already tied to a logged prediction (write_manifest
    # correctly refuses to do that).
    from src.predictions import read_predictions

    pid_str = (
        f"{ticker_key.upper().replace('^', '').replace('/', '-')}"
        f"_{origin_date}_{args.horizon}"
    )

    already_logged = False
    if not args.no_record:
        existing = read_predictions(ticker_key)
        if not existing.empty and "prediction_id" in existing.columns:
            already_logged = (existing["prediction_id"] == pid_str).any()

    if already_logged:
        print(f"[log] prediction {pid_str} already logged; "
              f"skipping manifest and log write")
        prediction_id = pid_str
    elif not args.no_record:
        try:
            import pandas as _pd

            returns_df = _pd.DataFrame({
                "date": dates,
                "log_return": returns,
            })
            price_path, price_sha = store_snapshot(
                ticker_key, "prices", returns_df, "parquet")

            garch_params = {
                k: state[k] for k in (
                    "mu", "omega", "alpha", "beta",
                    "sigma2_last", "eps_last", "n_returns_at_fit",
                )
            }
            garch_path, garch_sha = store_snapshot(
                ticker_key, "garch", garch_params, "json")

            manifest = build_manifest(
                prediction_id=pid_str,
                ticker=ticker_key,
                origin_date=origin_date,
                horizon_days=args.horizon,
                window_size=args.window,
                price_snapshot={
                    "path": str(price_path),
                    "sha256": price_sha,
                    "n_observations": int(len(returns)),
                    "first_date": str(dates[0])[:10],
                    "last_date": str(dates[-1])[:10],
                },
                garch_parameters={
                    "path": str(garch_path),
                    "sha256": garch_sha,
                    **garch_params,
                },
                model_config={
                    "vol": "Garch", "p": 1, "q": 1,
                    "dist": "normal", "mean": "Constant",
                },
            )
            manifest_path = write_manifest(manifest)
            manifest_sha = hash_file(manifest_path)

            prediction_id = record_prediction(
                ticker=ticker_key,
                origin_date=origin_date,
                horizon_days=args.horizon,
                forecast_vol=float(forecast_vol),
                screen_flagged=is_flagged,
                screen_energy=float(cete_energy),
                screen_cutoff=float(flag_cut),
                garch_params_refit=bool(was_refit),
                manifest_path=str(manifest_path),
                manifest_sha256=manifest_sha,
            )
            print(f"[log] prediction recorded: {prediction_id}")
            print(f"[log] manifest: {manifest_path} "
                  f"(sha256={manifest_sha[:12]}...)")
        except Exception as e:
            print(f"FATAL: could not write prediction log or manifest: "
                  f"{e}", file=sys.stderr)
            sys.exit(EXIT_LOG_FAILED)
    else:
        print("[log] --no-record set; nothing written to log or manifests")
        prediction_id = "(not recorded)"

    # ----------------------------------------------------------------
    # 5. Report
    # ----------------------------------------------------------------
    if is_flagged:
        decision = (
            f"SCREEN FLAGGED (>= {args.flag_percentile:.0f}th pct of "
            f"trailing variance history). Historically ~100% precision "
            f"at this threshold but ~21% recall: treat as a strong "
            f"signal when it fires, not as full coverage."
        )
    else:
        decision = (
            "Screen quiet -- no anomaly signal beyond the scheduled "
            "GARCH forecast."
        )

    as_of = dates[-1] if hasattr(dates[-1], "strftime") else dates[-1]

    print("\n" + "=" * 64)
    print("LIVE MONITOR RESULT")
    print("=" * 64)
    print(f"Prediction ID:            {prediction_id}")
    print(f"Ticker:                   {ticker_key}")
    print(f"As of:                    {as_of}")
    print(f"Data source:              {source}")
    print(f"GARCH(1,1) {args.horizon}-day vol forecast: "
          f"{forecast_vol:.6f}")
    print(f"CETE screen:              "
          f"{'FLAGGED' if is_flagged else 'quiet'} "
          f"(energy={cete_energy:.6f}, "
          f"{args.flag_percentile:.0f}th-pct cutoff={flag_cut:.6f})")
    print(f"Decision:                 {decision}")

    if args.out:
        payload = {
            "prediction_id": prediction_id,
            "ticker": ticker_key,
            "as_of": str(as_of),
            "origin_date": origin_date,
            "data_source": source,
            "n_observations": int(len(returns)),
            "garch_vol_forecast": float(forecast_vol),
            "forecast_horizon_days": int(args.horizon),
            "cete_flagged": is_flagged,
            "cete_energy": float(cete_energy),
            "cete_flag_cutoff": float(flag_cut),
            "flag_percentile_used": float(args.flag_percentile),
            "window": int(args.window),
            "garch_params_refit": bool(was_refit),
            "decision": decision,
        }
        try:
            with open(args.out, "w") as f:
                json.dump(payload, f, indent=2)
            print(f"\nJSON written to {args.out}")
        except OSError as e:
            print(f"WARNING: could not write {args.out}: {e}",
                  file=sys.stderr)

    sys.exit(EXIT_OK)


if __name__ == "__main__":
    main()