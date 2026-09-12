"""
reproduce.py
============

Verify a logged prediction against its provenance manifest.

Usage:
    python scripts/reproduce.py GSPC_2026-09-11_10
    python scripts/reproduce.py GSPC_2026-09-11_10 --verbose
    python scripts/reproduce.py --latest GSPC
    python scripts/reproduce.py --all GSPC

What it does:

  1. Reads the prediction from the log.
  2. Reads its manifest.
  3. Verifies that the manifest file on disk still hashes to the
     value recorded in the log.
  4. Verifies that the two snapshots (price series, GARCH parameters)
     still hash to the values recorded in the manifest.
  5. Loads the frozen GARCH parameters and the price snapshot.
  6. Recomputes the forecast from those inputs.
  7. Compares the recomputed forecast to the logged forecast, within
     the tolerance recorded in the manifest.

Exit codes:
    0  VERIFIED  -- all checks passed, forecast reproduces
    1  NOT VERIFIED -- at least one check failed
    2  CANNOT VERIFY -- a required file (log entry, manifest, or
                        snapshot) is missing; provenance is incomplete

The distinction matters. A prediction logged before the manifest
system existed is CANNOT VERIFY, not NOT VERIFIED. The former means
"the evidence was never recorded"; the latter means "the recorded
evidence does not support the claim."

------------------------------------------------------------------
WHAT "VERIFIED" MEANS
------------------------------------------------------------------
It means: given the recorded snapshot bytes and the recorded fitted
parameters, the recorded forecast is what the model produces, within
the recorded tolerance.

It does NOT mean: the fit would produce the same parameters if re-run
today. That is a separate and much stronger claim, deliberately not
made (see src/manifest.py).

If the forecast matches but the snapshots are missing, the result is
CANNOT VERIFY -- a numerical match against unverifiable inputs proves
nothing.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.manifest import (
    hash_file, read_manifest, verify_snapshot,
)
from src.predictions import read_predictions
from src.garch_filter import multistep_forecast


EXIT_VERIFIED = 0
EXIT_NOT_VERIFIED = 1
EXIT_CANNOT_VERIFY = 2


# ----------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------

def _load_prediction(ticker: str, prediction_id: str) -> dict:
    """Find the prediction row in the log. Raises SystemExit(2) if not
    found or if the log line does not carry manifest provenance."""
    df = read_predictions(ticker)
    if df.empty:
        print(f"FATAL: no log entries for {ticker}", file=sys.stderr)
        sys.exit(EXIT_CANNOT_VERIFY)

    rows = df[
        (df["row_type"] == "prediction")
        & (df["prediction_id"] == prediction_id)
    ]
    if rows.empty:
        print(f"FATAL: prediction {prediction_id} not found in the "
              f"{ticker} log", file=sys.stderr)
        sys.exit(EXIT_CANNOT_VERIFY)

    pred = rows.iloc[0].to_dict()

    if not pred.get("manifest_path") or not pred.get("manifest_sha256"):
        print(f"CANNOT VERIFY: prediction {prediction_id} has no "
              f"manifest. It was logged before provenance was recorded, "
              f"or it was written by a caller that did not supply one.",
              file=sys.stderr)
        sys.exit(EXIT_CANNOT_VERIFY)

    return pred


def _load_manifest(pred: dict) -> dict:
    """Read the manifest and verify its own hash against the log entry."""
    manifest_path = Path(pred["manifest_path"])
    if not manifest_path.exists():
        print(f"CANNOT VERIFY: manifest file missing: {manifest_path}",
              file=sys.stderr)
        sys.exit(EXIT_CANNOT_VERIFY)

    actual = hash_file(manifest_path)
    if actual != pred["manifest_sha256"]:
        print(f"NOT VERIFIED: manifest file has been modified since the "
              f"prediction was logged.", file=sys.stderr)
        print(f"  logged  sha256: {pred['manifest_sha256']}",
              file=sys.stderr)
        print(f"  actual  sha256: {actual}", file=sys.stderr)
        sys.exit(EXIT_NOT_VERIFIED)

    return read_manifest(pred["prediction_id"])


# ----------------------------------------------------------------------
# Recompute
# ----------------------------------------------------------------------

def _recompute_forecast(manifest: dict) -> float:
    """Recompute the forecast from the manifest's recorded snapshots.

    Loads the price snapshot (for record-keeping; the forecast does not
    actually need the historical prices because the GARCH parameters
    are frozen), loads the frozen parameters, and runs the multi-step
    forecast.

    The price snapshot is loaded and its hash verified by the caller;
    it is not passed to multistep_forecast because the frozen
    parameters fully determine the forecast path. That is the point
    of freezing parameters: reproduction does not depend on re-fitting,
    and therefore does not depend on the optimizer.
    """
    garch_entry = manifest["garch_parameters"]
    params_path = Path(garch_entry["path"])
    params = json.loads(params_path.read_text(encoding="utf-8"))

    # multistep_forecast expects a dict with mu/omega/alpha/beta/
    # sigma2_last/eps_last. The snapshot stores exactly that.
    return float(multistep_forecast(params, horizon=manifest["horizon_days"]))


# ----------------------------------------------------------------------
# Report
# ----------------------------------------------------------------------

def _report(pred: dict, manifest: dict, recomputed: float,
            verbose: bool) -> int:
    logged = float(pred["forecast_vol"])
    tolerance = float(manifest["verification"]["tolerance_abs"])
    diff = abs(recomputed - logged)

    print()
    print("=" * 64)
    print("PREDICTION VERIFICATION")
    print("=" * 64)
    print()
    print(f"Prediction:             {pred['prediction_id']}")
    print(f"Ticker:                 {pred['ticker']}")
    print(f"Origin date:            {pred['origin_date']}")
    print(f"Horizon:                {manifest['horizon_days']} days")
    print()
    print(f"Logged forecast:        {logged:.10f}")
    print(f"Recomputed forecast:    {recomputed:.10f}")
    print()
    print(f"Absolute difference:    {diff:.3e}")
    print(f"Tolerance:              {tolerance:.3e}")
    print()

    if verbose:
        print("PROVENANCE")
        print("-" * 64)
        ps = manifest["price_snapshot"]
        gp = manifest["garch_parameters"]
        print(f"  Price snapshot:       {ps['path']}")
        print(f"    sha256              {ps['sha256']}")
        print(f"    observations        {ps.get('n_observations', '--')}")
        print(f"    range               {ps.get('first_date', '--')} "
              f".. {ps.get('last_date', '--')}")
        print(f"  GARCH parameters:     {gp['path']}")
        print(f"    sha256              {gp['sha256']}")
        print(f"    omega               {gp.get('omega', '--')}")
        print(f"    alpha               {gp.get('alpha', '--')}")
        print(f"    beta                {gp.get('beta', '--')}")
        print()
        print("ENVIRONMENT")
        print("-" * 64)
        env = manifest.get("environment", {})
        for k in ("python", "numpy", "pandas", "arch", "yfinance"):
            print(f"  {k:<20} {env.get(k, '--')}")
        print()
        print("CODE")
        print("-" * 64)
        code = manifest.get("code", {})
        print(f"  git_sha               {code.get('git_sha', '--')}")
        print(f"  code_dirty             {code.get('code_dirty', '--')}")
        print()

    print("=" * 64)
    if diff <= tolerance:
        print("RESULT: VERIFIED")
        print("=" * 64)
        print()
        print("The logged forecast is numerically reproducible from the "
              "recorded snapshot and the recorded fitted parameters, "
              "within the declared tolerance.")
        print()
        return EXIT_VERIFIED
    else:
        print("RESULT: NOT VERIFIED")
        print("=" * 64)
        print()
        print(f"Recomputed forecast differs from the logged value by "
              f"{diff:.3e}, which exceeds the tolerance of "
              f"{tolerance:.3e}.")
        print()
        print("This means the recorded inputs and parameters do not "
              "produce the recorded forecast. Either the log entry, "
              "the manifest, or one of the snapshots has been modified.")
        print()
        return EXIT_NOT_VERIFIED


# ----------------------------------------------------------------------
# Per-prediction driver
# ----------------------------------------------------------------------

def verify_one(ticker: str, prediction_id: str, verbose: bool) -> int:
    pred = _load_prediction(ticker, prediction_id)
    manifest = _load_manifest(pred)

    # Verify both snapshots exist and still hash to their recorded
    # values. A missing or mismatched snapshot means CANNOT VERIFY --
    # we may still be able to compute a number, but the inputs are not
    # what the manifest says they are.
    try:
        verify_snapshot(manifest["price_snapshot"])
    except ValueError as e:
        print(f"NOT VERIFIED: price snapshot failed: {e}", file=sys.stderr)
        return EXIT_NOT_VERIFIED

    try:
        verify_snapshot(manifest["garch_parameters"])
    except ValueError as e:
        print(f"NOT VERIFIED: garch snapshot failed: {e}", file=sys.stderr)
        return EXIT_NOT_VERIFIED

    try:
        recomputed = _recompute_forecast(manifest)
    except Exception as e:
        print(f"NOT VERIFIED: could not recompute forecast: {e}",
              file=sys.stderr)
        return EXIT_NOT_VERIFIED

    return _report(pred, manifest, recomputed, verbose)


# ----------------------------------------------------------------------
# Ticker-level helpers
# ----------------------------------------------------------------------

def _latest_with_manifest(ticker: str) -> str | None:
    df = read_predictions(ticker)
    if df.empty:
        return None
    preds = df[df["row_type"] == "prediction"]
    with_manifest = preds[
        preds["manifest_path"].notna()
        if "manifest_path" in preds.columns else []
    ]
    if with_manifest.empty:
        return None
    with_manifest = with_manifest.sort_values("origin_date")
    return str(with_manifest.iloc[-1]["prediction_id"])


def _all_with_manifest(ticker: str) -> list[str]:
    df = read_predictions(ticker)
    if df.empty:
        return []
    preds = df[df["row_type"] == "prediction"]
    if "manifest_path" not in preds.columns:
        return []
    with_manifest = preds[preds["manifest_path"].notna()]
    with_manifest = with_manifest.sort_values("origin_date")
    return list(with_manifest["prediction_id"].astype(str))


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Verify a logged prediction against its manifest.")
    parser.add_argument("prediction_id", nargs="?",
                        help="Prediction id, e.g. GSPC_2026-09-11_10")
    parser.add_argument("--ticker", default=None,
                        help="Ticker (inferred from prediction_id if "
                             "not given)")
    parser.add_argument("--latest", metavar="TICKER",
                        help="Verify the most recent prediction for "
                             "this ticker")
    parser.add_argument("--all", metavar="TICKER",
                        help="Verify every prediction for this ticker "
                             "that has a manifest")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Print full provenance details")
    args = parser.parse_args()

    if args.all:
        ids = _all_with_manifest(args.all)
        if not ids:
            print(f"No predictions with manifests for {args.all}.",
                  file=sys.stderr)
            sys.exit(EXIT_CANNOT_VERIFY)
        codes = []
        for pid in ids:
            print(f"\n### {pid}")
            codes.append(verify_one(args.all, pid, args.verbose))
        print()
        print("=" * 64)
        print(f"SUMMARY: {sum(c == 0 for c in codes)}/{len(codes)} verified")
        print("=" * 64)
        sys.exit(EXIT_VERIFIED if all(c == 0 for c in codes)
                 else EXIT_NOT_VERIFIED)

    if args.latest:
        pid = _latest_with_manifest(args.latest)
        if pid is None:
            print(f"No predictions with manifests for {args.latest}.",
                  file=sys.stderr)
            sys.exit(EXIT_CANNOT_VERIFY)
        sys.exit(verify_one(args.latest, pid, args.verbose))

    if not args.prediction_id:
        parser.error("provide a prediction_id, or --latest/--all TICKER")

    # Infer the ticker from the prediction_id if not given:
    # format is TICKER_YYYY-MM-DD_HH, so the ticker is everything up to
    # the last two underscore-separated tokens.
    ticker = args.ticker
    if ticker is None:
        parts = args.prediction_id.rsplit("_", 2)
        if len(parts) < 3:
            parser.error(
                f"could not infer ticker from {args.prediction_id!r}; "
                f"pass --ticker explicitly")
        ticker = parts[0]

    sys.exit(verify_one(ticker, args.prediction_id, args.verbose))


if __name__ == "__main__":
    main()