"""
predictions.py
==============

Append-only prediction log for the live volatility monitor.

Design constraints, and why each one is load-bearing:

1. APPEND-ONLY. Predictions are written once, at the moment they are
   made, and never edited. A prediction that could be revised after
   the fact is not a prediction. The log is the record.

2. PLAIN TEXT, ONE FILE PER TICKER. JSONL (one JSON object per line).
   Binary formats (SQLite, pickle) are not diffable and not auditable.
   With JSONL, `git log -p data/predictions/GSPC.jsonl` shows every
   prediction ever made, in order, unmodified.

3. TARGET IS NOT STORED AT PREDICTION TIME. The realized volatility
   the forecast will be scored against is unknown when the forecast
   is made. It is filled in later, by score_outcomes.py, and appended
   as a SEPARATE line with the same prediction_id.

4. PREDICTION ID IS DETERMINISTIC. Format:
       {ticker}_{origin_date}_{horizon}
   e.g. "GSPC_2024-09-13_10". Duplicate inserts are detectable
   without a database; record_prediction is idempotent under re-runs.

5. PROVENANCE IS OPTIONAL BUT RECORDED WHEN AVAILABLE. Two fields,
   manifest_path and manifest_sha256, point to the provenance record
   for this prediction (see src/manifest.py). A prediction without a
   manifest is not auditable -- scripts/reproduce.py will refuse to
   verify it -- but the fields are optional so that callers that do
   not have a manifest can still write to the log.

6. NO DEPENDENCY ON src.data OR src.regime. This module is pure I/O.
"""

from __future__ import annotations

import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd


# ----------------------------------------------------------------------
# Storage location
# ----------------------------------------------------------------------

_DEFAULT_DIR = Path(os.environ.get(
    "CETE_PREDICTIONS_DIR", "data/predictions"))


def _predictions_dir() -> Path:
    return Path(os.environ.get("CETE_PREDICTIONS_DIR", str(_DEFAULT_DIR)))


def _normalize_ticker(ticker: str) -> str:
    return "".join(
        c for c in ticker.upper().replace("^", "").replace("/", "-")
        if c.isalnum() or c == "-"
    )


def _log_path(ticker: str) -> Path:
    d = _predictions_dir()
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{_normalize_ticker(ticker)}.jsonl"


# ----------------------------------------------------------------------
# Prediction ID
# ----------------------------------------------------------------------

def make_prediction_id(ticker: str, origin_date: str, horizon_days: int) -> str:
    """Deterministic ID: {NORMALIZED_TICKER}_{YYYY-MM-DD}_{horizon}."""
    if horizon_days <= 0:
        raise ValueError(f"horizon_days must be positive, got {horizon_days}")
    return f"{_normalize_ticker(ticker)}_{origin_date}_{int(horizon_days)}"


# ----------------------------------------------------------------------
# Low-level I/O
# ----------------------------------------------------------------------

def _read_all_lines(ticker: str) -> list[dict]:
    """Read every line of the ticker's log. Tolerates corrupt lines by
    representing them as {"_corrupt": True, "_raw": ...}."""
    path = _log_path(ticker)
    if not path.exists():
        return []

    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                out.append({"_corrupt": True, "_raw": line[:200]})
    return out


def _append_line(ticker: str, obj: dict) -> None:
    path = _log_path(ticker)
    line = json.dumps(obj, sort_keys=True) + "\n"
    with open(path, "a", encoding="utf-8") as f:
        f.write(line)


# ----------------------------------------------------------------------
# Write side
# ----------------------------------------------------------------------

def record_prediction(
    ticker: str,
    origin_date: str,
    horizon_days: int,
    forecast_vol: float,
    screen_flagged: bool,
    screen_energy: float,
    screen_cutoff: float,
    garch_params_refit: bool = False,
    model: str = "GARCH(1,1)",
    manifest_path: Optional[str] = None,
    manifest_sha256: Optional[str] = None,
    extra: Optional[dict] = None,
) -> str:
    """Append a prediction to the log. Returns the prediction_id.

    Idempotent: if a prediction with this ID already exists (resolved
    or unresolved), the call is a no-op and returns the existing ID.

    `manifest_path` and `manifest_sha256` point to the provenance
    record for this prediction (see src/manifest.py). They are
    optional so older callers keep working, but a prediction without
    a manifest is NOT auditable -- scripts/reproduce.py will refuse
    to verify it.

    Raises ValueError if forecast_vol is not finite and non-negative,
    or if exactly one of manifest_path / manifest_sha256 is provided.
    """
    if not math.isfinite(forecast_vol) or forecast_vol < 0:
        raise ValueError(
            f"forecast_vol must be finite and non-negative, "
            f"got {forecast_vol!r}"
        )

    if (manifest_path is None) != (manifest_sha256 is None):
        raise ValueError(
            "manifest_path and manifest_sha256 must be provided "
            "together or not at all"
        )

    prediction_id = make_prediction_id(ticker, origin_date, horizon_days)

    existing = _read_all_lines(ticker)
    for row in existing:
        if row.get("prediction_id") == prediction_id:
            return prediction_id

    entry = {
        "prediction_id": prediction_id,
        "ticker": _normalize_ticker(ticker),
        "origin_date": origin_date,
        "horizon_days": int(horizon_days),
        "forecast_vol": float(forecast_vol),
        "screen_flagged": bool(screen_flagged),
        "screen_energy": float(screen_energy),
        "screen_cutoff": float(screen_cutoff),
        "garch_params_refit": bool(garch_params_refit),
        "model": str(model),
        "recorded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    if manifest_path is not None:
        entry["manifest_path"] = str(manifest_path)
        entry["manifest_sha256"] = str(manifest_sha256)
    if extra:
        for k, v in extra.items():
            if k in entry:
                raise ValueError(
                    f"extra key {k!r} collides with a required field"
                )
            entry[k] = v

    _append_line(ticker, entry)
    return prediction_id


def record_resolution(
    ticker: str,
    origin_date: str,
    horizon_days: int,
    target_vol: float,
) -> str:
    """Append a resolution for a previously-recorded prediction.

    Raises ValueError if no prediction with this ID exists, or if it
    has already been resolved.
    """
    if not math.isfinite(target_vol) or target_vol < 0:
        raise ValueError(
            f"target_vol must be finite and non-negative, "
            f"got {target_vol!r}"
        )

    prediction_id = make_prediction_id(ticker, origin_date, horizon_days)

    existing = _read_all_lines(ticker)
    has_prediction = False
    already_resolved = False
    forecast_vol = None

    for row in existing:
        if row.get("prediction_id") != prediction_id:
            continue
        if row.get("resolved") is True:
            already_resolved = True
        else:
            has_prediction = True
            forecast_vol = row.get("forecast_vol")

    if already_resolved:
        raise ValueError(
            f"Prediction {prediction_id} is already resolved."
        )
    if not has_prediction:
        raise ValueError(
            f"No prediction {prediction_id} in the log to resolve."
        )

    error = float(forecast_vol) - float(target_vol)
    entry = {
        "prediction_id": prediction_id,
        "resolved": True,
        "target_vol": float(target_vol),
        "error": error,
        "abs_error": abs(error),
        "squared_error": error * error,
        "resolved_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    _append_line(ticker, entry)
    return prediction_id


# ----------------------------------------------------------------------
# Read side
# ----------------------------------------------------------------------

def read_predictions(ticker: str) -> pd.DataFrame:
    """Every line of the log as a DataFrame, with a `row_type` column
    in {'prediction', 'resolution', 'corrupt'}."""
    rows = _read_all_lines(ticker)
    if not rows:
        return pd.DataFrame()

    for r in rows:
        if r.get("_corrupt"):
            r["row_type"] = "corrupt"
        elif r.get("resolved") is True:
            r["row_type"] = "resolution"
        else:
            r["row_type"] = "prediction"

    return pd.DataFrame(rows)


def read_pending(ticker: str) -> pd.DataFrame:
    """Predictions recorded but not yet resolved, sorted by
    origin_date ascending."""
    df = read_predictions(ticker)
    if df.empty:
        return df

    preds = df[df["row_type"] == "prediction"].copy()
    resolutions = df[df["row_type"] == "resolution"]

    resolved_ids = (set(resolutions["prediction_id"])
                    if not resolutions.empty else set())
    pending = preds[~preds["prediction_id"].isin(resolved_ids)].copy()

    if "origin_date" in pending.columns:
        pending = pending.sort_values("origin_date").reset_index(drop=True)
    return pending


def read_resolved(ticker: str) -> pd.DataFrame:
    """Predictions joined to their resolutions."""
    df = read_predictions(ticker)
    if df.empty:
        return df

    preds = df[df["row_type"] == "prediction"].copy()
    resolutions = df[df["row_type"] == "resolution"].copy()

    if resolutions.empty:
        return pd.DataFrame()

    resolution_only = [
        "resolved", "target_vol", "error",
        "abs_error", "squared_error", "resolved_at",
    ]
    preds = preds.drop(
        columns=[c for c in resolution_only if c in preds.columns])

    resolutions = resolutions.drop_duplicates(
        subset="prediction_id", keep="first")

    merged = preds.merge(
        resolutions[[
            "prediction_id", "target_vol", "error",
            "abs_error", "squared_error", "resolved_at",
        ]],
        on="prediction_id",
        how="inner",
    )
    if "origin_date" in merged.columns:
        merged = merged.sort_values("origin_date").reset_index(drop=True)
    return merged


def latest_for_ticker(ticker: str) -> Optional[dict]:
    """Most recent prediction (resolved or not), or None if empty."""
    df = read_predictions(ticker)
    if df.empty:
        return None
    preds = df[df["row_type"] == "prediction"]
    if preds.empty:
        return None
    preds = preds.sort_values("origin_date")
    return preds.iloc[-1].to_dict()


# ----------------------------------------------------------------------
# Reporting helper
# ----------------------------------------------------------------------

def accuracy_summary(ticker: str) -> dict:
    """Rolling accuracy stats from resolved predictions only."""
    import numpy as np

    pending = read_pending(ticker)
    resolved = read_resolved(ticker)

    out = {
        "ticker": _normalize_ticker(ticker),
        "n_pending": int(len(pending)),
        "n_resolved": int(len(resolved)),
        "mean_abs_error": float("nan"),
        "rmse": float("nan"),
        "mean_error": float("nan"),
        "pearson": float("nan"),
        "first_origin": None,
        "last_origin": None,
    }

    if resolved.empty:
        return out

    out["mean_abs_error"] = float(resolved["abs_error"].mean())
    out["rmse"] = float(np.sqrt(resolved["squared_error"].mean()))
    out["mean_error"] = float(resolved["error"].mean())

    if len(resolved) >= 2:
        f = resolved["forecast_vol"].to_numpy(dtype=float)
        t = resolved["target_vol"].to_numpy(dtype=float)
        if np.std(f) > 0 and np.std(t) > 0:
            out["pearson"] = float(np.corrcoef(f, t)[0, 1])

    out["first_origin"] = str(resolved["origin_date"].iloc[0])
    out["last_origin"] = str(resolved["origin_date"].iloc[-1])
    return out