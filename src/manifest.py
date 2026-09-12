"""
manifest.py
===========

Content-addressed provenance for forecasts.

The claim this module supports:

    The logged forecast for a prediction_id is numerically
    reproducible from the logged price snapshot and the logged fitted
    GARCH parameters, within the tolerance recorded in the manifest.

It deliberately does NOT support the stronger claim:

    The logged fitted parameters would be reproduced if the GARCH fit
    were run again today.

The second claim is fragile -- numerical optimizers vary across
library versions, BLAS backends, and platforms -- and it is not what
an auditor actually needs. An auditor needs to know that the
prediction is what the recorded inputs and the recorded parameters
produce. Whether the parameters themselves are reproducible is a
separate question, answered by pinning dependency versions in the
manifest and letting a reader attempt a refit if they want to.

------------------------------------------------------------------
DIRECTORY LAYOUT
------------------------------------------------------------------
    data/snapshots/<TICKER>/prices/<sha256>.parquet
    data/snapshots/<TICKER>/garch/<sha256>.json
    data/manifests/<prediction_id>.json

The filename IS the hash. Two predictions that use byte-identical
inputs share the same snapshot file -- no duplication, and the
filename itself is a check: if the bytes hash to something other than
the filename, the file is corrupt.

Manifests are one per prediction and are referenced from the
prediction log by path + hash.

------------------------------------------------------------------
SCHEMA
------------------------------------------------------------------
    {
      "schema_version": 1,
      "prediction_id": "GSPC_2026-09-13_10",
      "ticker": "GSPC",
      "origin_date": "2026-09-13",
      "horizon_days": 10,
      "window_size": 64,

      "price_snapshot": {
        "path": "data/snapshots/GSPC/prices/<sha256>.parquet",
        "sha256": "<sha256>",
        "n_observations": 750,
        "first_date": "2024-09-09",
        "last_date": "2026-09-13"
      },

      "garch_parameters": {
        "path": "data/snapshots/GSPC/garch/<sha256>.json",
        "sha256": "<sha256>",
        "omega": 0.000001234,
        "alpha": 0.087,
        "beta": 0.905,
        "mu": 0.0002,
        "sigma2_last": 0.0000412,
        "eps_last": -0.00314,
        "n_returns_at_fit": 750
      },

      "model_config": {
        "vol": "Garch",
        "p": 1,
        "q": 1,
        "dist": "normal",
        "mean": "Constant"
      },

      "code": {
        "git_sha": "abc123...",
        "git_dirty": false
      },

      "environment": {
        "python": "3.11.7",
        "numpy": "1.26.4",
        "pandas": "2.2.0",
        "arch": "6.3.0",
        "yfinance": "0.2.40"
      },

      "verification": {
        "tolerance_abs": 1e-6,
        "note": "Absolute tolerance on forecast_vol. Recomputed value must satisfy |recomputed - logged| <= tolerance_abs."
      }
    }

------------------------------------------------------------------
USAGE
------------------------------------------------------------------
    from src.manifest import (
        hash_bytes, hash_file, store_snapshot,
        build_manifest, write_manifest, read_manifest,
        verify_snapshot,
    )

    # At prediction time:
    price_path, price_hash = store_snapshot(ticker, kind="prices",
                                            data=price_df, fmt="parquet")
    garch_path, garch_hash = store_snapshot(ticker, kind="garch",
                                            data=params_dict, fmt="json")
    manifest = build_manifest(
        prediction_id=..., ticker=..., origin_date=..., ...,
        price_snapshot=(price_path, price_hash, n_obs, first_date, last_date),
        garch_parameters=(garch_path, garch_hash, params_dict),
        model_config=..., code=..., environment=...,
    )
    write_manifest(manifest)

    # Later, at verification time:
    m = read_manifest(prediction_id)
    verify_snapshot(m["price_snapshot"])   # raises on hash mismatch
    verify_snapshot(m["garch_parameters"]) # raises on hash mismatch
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Optional, Union

import pandas as pd


SCHEMA_VERSION = 1

# Where snapshots and manifests live. Overridable for tests.
_DEFAULT_SNAPSHOT_DIR = Path(os.environ.get(
    "CETE_SNAPSHOT_DIR", "data/snapshots"))
_DEFAULT_MANIFEST_DIR = Path(os.environ.get(
    "CETE_MANIFEST_DIR", "data/manifests"))


def _snapshot_dir() -> Path:
    return Path(os.environ.get("CETE_SNAPSHOT_DIR",
                               str(_DEFAULT_SNAPSHOT_DIR)))


def _manifest_dir() -> Path:
    return Path(os.environ.get("CETE_MANIFEST_DIR",
                               str(_DEFAULT_MANIFEST_DIR)))


def _normalize_ticker(ticker: str) -> str:
    return "".join(
        c for c in ticker.upper().replace("^", "").replace("/", "-")
        if c.isalnum() or c == "-"
    )


# ----------------------------------------------------------------------
# Hashing
# ----------------------------------------------------------------------

def hash_bytes(b: bytes) -> str:
    """SHA-256 of a byte string, hex-encoded."""
    return hashlib.sha256(b).hexdigest()


def hash_file(path: Union[str, Path]) -> str:
    """SHA-256 of a file's bytes, hex-encoded."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


# ----------------------------------------------------------------------
# Snapshot storage (content-addressed)
# ----------------------------------------------------------------------

def _serialize(data, fmt: str) -> bytes:
    """Serialize a snapshot payload to bytes deterministically.

    'parquet' expects a pandas DataFrame.
    'json' expects a dict, and serializes with sort_keys=True so the
    same dict always produces the same bytes.
    """
    if fmt == "parquet":
        if not isinstance(data, pd.DataFrame):
            raise TypeError("parquet snapshot requires a DataFrame")
        import io
        buf = io.BytesIO()
        data.to_parquet(buf, index=False)
        return buf.getvalue()
    elif fmt == "json":
        if not isinstance(data, dict):
            raise TypeError("json snapshot requires a dict")
        return json.dumps(data, sort_keys=True, indent=2).encode("utf-8")
    else:
        raise ValueError(f"unknown snapshot format: {fmt!r}")


def _deserialize(path: Path, fmt: str):
    if fmt == "parquet":
        return pd.read_parquet(path)
    elif fmt == "json":
        return json.loads(path.read_text(encoding="utf-8"))
    else:
        raise ValueError(f"unknown snapshot format: {fmt!r}")


def store_snapshot(ticker: str, kind: str, data, fmt: str) -> tuple[Path, str]:
    """Serialize `data`, hash it, and store it at
    data/snapshots/<TICKER>/<kind>/<sha256>.<ext>.

    Returns (path, sha256). If a file with that hash already exists,
    it is not rewritten -- content-addressed storage is idempotent.

    The filename IS the hash, so if the file on disk ever hashes to
    something else, the file has been corrupted and verification will
    catch it.
    """
    if kind not in ("prices", "garch"):
        raise ValueError(f"unknown snapshot kind: {kind!r}")
    ext = {"parquet": "parquet", "json": "json"}[fmt]
    payload = _serialize(data, fmt)
    sha = hash_bytes(payload)

    d = _snapshot_dir() / _normalize_ticker(ticker) / kind
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{sha}.{ext}"

    if not path.exists():
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_bytes(payload)
        os.replace(tmp, path)

    return path, sha


def verify_snapshot(entry: dict) -> None:
    """Check that a snapshot file still hashes to the value recorded
    in the manifest. Raises ValueError on mismatch, missing file, or
    filename/hash disagreement."""
    path = Path(entry["path"])
    expected = entry["sha256"]

    if not path.exists():
        raise ValueError(f"snapshot missing: {path}")

    actual = hash_file(path)
    if actual != expected:
        raise ValueError(
            f"snapshot hash mismatch for {path}:\n"
            f"  expected {expected}\n"
            f"  actual   {actual}"
        )

    # Also check that the filename matches the hash, since the whole
    # point of content addressing is that the name is the hash.
    if path.stem != expected:
        raise ValueError(
            f"snapshot filename does not match its hash: "
            f"{path.name} vs {expected}"
        )


# ----------------------------------------------------------------------
# Environment / code provenance
# ----------------------------------------------------------------------

def _git_sha() -> tuple[Optional[str], bool]:
    """Return (sha, dirty). sha is None if not in a git repo."""
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            cwd=Path(__file__).resolve().parent.parent,
        ).decode().strip()
        dirty_out = subprocess.check_output(
            ["git", "status", "--porcelain"],
            stderr=subprocess.DEVNULL,
            cwd=Path(__file__).resolve().parent.parent,
        ).decode().strip()
        return sha, bool(dirty_out)
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None, False


def _env_versions() -> dict:
    """Record the versions of the packages the forecast depends on."""
    out = {"python": platform.python_version()}
    for name in ("numpy", "pandas", "arch", "yfinance"):
        try:
            mod = __import__(name)
            out[name] = getattr(mod, "__version__", "unknown")
        except ImportError:
            out[name] = None
    return out


# ----------------------------------------------------------------------
# Manifest assembly
# ----------------------------------------------------------------------

def build_manifest(
    prediction_id: str,
    ticker: str,
    origin_date: str,
    horizon_days: int,
    window_size: int,
    price_snapshot: dict,
    garch_parameters: dict,
    model_config: dict,
    tolerance_abs: float = 1e-6,
) -> dict:
    """Assemble a manifest dict. The caller supplies the two snapshot
    entries (already stored via store_snapshot) and the model config;
    this function adds the code and environment provenance and the
    schema version.
    """
    sha, dirty = _git_sha()
    return {
        "schema_version": SCHEMA_VERSION,
        "prediction_id": prediction_id,
        "ticker": _normalize_ticker(ticker),
        "origin_date": origin_date,
        "horizon_days": int(horizon_days),
        "window_size": int(window_size),
        "price_snapshot": price_snapshot,
        "garch_parameters": garch_parameters,
        "model_config": model_config,
        "code": {
            "git_sha": sha,
            "git_dirty": dirty,
        },
        "environment": _env_versions(),
        "verification": {
            "tolerance_abs": float(tolerance_abs),
            "note": (
                "Absolute tolerance on forecast_vol. Recomputed value "
                "must satisfy |recomputed - logged| <= tolerance_abs."
            ),
        },
    }

def write_manifest(manifest: dict) -> Path:
    pid = manifest["prediction_id"]
    d = _manifest_dir()
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{pid}.json"

    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(manifest, sort_keys=True, indent=2),
                   encoding="utf-8")
    os.replace(tmp, path)
    return path


def read_manifest(prediction_id: str) -> dict:
    """Read a manifest by prediction id. Raises FileNotFoundError if
    missing."""
    path = _manifest_dir() / f"{prediction_id}.json"
    if not path.exists():
        raise FileNotFoundError(f"manifest not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))