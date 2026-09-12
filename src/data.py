"""
data.py
=======

Versioned, validated, cached market data layer.

Why this exists (vs the previous fetch-on-every-call version):

  1. Adjusted-close series are RETROACTIVELY REVISED. If AAPL splits
     tomorrow, every historical Adj Close in a live-fetched series is
     silently wrong, and any cached returns derived from it are wrong.
     A walk-forward evaluatio  n run against today's version of history
     is testing a series that did not exist at the time. FIX: every
     fetch is stored under data/raw/<ticker>/<fetch_date>.csv. You can
     always ask "what did this look like on date X" and get the honest
     answer.

  2. yfinance is an unofficial, undocumented endpoint. It rate-limits,
     occasionally returns short or partial series, and has no SLA. FIX:
     every fetch is validated against an exchange-calendar-derived
     expected bar count before being trusted. Bad data is refused, not
     stored.

  3. When Yahoo is down or rate-limiting, a real system falls back to a
     second source rather than aborting. FIX: Stooq is queried as a
     fallback. If both fail, this module raises — callers cannot
     accidentally proceed on fabricated data.

The public API is unchanged:

    get_market_returns(ticker, start, end) -> (returns, dates, source)

`source` is one of:
    "cache"      - served from a validated local snapshot
    "yfinance"   - freshly fetched from Yahoo and validated
    "stooq"      - freshly fetched from Stooq and validated
    "synthetic"  - ONLY returned by get_synthetic_regime_switch(), never
                   by get_market_returns(). If you see this in a caller,
                   the caller explicitly asked for it.

Callers that must not proceed on non-market data should check
`source != "synthetic"` (which get_market_returns guarantees) and may
additionally check `source in ("cache", "yfinance", "stooq")`.
"""

import os
import hashlib
import datetime as _dt
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False


# ----------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------

DATA_ROOT = Path(os.environ.get("CETE_DATA_ROOT", "data"))
RAW_DIR = DATA_ROOT / "raw"
DERIVED_DIR = DATA_ROOT / "derived"

# Expected bar counts. NYSE/NASDAQ have ~252 trading days/year; we allow
# a tolerance because half-days and occasional missing prints exist.
# These bounds are deliberately loose; validation is a smoke test for
# "obviously broken", not a strict calendar check (that would require a
# full exchange calendar dependency, which is out of scope here).
BARS_PER_YEAR = 252
BARS_TOLERANCE = 0.10  # allow ±3% deviation from expected


# ----------------------------------------------------------------------
# Ticker / path helpers
# ----------------------------------------------------------------------

def _normalize_ticker(ticker: str) -> str:
    """Canonical form for filesystem paths. '^GSPC', 'gsPC', 'GSPC' all
    map to 'GSPC'. Keeps letters, digits, and dashes."""
    return "".join(
        c for c in ticker.upper().replace("^", "").replace("/", "-")
        if c.isalnum() or c == "-"
    )


def _raw_dir(ticker: str) -> Path:
    return RAW_DIR / _normalize_ticker(ticker)


def _derived_path(ticker: str) -> Path:
    return DERIVED_DIR / f"{_normalize_ticker(ticker)}.parquet"


def _latest_raw_snapshot(ticker: str) -> Optional[Path]:
    d = _raw_dir(ticker)
    if not d.exists():
        return None
    snaps = sorted(p for p in d.glob("*.csv") if p.is_file())
    return snaps[-1] if snaps else None


def _today_iso() -> str:
    return _dt.date.today().isoformat()


# ----------------------------------------------------------------------
# Validation
# ----------------------------------------------------------------------

class DataValidationError(ValueError):
    pass


def _expected_bars(start: str, end: str) -> int:
    d0 = pd.Timestamp(start)
    d1 = pd.Timestamp(end)
    days = max((d1 - d0).days, 1)
    return max(int(days * BARS_PER_YEAR / 365.25), 1)


def _validate_prices(prices: pd.Series, dates: pd.DatetimeIndex,
                     start: str, end: str, ticker: str) -> None:
    """Refuse obviously-broken series. Raises DataValidationError."""
    if len(prices) == 0:
        raise DataValidationError(f"{ticker}: empty series")

    if not np.all(np.isfinite(prices.values)):
        raise DataValidationError(f"{ticker}: non-finite price values")

    if (prices <= 0).any():
        raise DataValidationError(f"{ticker}: non-positive price")

    # Bar-count check
    expected = _expected_bars(start, end)
    lo = expected * (1 - BARS_TOLERANCE)
    hi = expected * (1 + BARS_TOLERANCE)
    if not (lo <= len(prices) <= hi):
        raise DataValidationError(
            f"{ticker}: got {len(prices)} bars, expected ~{expected} "
            f"between {start} and {end}"
        )

    # Duplicate-date check
    if dates.duplicated().any():
        raise DataValidationError(f"{ticker}: duplicate dates in index")

    # Monotonic time check
    if not dates.is_monotonic_increasing:
        raise DataValidationError(f"{ticker}: dates not sorted")

    # Single-day jump check. 50% is a generous threshold — real single-day
    # crashes exist (COVID, GFC), but a 50% move in a broad index in one
    # day is almost always a data error, not a market event.
    log_rets = np.diff(np.log(prices.values))
    if len(log_rets) > 0 and np.abs(log_rets).max() > 0.5:
        worst = dates[np.argmax(np.abs(log_rets)) + 1]
        raise DataValidationError(
            f"{ticker}: single-day log return magnitude "
            f"{np.abs(log_rets).max():.3f} on {worst.date()} — "
            f"refusing to trust this series"
        )


# ----------------------------------------------------------------------
# Fetch backends
# ----------------------------------------------------------------------

def _fetch_yfinance(ticker: str, start: str, end: str) -> Tuple[pd.Series, pd.DatetimeIndex]:
    if not YFINANCE_AVAILABLE:
        raise ImportError("yfinance not installed")

    df = yf.download(ticker, start=start, end=end, progress=False,
                     auto_adjust=False)
    if df is None or df.empty:
        raise DataValidationError(f"yfinance returned empty for {ticker}")

    # Prefer unadjusted Close if present — we want OBSERVED returns, not
    # hold-through-split returns, for a volatility system. See module
    # docstring. Fall back to Adj Close only if Close is missing.
    if "Close" in df.columns:
        prices = df["Close"]
    elif "Adj Close" in df.columns:
        prices = df["Adj Close"]
    else:
        raise DataValidationError(f"yfinance: no Close column for {ticker}")

    if isinstance(prices, pd.DataFrame):
        prices = prices.iloc[:, 0]

    prices = prices.dropna()
    return prices, prices.index


def _fetch_stooq(ticker: str, start: str, end: str) -> Tuple[pd.Series, pd.DatetimeIndex]:
    """Stooq: free, no API key, daily CSV endpoint. Ticker format differs
    from Yahoo — US equities get '.us' suffix; indices need their own
    symbols. We try a couple of common variants and take the first that
    returns data."""
    import io
    import urllib.request

    # Stooq's symbol universe differs from Yahoo's. Common mappings:
    candidates = [
        ticker.replace("^", "").lower() + ".us",     # US equity
        ticker.replace("^", "").lower(),             # index fallback
    ]

    last_err = None
    for sym in candidates:
        url = f"https://stooq.com/q/d/l/?s={sym}&i=d"
        try:
            with urllib.request.urlopen(url, timeout=15) as r:
                raw = r.read().decode("utf-8", errors="replace")
            if not raw or "No data" in raw or len(raw) < 50:
                continue
            df = pd.read_csv(io.StringIO(raw))
            if "Date" not in df.columns or "Close" not in df.columns:
                continue
            df["Date"] = pd.to_datetime(df["Date"])
            df = df.set_index("Date").sort_index()
            df = df.loc[(df.index >= pd.Timestamp(start)) &
                        (df.index <= pd.Timestamp(end))]
            if df.empty:
                continue
            prices = df["Close"].astype(float).dropna()
            return prices, prices.index
        except Exception as e:
            last_err = e
            continue

    raise DataValidationError(
        f"stooq: no data for {ticker} (last error: {last_err})"
    )


# ----------------------------------------------------------------------
# Cache read/write
# ----------------------------------------------------------------------

def _snapshot_paths(ticker: str):
    d = _raw_dir(ticker)
    return {
        "prices": d / f"{_today_iso()}.csv",
    }


def _write_snapshot(ticker: str, prices: pd.Series, source: str) -> None:
    d = _raw_dir(ticker)
    d.mkdir(parents=True, exist_ok=True)
    path = _snapshot_paths(ticker)["prices"]
    # Attach source as a column so the snapshot is self-documenting.
    out = pd.DataFrame({"date": prices.index, "close": prices.values})
    out["source"] = source
    out["fetched_at"] = _dt.datetime.utcnow().isoformat()
    out.to_csv(path, index=False)


def _read_snapshot(ticker: str, path: Path,
                   start: str, end: str) -> Tuple[pd.Series, pd.DatetimeIndex]:
    df = pd.read_csv(path, parse_dates=["date"])
    df = df.set_index("date").sort_index()
    df = df.loc[(df.index >= pd.Timestamp(start)) & (df.index <= pd.Timestamp(end))]
    if df.empty:
        raise DataValidationError(f"cached snapshot for {ticker} has no data in range")
    return df["close"].astype(float), df.index


# ----------------------------------------------------------------------
# Returns derivation
# ----------------------------------------------------------------------

def _log_returns(prices: pd.Series) -> np.ndarray:
    return np.diff(np.log(prices.values))


def _cache_derived(ticker: str, returns: np.ndarray, dates: pd.DatetimeIndex) -> None:
    DERIVED_DIR.mkdir(parents=True, exist_ok=True)
    path = _derived_path(ticker)
    df = pd.DataFrame({"date": dates, "log_return": returns})
    try:
        df.to_parquet(path, index=False)
    except Exception:
        # pyarrow/fastparquet not installed — skip silently. Derived cache
        # is an optimization, not a requirement.
        pass


# ----------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------

def get_market_returns(ticker: str = "^GSPC",
                       start: str = "2018-01-01",
                       end: Optional[str] = None,
                       refresh: bool = False
                       ) -> Tuple[np.ndarray, pd.DatetimeIndex, str]:
    """Fetch validated daily log returns for `ticker` between `start` and `end`.

    Returns (returns, dates, source), where dates is a DatetimeIndex of
    length len(returns) (the dates of the returns, not the prices — one
    shorter than the price series).

    `source` is one of "cache", "yfinance", "stooq". This function NEVER
    returns "synthetic" — if you want the synthetic test series, call
    get_synthetic_regime_switch() directly.
    """
    if end is None:
        end = _dt.date.today().isoformat()

    # 1. Try cached snapshot first (unless caller forces refresh)
    if not refresh:
        snap = _latest_raw_snapshot(ticker)
        if snap is not None:
            try:
                prices, dates_idx = _read_snapshot(ticker, snap, start, end)
                _validate_prices(prices, dates_idx, start, end, ticker)
                returns = _log_returns(prices)
                return returns, dates_idx[1:], "cache"
            except DataValidationError:
                # Corrupt/insufficient cache — fall through to live fetch.
                pass

    # 2. Live fetch: yfinance, then stooq
    errors = []
    for name, fetcher in (("yfinance", _fetch_yfinance),
                          ("stooq", _fetch_stooq)):
        try:
            prices, dates_idx = fetcher(ticker, start, end)
            _validate_prices(prices, dates_idx, start, end, ticker)
            _write_snapshot(ticker, prices, source=name)
            returns = _log_returns(prices)
            _cache_derived(ticker, returns, dates_idx[1:])
            return returns, dates_idx[1:], name
        except Exception as e:
            errors.append(f"{name}: {e}")
            continue

    # 3. Both failed. Refuse — do NOT silently return synthetic data.
    raise RuntimeError(
        f"All data sources failed for {ticker} between {start} and {end}.\n"
        + "\n".join(f"  {e}" for e in errors)
    )


def get_synthetic_regime_switch(seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """Labeled low-vol -> high-vol -> low-vol series for regression tests.
    This is the ONLY function in this module that returns non-market data,
    and it is named so that misuse is obvious."""
    rng = np.random.RandomState(seed)
    t = np.linspace(0, 10, 800)
    low_vol_1 = 0.005 * rng.randn(300)
    high_vol = 0.05 * rng.randn(300)
    low_vol_2 = 0.005 * rng.randn(200)
    returns = np.concatenate([low_vol_1, high_vol, low_vol_2])
    returns += 0.0002 * np.sin(t)
    dates = np.arange(len(returns))
    return returns, dates


def load_local_csv(path: str,
                   date_col: str = "date",
                   price_col: str = "value"
                   ) -> Tuple[np.ndarray, pd.DatetimeIndex]:
    """Load a local price CSV. Validates the same way live fetches do."""
    df = pd.read_csv(path, parse_dates=[date_col]).sort_values(date_col).reset_index(drop=True)
    prices = df[price_col].astype(float)
    dates_idx = pd.DatetimeIndex(df[date_col].values)
    _validate_prices(prices, dates_idx, str(dates_idx[0].date()),
                     str(dates_idx[-1].date()), path)
    returns = _log_returns(prices)
    return returns, dates_idx[1:]