"""
test_data.py
============

Tests for the versioned, validated data layer in src/data.py.

These tests are written to catch the specific failure modes the module
was designed to prevent:

  - silent fallback to fabricated data on fetch failure
  - trusting a short/partial series from yfinance
  - re-fetching adjusted-close history instead of serving the cached
    point-in-time snapshot
  - corrupting the cache when a fetch fails validation
  - accepting obviously-broken prices (zeros, negatives, 50%+ jumps,
    duplicate dates, unsorted index)

Network-dependent tests are marked with @pytest.mark.network and skipped
by default. Run them explicitly with:

    pytest tests/test_data.py -m network

The offline tests (validation logic, cache round-trip) run everywhere.
"""

import os
import sys
from unittest import mock

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import src.data as data_mod
from src.data import (
    DataValidationError,
    _normalize_ticker,
    _validate_prices,
    get_market_returns,
    get_synthetic_regime_switch,
    load_local_csv,
)


# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------

@pytest.fixture
def tmp_data_root(monkeypatch, tmp_path):
    """Redirect the module's DATA_ROOT to a temp directory for the
    duration of a test, so tests don't write to the real data/ dir."""
    monkeypatch.setattr(data_mod, "DATA_ROOT", tmp_path)
    monkeypatch.setattr(data_mod, "RAW_DIR", tmp_path / "raw")
    monkeypatch.setattr(data_mod, "DERIVED_DIR", tmp_path / "derived")
    return tmp_path


def _make_clean_prices(start="2018-01-01", end="2020-01-01",
                       seed=0, extra_bars=0, n=None):
    """Well-formed synthetic price series: business-day index matching
    `start`..`end`, positive prices, small daily moves, no duplicates.

    `n` is a legacy argument kept for compatibility with older call
    sites: if given, the series is truncated to (or extended to) that
    many bars. New code should prefer specifying start/end directly.
    """
    rng = np.random.RandomState(seed)
    dates = pd.bdate_range(start=start, end=end)
    total = len(dates) + extra_bars
    rets = 0.0005 + 0.01 * rng.randn(total)
    prices = 100.0 * np.exp(np.cumsum(rets))

    if extra_bars > 0:
        extra_dates = pd.bdate_range(
            start=dates[-1] + pd.Timedelta(days=1), periods=extra_bars)
        dates = dates.append(extra_dates)

    series = pd.Series(prices[:total], index=dates[:total])

    if n is not None:
        if n <= len(series):
            series = series.iloc[:n]
        else:
            extra = pd.bdate_range(
                start=series.index[-1] + pd.Timedelta(days=1),
                periods=n - len(series))
            extra_vals = series.iloc[-1] * np.exp(
                np.cumsum(0.0005 + 0.01 * rng.randn(len(extra))))
            series = pd.concat([series, pd.Series(extra_vals, index=extra)])

    return series


# ----------------------------------------------------------------------
# Ticker normalization
# ----------------------------------------------------------------------

class TestTickerNormalization:
    def test_strips_caret(self):
        assert _normalize_ticker("^GSPC") == "GSPC"

    def test_case_insensitive(self):
        assert _normalize_ticker("gspc") == "GSPC"
        assert _normalize_ticker("GsPc") == "GSPC"

    def test_slash_to_dash(self):
        assert _normalize_ticker("BRK/B") == "BRK-B"

    def test_stable_round_trip(self):
        for t in ["^GSPC", "AAPL", "brk/b", "EURUSD=X"]:
            once = _normalize_ticker(t)
            twice = _normalize_ticker(once)
            assert once == twice, f"normalization not idempotent for {t}"


# ----------------------------------------------------------------------
# Validation
# ----------------------------------------------------------------------

class TestValidation:
    def test_accepts_clean_series(self):
        prices = _make_clean_prices()
        _validate_prices(prices, prices.index,
                         str(prices.index[0].date()),
                         str(prices.index[-1].date()), "TEST")

    def test_rejects_empty(self):
        empty = pd.Series([], dtype=float)
        with pytest.raises(DataValidationError, match="empty"):
            _validate_prices(empty, pd.DatetimeIndex([]),
                             "2018-01-01", "2020-01-01", "T")

    def test_rejects_non_finite(self):
        prices = _make_clean_prices()
        prices.iloc[10] = np.nan
        with pytest.raises(DataValidationError, match="non-finite"):
            _validate_prices(prices, prices.index,
                             "2018-01-01", "2020-01-01", "T")

    def test_rejects_zero_or_negative(self):
        prices = _make_clean_prices()
        prices.iloc[10] = 0.0
        with pytest.raises(DataValidationError, match="non-positive"):
            _validate_prices(prices, prices.index,
                             "2018-01-01", "2020-01-01", "T")

    def test_rejects_too_few_bars(self):
        # 20 bars but claim a 5-year range — should fail the bar-count check
        prices = _make_clean_prices(start="2020-01-01", end="2020-01-29")
        with pytest.raises(DataValidationError, match="expected"):
            _validate_prices(prices, prices.index,
                             "2018-01-01", "2023-01-01", "T")

    def test_rejects_duplicate_dates(self):
        prices = _make_clean_prices()
        idx = prices.index.tolist()
        idx[10] = idx[9]  # duplicate
        prices.index = pd.DatetimeIndex(idx)
        with pytest.raises(DataValidationError, match="duplicate"):
            _validate_prices(prices, prices.index,
                             "2018-01-01", "2020-01-01", "T")

    def test_rejects_unsorted(self):
        prices = _make_clean_prices()
        shuffled = prices.sample(frac=1.0, random_state=1)
        with pytest.raises(DataValidationError, match="sorted"):
            _validate_prices(shuffled, shuffled.index,
                             "2018-01-01", "2020-01-01", "T")

    def test_rejects_giant_single_day_jump(self):
        # A 60% move in one day, on a broad index, is almost always a
        # data error (unadjusted split, bad print). 50% threshold.
        prices = _make_clean_prices()
        prices.iloc[100] = prices.iloc[99] * 0.4
        with pytest.raises(DataValidationError, match="single-day"):
            _validate_prices(prices, prices.index,
                             "2018-01-01", "2020-01-01", "T")

    def test_allows_realistic_large_move(self):
        # A 15% single-day move is real (COVID March 2020, Volmageddon).
        # Validation must not reject it.
        prices = _make_clean_prices()
        prices.iloc[100] = prices.iloc[99] * 0.85
        _validate_prices(prices, prices.index,
                         "2018-01-01", "2020-01-01", "T")


# ----------------------------------------------------------------------
# Cache behavior (uses mocked fetchers, no network)
# ----------------------------------------------------------------------

class TestCaching:
    def test_first_call_fetches_then_writes_snapshot(self, tmp_data_root):
        prices = _make_clean_prices()

        def fake_yf(ticker, start, end):
            return prices, prices.index

        with mock.patch.object(data_mod, "_fetch_yfinance", side_effect=fake_yf), \
             mock.patch.object(data_mod, "_fetch_stooq",
                               side_effect=AssertionError("stooq should not be called")):
            r1, d1, s1 = get_market_returns("TEST", "2018-01-01", "2020-01-01")

        assert s1 == "yfinance"
        assert len(r1) == len(prices) - 1

        snap_dir = tmp_data_root / "raw" / "TEST"
        assert snap_dir.exists()
        assert len(list(snap_dir.glob("*.csv"))) == 1

    def test_second_call_serves_from_cache(self, tmp_data_root):
        prices = _make_clean_prices()
        call_count = {"n": 0}

        def counting_fetch(ticker, start, end):
            call_count["n"] += 1
            return prices, prices.index

        with mock.patch.object(data_mod, "_fetch_yfinance", side_effect=counting_fetch), \
             mock.patch.object(data_mod, "_fetch_stooq",
                               side_effect=AssertionError("no fallback expected")):
            _, _, s1 = get_market_returns("TEST", "2018-01-01", "2020-01-01")
            _, _, s2 = get_market_returns("TEST", "2018-01-01", "2020-01-01")

        assert s1 == "yfinance"
        assert s2 == "cache"
        assert call_count["n"] == 1, "second call must not re-fetch"

    def test_refresh_forces_refetch(self, tmp_data_root):
        prices = _make_clean_prices()
        call_count = {"n": 0}

        def counting_fetch(ticker, start, end):
            call_count["n"] += 1
            return prices, prices.index

        with mock.patch.object(data_mod, "_fetch_yfinance", side_effect=counting_fetch), \
             mock.patch.object(data_mod, "_fetch_stooq",
                               side_effect=AssertionError("no fallback expected")):
            get_market_returns("TEST", "2018-01-01", "2020-01-01")
            get_market_returns("TEST", "2018-01-01", "2020-01-01", refresh=True)

        assert call_count["n"] == 2

    def test_corrupt_cache_falls_through_to_live_fetch(self, tmp_data_root):
        """If the cached snapshot exists but fails validation (e.g. someone
        edited it, or it's a partial write), the module must not return
        garbage — it must fall through to a live fetch."""
        prices = _make_clean_prices()

        with mock.patch.object(data_mod, "_fetch_yfinance",
                               return_value=(prices, prices.index)):
            get_market_returns("TEST", "2018-01-01", "2020-01-01")

        # Corrupt the snapshot: introduce a NaN
        snap_dir = tmp_data_root / "raw" / "TEST"
        snap = next(snap_dir.glob("*.csv"))
        df = pd.read_csv(snap, parse_dates=["date"])
        df.loc[10, "close"] = np.nan
        df.to_csv(snap, index=False)

        call_count = {"n": 0}

        def counting_fetch(ticker, start, end):
            call_count["n"] += 1
            return prices, prices.index

        with mock.patch.object(data_mod, "_fetch_yfinance", side_effect=counting_fetch), \
             mock.patch.object(data_mod, "_fetch_stooq",
                               side_effect=AssertionError("stooq should not be reached")):
            _, _, s = get_market_returns("TEST", "2018-01-01", "2020-01-01")

        assert s == "yfinance"
        assert call_count["n"] == 1


# ----------------------------------------------------------------------
# Fallback behavior
# ----------------------------------------------------------------------

class TestFallback:
    def test_falls_back_to_stooq_when_yfinance_fails(self, tmp_data_root):
        prices = _make_clean_prices()

        with mock.patch.object(data_mod, "_fetch_yfinance",
                               side_effect=RuntimeError("yf down")), \
             mock.patch.object(data_mod, "_fetch_stooq",
                               return_value=(prices, prices.index)):
            _, _, s = get_market_returns("TEST", "2018-01-01", "2020-01-01")

        assert s == "stooq"

    def test_raises_when_both_sources_fail(self, tmp_data_root):
        with mock.patch.object(data_mod, "_fetch_yfinance",
                               side_effect=RuntimeError("yf down")), \
             mock.patch.object(data_mod, "_fetch_stooq",
                               side_effect=RuntimeError("stooq down")):
            with pytest.raises(RuntimeError, match="All data sources failed"):
                get_market_returns("TEST", "2018-01-01", "2020-01-01")

    def test_never_returns_synthetic(self, tmp_data_root):
        """The single most important property of this module: fetch
        failure must raise, not silently return synthetic data."""
        with mock.patch.object(data_mod, "_fetch_yfinance",
                               side_effect=RuntimeError("fail")), \
             mock.patch.object(data_mod, "_fetch_stooq",
                               side_effect=RuntimeError("fail")):
            try:
                _, _, s = get_market_returns("TEST", "2018-01-01", "2020-01-01")
                assert s != "synthetic", \
                    "get_market_returns must never return synthetic"
            except RuntimeError:
                pass  # expected path


# ----------------------------------------------------------------------
# Synthetic series (must remain accessible but named explicitly)
# ----------------------------------------------------------------------

class TestSynthetic:
    def test_synthetic_series_shape(self):
        r, d = get_synthetic_regime_switch()
        assert len(r) == 800
        assert len(d) == 800

    def test_synthetic_has_high_vol_middle(self):
        r, _ = get_synthetic_regime_switch()
        head = np.std(r[:200])
        mid = np.std(r[350:550])
        tail = np.std(r[-150:])
        assert mid > head * 5
        assert mid > tail * 5


# ----------------------------------------------------------------------
# Local CSV loader
# ----------------------------------------------------------------------

class TestLocalCSV:
    def test_loads_clean_csv(self, tmp_path):
        prices = _make_clean_prices(start="2018-01-01", end="2019-11-29")
        csv = tmp_path / "prices.csv"
        pd.DataFrame({
            "date": prices.index,
            "value": prices.values,
        }).to_csv(csv, index=False)

        r, d = load_local_csv(str(csv))
        assert len(r) == len(prices) - 1
        assert len(d) == len(prices) - 1

    def test_rejects_csv_with_bad_prices(self, tmp_path):
        prices = _make_clean_prices(start="2018-01-01", end="2019-11-29")
        prices.iloc[5] = -1.0
        csv = tmp_path / "prices.csv"
        pd.DataFrame({
            "date": prices.index,
            "value": prices.values,
        }).to_csv(csv, index=False)

        with pytest.raises(DataValidationError):
            load_local_csv(str(csv))


# ----------------------------------------------------------------------
# Network tests (opt-in)
# ----------------------------------------------------------------------

@pytest.mark.network
class TestNetwork:
    def test_real_fetch_yfinance(self, tmp_data_root):
        r, d, s = get_market_returns("^GSPC", "2023-01-01", "2023-12-31")
        assert s in ("yfinance", "stooq", "cache")
        assert len(r) > 200
        assert len(r) == len(d)

    def test_real_second_call_uses_cache(self, tmp_data_root):
        get_market_returns("^GSPC", "2023-01-01", "2023-12-31", refresh=True)
        _, _, s = get_market_returns("^GSPC", "2023-01-01", "2023-12-31")
        assert s == "cache"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-m", "not network"]))