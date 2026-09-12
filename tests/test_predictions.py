"""
test_predictions.py
===================

Tests for the append-only prediction log.

Encodes the properties the log is supposed to have, including the two
bugs found while building it:

  1. read_resolved returned a DataFrame without abs_error/error/etc.,
     because pandas took the union of keys across prediction and
     resolution lines and then suffixed the merge columns to _x/_y.
     Fixed by dropping resolution-only columns from the prediction
     slice before merging. Covered by
     test_read_resolved_has_resolution_columns.

  2. record_resolution's "is this already resolved" check was
     unreachable: has_unresolved was set by the presence of a
     prediction line, which always exists if a resolution is being
     attempted, so the double-resolve guard never fired. Fixed by
     checking already_resolved first. Covered by
     test_double_resolve_raises.

Both were caught by running a test that measured the property, not by
reading the code. Keeping those two cases in the suite is the point.
"""

import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ----------------------------------------------------------------------
# Fixture: redirect the log directory to a temp dir per test
# ----------------------------------------------------------------------

@pytest.fixture
def pred_dir(monkeypatch, tmp_path):
    """Every test gets its own empty predictions directory."""
    monkeypatch.setenv("CETE_PREDICTIONS_DIR", str(tmp_path))
    import importlib
    import src.predictions as pred_mod
    importlib.reload(pred_mod)
    return tmp_path


# ----------------------------------------------------------------------
# Prediction ID
# ----------------------------------------------------------------------

class TestPredictionId:
    def test_deterministic(self, pred_dir):
        from src.predictions import make_prediction_id
        a = make_prediction_id("^GSPC", "2024-09-13", 10)
        b = make_prediction_id("GSPC", "2024-09-13", 10)
        assert a == b == "GSPC_2024-09-13_10"

    def test_rejects_nonpositive_horizon(self, pred_dir):
        from src.predictions import make_prediction_id
        with pytest.raises(ValueError, match="horizon_days"):
            make_prediction_id("GSPC", "2024-09-13", 0)
        with pytest.raises(ValueError, match="horizon_days"):
            make_prediction_id("GSPC", "2024-09-13", -1)


# ----------------------------------------------------------------------
# record_prediction
# ----------------------------------------------------------------------

class TestRecordPrediction:
    def _rec(self, ticker="^GSPC", origin="2024-09-01", horizon=10,
             fv=0.008):
        from src.predictions import record_prediction
        return record_prediction(
            ticker, origin, horizon, fv,
            screen_flagged=False,
            screen_energy=0.0004, screen_cutoff=0.0006)

    def test_creates_file_and_returns_id(self, pred_dir):
        pid = self._rec()
        assert pid == "GSPC_2024-09-01_10"
        log = pred_dir / "GSPC.jsonl"
        assert log.exists()
        lines = log.read_text().strip().split("\n")
        assert len(lines) == 1
        obj = json.loads(lines[0])
        assert obj["prediction_id"] == pid
        assert obj["forecast_vol"] == 0.008

    def test_idempotent(self, pred_dir):
        """Calling twice with the same (ticker, date, horizon) does not
        create a second line, regardless of the other arguments."""
        from src.predictions import read_pending
        self._rec(fv=0.008)
        self._rec(fv=0.999)  # different value, same ID
        assert len(read_pending("^GSPC")) == 1
        log = pred_dir / "GSPC.jsonl"
        assert len(log.read_text().strip().split("\n")) == 1

    def test_rejects_negative_forecast(self, pred_dir):
        from src.predictions import record_prediction
        with pytest.raises(ValueError, match="forecast_vol"):
            record_prediction(
                "^GSPC", "2024-09-01", 10, -0.01,
                screen_flagged=False,
                screen_energy=0.0, screen_cutoff=0.0)

    def test_rejects_nan_forecast(self, pred_dir):
        from src.predictions import record_prediction
        with pytest.raises(ValueError, match="forecast_vol"):
            record_prediction(
                "^GSPC", "2024-09-01", 10, float("nan"),
                screen_flagged=False,
                screen_energy=0.0, screen_cutoff=0.0)

    def test_extra_collision_raises(self, pred_dir):
        from src.predictions import record_prediction
        with pytest.raises(ValueError, match="collides"):
            record_prediction(
                "^GSPC", "2024-09-01", 10, 0.008,
                screen_flagged=False,
                screen_energy=0.0, screen_cutoff=0.0,
                extra={"forecast_vol": 999.0})

    def test_extra_noncolliding_is_stored(self, pred_dir):
        from src.predictions import record_prediction, read_predictions
        record_prediction(
            "^GSPC", "2024-09-01", 10, 0.008,
            screen_flagged=False,
            screen_energy=0.0, screen_cutoff=0.0,
            extra={"note": "manual run"})
        df = read_predictions("^GSPC")
        assert df.iloc[0]["note"] == "manual run"


# ----------------------------------------------------------------------
# record_resolution
# ----------------------------------------------------------------------

class TestRecordResolution:
    def test_resolves_a_prediction(self, pred_dir):
        from src.predictions import (
            record_prediction, record_resolution, read_resolved)
        record_prediction(
            "^GSPC", "2024-09-01", 10, 0.008,
            screen_flagged=False,
            screen_energy=0.0, screen_cutoff=0.0)
        record_resolution("^GSPC", "2024-09-01", 10, 0.010)
        resolved = read_resolved("^GSPC")
        assert len(resolved) == 1
        assert resolved.iloc[0]["target_vol"] == 0.010
        assert abs(resolved.iloc[0]["error"] - (-0.002)) < 1e-12

    def test_double_resolve_raises(self, pred_dir):
        """This was the second bug found while building the module.
        Before the fix, has_unresolved was set by the presence of the
        prediction line (which always exists), so the check never
        fired and a second resolution line was appended silently."""
        from src.predictions import record_prediction, record_resolution
        record_prediction(
            "^GSPC", "2024-09-01", 10, 0.008,
            screen_flagged=False,
            screen_energy=0.0, screen_cutoff=0.0)
        record_resolution("^GSPC", "2024-09-01", 10, 0.009)
        with pytest.raises(ValueError, match="already resolved"):
            record_resolution("^GSPC", "2024-09-01", 10, 0.009)

    def test_resolve_nonexistent_raises(self, pred_dir):
        from src.predictions import record_resolution
        with pytest.raises(ValueError, match="No prediction"):
            record_resolution("^GSPC", "2099-01-01", 10, 0.009)

    def test_rejects_negative_target(self, pred_dir):
        from src.predictions import record_prediction, record_resolution
        record_prediction(
            "^GSPC", "2024-09-01", 10, 0.008,
            screen_flagged=False,
            screen_energy=0.0, screen_cutoff=0.0)
        with pytest.raises(ValueError, match="target_vol"):
            record_resolution("^GSPC", "2024-09-01", 10, -0.01)


# ----------------------------------------------------------------------
# read_pending, read_resolved, accuracy_summary
# ----------------------------------------------------------------------

class TestReadSide:
    def _seed(self, n=3):
        from src.predictions import record_prediction
        for d, fv in [
                ("2024-09-01", 0.008),
                ("2024-09-02", 0.012),
                ("2024-09-03", 0.010)][:n]:
            record_prediction(
                "^GSPC", d, 10, fv,
                screen_flagged=False,
                screen_energy=0.0004, screen_cutoff=0.0006)

    def test_read_pending_counts(self, pred_dir):
        from src.predictions import read_pending
        self._seed(3)
        assert len(read_pending("^GSPC")) == 3

    def test_read_pending_excludes_resolved(self, pred_dir):
        from src.predictions import read_pending, record_resolution
        self._seed(3)
        record_resolution("^GSPC", "2024-09-01", 10, 0.009)
        assert len(read_pending("^GSPC")) == 2

    def test_read_resolved_empty_when_none(self, pred_dir):
        from src.predictions import read_resolved
        self._seed(3)
        assert read_resolved("^GSPC").empty

    def test_read_resolved_has_resolution_columns(self, pred_dir):
        """This was the first bug found while building the module.
        Before the fix, read_resolved returned a DataFrame without
        abs_error because the merge suffixed the shared column names.
        Direct assertion here so the fix cannot silently regress."""
        from src.predictions import read_resolved, record_resolution
        self._seed(3)
        record_resolution("^GSPC", "2024-09-01", 10, 0.009)
        record_resolution("^GSPC", "2024-09-02", 10, 0.011)
        df = read_resolved("^GSPC")
        assert len(df) == 2
        for col in ("target_vol", "error", "abs_error",
                    "squared_error", "resolved_at",
                    "forecast_vol", "origin_date"):
            assert col in df.columns, (
                f"read_resolved is missing column {col!r}. "
                f"Columns present: {list(df.columns)}"
            )

    def test_read_resolved_values(self, pred_dir):
        from src.predictions import read_resolved, record_resolution
        self._seed(3)
        record_resolution("^GSPC", "2024-09-01", 10, 0.009)
        df = read_resolved("^GSPC")
        row = df.iloc[0]
        assert row["forecast_vol"] == 0.008
        assert row["target_vol"] == 0.009
        assert abs(row["error"] - (-0.001)) < 1e-12
        assert abs(row["abs_error"] - 0.001) < 1e-12

    def test_accuracy_summary_empty_log(self, pred_dir):
        from src.predictions import accuracy_summary
        s = accuracy_summary("^GSPC")
        assert s["n_pending"] == 0
        assert s["n_resolved"] == 0
        assert np.isnan(s["mean_abs_error"])

    def test_accuracy_summary_with_resolutions(self, pred_dir):
        from src.predictions import accuracy_summary, record_resolution
        self._seed(3)
        record_resolution("^GSPC", "2024-09-01", 10, 0.009)
        record_resolution("^GSPC", "2024-09-02", 10, 0.011)
        s = accuracy_summary("^GSPC")
        assert s["n_pending"] == 1
        assert s["n_resolved"] == 2
        assert s["mean_abs_error"] == pytest.approx(0.001, abs=1e-12)
        assert s["rmse"] == pytest.approx(0.001, abs=1e-12)
        assert s["first_origin"] == "2024-09-01"
        assert s["last_origin"] == "2024-09-02"

    def test_accuracy_summary_pearson_nan_for_one_point(self, pred_dir):
        from src.predictions import accuracy_summary, record_resolution
        self._seed(1)
        record_resolution("^GSPC", "2024-09-01", 10, 0.009)
        s = accuracy_summary("^GSPC")
        assert np.isnan(s["pearson"]), (
            "pearson should be NaN with <2 resolved points; a number "
            "here would look authoritative while meaning nothing"
        )


# ----------------------------------------------------------------------
# Corruption tolerance
# ----------------------------------------------------------------------

class TestCorruption:
    def test_corrupt_line_is_tolerated(self, pred_dir):
        """A truncated write (process killed mid-append) leaves an
        unparseable line. The reader must not fail; it must surface
        the corrupt line as row_type='corrupt'."""
        from src.predictions import record_prediction, read_predictions

        record_prediction("^GSPC", "2024-09-01", 10, 0.008,
                          screen_flagged=False,
                          screen_energy=0.0, screen_cutoff=0.0)

        log = pred_dir / "GSPC.jsonl"
        with open(log, "a") as f:
            # Truncated JSON, WITH a trailing newline so the corruption
            # is confined to one line. Without the newline, the next
            # append would merge onto the same line, producing a single
            # unparseable blob that looks like a different failure.
            f.write('{"prediction_id": "GSPC_2024-09-02_10", "fore\n')

        record_prediction("^GSPC", "2024-09-03", 10, 0.010,
                          screen_flagged=False,
                          screen_energy=0.0, screen_cutoff=0.0)

        df = read_predictions("^GSPC")
        types = list(df["row_type"])
        assert "corrupt" in types, (
            "truncated line should be surfaced as row_type='corrupt'"
        )
        assert types.count("prediction") == 2, (
            f"expected 2 valid predictions around the corrupt line, "
            f"got {types.count('prediction')}; types={types}"
        )

    def test_corrupt_line_does_not_block_pending(self, pred_dir):
        """A corrupt line in the middle must not prevent the module
        from reading valid predictions that follow it."""
        from src.predictions import record_prediction, read_pending

        record_prediction("^GSPC", "2024-09-01", 10, 0.008,
                          screen_flagged=False,
                          screen_energy=0.0, screen_cutoff=0.0)

        log = pred_dir / "GSPC.jsonl"
        with open(log, "a") as f:
            f.write("this is not json\n")

        record_prediction("^GSPC", "2024-09-03", 10, 0.010,
                          screen_flagged=False,
                          screen_energy=0.0, screen_cutoff=0.0)

        assert len(read_pending("^GSPC")) == 2


# ----------------------------------------------------------------------
# Storage layout
# ----------------------------------------------------------------------

class TestStorageLayout:
    def test_one_file_per_ticker(self, pred_dir):
        from src.predictions import record_prediction
        record_prediction("^GSPC", "2024-09-01", 10, 0.008,
                          screen_flagged=False,
                          screen_energy=0.0, screen_cutoff=0.0)
        record_prediction("AAPL", "2024-09-01", 10, 0.012,
                          screen_flagged=False,
                          screen_energy=0.0, screen_cutoff=0.0)
        assert (pred_dir / "GSPC.jsonl").exists()
        assert (pred_dir / "AAPL.jsonl").exists()

    def test_ticker_normalization(self, pred_dir):
        """'^GSPC', 'GSPC', 'gspc' must all write to GSPC.jsonl."""
        from src.predictions import record_prediction
        for t in ("^GSPC", "GSPC", "gspc"):
            record_prediction(t, "2024-09-01", 10, 0.008,
                              screen_flagged=False,
                              screen_energy=0.0, screen_cutoff=0.0)
        files = sorted(p.name for p in pred_dir.glob("*.jsonl"))
        assert files == ["GSPC.jsonl"]

    def test_jsonl_is_line_delimited(self, pred_dir):
        """Each record must occupy exactly one line. A record
        containing an embedded newline would break every downstream
        reader, including `git log -p`."""
        from src.predictions import record_prediction, record_resolution
        record_prediction("^GSPC", "2024-09-01", 10, 0.008,
                          screen_flagged=False,
                          screen_energy=0.0, screen_cutoff=0.0,
                          extra={"note": "a note with\nembedded newline"})
        record_resolution("^GSPC", "2024-09-01", 10, 0.009)

        log = pred_dir / "GSPC.jsonl"
        lines = log.read_text().strip().split("\n")
        assert len(lines) == 2, (
            f"expected 2 lines (1 prediction + 1 resolution), got "
            f"{len(lines)}; an embedded newline broke line-delimiting"
        )
        for line in lines:
            json.loads(line)  # must parse


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))