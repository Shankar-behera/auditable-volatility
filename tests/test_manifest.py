"""
test_manifest.py
================

Tests for the content-addressed provenance layer.

The properties the manifest system must have, each encoded as a test:

  1. Content addressing: storing the same bytes twice produces the
     same filename, and does not duplicate the file.

  2. Atomic overwrite: writing a manifest when one already exists
     succeeds and replaces the file (this caught a Windows-specific
     bug: Path.rename refuses to overwrite, os.replace overwrites).

  3. Hash verification: a tampered snapshot is detected. This is the
     property that makes the whole chain auditable.

  4. Filename matches hash: the content-addressed filename is itself
     a check. A file renamed to the wrong hash is detected.

  5. Deterministic serialization: the same dict always produces the
     same bytes, so re-running a prediction produces the same snapshot
     hash.
"""

import json
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


@pytest.fixture
def manifest_dirs(monkeypatch, tmp_path):
    """Redirect snapshot and manifest storage to temp dirs."""
    snap = tmp_path / "snapshots"
    mani = tmp_path / "manifests"
    monkeypatch.setenv("CETE_SNAPSHOT_DIR", str(snap))
    monkeypatch.setenv("CETE_MANIFEST_DIR", str(mani))

    import importlib
    import src.manifest as m
    importlib.reload(m)
    return snap, mani


# ----------------------------------------------------------------------
# Hashing
# ----------------------------------------------------------------------

class TestHashing:
    def test_hash_bytes_deterministic(self, manifest_dirs):
        from src.manifest import hash_bytes
        a = hash_bytes(b"hello")
        b = hash_bytes(b"hello")
        assert a == b
        assert len(a) == 64  # sha256 hex

    def test_hash_bytes_differs_on_content(self, manifest_dirs):
        from src.manifest import hash_bytes
        assert hash_bytes(b"hello") != hash_bytes(b"world")

    def test_hash_file_matches_hash_bytes(self, manifest_dirs, tmp_path):
        from src.manifest import hash_bytes, hash_file
        p = tmp_path / "x.bin"
        p.write_bytes(b"some bytes")
        assert hash_file(p) == hash_bytes(b"some bytes")


# ----------------------------------------------------------------------
# Content-addressed snapshot storage
# ----------------------------------------------------------------------

class TestStoreSnapshot:
    def test_json_snapshot_roundtrip(self, manifest_dirs):
        from src.manifest import store_snapshot
        data = {"omega": 1.2e-6, "alpha": 0.087, "beta": 0.905}
        path, sha = store_snapshot("GSPC", "garch", data, "json")
        assert path.exists()
        assert path.stem == sha
        loaded = json.loads(path.read_text())
        assert loaded == data

    def test_parquet_snapshot_roundtrip(self, manifest_dirs):
        from src.manifest import store_snapshot
        df = pd.DataFrame({
            "date": pd.bdate_range("2026-01-01", periods=10),
            "log_return": np.linspace(-0.01, 0.01, 10),
        })
        path, sha = store_snapshot("GSPC", "prices", df, "parquet")
        assert path.exists()
        assert path.stem == sha
        loaded = pd.read_parquet(path)
        pd.testing.assert_frame_equal(loaded, df)

    def test_idempotent(self, manifest_dirs):
        """Storing the same bytes twice returns the same path and hash,
        and does not create a second file."""
        from src.manifest import store_snapshot
        data = {"omega": 1.2e-6, "alpha": 0.087}
        p1, h1 = store_snapshot("GSPC", "garch", data, "json")
        p2, h2 = store_snapshot("GSPC", "garch", data, "json")
        assert p1 == p2
        assert h1 == h2
        # Only one file in the directory
        files = list(p1.parent.glob("*.json"))
        assert len(files) == 1

    def test_different_data_different_hash(self, manifest_dirs):
        from src.manifest import store_snapshot
        p1, h1 = store_snapshot("GSPC", "garch", {"omega": 1.0}, "json")
        p2, h2 = store_snapshot("GSPC", "garch", {"omega": 2.0}, "json")
        assert h1 != h2
        assert p1 != p2

    def test_deterministic_json_key_order(self, manifest_dirs):
        """The same dict with keys in different insertion order must
        produce the same bytes and therefore the same hash. This is
        what makes the idempotency property hold for dicts built by
        different code paths."""
        from src.manifest import store_snapshot
        d1 = {"a": 1, "b": 2, "c": 3}
        d2 = {"c": 3, "a": 1, "b": 2}
        _, h1 = store_snapshot("GSPC", "garch", d1, "json")
        _, h2 = store_snapshot("GSPC", "garch", d2, "json")
        assert h1 == h2

    def test_atomic_write_no_tmp_left(self, manifest_dirs):
        from src.manifest import store_snapshot
        store_snapshot("GSPC", "garch", {"x": 1}, "json")
        # No .tmp files should remain
        snapshot_dir = manifest_dirs[0] / "GSPC" / "garch"
        tmp_files = list(snapshot_dir.glob("*.tmp"))
        assert not tmp_files, f"leftover tmp files: {tmp_files}"


# ----------------------------------------------------------------------
# Hash verification
# ----------------------------------------------------------------------

class TestVerifySnapshot:
    def _store_and_entry(self, data=None):
        from src.manifest import store_snapshot
        if data is None:
            data = {"omega": 1.2e-6, "alpha": 0.087, "beta": 0.905}
        path, sha = store_snapshot("GSPC", "garch", data, "json")
        return {"path": str(path), "sha256": sha}

    def test_verifies_intact_snapshot(self, manifest_dirs):
        from src.manifest import verify_snapshot
        entry = self._store_and_entry()
        # Should not raise
        verify_snapshot(entry)

    def test_detects_tampered_snapshot(self, manifest_dirs):
        """Appending even one byte to a snapshot must be detected."""
        from src.manifest import verify_snapshot
        entry = self._store_and_entry()
        path = entry["path"]
        with open(path, "ab") as f:
            f.write(b"X")
        with pytest.raises(ValueError, match="hash mismatch"):
            verify_snapshot(entry)

    def test_detects_missing_snapshot(self, manifest_dirs):
        from src.manifest import verify_snapshot
        entry = self._store_and_entry()
        os.remove(entry["path"])
        with pytest.raises(ValueError, match="missing"):
            verify_snapshot(entry)

    def test_detects_renamed_file(self, manifest_dirs):
        """If the filename doesn't match the recorded hash, that's a
        violation of content addressing and must be caught."""
        from pathlib import Path
        from src.manifest import verify_snapshot
        entry = self._store_and_entry()
        path = Path(entry["path"])
        wrong_name = path.parent / ("0" * 64 + ".json")
        os.rename(path, wrong_name)
        entry["path"] = str(wrong_name)
        with pytest.raises(ValueError, match="filename"):
            verify_snapshot(entry)


# ----------------------------------------------------------------------
# Manifest writing
# ----------------------------------------------------------------------

class TestWriteManifest:
    def _make_manifest(self, manifest_dirs, pid="GSPC_2026-09-13_10"):
        from src.manifest import store_snapshot, build_manifest
        data = {"omega": 1.2e-6, "alpha": 0.087, "beta": 0.905,
                "mu": 0.0002, "sigma2_last": 4.1e-5,
                "eps_last": -0.003, "n_returns_at_fit": 750}
        gp, gh = store_snapshot("GSPC", "garch", data, "json")

        df = pd.DataFrame({
            "date": pd.bdate_range("2025-01-01", periods=10),
            "log_return": np.zeros(10),
        })
        pp, ph = store_snapshot("GSPC", "prices", df, "parquet")

        return build_manifest(
            prediction_id=pid,
            ticker="GSPC",
            origin_date="2026-09-13",
            horizon_days=10,
            window_size=64,
            price_snapshot={"path": str(pp), "sha256": ph,
                            "n_observations": 10},
            garch_parameters={"path": str(gp), "sha256": gh, **data},
            model_config={"vol": "Garch", "p": 1, "q": 1},
        )

    def test_writes_and_reads(self, manifest_dirs):
        from src.manifest import write_manifest, read_manifest
        m = self._make_manifest(manifest_dirs)
        path = write_manifest(m)
        assert path.exists()
        loaded = read_manifest("GSPC_2026-09-13_10")
        assert loaded["prediction_id"] == "GSPC_2026-09-13_10"
        assert loaded["horizon_days"] == 10

    def test_overwrites_existing(self, manifest_dirs):
        """This is the bug that bit on Windows: Path.rename refuses to
        overwrite an existing target, os.replace overwrites. Writing
        a manifest twice must succeed."""
        from src.manifest import write_manifest
        m = self._make_manifest(manifest_dirs)
        write_manifest(m)
        # Second write must not raise
        write_manifest(m)

    def test_overwrite_produces_same_bytes(self, manifest_dirs):
        """Because the manifest is a deterministic function of its
        inputs, writing it twice produces the same bytes and therefore
        the same hash. That's what makes re-running a prediction
        produce an identical manifest."""
        from src.manifest import write_manifest, hash_file
        m = self._make_manifest(manifest_dirs)
        p1 = write_manifest(m)
        h1 = hash_file(p1)
        p2 = write_manifest(m)
        h2 = hash_file(p2)
        assert h1 == h2

    def test_no_tmp_left_behind(self, manifest_dirs):
        from src.manifest import write_manifest
        m = self._make_manifest(manifest_dirs)
        write_manifest(m)
        tmp = list(manifest_dirs[1].glob("*.tmp"))
        assert not tmp, f"leftover tmp: {tmp}"


# ----------------------------------------------------------------------
# Manifest content
# ----------------------------------------------------------------------

class TestManifestContent:
    def test_records_code_and_environment(self, manifest_dirs):
        from src.manifest import store_snapshot, build_manifest
        data = {"omega": 1.2e-6}
        gp, gh = store_snapshot("GSPC", "garch", data, "json")
        df = pd.DataFrame({"date": [1], "log_return": [0.0]})
        pp, ph = store_snapshot("GSPC", "prices", df, "parquet")

        m = build_manifest(
            prediction_id="T_2026-01-01_10",
            ticker="GSPC", origin_date="2026-01-01",
            horizon_days=10, window_size=64,
            price_snapshot={"path": str(pp), "sha256": ph},
            garch_parameters={"path": str(gp), "sha256": gh, **data},
            model_config={"vol": "Garch"},
        )
        assert "environment" in m
        assert "code" in m
        assert "python" in m["environment"]
        assert "numpy" in m["environment"]
        assert m["verification"]["tolerance_abs"] == 1e-6


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))