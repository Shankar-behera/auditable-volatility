# auditable-volatility

A GARCH volatility forecaster whose every prediction is logged with
content-addressed provenance and can be independently reproduced and
verified by a third party.

The claim: **any prediction this system has made can be checked.** The
prediction, the inputs that produced it, and the code that ran are all
recorded, hashed, and committed to git before the outcome is known. A
stranger can clone the repo, pick a prediction ID, and run one command
to verify it.

```bash
git clone <this-repo>
cd auditable-volatility
pip install -r requirements.txt
python scripts/reproduce.py --latest GSPC --verbose
```

---

## What this system does

Once per trading day, for a configured ticker, it produces:

* a **volatility forecast** for the next 10 trading days, from GARCH(1,1)
* a **screen flag**, indicating whether the current trailing window's
  variance is in the top 5% of its own historical distribution

Every prediction is written to an append-only log *before* its outcome
is known. When the forecast horizon elapses, realized volatility is
computed and appended as a resolution. The log is committed to git at
prediction time, so the git history is the public record.

## What makes it auditable

Three properties, each enforced by code:

**1. Predictions are recorded before outcomes.**

The log line exists before anyone knows whether the forecast was right.
There is no code path that can revise a logged prediction.

**2. Inputs are content-addressed.**

Every prediction writes snapshots of the exact bytes the model consumed
-- the return series and the fitted GARCH parameters -- under filenames
that *are* their SHA-256 hashes. A manifest records the hashes, plus the
git SHA and dependency versions. Modifying any input is detectable.

**3. Verification is one command.**

`scripts/reproduce.py` reads a manifest, verifies every hash in the
chain, recomputes the forecast from the recorded parameters, and reports
`VERIFIED` or the specific reason it isn't.

See [docs/AUDIT.md](docs/AUDIT.md) for the full guide and
[docs/METHODOLOGY.md](docs/METHODOLOGY.md) for the model and evaluation
description.

---

## Quick start

```bash
python -m venv .venv

# Linux/macOS
source .venv/bin/activate

# Windows
.venv\Scripts\activate

pip install -r requirements.txt
```

Run the monitor once:

```bash
python scripts/live_monitor.py --ticker ^GSPC
```

Verify any prediction:

```bash
python scripts/reproduce.py --latest GSPC --verbose
python scripts/reproduce.py GSPC_2026-09-11_10
python scripts/reproduce.py --all GSPC
```

Resolve matured predictions and regenerate the track record:

```bash
python scripts/score_outcomes.py --ticker ^GSPC
python scripts/publish_report.py --outdir docs/track_record
```

---

## Repository layout

```text
src/
  cete.py              The CETE score (entropy-weighted spectral power).
  data.py              Versioned, validated, cached data layer.
  regime.py            Scoring, evaluation, operating curves.
  garch_filter.py      Recursive GARCH state update between refits.
  predictions.py       Append-only prediction log (JSONL).
  manifest.py          Content-addressed provenance.
  metrics.py           Shared correlation and error metrics.

scripts/
  live_monitor.py            Daily prediction entrypoint.
  score_outcomes.py          Resolve matured predictions.
  publish_report.py          Generate Markdown track record.
  reproduce.py               Verify a prediction end to end.
  causal_evaluation.py       Leakage-free forecasting + screening eval.
  baseline_comparison.py     Retrospective estimator comparison.
  transition_evaluation.py   Changepoint detection comparison.
  run_analysis.py            Basic pipeline: fetch -> score -> plot.

data/
  predictions/<TICKER>.jsonl     Append-only prediction log (committed).
  snapshots/<TICKER>/...         Content-addressed inputs (committed).
  manifests/<prediction_id>.json Provenance records (committed).

docs/
  AUDIT.md            How a third party verifies a prediction.
  METHODOLOGY.md      Model, target definition, evaluation.
  RESULTS.md          Research write-up, including the negative result.
  track_record/       Generated accuracy reports.

tests/
  test_data.py        Data layer: caching, validation, fallback.
  test_cete.py        Engine, scoring, leakage guards.
  test_predictions.py Prediction log: idempotency, corruption tolerance.
  test_manifest.py    Provenance: content addressing, hash verification.
```

---

## The research behind it

This project began as **CETE** (Chrono-Entropic Topodynamic Engine), a
hand-designed spectral-entropy volatility model with a relaxation
dynamics layered on top.

The evaluation showed it reduces to a variance calculation and loses to
GARCH on every honest test. That negative result is documented in full
in [docs/RESULTS.md](docs/RESULTS.md).

The production system uses GARCH(1,1), because it works. The interesting
engineering is the provenance layer around it.

---

## Testing

```bash
# Offline tests (default)
pytest tests/ -v

# Include live-fetch tests
pytest tests/ -v -m network
```

`pytest.ini` registers the `network` marker and deselects those tests by
default, so the offline suite is deterministic.

---

## Scope

This is a research codebase, not a production trading system.

What it does well:

* versioned data
* honest evaluation
* immutable prediction logging
* content-addressed provenance
* independent verification of predictions

What it does not do:

* portfolio risk
* position sizing
* execution
* compliance
* complete external audit trails

It is not registered as a financial model and has no named owner. It
should not be the basis of any trading decision without independent
review.

---

## License

MIT -- see [LICENSE](LICENSE).
