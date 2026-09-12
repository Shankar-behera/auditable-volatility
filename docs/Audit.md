# Audit Guide

How to independently verify any prediction this system has made.

The claim being checked is:

> The logged forecast for a given prediction is numerically reproducible
> from the recorded input data and the recorded fitted model parameters,
> within a declared tolerance.

If that claim holds, the prediction was what it says it was: it was not
edited after the fact, computed from different recorded inputs, or
retrospectively adjusted.

This document describes the complete verification procedure and its
limitations.

---

## Prerequisites

```bash
git clone https://github.com/Shankar-behera/auditable-volatility.git
cd auditable-volatility
pip install -r requirements.txt
```

Verification does **not** require network access.

Every input required to reproduce a supported forecast is stored in the
repository:

* the return-series snapshot consumed by the model
* the fitted GARCH parameters used for the forecast
* the manifest tying those artifacts together
* the code SHA recorded at prediction time
* the dependency and configuration metadata recorded by the manifest

If any required artifact is missing, the prediction may be
unverifiable. See [Result codes](#result-codes) below.

---

## Verify the latest prediction

The simplest verification command is:

```bash
python scripts/reproduce.py --latest GSPC
```

This automatically selects the latest prediction for which the required
manifest is available.

A successful verification looks like:

```text
================================================================
PREDICTION VERIFICATION
================================================================

Prediction:             GSPC_2026-09-11_10
Ticker:                 GSPC
Origin date:            2026-09-11
Horizon:                10 days

Logged forecast:        0.0079869874
Recomputed forecast:    0.0079869874

Absolute difference:    0.000e+00
Tolerance:              1.000e-06

================================================================
RESULT: VERIFIED
================================================================
```

The exact prediction ID, forecast value, and date will depend on the
repository state when the verification is performed.

---

## Verify a specific prediction

If you know the prediction ID:

```bash
python scripts/reproduce.py GSPC_2026-09-11_10
```

The verifier checks the complete provenance chain for that prediction.

Use `--verbose` to additionally display snapshot paths, hashes,
environment information, and the recorded code SHA:

```bash
python scripts/reproduce.py GSPC_2026-09-11_10 --verbose
```

---

## Verify every prediction for a ticker

To verify every supported prediction for a ticker:

```bash
python scripts/reproduce.py --all GSPC
```

The command prints one verification block per prediction and a summary
at the end.

For example:

```text
SUMMARY: 42/42 verified
```

A prediction that predates the manifest system may instead produce
`CANNOT VERIFY`. This is expected for historical records that do not
contain the artifacts required by the current verification scheme.

---

# Result Codes

The verifier has three meaningful result states.

## `RESULT: VERIFIED` — exit code 0

Every required check succeeded:

1. The prediction exists in the prediction log.
2. The manifest still hashes to the value recorded in the prediction
   log.
3. The recorded input snapshot still hashes to the value recorded in
   the manifest.
4. The frozen GARCH parameter snapshot still hashes to the value
   recorded in the manifest.
5. Recomputing the forecast from the recorded parameters produces a
   value within the recorded numerical tolerance of the logged forecast.

Every link in the chain from the logged prediction to the recomputed
forecast has therefore been checked.

The logged number is consistent with the recorded evidence.

---

## `RESULT: NOT VERIFIED` — exit code 1

At least one verification check failed.

Common causes include:

* a snapshot file was modified after the prediction was recorded
* the snapshot hash no longer matches the manifest
* the manifest itself was modified
* a content-addressed snapshot was replaced with different bytes
* the frozen GARCH parameter file was modified
* the recomputed forecast differs from the logged forecast by more than
  the declared tolerance
* the code used for recomputation is inconsistent with the recorded
  provenance

`NOT VERIFIED` does **not** necessarily mean the prediction was
fraudulent.

It means the available evidence no longer supports the reproducibility
claim.

The important property is that the discrepancy is detected instead of
being silently accepted.

---

## `CANNOT VERIFY` — exit code 2

A required artifact is unavailable.

Typical causes include:

* the prediction was recorded before the manifest system existed
* a referenced snapshot was deleted
* a required manifest cannot be found
* the repository does not contain enough information to reconstruct
  the provenance chain

The prediction is therefore **unverifiable**, not falsified.

Predictions recorded before the manifest system existed will remain
unverifiable under this audit procedure unless their missing provenance
artifacts are recovered.

---

# What `VERIFIED` Means — Precisely

A successful verification means:

> Given the recorded snapshot bytes and the recorded fitted model
> parameters, the recorded forecast is what the model produces, within
> the recorded numerical tolerance.

That is the complete reproducibility claim.

It does **not** mean:

* the model is good
* the forecast was accurate
* the input data was externally correct
* the fitted parameters would be reproduced by refitting
* the future outcome was calculated correctly
* every prediction that should have existed was recorded
* the repository history has never been rewritten

Each of these is a separate claim.

Keeping these claims separate is intentional.

---

# How the Provenance Chain Works

Each prediction is connected to its evidence through hashes.

Conceptually:

```text
prediction log
      |
      | manifest hash
      v
   manifest
      |
      +--------------------+
      |                    |
      | input hash         | parameter hash
      v                    v
return snapshot      GARCH parameter snapshot
      |                    |
      +---------+----------+
                |
                v
       deterministic forecast
                |
                v
        logged forecast
```

The verifier checks every link.

This means that changing an input snapshot without changing the
manifest is detectable.

Changing the manifest without changing the prediction record is
detectable.

Changing the frozen parameters is detectable.

Changing the forecast itself is detectable because the recomputed value
will no longer match within tolerance.

The hashes therefore provide **content integrity** for the recorded
artifacts.

---

# How to Check the Log Itself Has Not Been Rewritten

Verifying an individual prediction proves that its logged forecast is
internally consistent.

It does not, by itself, prove that the prediction log has never been
rewritten to remove or reorder records.

That property comes from the repository's version history.

Every prediction should be committed to git at the time it is made,
before its future outcome is known.

Inspect the prediction history with:

```bash
git log -p data/predictions/GSPC.jsonl
```

Also inspect the commits affecting the provenance artifacts:

```bash
git log --stat -- data/predictions/GSPC.jsonl
git log --stat -- data/manifests
git log --stat -- data/snapshots
```

When reviewing the history, look for:

### 1. Prediction timing

The commit containing a prediction should be consistent with the
prediction's `origin_date`.

A prediction that appears in a commit long after it claims to have been
generated deserves investigation.

### 2. Removed prediction lines

A prediction appearing in an earlier commit but disappearing from a
later commit is inconsistent with an append-only record unless the
removal is explicitly documented.

### 3. Rewritten history

Previously observed commit SHAs should remain stable.

Force-pushing can rewrite repository history and therefore weakens the
tamper-evidence of the public record.

Git permits history rewriting.

Therefore, the audit property is partly technical and partly
operational: the repository must not be force-pushed after predictions
have been published.

---

# What Verification Does Not Prove

## It does not prove the model is good

Verification proves that the logged forecast is reproducible from the
recorded evidence.

It says nothing about predictive quality.

Model quality must be evaluated separately using the project's
historical track record and causal evaluation procedures.

---

## It does not prove the input data is correct

The recorded snapshot is whatever data the system consumed at
prediction time.

If the upstream data source contained an error, the forecast can still
be perfectly reproducible from that incorrect data.

Therefore:

```text
hash verification
    !=
external data validation
```

The system proves internal consistency, not external truth.

---

## It does not prove the parameters would be reproduced by refitting

The GARCH parameters used for a prediction are frozen and recorded at
prediction time.

The audit claim is:

> Given these recorded parameters, does the model reproduce the
> recorded forecast?

It is **not**:

> If I refit GARCH today using the same historical data, do I obtain
> exactly the same parameters?

Those are different experiments.

Refitting can differ because of:

* library versions
* optimization behavior
* numerical differences
* data revisions
* configuration changes
* platform differences

The audit system intentionally freezes the parameters used by the
original prediction so that the original numerical forecast can be
reconstructed directly.

---

## It does not prove the outcome was computed correctly

The future realized-volatility outcome is calculated later by:

```bash
python scripts/score_outcomes.py
```

The outcome is separate from the forecast.

Verification therefore does not establish that the later outcome
calculation correctly implemented the methodology.

That is a separate methodological check.

See [METHODOLOGY.md](METHODOLOGY.md) for the target definition.

---

## It does not prove the log is complete

The record proves what it contains.

It cannot prove that a prediction which was never written to the log
should have existed.

This is an inherent limitation of append-only records.

A missing prediction leaves no prediction record to verify.

Repository history, scheduled execution logs, and external archival
systems can provide additional evidence of completeness, but the
prediction log alone cannot establish it.

---

# Reproducing a Forecast Manually

The automated verifier should be the primary audit tool.

A technically inclined reviewer can also reproduce the numerical
forecast manually.

For example:

```bash
python - <<'PY'
import json
from src.manifest import read_manifest
from src.garch_filter import multistep_forecast

prediction_id = "GSPC_2026-09-11_10"

m = read_manifest(prediction_id)

print("horizon:", m["horizon_days"])
print("tolerance:", m["verification"]["tolerance_abs"])

params_path = m["garch_parameters"]["path"]

with open(params_path, "r", encoding="utf-8") as f:
    params = json.load(f)

print("frozen parameters:", params)

recomputed = multistep_forecast(
    params,
    horizon=m["horizon_days"],
)

print(f"recomputed forecast: {recomputed:.10f}")
```

`multistep_forecast` is the same deterministic forecasting function used
by the live prediction pipeline.

Given the same frozen parameters and horizon, it should reproduce the
logged forecast within the declared tolerance.

If it does not, first check:

1. whether the correct prediction ID was used
2. whether the repository is at the recorded code SHA
3. whether the parameter snapshot hash passes verification
4. whether the installed dependency versions match the recorded
   environment

The automated `reproduce.py` command performs these checks as part of
the complete verification process.

---

# Checking Out the Exact Code

The manifest records the git SHA associated with the prediction.

Retrieve it with:

```bash
python - <<'PY'
from src.manifest import read_manifest

prediction_id = "GSPC_2026-09-11_10"

m = read_manifest(prediction_id)

print(m["code"]["git_sha"])
PY
```

Then check out that revision:

```bash
git checkout <sha-from-above>
```

Run the verification:

```bash
python scripts/reproduce.py GSPC_2026-09-11_10
```

Return to the previous branch or commit afterward:

```bash
git checkout -
```

---

## Dirty working trees

The manifest records whether the prediction was generated from a clean
or dirty git working tree.

If:

```text
code_dirty: false
```

the recorded git SHA identifies the committed code state used for the
prediction.

If:

```text
code_dirty: true
```

the prediction was generated while uncommitted changes existed.

The recorded SHA therefore identifies the repository's committed state,
but not necessarily every source-file byte that was actually executed.

Numerical verification can still succeed because the snapshots and
frozen parameters may be sufficient to reproduce the forecast.

However, the code provenance is weaker.

For the strongest audit trail, predictions should be generated from a
clean working tree.

---

## Missing git SHA

If:

```text
git_sha: null
```

the prediction was made before the repository was placed under version
control or before code provenance was recorded.

The numerical forecast may still be reproducible from its snapshots and
parameters.

However, the exact source-code revision cannot be identified.

This is why the audit system distinguishes numerical reproducibility
from complete code provenance.

---

# Environment Information

The manifest records relevant environment information, including
dependency versions used by the prediction pipeline.

The primary dependencies relevant to the forecasting calculation
include:

* `arch`
* `numpy`
* `pandas`
* `yfinance`

Dependency information helps a reviewer identify environmental
differences that could affect numerical behavior.

It should not be interpreted as a guarantee of bit-identical execution
across all operating systems, CPUs, Python versions, or library builds.

The verification target is numerical agreement within the declared
tolerance.

---

# What the Hashes Prove

A SHA-256 hash proves that the bytes currently present are consistent
with the bytes that were hashed when the manifest was created.

For example:

```text
recorded SHA-256
       |
       v
snapshot bytes
       |
       v
current SHA-256
```

If the values match, the snapshot bytes have not changed relative to
the recorded hash.

This establishes content identity.

It does **not** prove that:

* the original upstream source was truthful
* the original data provider was correct
* the prediction was economically useful
* the model was statistically valid
* the repository was never copied incorrectly
* a missing prediction never existed

Hashes provide integrity for the artifacts they cover, not universal
truth.

---

# Operator Requirements

For the audit trail to remain meaningful over time, the operator must
follow these rules.

## 1. Commit every prediction

Every prediction should be committed to the repository when it is made,
before the future outcome is known.

Scheduled workflows should perform this automatically.

Manual runs must also commit the resulting prediction and provenance
artifacts.

---

## 2. Never modify or delete a committed prediction

The prediction log is append-only.

If a correction is necessary, record a new entry explaining the
correction rather than silently editing the original prediction.

Historical records should remain inspectable.

---

## 3. Never force-push published prediction history

Force-pushing can rewrite commit SHAs associated with previously
published predictions.

Do not rewrite published prediction history.

If a historical correction is required, make a new commit that
documents the correction.

---

## 4. Never delete referenced snapshots

Every snapshot referenced by a manifest is part of the evidence required
to reproduce the prediction.

Deleting it can turn a previously verifiable prediction into
`CANNOT VERIFY`.

Snapshots should therefore be retained for as long as their associated
predictions remain part of the public record.

---

## 5. Prefer a clean working tree

Run the live monitor from a clean git working tree whenever possible.

A clean tree gives the strongest connection between:

```text
prediction
    |
    v
git SHA
    |
    v
source code
```

Predictions created from dirty trees should be treated as having weaker
code provenance.

---

# Reporting a Failed Verification

If:

```text
RESULT: NOT VERIFIED
```

is reported, preserve the evidence.

Capture:

1. the prediction ID
2. the complete output of:

```bash
python scripts/reproduce.py <prediction-id> --verbose
```

3. the current repository SHA:

```bash
git rev-parse HEAD
```

4. the manifest contents
5. the hashes recorded by the manifest
6. the hashes of the corresponding files currently on disk
7. the current dependency versions

For example:

```bash
git rev-parse HEAD
python scripts/reproduce.py GSPC_2026-09-11_10 --verbose
pip freeze
```

A failed verification is itself a useful audit finding.

It can indicate:

* data modification
* parameter modification
* manifest modification
* code changes
* numerical incompatibility
* accidental deletion
* an implementation bug
* an operational mistake

The important property is that the system exposes the discrepancy.

---

# Recommended Independent Audit Procedure

A skeptical third party can perform the following sequence without
trusting the project operator's interpretation.

### Step 1 — obtain the repository

```bash
git clone <this-repo>
cd auditable-volatility
pip install -r requirements.txt
```

### Step 2 — inspect the repository history

```bash
git log --oneline --all --decorate
```

### Step 3 — inspect the prediction record

```bash
git log -p -- data/predictions/GSPC.jsonl
```

### Step 4 — verify the latest prediction

```bash
python scripts/reproduce.py --latest GSPC --verbose
```

### Step 5 — verify all available predictions

```bash
python scripts/reproduce.py --all GSPC
```

### Step 6 — inspect the recorded code SHA

Read the manifest and compare its `git_sha` with the repository history.

### Step 7 — manually reproduce if desired

Use the `multistep_forecast` procedure described above.

### Step 8 — inspect methodology separately

Read:

* [METHODOLOGY.md](METHODOLOGY.md)
* [RESULTS.md](RESULTS.md)

The audit establishes reproducibility.

The methodology establishes what the forecast is supposed to mean.

The results establish how the forecasting system performed.

These are deliberately separate claims.

---

# Summary

A reviewer can use this document to:

* verify the latest prediction
* verify a specific prediction
* verify every available prediction for a ticker
* inspect the content-addressed input and parameter snapshots
* check the git history for prediction-log rewrites
* reproduce the forecast manually
* inspect the exact code SHA associated with a prediction
* distinguish numerical verification from model validation
* understand the limitations of the audit claim
* diagnose and report failed verification

The core claim is deliberately narrow:

> **The recorded forecast can be independently reproduced from the
> recorded evidence within a declared tolerance.**

That is what this audit system establishes.

It does not claim that the forecast is correct, that the model is good,
or that the external data source was truthful.

Those questions require separate evidence.

That separation is intentional: an auditable system should make clear
not only what can be verified, but also what cannot.
