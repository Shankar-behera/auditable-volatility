import json
from src.manifest import hash_file, read_manifest

with open("data/predictions/GSPC.jsonl") as f:
    lines = [json.loads(l) for l in f if l.strip()]

# Predictions are rows without a "resolved" key; the latest one with a
# manifest is what we want to verify.
preds = [p for p in lines if p.get("resolved") is not True]
with_manifest = [p for p in preds if p.get("manifest_path")]

if not with_manifest:
    print("no predictions have a manifest_path -- nothing to check")
    raise SystemExit(1)

latest = with_manifest[-1]
print(f"prediction_id:          {latest['prediction_id']}")
print(f"logged manifest_sha256: {latest['manifest_sha256'][:16]}...")

actual = hash_file(latest["manifest_path"])
print(f"actual file sha256:     {actual[:16]}...")
assert actual == latest["manifest_sha256"], "MISMATCH"
print("manifest hash matches log: PASS")

m = read_manifest(latest["prediction_id"])
print(f"  price_snapshot: {m['price_snapshot']['path']}")
print(f"    sha256={m['price_snapshot']['sha256'][:16]}...")
print(f"  garch_params:   {m['garch_parameters']['path']}")
print(f"    sha256={m['garch_parameters']['sha256'][:16]}...")
print(f"  git_sha:        {m['code']['git_sha']}")
print(f"  tolerance_abs:  {m['verification']['tolerance_abs']}")