from pathlib import Path

root = Path(__file__).resolve().parent
if not (root / "instance_manifest.json").is_file():
    raise SystemExit("instance_manifest_missing")
print("derivation_check:ok")

