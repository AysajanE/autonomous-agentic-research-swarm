from pathlib import Path

required = ("instance_manifest.json", "experiment_design.json", "convergence.log")
root = Path(__file__).resolve().parent
missing = [name for name in required if not (root / name).is_file()]
if missing:
    raise SystemExit("missing:" + ",".join(missing))
print("derivation_check:ok")

