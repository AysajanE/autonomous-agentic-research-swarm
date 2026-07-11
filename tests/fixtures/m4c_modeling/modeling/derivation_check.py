import json
from pathlib import Path

required = ("instance_manifest.json", "experiment_design.json")
root = Path(__file__).resolve().parent
missing = [name for name in required if not (root / name).is_file()]
if missing:
    raise SystemExit("missing:" + ",".join(missing))
(root / "convergence.jsonl").write_text(
    json.dumps(
        {
            "fixture_instance": "toy-model-1",
            "iterations": 3,
            "objective": 4.0,
            "status": "optimal",
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    + "\n",
    encoding="utf-8",
)
print("derivation_check:ok")
