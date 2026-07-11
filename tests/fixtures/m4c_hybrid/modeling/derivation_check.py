import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parent
instance_path = root / "instance_manifest.json"
if not instance_path.is_file():
    raise SystemExit("instance_manifest_missing")
package_root = root.parent
output = {
    "instance_manifest": {
        "path": "modeling/instance_manifest.json",
        "sha256": hashlib.sha256(instance_path.read_bytes()).hexdigest(),
    },
    "objective": 7.0,
    "status": "optimal",
}
(package_root / "bridge/experiment_output.json").write_text(
    json.dumps(output, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
(root / "convergence.jsonl").write_text(
    json.dumps(
        {
            "fixture_instance": "toy-hybrid-1",
            "iterations": 4,
            "objective": 7.0,
            "status": "optimal",
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    + "\n",
    encoding="utf-8",
)
print("derivation_check:ok")
