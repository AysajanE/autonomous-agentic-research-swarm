import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
source = root / "data/processed_manifest/source.json"
destination = root / "modeling/instance_manifest.json"
payload = {
    "instance_id": "toy-hybrid-1",
    "seed": 23,
    "source_manifest": {
        "path": "data/processed_manifest/source.json",
        "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
    },
}
destination.parent.mkdir(parents=True, exist_ok=True)
destination.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(f"generated:{destination.relative_to(root)}")
