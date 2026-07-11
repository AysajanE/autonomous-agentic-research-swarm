from pathlib import Path
import hashlib

source = Path(__file__).resolve().parents[1] / "data/processed_manifest/source.json"
print(hashlib.sha256(source.read_bytes()).hexdigest())

