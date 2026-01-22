from __future__ import annotations

import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _ensure_within_repo(root: Path, target: Path) -> Path:
    try:
        return target.resolve().relative_to(root.resolve())
    except ValueError as exc:
        raise SystemExit(f"snapshot_dir must be inside repo root: {root}") from exc


def build_manifest(source: str, snapshot_dir: Path, command: str) -> dict[str, object]:
    root = _repo_root()
    snap = snapshot_dir if snapshot_dir.is_absolute() else (root / snapshot_dir)
    if not snap.exists():
        raise SystemExit(f"snapshot_dir does not exist: {snap}")
    if not snap.is_dir():
        raise SystemExit(f"snapshot_dir is not a directory: {snap}")

    files: list[dict[str, object]] = []
    for p in sorted(snap.rglob("*")):
        if not p.is_file():
            continue
        rel = _ensure_within_repo(root, p)
        files.append(
            {
                "path": str(rel),
                "sha256": _sha256_file(p),
                "bytes": p.stat().st_size,
            }
        )

    now = datetime.now(timezone.utc)
    return {
        "source": source,
        "fetched_at_utc": now.isoformat(),
        "command": command,
        "files": files,
        "environment": {
            "python_version": sys.version.split()[0],
            "python_implementation": platform.python_implementation(),
            "platform": platform.platform(),
        },
    }


def main(argv: list[str]) -> None:
    if len(argv) < 4:
        raise SystemExit(
            "usage: python scripts/make_raw_manifest.py <source> <snapshot_dir> <command>"
        )
    source = argv[1]
    snapshot_dir = Path(argv[2])
    command = " ".join(argv[3:])

    manifest = build_manifest(source=source, snapshot_dir=snapshot_dir, command=command)
    now = datetime.now(timezone.utc)
    out_dir = _repo_root() / "data/raw_manifest"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{source}_{now.date().isoformat()}.json"
    out_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main(sys.argv)
