from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import shlex
import sys
from datetime import date, datetime, timezone
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


def _parse_utc_date(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise SystemExit(f"Invalid --as-of date (expected YYYY-MM-DD): {value!r}") from exc


def _infer_as_of_from_snapshot_dir(snapshot_dir: Path) -> date | None:
    m = re.fullmatch(r"\d{4}-\d{2}-\d{2}", snapshot_dir.name)
    if m is None:
        return None
    return _parse_utc_date(snapshot_dir.name)


def build_manifest(source: str, snapshot_dir: Path, command: str, *, as_of: date) -> dict[str, object]:
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
        "as_of_utc_date": as_of.isoformat(),
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
    p = argparse.ArgumentParser(prog="make_raw_manifest.py")
    p.add_argument("source")
    p.add_argument("snapshot_dir")
    p.add_argument("--as-of", dest="as_of", default=None, help="UTC snapshot date (YYYY-MM-DD)")
    p.add_argument("--out", dest="out_path", default=None, help="Optional output path for the manifest JSON")
    p.add_argument(
        "command",
        nargs=argparse.REMAINDER,
        help="Command used to produce the snapshot (precede with --, e.g. -- python src/etl/foo.py ...)",
    )
    args = p.parse_args(argv[1:])

    source = args.source
    snapshot_dir = Path(args.snapshot_dir)
    snap_abs = snapshot_dir if snapshot_dir.is_absolute() else (_repo_root() / snapshot_dir)

    as_of: date | None
    if args.as_of is not None:
        as_of = _parse_utc_date(args.as_of)
    else:
        as_of = _infer_as_of_from_snapshot_dir(snap_abs)
    if as_of is None:
        raise SystemExit("Missing --as-of and could not infer date from snapshot_dir folder name (expected .../<YYYY-MM-DD>/)")

    if not args.command:
        raise SystemExit("Missing command. Provide it after --, e.g. -- python src/etl/foo.py --run-date ...")
    if args.command and args.command[0] == "--":
        cmd_tokens = args.command[1:]
    else:
        cmd_tokens = args.command
    if not cmd_tokens:
        raise SystemExit("Missing command tokens after --")
    command = " ".join(shlex.quote(t) for t in cmd_tokens)

    manifest = build_manifest(source=source, snapshot_dir=snapshot_dir, command=command, as_of=as_of)
    out_dir = _repo_root() / "data/raw_manifest"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.out_path) if args.out_path else (out_dir / f"{source}_{as_of.isoformat()}.json")
    out_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main(sys.argv)
