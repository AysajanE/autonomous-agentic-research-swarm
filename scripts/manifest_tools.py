#!/usr/bin/env python3
"""Writers for content-bound research manifests."""

from __future__ import annotations

import argparse
import hashlib
from importlib import metadata
import json
import re
from pathlib import Path
import subprocess
import sys
from typing import Iterable


PROCESSED_MANIFEST_SCHEMA_VERSION = "research_swarm.processed_manifest.v2"


def _run_git(repo: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout


def _repo_path(repo: Path, value: str | Path) -> tuple[str, Path]:
    raw_path = Path(value)
    disk_path = raw_path if raw_path.is_absolute() else repo / raw_path
    resolved = disk_path.resolve()
    try:
        relative = resolved.relative_to(repo).as_posix()
    except ValueError as exc:
        raise ValueError(f"path_outside_repo:{value}") from exc
    return relative, resolved


def _sha256_and_bytes(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def _dependency_versions(names: Iterable[str]) -> dict[str, str]:
    versions: dict[str, str] = {}
    for name in names:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            continue
    return versions


def _declared_dependencies(repo: Path) -> tuple[str, ...]:
    """Names of the project's declared runtime dependencies (pyproject.toml).
    Falls back to the known core pair if the file is unreadable."""
    pyproject = repo / "pyproject.toml"
    try:
        import tomllib

        payload = tomllib.loads(pyproject.read_text(encoding="utf-8"))
        deps = payload.get("project", {}).get("dependencies", [])
        names = []
        for spec in deps:
            if isinstance(spec, str) and spec.strip():
                name = re.split(r"[<>=!~\[; ]", spec.strip(), maxsplit=1)[0]
                if name:
                    names.append(name)
        if names:
            return tuple(names)
    except Exception:
        pass
    return ("pandas", "matplotlib")


def write_processed_manifest(
    *,
    repo: Path,
    manifest_path: str | Path,
    as_of_utc_date: str,
    inputs: Iterable[str | Path],
    script_path: str | Path,
    command: str,
    outputs: Iterable[str | Path],
    allow_dirty_with_diff: bool = False,
) -> dict[str, object]:
    repo = repo.resolve()
    dirty_status = _run_git(repo, "status", "--porcelain")
    dirty = bool(dirty_status.strip())
    if dirty and not allow_dirty_with_diff:
        raise SystemExit(f"dirty_tree_manifest_refused:{dirty_status.strip()}")

    normalized_script_path, script_disk_path = _repo_path(repo, script_path)
    script_sha256, _ = _sha256_and_bytes(script_disk_path)

    output_entries: list[dict[str, object]] = []
    for output in outputs:
        normalized_output, output_disk_path = _repo_path(repo, output)
        output_sha256, output_bytes = _sha256_and_bytes(output_disk_path)
        output_entries.append(
            {
                "path": normalized_output,
                "sha256": output_sha256,
                "bytes": output_bytes,
            }
        )

    transform: dict[str, object] = {
        "script_path": normalized_script_path,
        "script_sha256": script_sha256,
        "git_sha": _run_git(repo, "rev-parse", "HEAD").strip(),
        "command": command,
        "dirty": dirty,
    }
    if dirty:
        transform["tree_diff"] = _run_git(repo, "diff", "--binary", "--no-ext-diff", "HEAD", "--")
        transform["tree_status"] = dirty_status
        untracked_entries: list[dict[str, object]] = []
        for line in _run_git(repo, "ls-files", "--others", "--exclude-standard").splitlines():
            rel = line.strip()
            if not rel:
                continue
            candidate = repo / rel
            if candidate.is_file():
                sha256, size = _sha256_and_bytes(candidate)
                untracked_entries.append({"path": rel, "sha256": sha256, "bytes": size})
        transform["untracked_files"] = untracked_entries

    normalized_inputs = [_repo_path(repo, input_path)[0] for input_path in inputs]
    payload: dict[str, object] = {
        "schema_version": PROCESSED_MANIFEST_SCHEMA_VERSION,
        "as_of_utc_date": as_of_utc_date,
        "inputs": normalized_inputs,
        "transform": transform,
        "outputs": output_entries,
        "environment": {
            "python": sys.version.split()[0],
            "dependencies": _dependency_versions(_declared_dependencies(repo)),
        },
    }

    _, manifest_disk_path = _repo_path(repo, manifest_path)
    manifest_disk_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_disk_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return payload


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="manifest_tools.py")
    subparsers = parser.add_subparsers(dest="subcommand", required=True)
    write_parser = subparsers.add_parser("write-processed")
    write_parser.add_argument("--manifest", required=True)
    write_parser.add_argument("--as-of", required=True)
    write_parser.add_argument("--script", required=True)
    write_parser.add_argument("--command", required=True)
    write_parser.add_argument("--input", action="append", default=[])
    write_parser.add_argument("--output", action="append", required=True)
    write_parser.add_argument("--allow-dirty-with-diff", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.subcommand == "write-processed":
        payload = write_processed_manifest(
            repo=Path.cwd(),
            manifest_path=args.manifest,
            as_of_utc_date=args.as_of,
            inputs=args.input,
            script_path=args.script,
            command=args.command,
            outputs=args.output,
            allow_dirty_with_diff=args.allow_dirty_with_diff,
        )
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    raise AssertionError(f"unknown_subcommand:{args.subcommand}")


if __name__ == "__main__":
    raise SystemExit(main())
