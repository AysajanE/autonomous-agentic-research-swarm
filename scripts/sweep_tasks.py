#!/usr/bin/env python3
"""
Planner/Operator sweep tool for lifecycle-folder projection.

`State:` inside each task file is authoritative.
Folder placement under `.orchestrator/` is only a projection.

This tool makes no network calls.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import subprocess
import sys


DEFAULT_PROJECTION_DIRS = (
    "backlog",
    "active",
    "integration_ready",
    "ready_for_review",
    "blocked",
    "done",
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _run(cmd: list[str], *, cwd: Path, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=str(cwd),
        check=check,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )


def _load_projection_dirs(repo: Path) -> tuple[str, ...]:
    framework_path = repo / "contracts" / "framework.json"
    if not framework_path.exists():
        return DEFAULT_PROJECTION_DIRS

    try:
        payload = json.loads(_read_text(framework_path))
    except Exception:
        return DEFAULT_PROJECTION_DIRS

    if not isinstance(payload, dict):
        return DEFAULT_PROJECTION_DIRS

    states = payload.get("states")
    if not isinstance(states, dict):
        return DEFAULT_PROJECTION_DIRS

    raw_dirs = states.get("projection_dirs")
    if not isinstance(raw_dirs, list):
        return DEFAULT_PROJECTION_DIRS

    normalized = [Path(str(item)).name for item in raw_dirs if isinstance(item, str) and str(item).strip()]
    return tuple(normalized) or DEFAULT_PROJECTION_DIRS


def _parse_state(text: str) -> str | None:
    match = re.search(r"^\s*-\s*State:\s*(\S+)\s*$", text, flags=re.MULTILINE)
    return match.group(1).strip() if match else None


def _iter_task_files(orchestrator_dir: Path, projection_dirs: tuple[str, ...]) -> list[Path]:
    paths: list[Path] = []
    for folder_name in projection_dirs:
        folder = orchestrator_dir / folder_name
        if not folder.exists():
            continue
        for path in sorted(folder.glob("*.md")):
            if path.name == "README.md":
                continue
            paths.append(path)
    return paths


def plan_sweep(repo: Path) -> tuple[list[tuple[Path, Path]], list[str]]:
    projection_dirs = _load_projection_dirs(repo)
    valid_states = set(projection_dirs)
    orchestrator_dir = repo / ".orchestrator"

    moves: list[tuple[Path, Path]] = []
    problems: list[str] = []

    for path in _iter_task_files(orchestrator_dir, projection_dirs):
        current_folder = path.parent.name
        state = _parse_state(_read_text(path))

        if state is None:
            problems.append(f"{path}:missing_state")
            continue
        if state not in valid_states:
            problems.append(f"{path}:invalid_state:{state}")
            continue
        if current_folder == state:
            continue

        destination = orchestrator_dir / state / path.name
        moves.append((path, destination))

    return moves, problems


def _apply_moves(repo: Path, moves: list[tuple[Path, Path]]) -> None:
    for source, destination in moves:
        destination.parent.mkdir(parents=True, exist_ok=True)
        cp = _run(["git", "mv", str(source), str(destination)], cwd=repo, check=False)
        if cp.returncode != 0:
            source.rename(destination)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="sweep_tasks.py")
    parser.add_argument("--dry-run", action="store_true", help="Report planned moves without changing files")
    parser.add_argument("--check", action="store_true", help="Exit nonzero if any move is required or any problem exists")
    parser.add_argument("--json", action="store_true", help="Print machine-readable output")
    args = parser.parse_args(argv)

    repo = _repo_root()
    moves, problems = plan_sweep(repo)

    if not args.dry_run and not args.check:
        _apply_moves(repo, moves)

    payload = {
        "moves": [{"source": str(source), "target": str(target)} for source, target in moves],
        "problems": problems,
        "dry_run": bool(args.dry_run),
        "check": bool(args.check),
    }

    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        for source, target in moves:
            print(f"[move] {source} -> {target}")
        for problem in problems:
            print(f"[problem] {problem}", file=sys.stderr)

        if args.check:
            print(f"Check complete. Pending moves: {len(moves)}; problems: {len(problems)}")
        elif args.dry_run:
            print(f"Dry-run complete. Planned moves: {len(moves)}; problems: {len(problems)}")
        else:
            print(f"Sweep complete. Moves: {len(moves)}; problems: {len(problems)}")

    if problems:
        return 1
    if args.check and moves:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
