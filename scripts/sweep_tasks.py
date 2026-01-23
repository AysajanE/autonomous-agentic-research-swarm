#!/usr/bin/env python3
"""
Planner sweep tool: move task files between lifecycle folders based on `State:`.

Why:
- Repo governance says ONLY the Planner moves tasks across lifecycle folders.
- Workers/Judges update `State:` in their branch/worktree; this sweeper enforces folder/state alignment.

This tool is deterministic and makes no network calls.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import subprocess
import sys


VALID_TASK_STATES = {"backlog", "active", "blocked", "ready_for_review", "done"}
LIFECYCLE_DIRS = ["backlog", "active", "ready_for_review", "blocked", "done"]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _run(cmd: list[str], *, cwd: Path, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, cwd=str(cwd), check=check, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)


def _parse_state(text: str) -> str | None:
    m = re.search(r"^\s*-\s*State:\s*(\S+)\s*$", text, flags=re.MULTILINE)
    return m.group(1).strip() if m else None


def iter_task_files(orchestrator_dir: Path) -> list[Path]:
    paths: list[Path] = []
    for sub in LIFECYCLE_DIRS:
        d = orchestrator_dir / sub
        if not d.exists():
            continue
        for p in sorted(d.glob("*.md")):
            if p.name == "README.md":
                continue
            paths.append(p)
    return paths


def sweep(*, repo: Path, dry_run: bool) -> tuple[list[str], list[str]]:
    orch = repo / ".orchestrator"
    moves: list[str] = []
    problems: list[str] = []

    for path in iter_task_files(orch):
        current_folder = path.parent.name
        text = _read_text(path)
        state = _parse_state(text)
        if state is None:
            problems.append(f"{path}: missing State line")
            continue
        if state not in VALID_TASK_STATES:
            problems.append(f"{path}: invalid State {state!r}")
            continue
        if current_folder == state:
            continue

        dest_dir = orch / state
        dest = dest_dir / path.name
        moves.append(f"{path} -> {dest}")
        if dry_run:
            continue

        dest_dir.mkdir(parents=True, exist_ok=True)
        _run(["git", "mv", str(path), str(dest)], cwd=repo, check=True)

    return moves, problems


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(prog="sweep_tasks.py")
    p.add_argument("--dry-run", action="store_true", help="Print planned moves without changing files")
    args = p.parse_args(argv)

    repo = _repo_root()
    moves, problems = sweep(repo=repo, dry_run=bool(args.dry_run))

    for line in moves:
        print(f"[move] {line}")
    for line in problems:
        print(f"[problem] {line}", file=sys.stderr)

    if args.dry_run:
        print(f"Dry-run complete. Planned moves: {len(moves)}; problems: {len(problems)}")
    else:
        print(f"Sweep complete. Moves: {len(moves)}; problems: {len(problems)}")

    return 0 if len(problems) == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

