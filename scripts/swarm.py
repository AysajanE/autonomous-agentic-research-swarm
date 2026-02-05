#!/usr/bin/env python3
"""
Unattended research swarm supervisor (tmux-friendly).

Design goals:
- Use the repo as shared memory: tasks live in `.orchestrator/`.
- Run tasks in isolated git worktrees/branches.
- Support "Planner → Worker → Judge" with CLI agents:
  - Planner: Claude Code (optional; heuristic fallback)
  - Worker:  Codex CLI
  - Judge:   deterministic gates (make) + optional Codex review

Governance (Option A):
- This script updates task `State:` and notes, but does NOT move task files across lifecycle folders.
- Use `python scripts/sweep_tasks.py` (Planner) to move task files based on `State:`.

Safety:
- Unattended mode disables approval prompts in Codex/Claude. ONLY run in an external sandbox
  (VM/devcontainer/Codespaces) that contains ONLY this repo and no sensitive files (see AGENTS.md).
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as _dt
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time
from typing import Any, Iterable


VALID_TASK_STATES = {"backlog", "active", "blocked", "ready_for_review", "done"}
VALID_TASK_PRIORITIES = {"low", "medium", "high"}
_PREFLIGHT_STRICT_SYNC_CACHE: set[tuple[str, bool, bool]] = set()


@dataclasses.dataclass(frozen=True)
class Task:
    path: Path
    task_id: str
    title: str
    workstream: str
    role: str
    priority: str
    dependencies: list[str]
    parallel_ok: bool
    allowed_paths: list[str]
    disallowed_paths: list[str]
    outputs: list[str]
    gates: list[str]
    stop_conditions: list[str]
    state: str | None
    last_updated: str | None


def _utc_today() -> str:
    return _dt.datetime.now(tz=_dt.timezone.utc).date().isoformat()


def _utc_timestamp_compact() -> str:
    return _dt.datetime.now(tz=_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _run(
    cmd: list[str],
    *,
    cwd: Path | None = None,
    check: bool = True,
    capture: bool = False,
    env: dict[str, str] | None = None,
    timeout_seconds: int | None = None,
) -> subprocess.CompletedProcess[str]:
    if capture:
        return subprocess.run(
            cmd,
            cwd=str(cwd) if cwd else None,
            check=check,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=env,
            timeout=timeout_seconds,
        )
    return subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        check=check,
        text=True,
        env=env,
        timeout=timeout_seconds,
    )


def _which_or_none(name: str) -> str | None:
    for p in os.environ.get("PATH", "").split(os.pathsep):
        cand = Path(p) / name
        if cand.exists() and os.access(cand, os.X_OK):
            return str(cand)
    return None


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _parse_task_frontmatter(text: str) -> dict[str, object] | None:
    """Minimal YAML frontmatter parser (no external deps).

    Supports:
    - `key: value`
    - `key: [a, b]`
    - `key:` followed by indented `- item` lines
    """
    lines = text.splitlines()
    if len(lines) < 3 or lines[0].strip() != "---":
        return None
    end_idx = None
    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            end_idx = i
            break
    if end_idx is None:
        return None

    data: dict[str, object] = {}
    current_list_key: str | None = None
    for raw_line in lines[1:end_idx]:
        line = raw_line.split("#", 1)[0].rstrip()
        if line.strip() == "":
            continue

        list_item_match = re.match(r"^\s*-\s+(.*)\s*$", line)
        if current_list_key is not None and list_item_match is not None:
            item = list_item_match.group(1).strip().strip("'\"")
            current_list = data.get(current_list_key)
            if isinstance(current_list, list):
                current_list.append(item)
            continue

        current_list_key = None
        if ":" not in line:
            continue
        key, rest = line.split(":", 1)
        key = key.strip()
        rest = rest.strip()

        if rest == "":
            data[key] = []
            current_list_key = key
            continue

        if rest.startswith("[") and rest.endswith("]"):
            inner = rest[1:-1].strip()
            if inner == "":
                data[key] = []
            else:
                items = [x.strip().strip("'\"") for x in inner.split(",") if x.strip()]
                data[key] = items
            continue

        data[key] = rest.strip("'\"")

    return data


def _parse_task_state(text: str) -> str | None:
    match = re.search(r"^\s*-\s*State:\s*(\S+)\s*$", text, flags=re.MULTILINE)
    return match.group(1).strip() if match else None


def _parse_task_last_updated(text: str) -> str | None:
    match = re.search(r"^\s*-\s*Last updated:\s*(\d{4}-\d{2}-\d{2})\s*$", text, flags=re.MULTILINE)
    return match.group(1).strip() if match else None


def _coerce_list(value: object) -> list[str]:
    if isinstance(value, list):
        out: list[str] = []
        for x in value:
            if isinstance(x, str):
                out.append(x)
        return out
    return []


def load_task(path: Path) -> Task:
    text = _read_text(path)
    fm = _parse_task_frontmatter(text)
    if fm is None:
        raise ValueError(f"Task missing YAML frontmatter: {path}")

    def _get_str(key: str) -> str:
        v = fm.get(key)
        if not isinstance(v, str):
            raise ValueError(f"Task {path} missing/invalid frontmatter key: {key}")
        return v

    task_id = _get_str("task_id")
    title = _get_str("title")
    workstream = _get_str("workstream")
    role = _get_str("role")
    priority = _get_str("priority").lower()

    dependencies = _coerce_list(fm.get("dependencies"))
    parallel_ok = False
    raw_parallel_ok = fm.get("parallel_ok")
    if isinstance(raw_parallel_ok, str):
        parallel_ok = raw_parallel_ok.strip().lower() in {"1", "true", "yes"}
    allowed_paths = _coerce_list(fm.get("allowed_paths"))
    disallowed_paths = _coerce_list(fm.get("disallowed_paths"))
    outputs = _coerce_list(fm.get("outputs"))
    gates = _coerce_list(fm.get("gates"))
    stop_conditions = _coerce_list(fm.get("stop_conditions"))

    state = _parse_task_state(text)
    last_updated = _parse_task_last_updated(text)

    return Task(
        path=path,
        task_id=task_id,
        title=title,
        workstream=workstream,
        role=role,
        priority=priority,
        dependencies=dependencies,
        parallel_ok=parallel_ok,
        allowed_paths=allowed_paths,
        disallowed_paths=disallowed_paths,
        outputs=outputs,
        gates=gates,
        stop_conditions=stop_conditions,
        state=state,
        last_updated=last_updated,
    )


def iter_task_files(dir_path: Path) -> Iterable[Path]:
    if not dir_path.exists():
        return []
    for p in sorted(dir_path.glob("*.md")):
        if p.name == "README.md":
            continue
        yield p


def list_tasks(dir_path: Path) -> list[Task]:
    tasks: list[Task] = []
    for p in iter_task_files(dir_path):
        tasks.append(load_task(p))
    return tasks


def task_dir(name: str) -> Path:
    return _repo_root() / ".orchestrator" / name


def done_task_ids() -> set[str]:
    out: set[str] = set()
    # State-based completion: treat tasks as "done" if their `State:` is `done`,
    # regardless of which lifecycle folder they currently live in. Folder moves are
    # still useful for hygiene, but should not be required for dependency progress.
    for sub in ["active", "backlog", "ready_for_review", "blocked", "done"]:
        for t in list_tasks(task_dir(sub)):
            if t.state == "done":
                out.add(t.task_id)
    return out


def _parse_task_id_from_branch(name: str) -> str | None:
    m = re.match(r"^(T\d{3})\b", name)
    return m.group(1) if m else None


def claimed_task_ids(remote: str, base_branch: str) -> set[str]:
    """Detect claimed tasks via open PRs (preferred) or remote branches (fallback)."""
    claimed: set[str] = set()

    # Local fallback (no network): any task-prefixed branch currently attached to a worktree.
    try:
        cp = _run(
            ["git", "worktree", "list", "--porcelain"],
            capture=True,
            check=True,
            cwd=_repo_root(),
        )
        for line in (cp.stdout or "").splitlines():
            if not line.startswith("branch "):
                continue
            ref = line.split(" ", 1)[1].strip()
            if ref.startswith("refs/heads/"):
                branch = ref.removeprefix("refs/heads/")
                tid = _parse_task_id_from_branch(branch)
                if tid is not None:
                    claimed.add(tid)
    except Exception:
        pass

    gh = _which_or_none("gh")
    if gh is not None:
        try:
            cp = _run(
                [gh, "pr", "list", "--state", "open", "--base", base_branch, "--json", "headRefName"],
                capture=True,
                check=True,
                cwd=_repo_root(),
            )
            prs = json.loads(cp.stdout or "[]")
            if isinstance(prs, list):
                for pr in prs:
                    if not isinstance(pr, dict):
                        continue
                    head = pr.get("headRefName")
                    if isinstance(head, str):
                        tid = _parse_task_id_from_branch(head)
                        if tid is not None:
                            claimed.add(tid)
        except Exception:
            pass

    # Fallback: any remote branch with prefix T###_
    try:
        cp = _run(
            ["git", "ls-remote", "--heads", remote, "T[0-9][0-9][0-9]_*"],
            capture=True,
            check=True,
            cwd=_repo_root(),
        )
        for line in (cp.stdout or "").splitlines():
            parts = line.split("\t")
            if len(parts) != 2:
                continue
            ref = parts[1].strip()
            if ref.startswith("refs/heads/"):
                branch = ref.removeprefix("refs/heads/")
                tid = _parse_task_id_from_branch(branch)
                if tid is not None:
                    claimed.add(tid)
    except Exception:
        pass

    return claimed


def ready_backlog_tasks(*, done_ids: set[str], claimed_ids: set[str]) -> list[Task]:
    tasks = [t for t in list_tasks(task_dir("backlog")) if t.state == "backlog"]
    ready: list[Task] = []
    for t in tasks:
        if t.task_id in claimed_ids:
            continue
        if all(dep in done_ids for dep in t.dependencies):
            ready.append(t)
    return ready


def _compute_workstream_locks(*, repo: Path, claimed_ids: set[str]) -> tuple[set[str], set[str]]:
    """Return (locked_workstreams, parallel_only_workstreams).

    - If any claimed task in a workstream is not parallel_ok, the workstream is locked.
    - If only parallel_ok tasks are claimed for a workstream, the workstream is parallel-only.
    """
    locked: set[str] = set()
    parallel_only: set[str] = set()
    for tid in claimed_ids:
        tf = _find_task_file_anywhere(tid, repo)
        if tf is None:
            continue
        try:
            t = load_task(tf)
        except Exception:
            continue
        if t.parallel_ok:
            parallel_only.add(t.workstream)
        else:
            locked.add(t.workstream)
    parallel_only.difference_update(locked)
    return locked, parallel_only


def _apply_workstream_concurrency_filters(
    *,
    tasks: list[Task],
    locked_workstreams: set[str],
    parallel_only_workstreams: set[str],
    capacity: int,
) -> list[Task]:
    """Filter a task list to enforce simple workstream-level concurrency rules."""
    selected: list[Task] = []
    selected_workstreams: set[str] = set()
    for t in tasks:
        if t.workstream in locked_workstreams:
            continue
        if t.workstream in parallel_only_workstreams and not t.parallel_ok:
            continue
        if t.workstream in selected_workstreams and not t.parallel_ok:
            continue
        selected.append(t)
        selected_workstreams.add(t.workstream)
        if len(selected) >= max(0, capacity):
            break
    return selected


def _priority_rank(priority: str) -> int:
    return {"high": 0, "medium": 1, "low": 2}.get(priority, 9)


def choose_tasks_heuristic(ready: list[Task], capacity: int) -> list[Task]:
    ready_sorted = sorted(ready, key=lambda t: (_priority_rank(t.priority), t.task_id))
    return ready_sorted[: max(0, capacity)]


def choose_tasks_claude(
    *,
    ready: list[Task],
    capacity: int,
    model: str | None,
    unattended: bool,
) -> list[Task]:
    claude = _which_or_none("claude")
    if claude is None:
        print("claude not found; falling back to heuristic planner", file=sys.stderr)
        return choose_tasks_heuristic(ready, capacity)

    payload = [
        {
            "task_id": t.task_id,
            "title": t.title,
            "workstream": t.workstream,
            "priority": t.priority,
            "dependencies": t.dependencies,
            "parallel_ok": t.parallel_ok,
        }
        for t in ready
    ]

    schema = {
        "type": "object",
        "properties": {
            "selected_task_ids": {"type": "array", "items": {"type": "string"}},
            "rationale": {"type": "string"},
        },
        "required": ["selected_task_ids"],
        "additionalProperties": True,
    }

    prompt = "\n".join(
        [
            "Role: Planner.",
            "You are selecting which tasks to start right now for an autonomous research swarm.",
            "",
            "Rules:",
            f"- Select at most {capacity} task_ids.",
            "- Prefer higher priority tasks.",
            "- Prefer tasks that unblock dependencies.",
            "- Start at most ONE task per workstream unless tasks are marked parallel_ok=true.",
            "- Return ONLY the JSON object required by the schema (selected_task_ids, optional rationale).",
            "",
            "Ready tasks (JSON):",
            json.dumps(payload, indent=2, sort_keys=True),
        ]
    )

    cmd: list[str] = [
        claude,
        "-p",
        prompt,
        "--output-format",
        "json",
        "--json-schema",
        json.dumps(schema),
        "--tools",
        "",
    ]
    if model:
        cmd.extend(["--model", model])
    if unattended:
        # IMPORTANT: Only use unattended bypass modes in external sandboxes.
        cmd.extend(["--permission-mode", "bypassPermissions"])

    cp = _run(cmd, capture=True, check=True, cwd=_repo_root())
    data = json.loads(cp.stdout or "{}")
    structured = data.get("structured_output")
    if not isinstance(structured, dict):
        print("claude planner did not return structured_output; falling back", file=sys.stderr)
        return choose_tasks_heuristic(ready, capacity)
    selected = structured.get("selected_task_ids")
    if not isinstance(selected, list):
        print("claude planner missing selected_task_ids; falling back", file=sys.stderr)
        return choose_tasks_heuristic(ready, capacity)

    selected_ids = {x for x in selected if isinstance(x, str)}
    out = [t for t in ready if t.task_id in selected_ids]
    # Preserve a stable order (priority then id) if Claude returns many.
    out_sorted = sorted(out, key=lambda t: (_priority_rank(t.priority), t.task_id))
    return out_sorted[: max(0, capacity)]


def _slug_from_task_path(path: Path, task_id: str) -> str:
    stem = path.stem
    prefix = f"{task_id}_"
    if stem.startswith(prefix):
        return stem[len(prefix) :]
    return stem


def ensure_worktree(*, task: Task, worktree_parent: Path, base_ref: str) -> tuple[Path, str]:
    """Create a new worktree for a task branch.

    Returns (worktree_path, branch_name).
    """
    slug = _slug_from_task_path(task.path, task.task_id)
    branch = f"{task.task_id}_{slug}"
    wt_path = worktree_parent / f"wt-{task.task_id}"

    if wt_path.exists():
        raise SystemExit(f"Worktree path already exists: {wt_path}")

    # If branch exists locally, attach it; otherwise create from base_ref.
    branch_exists = False
    try:
        _run(["git", "show-ref", "--verify", "--quiet", f"refs/heads/{branch}"], cwd=_repo_root())
        branch_exists = True
    except subprocess.CalledProcessError:
        branch_exists = False

    if branch_exists:
        _run(["git", "worktree", "add", str(wt_path), branch], cwd=_repo_root())
    else:
        _run(["git", "worktree", "add", str(wt_path), "-b", branch, base_ref], cwd=_repo_root())

    return wt_path, branch


def _find_worktree_path_for_branch(branch: str) -> Path | None:
    try:
        cp = _run(
            ["git", "worktree", "list", "--porcelain"],
            cwd=_repo_root(),
            capture=True,
            check=True,
        )
        current_path: Path | None = None
        current_branch: str | None = None
        for line in (cp.stdout or "").splitlines():
            if line.startswith("worktree "):
                current_path = Path(line.split(" ", 1)[1].strip())
                current_branch = None
                continue
            if line.startswith("branch "):
                ref = line.split(" ", 1)[1].strip()
                if ref.startswith("refs/heads/"):
                    current_branch = ref.removeprefix("refs/heads/")
            if current_path is not None and current_branch == branch:
                return current_path
    except Exception:
        return None
    return None


def ensure_worktree_for_branch(*, branch: str, task_id: str, worktree_parent: Path, remote: str) -> Path:
    """Get or create a worktree for an existing task branch (used for repair runs)."""
    existing = _find_worktree_path_for_branch(branch)
    if existing is not None:
        return existing

    wt_path = worktree_parent / f"wt-{task_id}"
    if wt_path.exists():
        raise SystemExit(f"Worktree path already exists but is not registered for branch {branch}: {wt_path}")

    # Ensure remote refs are up-to-date for the branch.
    _run(["git", "fetch", remote], cwd=_repo_root(), check=True)

    branch_exists = False
    try:
        _run(["git", "show-ref", "--verify", "--quiet", f"refs/heads/{branch}"], cwd=_repo_root())
        branch_exists = True
    except subprocess.CalledProcessError:
        branch_exists = False

    if branch_exists:
        _run(["git", "worktree", "add", str(wt_path), branch], cwd=_repo_root())
    else:
        _run(["git", "worktree", "add", str(wt_path), "-b", branch, f"{remote}/{branch}"], cwd=_repo_root())

    return wt_path


def _tmux(*args: str, check: bool = True, capture: bool = False) -> subprocess.CompletedProcess[str]:
    tmux = _which_or_none("tmux")
    if tmux is None:
        raise SystemExit("tmux not found on PATH (install tmux or use runner=local)")
    return _run([tmux, *args], check=check, capture=capture)


def tmux_ensure_session(session: str, start_dir: Path) -> None:
    cp = _tmux("has-session", "-t", session, check=False, capture=False)
    if cp.returncode == 0:
        return
    _tmux("new-session", "-d", "-s", session, "-c", str(start_dir))


def tmux_spawn_task_window(
    *,
    session: str,
    window_name: str,
    workdir: Path,
    command: list[str],
) -> None:
    cmd_str = " ".join(shlex.quote(x) for x in command)
    # Use bash -lc so PATH/env behaves like a login shell (important in tmux automation).
    _tmux(
        "new-window",
        "-t",
        session,
        "-n",
        window_name,
        "-c",
        str(workdir),
        "bash",
        "-lc",
        cmd_str,
    )


def _git_current_branch(cwd: Path) -> str:
    cp = _run(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=cwd, capture=True, check=True)
    return (cp.stdout or "").strip()


def _git_config_get(cwd: Path, key: str) -> str | None:
    """Best-effort `git config --get <key>` helper (no external deps)."""
    cp = _run(["git", "config", "--get", key], cwd=cwd, capture=True, check=False)
    if cp.returncode != 0:
        return None
    value = (cp.stdout or "").strip()
    return value or None


def _git_remote_exists(cwd: Path, remote: str) -> bool:
    cp = _run(["git", "remote", "get-url", remote], cwd=cwd, capture=True, check=False)
    return cp.returncode == 0


def _require_git_identity(*, cwd: Path, reason: str) -> None:
    """Hard-fail if git identity is missing.

    Why:
    - `swarm.py` makes commits (task State/Notes, etc.). If identity is missing,
      `git commit` can fail, and prior versions of this repo used `check=False`
      in places (a deliberate "don't crash" choice that becomes a footgun).
    - In unattended or PR-driven operation, a "local-only" run is a governance
      problem: other machines may see tasks as unclaimed and duplicate work.
    """
    name = _git_config_get(cwd, "user.name")
    email = _git_config_get(cwd, "user.email")
    if name and email:
        return

    missing: list[str] = []
    if not name:
        missing.append("user.name")
    if not email:
        missing.append("user.email")

    raise SystemExit(
        "\n".join(
            [
                f"Preflight failed ({reason}): missing git identity: {', '.join(missing)}.",
                "Fix by configuring an identity (repo-local recommended):",
                '  git config user.name  "swarm-bot"',
                '  git config user.email "swarm-bot@users.noreply.github.com"',
                "Then re-run the swarm command.",
            ]
        )
    )


def _require_git_push_access(*, cwd: Path, remote: str, reason: str, timeout_seconds: int = 30) -> None:
    """Hard-fail if we cannot push to the configured remote (auth/permissions).

    Uses a dry-run push so we don't mutate remote state, but still exercises
    authentication. Also forces `GIT_TERMINAL_PROMPT=0` to fail fast instead of
    hanging on interactive credential prompts (critical for unattended runs).
    """
    if not _git_remote_exists(cwd, remote):
        raise SystemExit(
            "\n".join(
                [
                    f"Preflight failed ({reason}): git remote {remote!r} is not configured in this repo.",
                    "Fix by adding/configuring the remote, e.g.:",
                    f"  git remote add {remote} <url>",
                ]
            )
        )

    env = dict(os.environ)
    env["GIT_TERMINAL_PROMPT"] = "0"
    cp = _run(
        ["git", "push", "--dry-run", remote, "HEAD"],
        cwd=cwd,
        capture=True,
        check=False,
        env=env,
        timeout_seconds=timeout_seconds,
    )
    if cp.returncode == 0:
        return

    out = (cp.stdout or "").strip()
    if len(out) > 2000:
        out = out[-2000:]
    raise SystemExit(
        "\n".join(
            [
                f"Preflight failed ({reason}): cannot push to remote {remote!r} (auth/permission issue).",
                "This would cause a 'ghost run' where work happens locally but is never shared.",
                "Fix by authenticating git credentials for the remote, then verify:",
                f"  git push --dry-run {remote} HEAD",
                "",
                "Output:",
                out or "(no output)",
            ]
        )
    )


def _require_gh_auth(*, cwd: Path, reason: str, timeout_seconds: int = 20) -> None:
    """Hard-fail if GitHub CLI is missing or not authenticated.

    Required when `--create-pr` is requested: otherwise we would run tasks but
    fail to open PRs, breaking the PR-synchronized control plane.
    """
    gh = _which_or_none("gh")
    if gh is None:
        raise SystemExit(
            "\n".join(
                [
                    f"Preflight failed ({reason}): `gh` not found on PATH but --create-pr was requested.",
                    "Install GitHub CLI and authenticate (or rerun without --create-pr).",
                ]
            )
        )

    cp = _run([gh, "auth", "status"], cwd=cwd, capture=True, check=False, timeout_seconds=timeout_seconds)
    if cp.returncode == 0:
        return

    out = (cp.stdout or "").strip()
    if len(out) > 2000:
        out = out[-2000:]
    raise SystemExit(
        "\n".join(
            [
                f"Preflight failed ({reason}): GitHub CLI is not authenticated but --create-pr was requested.",
                "Fix by running:",
                "  gh auth login",
                "",
                "Output:",
                out or "(no output)",
            ]
        )
    )


def _git_commit(*, cwd: Path, message: str, strict: bool) -> None:
    """Run `git commit` with failure behavior appropriate to the swarm mode.

    In strict modes (unattended / create-pr), failures must abort to prevent
    "ghost runs" (local-only work that is never pushed).
    """
    cp = _run(["git", "commit", "-m", message], cwd=cwd, capture=True, check=False)
    if cp.returncode == 0:
        return
    out = (cp.stdout or "").strip()
    if len(out) > 2000:
        out = out[-2000:]
    if strict:
        raise SystemExit(
            "\n".join(
                [
                    f"git commit failed in strict mode: {message!r}",
                    out or "(no output)",
                ]
            )
        )
    print(f"[warn] git commit failed: {message!r}\n{out}", file=sys.stderr)


def _git_push(*, cwd: Path, remote: str, ref: str, set_upstream: bool, strict: bool, timeout_seconds: int = 60) -> None:
    """Run `git push` with failure behavior appropriate to the swarm mode.

    In strict modes we force non-interactive auth (`GIT_TERMINAL_PROMPT=0`) and
    abort on failure to prevent silently unshared work.
    """
    env = dict(os.environ)
    if strict:
        env["GIT_TERMINAL_PROMPT"] = "0"
    cmd: list[str] = ["git", "push"]
    if set_upstream:
        cmd.append("-u")
    cmd.extend([remote, ref])
    cp = _run(cmd, cwd=cwd, capture=True, check=False, env=env, timeout_seconds=timeout_seconds)
    if cp.returncode == 0:
        return
    out = (cp.stdout or "").strip()
    if len(out) > 2000:
        out = out[-2000:]
    if strict:
        raise SystemExit(
            "\n".join(
                [
                    f"git push failed in strict mode: remote={remote!r} ref={ref!r}",
                    out or "(no output)",
                ]
            )
        )
    print(f"[warn] git push failed: remote={remote!r} ref={ref!r}\n{out}", file=sys.stderr)


def _preflight_strict_sync_requirements(
    *,
    cwd: Path,
    remote: str,
    unattended: bool,
    create_pr: bool,
) -> None:
    """Preflight checks to prevent silent 'ghost runs' (expert feedback item B).

    Trigger conditions:
    - `--unattended` OR `--create-pr`

    Rationale:
    - In those modes we *must* be able to commit + push reliably; otherwise the
      repo-as-shared-memory model breaks (duplicate work across machines).
    """
    if not (unattended or create_pr):
        return
    cache_key = (remote, unattended, create_pr)
    if cache_key in _PREFLIGHT_STRICT_SYNC_CACHE:
        return
    reason = "unattended" if unattended else "create-pr"
    _require_git_identity(cwd=cwd, reason=reason)
    _require_git_push_access(cwd=cwd, remote=remote, reason=reason)
    if create_pr:
        _require_gh_auth(cwd=cwd, reason=reason)
    _PREFLIGHT_STRICT_SYNC_CACHE.add(cache_key)


def _git_has_changes(cwd: Path) -> bool:
    cp = _run(["git", "status", "--porcelain"], cwd=cwd, capture=True, check=True)
    return bool((cp.stdout or "").strip())


def _git_status_entries(cwd: Path) -> list[dict[str, str]]:
    cp = _run(["git", "status", "--porcelain=v1"], cwd=cwd, capture=True, check=True)
    entries: list[dict[str, str]] = []
    for line in (cp.stdout or "").splitlines():
        if len(line) < 4:
            continue
        xy = line[:2]
        path_part = line[3:].strip()
        old_path = ""
        new_path = path_part
        if " -> " in path_part:
            old_path, new_path = path_part.split(" -> ", 1)
            old_path = old_path.strip()
            new_path = new_path.strip()
        entries.append({"xy": xy, "path": new_path, "old_path": old_path})
    return entries


def _git_ref_exists(cwd: Path, ref: str) -> bool:
    """Return True if `ref` resolves to an object in this repo.

    We keep this best-effort and dependency-free. This is used to choose a base
    ref for diff-based ownership checks (committed changes).
    """
    cp = _run(["git", "rev-parse", "--verify", ref], cwd=cwd, capture=True, check=False)
    return cp.returncode == 0


def _resolve_base_ref_for_diff(*, cwd: Path, base_branch: str, remote: str) -> str | None:
    """Resolve a stable base ref for three-dot diffs.

    Prefer the remote-tracking ref when it exists (e.g. `origin/main`), otherwise
    fall back to the local branch name (`main`).
    """
    candidates = [f"{remote}/{base_branch}", base_branch]
    for ref in candidates:
        if _git_ref_exists(cwd, ref):
            return ref
    return None


def _git_diff_name_status_entries(cwd: Path, diff_args: list[str]) -> list[dict[str, str]]:
    """Parse `git diff --name-status -M ...` output into structured entries.

    Why:
    - We need explicit detection of deletions and renames (AC3).
    - We must evaluate path ownership on committed *and* uncommitted changes (AC1/AC2).

    Notes:
    - We avoid `-z` for simplicity, but we split on tabs so filenames with spaces
      still parse correctly.
    - `status` can include a similarity score (e.g. `R100`); `code` is the first
      letter (`R`, `D`, `A`, `M`, ...).
    """
    cp = _run(["git", "diff", "--name-status", "-M", *diff_args], cwd=cwd, capture=True, check=True)
    entries: list[dict[str, str]] = []
    for raw_line in (cp.stdout or "").splitlines():
        line = raw_line.strip("\n")
        if not line:
            continue

        parts = line.split("\t")
        status = parts[0].strip()
        if not status:
            continue
        code = status[:1]

        path = ""
        old_path = ""
        if code in {"R", "C"}:
            # Rename/Copy: status, old, new
            if len(parts) < 3:
                continue
            old_path = parts[1].strip()
            path = parts[2].strip()
        else:
            # Most statuses: status, path
            if len(parts) < 2:
                continue
            path = parts[1].strip()

        if not path and not old_path:
            continue

        entries.append({"status": status, "code": code, "path": path, "old_path": old_path})
    return entries


def _git_untracked_files(cwd: Path) -> list[str]:
    """List untracked, non-ignored files (AC2)."""
    cp = _run(["git", "ls-files", "--others", "--exclude-standard"], cwd=cwd, capture=True, check=True)
    return [line.strip() for line in (cp.stdout or "").splitlines() if line.strip()]


def _collect_changed_paths_with_sources(
    *,
    repo: Path,
    base_ref: str | None,
) -> tuple[dict[str, set[str]], list[dict[str, str]]]:
    """Collect changed paths across committed+staged+unstaged+untracked changes.

    Returns:
    - `path_sources`: map of path -> {sources...}
      where sources ∈ {"committed", "staged", "unstaged", "untracked"}
    - `ops`: a list of low-level change records including rename/delete metadata.

    This is the heart of the governance hardening:
    - Using only `git status` is porous if a Worker creates commits (AC1).
    - We must also include untracked files to prevent "write forbidden file but don't stage/commit" bypasses (AC2).
    - We capture rename/delete operations explicitly for task file integrity checks (AC3).
    """

    def _add_paths(
        *,
        source: str,
        entries: list[dict[str, str]],
        path_sources: dict[str, set[str]],
        ops: list[dict[str, str]],
    ) -> None:
        for e in entries:
            rec = dict(e)
            rec["source"] = source
            ops.append(rec)

            # For ownership, both the new and old path matter:
            # - Renaming from disallowed -> allowed still touched disallowed.
            # - Renaming from allowed -> disallowed must be caught by the new path.
            for p in [e.get("path", ""), e.get("old_path", "")]:
                if not p:
                    continue
                path_sources.setdefault(p, set()).add(source)

    path_sources: dict[str, set[str]] = {}
    ops: list[dict[str, str]] = []

    # 1) Committed branch diff (AC1) — only if we can resolve a base ref.
    if base_ref is not None:
        committed = _git_diff_name_status_entries(repo, [f"{base_ref}...HEAD"])
        _add_paths(source="committed", entries=committed, path_sources=path_sources, ops=ops)

    # 2) Staged (index) changes
    staged = _git_diff_name_status_entries(repo, ["--cached"])
    _add_paths(source="staged", entries=staged, path_sources=path_sources, ops=ops)

    # 3) Unstaged working tree changes
    unstaged = _git_diff_name_status_entries(repo, [])
    _add_paths(source="unstaged", entries=unstaged, path_sources=path_sources, ops=ops)

    # 4) Untracked (but not ignored) files (AC2)
    for p in _git_untracked_files(repo):
        path_sources.setdefault(p, set()).add("untracked")
        ops.append({"status": "??", "code": "?", "path": p, "old_path": "", "source": "untracked"})

    return path_sources, ops


_OUTPUT_WILDCARD_TOKENS = ("...", "YYYY-MM-DD", "<", ">", "*", "?")


def _normalize_repo_relative_path(value: str) -> str:
    """Normalize a repo-relative path-like string to POSIX separators."""
    s = value.strip().replace("\\", "/")
    if s.startswith("./"):
        s = s[2:]
    return s


def _output_spec_is_safe(spec: str) -> tuple[bool, str | None]:
    """Validate an output spec is a safe repo-relative path/pattern.

    Output specs come from task frontmatter and are used to touch the filesystem.
    We only support *repo-relative* specs and we fail closed on anything that
    could escape the repo (e.g. absolute paths, `..` segments).
    """
    s = _normalize_repo_relative_path(spec)
    if s == "":
        return False, "empty_output_spec"
    if s.startswith("~") or s.startswith("/"):
        return False, "absolute_output_spec_forbidden"
    if s.startswith("../") or "/../" in s or s == "..":
        return False, "path_traversal_forbidden"
    return True, None


def _segment_pattern_to_regex(segment: str) -> re.Pattern[str]:
    """Convert a single path segment pattern into a regex.

    Supported tokens (kept minimal to match task templates):
    - `YYYY-MM-DD` -> `\\d{4}-\\d{2}-\\d{2}`
    - `<...>`      -> wildcard for a single segment (no `/`)
    - `...`        -> wildcard within the segment
    - `*` / `?`    -> glob-style wildcards within the segment
    """
    # Replace `<...>` placeholders before escaping.
    s = re.sub(r"<[^>]+>", "{WILD}", segment)
    s = s.replace("YYYY-MM-DD", "{DATE}")
    s = s.replace("...", "{ELLIPSIS}")

    rx = re.escape(s)
    rx = rx.replace(re.escape("{WILD}"), r"[^/]+")
    rx = rx.replace(re.escape("{DATE}"), r"\d{4}-\d{2}-\d{2}")
    rx = rx.replace(re.escape("{ELLIPSIS}"), r".*")
    # Allow basic glob wildcards as a convenience.
    rx = rx.replace(r"\*", ".*").replace(r"\?", ".")
    return re.compile("^" + rx + "$")


def _has_wildcards(segment: str) -> bool:
    return any(tok in segment for tok in _OUTPUT_WILDCARD_TOKENS)


def _find_paths_matching_output_spec(*, repo: Path, spec: str) -> list[Path]:
    """Find existing paths that match a repo-relative output spec.

    This matcher is intentionally small and deterministic. It walks the spec
    segment-by-segment, expanding only when a segment contains known wildcard
    tokens. This keeps checks fast even when `data/raw/` is large.
    """
    s = _normalize_repo_relative_path(spec)
    segments = [seg for seg in s.split("/") if seg]
    current: list[Path] = [repo]

    for seg in segments:
        next_paths: list[Path] = []
        if not _has_wildcards(seg):
            for base in current:
                cand = base / seg
                if cand.exists():
                    next_paths.append(cand)
        else:
            rx = _segment_pattern_to_regex(seg)
            for base in current:
                if not base.is_dir():
                    continue
                try:
                    for child in base.iterdir():
                        if rx.match(child.name):
                            next_paths.append(child)
                except FileNotFoundError:
                    continue

        current = next_paths
        if not current:
            break

    return current


def _guess_output_kind(spec: str) -> str:
    """Return one of: file | dir | dir_nonempty | any."""
    s = _normalize_repo_relative_path(spec)
    if s.endswith("/...") or s.endswith("..."):
        return "dir_nonempty"
    if s.endswith("/"):
        return "dir"
    # Heuristic: common file extensions imply "file".
    for ext in (
        ".py",
        ".md",
        ".json",
        ".csv",
        ".tsv",
        ".yml",
        ".yaml",
        ".svg",
        ".png",
        ".jpg",
        ".jpeg",
        ".txt",
        ".pdf",
    ):
        if s.lower().endswith(ext):
            return "file"
    return "any"


def _strip_trailing_ellipsis(spec: str) -> str:
    s = _normalize_repo_relative_path(spec)
    if s.endswith("/..."):
        return s[:-4]
    if s.endswith("..."):
        return s[:-3].rstrip("/")
    return s


def _check_task_outputs_exist(*, repo: Path, task: Task) -> tuple[bool, list[dict[str, str]]]:
    """Validate that task-declared outputs exist before marking review/done.

    This is a governance hardening check: it prevents tasks from being marked as
    `ready_for_review` / `done` when they did not actually produce their declared
    outputs (e.g. an agent "forgot" to write a report, or only updated the task
    State line).
    """
    failures: list[dict[str, str]] = []
    if not task.outputs:
        return False, [{"output": "(none)", "reason": "no_outputs_declared"}]

    for raw_spec in task.outputs:
        spec = raw_spec.strip()
        ok, why = _output_spec_is_safe(spec)
        if not ok:
            failures.append({"output": spec or raw_spec, "reason": why or "invalid_output_spec"})
            continue

        kind = _guess_output_kind(spec)
        match_spec = _strip_trailing_ellipsis(spec) if kind == "dir_nonempty" else _normalize_repo_relative_path(spec)
        matches = _find_paths_matching_output_spec(repo=repo, spec=match_spec)

        if kind == "file":
            if not any(p.is_file() for p in matches):
                failures.append({"output": spec, "reason": "missing_file"})
            continue

        if kind == "dir":
            if not any(p.is_dir() for p in matches):
                failures.append({"output": spec, "reason": "missing_dir"})
            continue

        if kind == "dir_nonempty":
            ok_any = False
            for p in matches:
                if not p.is_dir():
                    continue
                try:
                    next(p.iterdir())
                    ok_any = True
                    break
                except StopIteration:
                    continue
                except FileNotFoundError:
                    continue
            if not ok_any:
                failures.append({"output": spec, "reason": "missing_or_empty_dir"})
            continue

        # kind == "any": require at least one match exists.
        if not matches:
            failures.append({"output": spec, "reason": "missing_path"})

    return (len(failures) == 0), failures


def _path_is_allowed(
    *,
    path: str,
    allowed_paths: list[str],
    disallowed_paths: list[str],
    task_file_paths: set[str],
) -> tuple[bool, str | None]:
    norm = path.replace("\\", "/")

    # Enforce control-plane governance:
    # - allow only the current task file, and handoff notes
    # - disallow everything else under `.orchestrator/`
    if norm.startswith(".orchestrator/"):
        if norm in task_file_paths:
            return True, None
        if norm.startswith(".orchestrator/handoff/"):
            return True, None
        return False, "orchestrator_write_forbidden"

    for bad in disallowed_paths:
        if bad and norm.startswith(bad):
            return False, f"disallowed_path:{bad}"

    for ok in allowed_paths:
        if ok and norm.startswith(ok):
            return True, None

    return False, "outside_allowed_paths"


def _update_task_status_and_notes(
    *,
    task_path: Path,
    new_state: str,
    note_line: str,
) -> None:
    """Edit ONLY Status and Notes sections (best effort)."""
    if new_state not in VALID_TASK_STATES:
        raise ValueError(f"invalid state: {new_state}")

    text = _read_text(task_path)
    today = _utc_today()

    # Update State
    text2, n1 = re.subn(
        r"^\s*-\s*State:\s*.*\s*$",
        f"- State: {new_state}",
        text,
        flags=re.MULTILINE,
    )
    if n1 == 0:
        raise SystemExit(f"Could not find State line to update in: {task_path}")

    # Update Last updated
    text3, n2 = re.subn(
        r"^\s*-\s*Last updated:\s*\d{4}-\d{2}-\d{2}\s*$",
        f"- Last updated: {today}",
        text2,
        flags=re.MULTILINE,
    )
    if n2 == 0:
        raise SystemExit(f"Could not find Last updated line to update in: {task_path}")

    # Append note under Notes / Decisions (append-only)
    marker = "## Notes / Decisions"
    idx = text3.find(marker)
    if idx < 0:
        raise SystemExit(f"Could not find Notes / Decisions heading in: {task_path}")
    insert_at = idx + len(marker)
    suffix = text3[insert_at:]
    note = f"\n\n- {today}: {note_line}".rstrip() + "\n"
    text4 = text3[:insert_at] + suffix + note

    task_path.write_text(text4, encoding="utf-8")


def _find_task_file_anywhere(task_id: str, cwd: Path) -> Path | None:
    orch = cwd / ".orchestrator"
    for sub in ["active", "backlog", "ready_for_review", "blocked", "done"]:
        for p in iter_task_files(orch / sub):
            if p.name.startswith(task_id):
                return p
    return None


def _require_unattended_ack() -> None:
    if os.environ.get("SWARM_UNATTENDED_I_UNDERSTAND") == "1":
        return
    raise SystemExit(
        "Refusing to run with --unattended without SWARM_UNATTENDED_I_UNDERSTAND=1 "
        "(safety interlock; run only in an external sandbox with no secrets)."
    )


def _supervisor_sync_to_remote_base(*, repo: Path, remote: str, base_branch: str) -> None:
    """Hard-sync the supervisor checkout to the remote base branch.

    This prevents "local main drift" in long-running unattended loops.
    """
    _run(["git", "fetch", remote], cwd=repo, check=True)
    _run(["git", "checkout", "-B", base_branch, f"{remote}/{base_branch}"], cwd=repo, check=True)


def _codex_exec_cmd(
    *,
    prompt: str,
    model: str | None,
    sandbox: str,
    unattended: bool,
    allow_network: bool,
    workdir: Path,
    output_last_message: Path | None,
) -> list[str]:
    if _which_or_none("codex") is None:
        raise SystemExit("codex not found on PATH")

    cmd: list[str] = ["codex"]
    if unattended:
        cmd.extend(["-a", "never"])
    cmd.extend(["exec", "--sandbox", sandbox])
    if model:
        cmd.extend(["-m", model])
    if allow_network:
        # Codex CLI supports config overrides via -c; allow networking for ETL tasks in workspace-write.
        cmd.extend(["-c", "sandbox_workspace_write.network_access=true"])
    if output_last_message is not None:
        cmd.extend(["-o", str(output_last_message)])
    cmd.extend(["-C", str(workdir)])
    cmd.append(prompt)
    return cmd


def _codex_review_cmd(
    *,
    prompt: str,
    unattended: bool,
    base_branch: str,
    workdir: Path,
) -> list[str]:
    if _which_or_none("codex") is None:
        raise SystemExit("codex not found on PATH")
    cmd: list[str] = ["codex"]
    if unattended:
        cmd.extend(["-a", "never"])
    cmd.extend(["review", "--base", base_branch, "--uncommitted", prompt])
    return cmd


def _gh_create_pr_if_missing(
    *,
    cwd: Path,
    base_branch: str,
    title: str,
    body: str,
) -> None:
    gh = _which_or_none("gh")
    if gh is None:
        return

    branch = _git_current_branch(cwd)
    # If PR already exists for this head branch, do nothing.
    try:
        cp = _run(
            [gh, "pr", "list", "--state", "open", "--head", branch, "--json", "number"],
            cwd=cwd,
            capture=True,
            check=True,
        )
        items = json.loads(cp.stdout or "[]")
        if isinstance(items, list) and len(items) > 0:
            return
    except Exception:
        pass

    _run(
        [
            gh,
            "pr",
            "create",
            "--base",
            base_branch,
            "--title",
            title,
            "--body",
            body,
        ],
        cwd=cwd,
        check=True,
    )


def _maybe_auto_merge(
    *,
    cwd: Path,
    squash: bool,
) -> None:
    gh = _which_or_none("gh")
    if gh is None:
        return
    branch = _git_current_branch(cwd)
    # Attempt auto-merge (requires repo settings/permissions). If it fails, we keep the PR open.
    args = [gh, "pr", "merge", "--auto"]
    args.append("--squash" if squash else "--merge")
    args.extend(["--delete-branch", branch])
    _run(args, cwd=cwd, check=False)


def _parse_iso_datetime(value: str) -> _dt.datetime | None:
    try:
        v = value.strip()
        if v.endswith("Z"):
            v = v[:-1] + "+00:00"
        dt = _dt.datetime.fromisoformat(v)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=_dt.timezone.utc)
        return dt.astimezone(_dt.timezone.utc)
    except Exception:
        return None


def _summarize_pr_checks(pr: dict[str, Any]) -> tuple[str, list[str]]:
    """Return (status, failing_check_names)."""
    rollup = pr.get("statusCheckRollup")
    if not isinstance(rollup, list):
        return "unknown", []

    failing: list[str] = []
    states: list[str] = []
    for item in rollup:
        if not isinstance(item, dict):
            continue
        state = item.get("state") or item.get("conclusion") or ""
        if isinstance(state, str):
            states.append(state.upper())
        name = item.get("name")
        if not isinstance(name, str):
            name = "unknown"

    failing_states = {"FAILURE", "ERROR", "CANCELLED", "TIMED_OUT"}
    pending_states = {"PENDING", "IN_PROGRESS", "EXPECTED"}
    success_states = {"SUCCESS", "SKIPPED", "NEUTRAL"}

    for item in rollup:
        if not isinstance(item, dict):
            continue
        state = item.get("state") or item.get("conclusion") or ""
        if not isinstance(state, str):
            continue
        s = state.upper()
        if s in failing_states:
            name = item.get("name")
            failing.append(name if isinstance(name, str) else "unknown")

    if any(s in failing_states for s in states):
        return "failing", sorted(set(failing))
    if any(s in pending_states for s in states):
        return "pending", []
    if states and all(s in success_states for s in states):
        return "success", []
    return "unknown", []


def _maybe_spawn_repairs(args: argparse.Namespace, repo: Path) -> None:
    """Best-effort recovery loop for stuck/failing task PRs.

    Crude but effective: if an open task PR is failing checks or merge-conflicting and has not
    changed recently, run a bounded "repair" worker pass in the PR branch worktree.
    """
    if not args.unattended:
        return
    if args.max_repairs_per_tick <= 0:
        return
    gh = _which_or_none("gh")
    if gh is None:
        return

    try:
        cp = _run(
            [
                gh,
                "pr",
                "list",
                "--state",
                "open",
                "--base",
                args.base_branch,
                "--json",
                "number,headRefName,url,updatedAt,mergeable,statusCheckRollup",
            ],
            capture=True,
            check=True,
            cwd=repo,
        )
        prs = json.loads(cp.stdout or "[]")
    except Exception:
        return

    if not isinstance(prs, list):
        return

    now = _dt.datetime.now(tz=_dt.timezone.utc)
    candidates: list[dict[str, Any]] = []
    for pr in prs:
        if not isinstance(pr, dict):
            continue
        head = pr.get("headRefName")
        if not isinstance(head, str):
            continue
        task_id = _parse_task_id_from_branch(head)
        if task_id is None:
            continue
        updated_at_raw = pr.get("updatedAt")
        updated_at = _parse_iso_datetime(updated_at_raw) if isinstance(updated_at_raw, str) else None
        if updated_at is None:
            continue
        age_seconds = (now - updated_at).total_seconds()
        if age_seconds < float(args.repair_after_seconds):
            continue

        checks_status, failing_checks = _summarize_pr_checks(pr)
        mergeable = pr.get("mergeable")
        mergeable_s = mergeable if isinstance(mergeable, str) else "unknown"

        needs_repair = checks_status == "failing" or mergeable_s.upper() == "CONFLICTING"
        if not needs_repair:
            continue

        candidates.append(
            {
                "task_id": task_id,
                "branch": head,
                "pr_number": pr.get("number"),
                "url": pr.get("url"),
                "checks_status": checks_status,
                "failing_checks": failing_checks,
                "mergeable": mergeable_s,
                "age_seconds": age_seconds,
            }
        )

    candidates_sorted = sorted(candidates, key=lambda x: float(x.get("age_seconds", 0.0)), reverse=True)
    if not candidates_sorted:
        return

    wt_parent = Path(args.worktree_parent).expanduser().resolve() if args.worktree_parent else repo.parent
    wt_parent.mkdir(parents=True, exist_ok=True)

    repairs_started = 0
    for cand in candidates_sorted:
        if repairs_started >= int(args.max_repairs_per_tick):
            break
        task_id = str(cand["task_id"])
        branch = str(cand["branch"])
        try:
            wt_path = ensure_worktree_for_branch(
                branch=branch,
                task_id=task_id,
                worktree_parent=wt_parent,
                remote=args.remote,
            )
        except Exception:
            continue

        reason = (
            f"Auto-repair: PR {cand.get('url')} "
            f"(checks={cand.get('checks_status')}, mergeable={cand.get('mergeable')}, "
            f"failing_checks={','.join(cand.get('failing_checks') or [])})"
        )

        run_cmd = [
            sys.executable,
            "scripts/swarm.py",
            "run-task",
            "--task-id",
            task_id,
            "--base-branch",
            args.base_branch,
            "--remote",
            args.remote,
            "--codex-sandbox",
            args.codex_sandbox,
            "--final-state",
            args.final_state,
            "--repair-context",
            reason,
        ]
        if args.unattended:
            run_cmd.append("--unattended")
        if args.max_worker_seconds:
            run_cmd.extend(["--max-worker-seconds", str(args.max_worker_seconds)])
        if args.max_review_seconds:
            run_cmd.extend(["--max-review-seconds", str(args.max_review_seconds)])
        if args.codex_model:
            run_cmd.extend(["--codex-model", args.codex_model])
        if args.create_pr:
            run_cmd.append("--create-pr")
        if args.auto_merge:
            run_cmd.append("--auto-merge")

        if args.runner == "tmux":
            tmux_ensure_session(args.tmux_session, repo)
            window_name = f"repair-{task_id}-{_utc_timestamp_compact()[9:15]}"
            tmux_spawn_task_window(
                session=args.tmux_session,
                window_name=window_name,
                workdir=wt_path,
                command=run_cmd,
            )
        else:
            _run(run_cmd, cwd=wt_path, check=False)

        repairs_started += 1


def cmd_plan(args: argparse.Namespace) -> int:
    done_ids = done_task_ids()
    claimed_ids = claimed_task_ids(args.remote, args.base_branch)
    ready = ready_backlog_tasks(done_ids=done_ids, claimed_ids=claimed_ids)
    print(json.dumps({"done": sorted(done_ids), "claimed": sorted(claimed_ids), "ready": [dataclasses.asdict(t) for t in ready]}, indent=2, sort_keys=True, default=str))
    return 0


def cmd_tick(args: argparse.Namespace) -> int:
    repo = _repo_root()
    if args.unattended:
        _require_unattended_ack()

    # Preflight (expert feedback item B): in unattended mode or when PR creation
    # is requested, hard-fail early if git identity/push auth (and gh auth) are
    # missing. This prevents "ghost runs" where work happens locally but never
    # gets pushed/PR'd, leading to duplicate work across machines.
    if not args.dry_run:
        _preflight_strict_sync_requirements(
            cwd=repo,
            remote=args.remote,
            unattended=bool(args.unattended),
            create_pr=bool(args.create_pr),
        )

    done_ids = done_task_ids()
    claimed_ids = claimed_task_ids(args.remote, args.base_branch)
    ready = ready_backlog_tasks(done_ids=done_ids, claimed_ids=claimed_ids)
    locked_workstreams, parallel_only_workstreams = _compute_workstream_locks(repo=repo, claimed_ids=claimed_ids)
    ready = _apply_workstream_concurrency_filters(
        tasks=sorted(ready, key=lambda t: (_priority_rank(t.priority), t.task_id)),
        locked_workstreams=locked_workstreams,
        parallel_only_workstreams=parallel_only_workstreams,
        capacity=len(ready),
    )

    if not ready:
        print("No ready tasks in backlog.")
        return 0

    capacity = max(0, args.max_workers)
    if capacity == 0:
        print("max_workers=0; nothing to do.")
        return 0

    if args.planner == "claude":
        selected = choose_tasks_claude(
            ready=ready,
            capacity=capacity,
            model=args.claude_model,
            unattended=args.unattended,
        )
    else:
        selected = choose_tasks_heuristic(ready, capacity)

    selected = _apply_workstream_concurrency_filters(
        tasks=selected,
        locked_workstreams=locked_workstreams,
        parallel_only_workstreams=parallel_only_workstreams,
        capacity=capacity,
    )
    if not selected:
        print("Planner selected no tasks.")
        return 0

    wt_parent = Path(args.worktree_parent).expanduser().resolve() if args.worktree_parent else repo.parent
    wt_parent.mkdir(parents=True, exist_ok=True)

    tasks_started: list[dict[str, str]] = []
    if args.runner == "tmux":
        tmux_ensure_session(args.tmux_session, repo)
        if args.unattended:
            # Robust tmux env propagation so unattended mode works inside tmux windows.
            _tmux("set-environment", "-g", "SWARM_UNATTENDED_I_UNDERSTAND", "1")
    for task in selected:
        if args.dry_run:
            print(f"[dry-run] would start {task.task_id}: {task.title}")
            continue

        wt_path, branch = ensure_worktree(task=task, worktree_parent=wt_parent, base_ref=args.base_branch)
        tasks_started.append({"task_id": task.task_id, "branch": branch, "worktree": str(wt_path)})

        run_cmd = [
            sys.executable,
            "scripts/swarm.py",
            "run-task",
            "--task-id",
            task.task_id,
            "--base-branch",
            args.base_branch,
            "--remote",
            args.remote,
            "--codex-sandbox",
            args.codex_sandbox,
            "--final-state",
            args.final_state,
        ]
        if args.unattended:
            run_cmd.append("--unattended")
        if args.max_worker_seconds:
            run_cmd.extend(["--max-worker-seconds", str(args.max_worker_seconds)])
        if args.max_review_seconds:
            run_cmd.extend(["--max-review-seconds", str(args.max_review_seconds)])
        if args.codex_model:
            run_cmd.extend(["--codex-model", args.codex_model])
        if args.create_pr:
            run_cmd.append("--create-pr")
        if args.auto_merge:
            run_cmd.append("--auto-merge")

        if args.runner == "tmux":
            tmux_spawn_task_window(
                session=args.tmux_session,
                window_name=task.task_id,
                workdir=wt_path,
                command=run_cmd,
            )
        else:
            # local (sequential)
            _run(run_cmd, cwd=wt_path, check=True)

    status = {
        "timestamp_utc": _utc_timestamp_compact(),
        "selected": [t.task_id for t in selected],
        "started": tasks_started,
    }
    print(json.dumps(status, indent=2, sort_keys=True))
    return 0


def cmd_tmux_start(args: argparse.Namespace) -> int:
    repo = _repo_root()
    if args.unattended:
        _require_unattended_ack()

    # Same preflight as `tick`: ensure we won't start an unattended/PR-driven
    # supervisor loop that cannot commit/push or create PRs.
    _preflight_strict_sync_requirements(
        cwd=repo,
        remote=args.remote,
        unattended=bool(args.unattended),
        create_pr=bool(args.create_pr),
    )
    tmux_ensure_session(args.tmux_session, repo)
    if args.unattended:
        # Robust tmux env propagation so unattended mode works inside tmux windows.
        _tmux("set-environment", "-g", "SWARM_UNATTENDED_I_UNDERSTAND", "1")
    loop_cmd = [
        sys.executable,
        "scripts/swarm.py",
        "loop",
        "--interval-seconds",
        str(args.interval_seconds),
        "--planner",
        args.planner,
        "--runner",
        "tmux",
        "--tmux-session",
        args.tmux_session,
        "--max-workers",
        str(args.max_workers),
        "--base-branch",
        args.base_branch,
        "--remote",
        args.remote,
        "--codex-sandbox",
        args.codex_sandbox,
        "--final-state",
        args.final_state,
    ]
    if args.worktree_parent:
        loop_cmd.extend(["--worktree-parent", args.worktree_parent])
    if args.unattended:
        loop_cmd.append("--unattended")
    if args.max_worker_seconds:
        loop_cmd.extend(["--max-worker-seconds", str(args.max_worker_seconds)])
    if args.max_review_seconds:
        loop_cmd.extend(["--max-review-seconds", str(args.max_review_seconds)])
    if args.repair_after_seconds:
        loop_cmd.extend(["--repair-after-seconds", str(args.repair_after_seconds)])
    if args.max_repairs_per_tick is not None:
        loop_cmd.extend(["--max-repairs-per-tick", str(args.max_repairs_per_tick)])
    if args.codex_model:
        loop_cmd.extend(["--codex-model", args.codex_model])
    if args.claude_model:
        loop_cmd.extend(["--claude-model", args.claude_model])
    if args.create_pr:
        loop_cmd.append("--create-pr")
    if args.auto_merge:
        loop_cmd.append("--auto-merge")

    tmux_spawn_task_window(
        session=args.tmux_session,
        window_name="supervisor",
        workdir=repo,
        command=loop_cmd,
    )
    print(f"Started supervisor loop in tmux session {args.tmux_session}.")
    if args.attach:
        _tmux("attach", "-t", args.tmux_session, check=True, capture=False)
    return 0


def cmd_loop(args: argparse.Namespace) -> int:
    repo = _repo_root()
    if args.unattended:
        _require_unattended_ack()

    # Preflight once at loop start; subsequent ticks are cached (see
    # `_PREFLIGHT_STRICT_SYNC_CACHE`).
    _preflight_strict_sync_requirements(
        cwd=repo,
        remote=args.remote,
        unattended=bool(args.unattended),
        create_pr=bool(args.create_pr),
    )
    interval = max(5, int(args.interval_seconds))
    print(f"Swarm loop started (interval={interval}s). Repo: {repo}")
    while True:
        try:
            _supervisor_sync_to_remote_base(repo=repo, remote=args.remote, base_branch=args.base_branch)
            cmd_tick(args)
            _maybe_spawn_repairs(args, repo)
        except Exception as exc:
            print(f"[loop] tick failed: {exc}", file=sys.stderr)
            if args.unattended:
                # Fail loudly in unattended mode; persistent sync/auth failures otherwise cause silent stalls.
                return 1
        try:
            time_to_sleep = interval
            # Sleep in small increments so Ctrl-C works responsively even in tmux.
            while time_to_sleep > 0:
                step = min(5, time_to_sleep)
                time_to_sleep -= step
                time.sleep(step)
        except KeyboardInterrupt:
            print("Loop stopped.")
            return 0


def cmd_run_task(args: argparse.Namespace) -> int:
    repo = _repo_root()

    # Guardrails for unattended mode.
    if args.unattended:
        _require_unattended_ack()
        print(
            "WARNING: --unattended disables approval prompts. ONLY run in an external sandbox with no secrets (see AGENTS.md).",
            file=sys.stderr,
        )

    # Preflight (expert feedback item B): hard-fail early when `--unattended` or
    # `--create-pr` is requested but the environment cannot commit/push (or
    # cannot create PRs). This prevents "ghost runs".
    _preflight_strict_sync_requirements(
        cwd=repo,
        remote=args.remote,
        unattended=bool(args.unattended),
        create_pr=bool(args.create_pr),
    )
    strict_sync = bool(args.unattended or args.create_pr)

    task_file = _find_task_file_anywhere(args.task_id, repo)
    if task_file is None:
        raise SystemExit(f"Could not find task file for {args.task_id} under .orchestrator/")

    task = load_task(task_file)
    allow_network = task.workstream in {"W1", "W2"}

    # Claim: set State=active (do NOT move lifecycle folders; Planner sweeps separately)
    if task.state == "backlog":
        _update_task_status_and_notes(
            task_path=task_file,
            new_state="active",
            note_line=f"Claimed by swarm runner; starting worker (branch: {_git_current_branch(repo)}).",
        )
        _run(["git", "add", str(task_file)], cwd=repo)
        _git_commit(cwd=repo, message=f"{task.task_id}: claim (active)", strict=strict_sync)
        _git_push(
            cwd=repo,
            remote=args.remote,
            ref=_git_current_branch(repo),
            set_upstream=True,
            strict=strict_sync,
        )

    # Worker: Codex exec
    logs_dir = repo / "data" / "tmp" / "swarm_logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    worker_last_msg = logs_dir / f"{task.task_id}_{_utc_timestamp_compact()}_worker_last_message.txt"

    worker_prompt = "\n".join(
        [
            f"Role: Worker. Task: {task_file.as_posix()}",
            "Follow AGENTS.md and nested AGENTS.md.",
            "Execute exactly ONE task (this task).",
            "Respect allowed/disallowed paths in the task frontmatter.",
            "Workers: edit ONLY the task file sections `## Status` and `## Notes / Decisions`.",
            "Run the task gates/commands before declaring success.",
            *(["Repair context: " + str(args.repair_context)] if args.repair_context else []),
            *(
                [
                    "This is an automated repair pass. Focus on making the PR mergeable and checks pass.",
                    "Do not broaden scope; make the smallest change that fixes the failure.",
                ]
                if args.repair_context
                else []
            ),
        ]
    )
    worker_cmd = _codex_exec_cmd(
        prompt=worker_prompt,
        model=args.codex_model,
        sandbox=args.codex_sandbox,
        unattended=args.unattended,
        allow_network=allow_network,
        workdir=repo,
        output_last_message=worker_last_msg,
    )
    worker_timeout = int(args.max_worker_seconds) if args.max_worker_seconds else None
    try:
        _run(worker_cmd, cwd=repo, check=False, timeout_seconds=worker_timeout)
    except subprocess.TimeoutExpired:
        timeout_note = (
            f"Worker timed out after {worker_timeout}s; leaving task active. "
            f"Last message: {worker_last_msg.as_posix()}"
        )
        _update_task_status_and_notes(task_path=task_file, new_state="active", note_line=timeout_note)
        if _git_has_changes(repo):
            _run(["git", "add", "-A"], cwd=repo)
            _git_commit(cwd=repo, message=f"{task.task_id}: worker timeout", strict=strict_sync)
            _git_push(
                cwd=repo,
                remote=args.remote,
                ref=_git_current_branch(repo),
                set_upstream=True,
                strict=strict_sync,
            )
        print(json.dumps({"task_id": task.task_id, "state": "active", "error": "worker_timeout"}, indent=2))
        return 1

    # Judge: run declared gates (deterministic) + enforce path ownership
    gate_ok = True
    gate_outputs: list[dict[str, Any]] = []
    for gate in task.gates:
        # gates are declared in task files; run as shell for simplicity
        print(f"[judge] running gate: {gate}")
        cp = subprocess.run(gate, cwd=str(repo), shell=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        gate_outputs.append({"command": gate, "returncode": cp.returncode, "output": (cp.stdout or "")[-2000:]})
        if cp.returncode != 0:
            gate_ok = False

    task_rel = task_file.relative_to(repo).as_posix()
    task_paths = {task_rel}

    # Ownership enforcement (hardened):
    #
    # Historically we only checked `git status` at the end of the run. That is
    # porous if the Worker (or its tooling) creates commits: committed changes
    # are invisible to `git status` and can bypass allowed/disallowed paths.
    #
    # We now collect changed paths from:
    # - committed branch diff vs base ref (`base...HEAD`)  [AC1]
    # - staged changes                                   [AC1]
    # - unstaged changes                                 [AC1]
    # - untracked, non-ignored files                     [AC2]
    #
    # Then we evaluate *every touched path* against the task's allowed/disallowed
    # prefixes and the `.orchestrator/` special rules. This is offline and fast
    # (git-only) [AC4].
    base_ref = _resolve_base_ref_for_diff(cwd=repo, base_branch=args.base_branch, remote=args.remote)
    path_sources, ops = _collect_changed_paths_with_sources(repo=repo, base_ref=base_ref)

    ownership_ok = True
    ownership_failures: list[dict[str, str]] = []

    # Fail closed if we cannot compute the committed diff base ref. Without a
    # base ref we'd lose AC1 protection (committed bypass). This should be rare
    # in normal operation (tasks are created from base_branch), but if it happens
    # we block with an actionable error.
    if base_ref is None:
        ownership_ok = False
        ownership_failures.append(
            {
                "path": args.base_branch,
                "reason": "base_ref_unresolved",
                "sources": "committed",
            }
        )

    # Task file integrity checks (AC3): detect deletes/renames even if committed.
    for op in ops:
        code = op.get("code", "")
        src = op.get("source", "")
        p = op.get("path", "")
        old = op.get("old_path", "")

        # Rename/move: `R* old -> new`
        if code == "R" and old == task_rel and p and p != task_rel:
            ownership_ok = False
            ownership_failures.append(
                {
                    "path": f"{old} -> {p}",
                    "reason": "task_file_moved",
                    "sources": src,
                }
            )
        # Delete: `D path`
        if code == "D" and p == task_rel:
            ownership_ok = False
            ownership_failures.append(
                {
                    "path": p,
                    "reason": "task_file_deleted",
                    "sources": src,
                }
            )

    # Evaluate all touched paths against allow/deny rules (AC1/AC2/AC5).
    # We collapse duplicates across sources to keep reports readable.
    seen_failures: set[tuple[str, str]] = set()
    for p in sorted(path_sources.keys()):
        ok, reason = _path_is_allowed(
            path=p,
            allowed_paths=task.allowed_paths,
            disallowed_paths=task.disallowed_paths,
            task_file_paths=set(task_paths),
        )
        if ok:
            continue
        r = reason or "unknown"
        key = (p, r)
        if key in seen_failures:
            continue
        seen_failures.add(key)
        ownership_ok = False
        ownership_failures.append(
            {
                "path": p,
                "reason": r,
                "sources": ",".join(sorted(path_sources.get(p, set()))),
            }
        )

    # Output existence gate (expert feedback item C): only allow a task to reach
    # `ready_for_review` / `done` if it actually produced its declared outputs.
    outputs_ok = True
    output_failures: list[dict[str, str]] = []
    if gate_ok and ownership_ok:
        outputs_ok, output_failures = _check_task_outputs_exist(repo=repo, task=task)

    # Optional: Codex review summary (best-effort, non-blocking)
    review_path = logs_dir / f"{task.task_id}_{_utc_timestamp_compact()}_judge_review.txt"
    try:
        review_prompt = "\n".join(
            [
                "Role: Judge.",
                "Review ONLY the uncommitted changes for this task.",
                "Check alignment with the task success criteria and any obvious contract violations.",
                "Return a short, actionable bullet list. Do not propose scope creep.",
            ]
        )
        review_cmd = _codex_review_cmd(
            prompt=review_prompt,
            unattended=args.unattended,
            base_branch=args.base_branch,
            workdir=repo,
        )
        try:
            cp = _run(
                review_cmd,
                cwd=repo,
                capture=True,
                check=False,
                timeout_seconds=int(args.max_review_seconds) if args.max_review_seconds else None,
            )
        except subprocess.TimeoutExpired:
            cp = subprocess.CompletedProcess(args=review_cmd, returncode=124, stdout="")
        review_path.write_text(cp.stdout or "", encoding="utf-8")
    except Exception:
        pass

    # Decide new state
    if gate_ok and ownership_ok and outputs_ok:
        new_state = args.final_state
        note = f"Judge: gates ok; ownership ok. Review log: {review_path.as_posix()}"
    else:
        new_state = "blocked"
        why: list[str] = []
        if not gate_ok:
            why.append("gates_failed")
        if not ownership_ok:
            why.append("path_ownership_violation")
        if not outputs_ok:
            why.append("missing_outputs")
        violations = ""
        if not ownership_ok and ownership_failures:
            # Include a clear, grep-able violations list in the task note (AC1).
            # Format: path[sources]=reason
            violations = " Violations: " + "; ".join(
                [
                    f"{v.get('path')}[{v.get('sources', 'unknown')}]={v.get('reason', 'unknown')}"
                    for v in ownership_failures
                ]
            )
        missing = ""
        if not outputs_ok and output_failures:
            missing = " Missing outputs: " + "; ".join(
                [f"{f.get('output')}={f.get('reason')}" for f in output_failures]
            )
        note = f"@human Judge blocked: {', '.join(why)}.{violations}{missing} Review log: {review_path.as_posix()}"
    if args.repair_context:
        note = f"{note} Repair context: {args.repair_context}"

    # Update task status (do NOT move file; Planner action is separate via sweep_tasks.py)
    _update_task_status_and_notes(task_path=task_file, new_state=new_state, note_line=note)

    # Commit + push
    if _git_has_changes(repo):
        _run(["git", "add", "-A"], cwd=repo)
        _git_commit(cwd=repo, message=f"{task.task_id}: {new_state}", strict=strict_sync)
        _git_push(
            cwd=repo,
            remote=args.remote,
            ref=_git_current_branch(repo),
            set_upstream=True,
            strict=strict_sync,
        )

    # PR (optional)
    if args.create_pr:
        pr_title = f"{task.task_id}: {task.title}"
        pr_body = "\n".join(
            [
                f"Task: `{task_file.as_posix()}`",
                f"State: `{new_state}`",
                "",
                "Gates run:",
                *(f"- `{g['command']}` (rc={g['returncode']})" for g in gate_outputs),
                "",
                "Notes:",
                "- This PR was generated by the swarm supervisor (unattended).",
                "- Review the task file Notes / Decisions for context.",
            ]
        )
        _gh_create_pr_if_missing(cwd=repo, base_branch=args.base_branch, title=pr_title, body=pr_body)
        if args.auto_merge and new_state in {"ready_for_review", "done"}:
            _maybe_auto_merge(cwd=repo, squash=True)

    print(
        json.dumps(
            {
                "task_id": task.task_id,
                "state": new_state,
                "branch": _git_current_branch(repo),
                "gate_ok": gate_ok,
                "ownership_ok": ownership_ok,
                "ownership_failures": ownership_failures,
                "outputs_ok": outputs_ok,
                "output_failures": output_failures,
                "review_log": str(review_path),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="swarm.py")
    sub = p.add_subparsers(dest="cmd", required=True)

    plan = sub.add_parser("plan", help="Print done/claimed/ready tasks (JSON)")
    plan.add_argument("--remote", default="origin")
    plan.add_argument("--base-branch", default="main")
    plan.set_defaults(func=cmd_plan)

    tick = sub.add_parser("tick", help="Start up to N ready tasks (spawns tmux windows by default)")
    tick.add_argument("--planner", choices=["heuristic", "claude"], default="heuristic")
    tick.add_argument("--runner", choices=["tmux", "local"], default="tmux")
    tick.add_argument("--tmux-session", default="swarm")
    tick.add_argument("--max-workers", type=int, default=1)
    tick.add_argument("--worktree-parent", default=None)
    tick.add_argument("--remote", default="origin")
    tick.add_argument("--base-branch", default="main")
    tick.add_argument("--codex-model", default=None)
    tick.add_argument("--claude-model", default=None)
    tick.add_argument("--codex-sandbox", choices=["read-only", "workspace-write", "danger-full-access"], default="workspace-write")
    tick.add_argument("--unattended", action="store_true")
    tick.add_argument("--max-worker-seconds", type=int, default=0)
    tick.add_argument("--max-review-seconds", type=int, default=0)
    tick.add_argument("--repair-after-seconds", type=int, default=14400)
    tick.add_argument("--max-repairs-per-tick", type=int, default=1)
    tick.add_argument("--create-pr", action="store_true")
    tick.add_argument("--auto-merge", action="store_true")
    tick.add_argument("--final-state", choices=["ready_for_review", "done"], default="ready_for_review")
    tick.add_argument("--dry-run", action="store_true")
    tick.set_defaults(func=cmd_tick)

    loop = sub.add_parser("loop", help="Run tick repeatedly (intended to be run inside tmux)")
    loop.add_argument("--interval-seconds", type=int, default=300)
    loop.add_argument("--planner", choices=["heuristic", "claude"], default="heuristic")
    loop.add_argument("--runner", choices=["tmux", "local"], default="tmux")
    loop.add_argument("--tmux-session", default="swarm")
    loop.add_argument("--max-workers", type=int, default=1)
    loop.add_argument("--worktree-parent", default=None)
    loop.add_argument("--remote", default="origin")
    loop.add_argument("--base-branch", default="main")
    loop.add_argument("--codex-model", default=None)
    loop.add_argument("--claude-model", default=None)
    loop.add_argument("--codex-sandbox", choices=["read-only", "workspace-write", "danger-full-access"], default="workspace-write")
    loop.add_argument("--unattended", action="store_true")
    loop.add_argument("--max-worker-seconds", type=int, default=0)
    loop.add_argument("--max-review-seconds", type=int, default=0)
    loop.add_argument("--repair-after-seconds", type=int, default=14400)
    loop.add_argument("--max-repairs-per-tick", type=int, default=1)
    loop.add_argument("--create-pr", action="store_true")
    loop.add_argument("--auto-merge", action="store_true")
    loop.add_argument("--final-state", choices=["ready_for_review", "done"], default="ready_for_review")
    loop.add_argument("--dry-run", action="store_true")
    loop.set_defaults(func=cmd_loop)

    tmux_start = sub.add_parser("tmux-start", help="Create tmux session + start supervisor loop window")
    tmux_start.add_argument("--tmux-session", default="swarm")
    tmux_start.add_argument("--attach", action="store_true")
    tmux_start.add_argument("--interval-seconds", type=int, default=300)
    tmux_start.add_argument("--planner", choices=["heuristic", "claude"], default="heuristic")
    tmux_start.add_argument("--max-workers", type=int, default=1)
    tmux_start.add_argument("--worktree-parent", default=None)
    tmux_start.add_argument("--remote", default="origin")
    tmux_start.add_argument("--base-branch", default="main")
    tmux_start.add_argument("--codex-model", default=None)
    tmux_start.add_argument("--claude-model", default=None)
    tmux_start.add_argument("--codex-sandbox", choices=["read-only", "workspace-write", "danger-full-access"], default="workspace-write")
    tmux_start.add_argument("--unattended", action="store_true")
    tmux_start.add_argument("--max-worker-seconds", type=int, default=0)
    tmux_start.add_argument("--max-review-seconds", type=int, default=0)
    tmux_start.add_argument("--repair-after-seconds", type=int, default=14400)
    tmux_start.add_argument("--max-repairs-per-tick", type=int, default=1)
    tmux_start.add_argument("--create-pr", action="store_true")
    tmux_start.add_argument("--auto-merge", action="store_true")
    tmux_start.add_argument("--final-state", choices=["ready_for_review", "done"], default="ready_for_review")
    tmux_start.set_defaults(func=cmd_tmux_start)

    run_task = sub.add_parser("run-task", help="Run a single task in the current worktree (Codex worker + gates + PR)")
    run_task.add_argument("--task-id", required=True)
    run_task.add_argument("--remote", default="origin")
    run_task.add_argument("--base-branch", default="main")
    run_task.add_argument("--codex-model", default=None)
    run_task.add_argument("--codex-sandbox", choices=["read-only", "workspace-write", "danger-full-access"], default="workspace-write")
    run_task.add_argument("--unattended", action="store_true")
    run_task.add_argument("--max-worker-seconds", type=int, default=0, help="If >0, timeout Codex worker execution")
    run_task.add_argument("--max-review-seconds", type=int, default=0, help="If >0, timeout optional Codex review")
    run_task.add_argument("--repair-context", default=None, help="Optional context string for automated repair passes")
    run_task.add_argument("--create-pr", action="store_true")
    run_task.add_argument("--auto-merge", action="store_true")
    run_task.add_argument("--final-state", choices=["ready_for_review", "done"], default="ready_for_review")
    run_task.set_defaults(func=cmd_run_task)

    return p


def main(argv: list[str]) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
