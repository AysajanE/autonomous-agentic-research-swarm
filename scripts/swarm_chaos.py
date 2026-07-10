#!/usr/bin/env python3
"""Bounded chaos and soak driver for a prepared synthetic swarm repository."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path
import random
import signal
import subprocess
import sys
import time


_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

import swarm_claims
import swarm_events
from swarm_taskfile import parse_status_value
from swarm_taskfile import parse_task_frontmatter


SWARM_PATH = _SCRIPTS_DIR / "swarm.py"
RECONCILE_PATH = _SCRIPTS_DIR / "swarm_reconcile.py"


def _base_env(repo: Path) -> dict[str, str]:
    env = dict(os.environ)
    env["SWARM_REPO_ROOT"] = str(repo)
    env["SWARM_EVENT_REPO_ROOT"] = str(repo)
    return env


def _worktree_records(repo: Path) -> list[dict[str, str]]:
    cp = subprocess.run(
        ["git", "-C", str(repo), "worktree", "list", "--porcelain"],
        check=False,
        capture_output=True,
        text=True,
    )
    records: list[dict[str, str]] = []
    current: dict[str, str] = {}
    for line in [*cp.stdout.splitlines(), ""]:
        if not line.strip():
            if current:
                records.append(current)
                current = {}
            continue
        key, _, value = line.partition(" ")
        current[key] = value.strip()
    return records


def _worktree_for_branch(repo: Path, branch: str) -> Path | None:
    expected = f"refs/heads/{branch}"
    for record in _worktree_records(repo):
        if record.get("branch") == expected and record.get("worktree"):
            return Path(record["worktree"]).resolve()
    return None


def _sleeping_worker_candidate(repo: Path) -> tuple[str, Path, swarm_claims.ClaimState] | None:
    claims = swarm_claims.read_claims(repo, "origin", fetch=False)
    for task_id, claim in sorted(claims.items()):
        branch = claim.payload.get("branch")
        if not isinstance(branch, str) or not branch:
            continue
        worktree = _worktree_for_branch(repo, branch)
        if worktree is None:
            continue
        transcript = worktree / ".orchestrator" / "mock_transcripts" / f"{task_id}.json"
        try:
            payload = json.loads(transcript.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        actions = payload.get("actions") if isinstance(payload, dict) else None
        if isinstance(actions, list) and any(
            isinstance(action, dict)
            and isinstance(action.get("sleep_seconds"), (int, float))
            and not isinstance(action.get("sleep_seconds"), bool)
            and 0 < action["sleep_seconds"] <= 30
            for action in actions
        ):
            return task_id, worktree, claim
    return None


def _wait_for_run_started(repo: Path, task_id: str, previous_count: int) -> bool:
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        events, _ = swarm_events.read_events(repo)
        count = sum(
            event.get("event") == "run_started" and event.get("task_id") == task_id
            for event in events
        )
        if count > previous_count:
            return True
        time.sleep(0.05)
    return False


def _kill_group(proc: subprocess.Popen[str]) -> bool:
    if proc.poll() is not None:
        return False
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        return False
    return True


def _wait_for_claim_expiry(repo: Path, task_id: str) -> str | None:
    claim = swarm_claims.read_claims(repo, "origin", fetch=False).get(task_id)
    if claim is None:
        return "worker_claim_disappeared_before_reap"
    heartbeat = claim.payload.get("heartbeat_at_utc")
    ttl = claim.payload.get("lease_ttl_seconds")
    if not isinstance(heartbeat, str) or not isinstance(ttl, int):
        return None
    try:
        beat = dt.datetime.fromisoformat(heartbeat.replace("Z", "+00:00"))
    except ValueError:
        return None
    remaining = ttl - (dt.datetime.now(tz=dt.timezone.utc) - beat).total_seconds() + 0.05
    if remaining > 5:
        return f"worker_kill_requires_short_lease_ttl:{ttl}"
    if remaining > 0:
        time.sleep(remaining)
    return None


def _inject_worker_kill(repo: Path, rng: random.Random) -> tuple[dict, list[str]]:
    errors: list[str] = []
    candidate = _sleeping_worker_candidate(repo)
    if candidate is None:
        return {"injected": False}, ["worker_kill_requires_claimed_sleeping_task"]
    task_id, worktree, claim = candidate
    prior_events, _ = swarm_events.read_events(repo)
    prior_started = sum(
        event.get("event") == "run_started" and event.get("task_id") == task_id
        for event in prior_events
    )
    env = _base_env(repo)
    session_id = claim.session_id
    if not session_id:
        return {"injected": False, "task_id": task_id}, ["worker_claim_missing_session"]
    env["SWARM_REPO_ROOT"] = str(worktree)
    env["SWARM_ACTOR_SESSION"] = session_id
    command = [
        sys.executable,
        str(SWARM_PATH),
        "run-task",
        "--task-id",
        task_id,
        "--remote",
        "origin",
        "--base-branch",
        "main",
        "--executor-backend",
        "mock",
        "--codex-sandbox",
        "workspace-write",
        "--final-state",
        "ready_for_review",
    ]
    proc = subprocess.Popen(
        command,
        cwd=worktree,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    observed_start = _wait_for_run_started(repo, task_id, prior_started)
    delay = rng.uniform(0.8, 1.2)
    if observed_start:
        time.sleep(delay)
    killed = _kill_group(proc)
    output, _ = proc.communicate()
    if not observed_start:
        errors.append("worker_run_started_not_observed")
    if not killed:
        errors.append(f"worker_exited_before_kill:{proc.returncode}")
    expiry_error = _wait_for_claim_expiry(repo, task_id) if killed else None
    if expiry_error:
        errors.append(expiry_error)
    return (
        {
            "injected": killed,
            "task_id": task_id,
            "delay_seconds": round(delay, 3),
            "returncode": proc.returncode,
            "output_tail": (output or "")[-1000:],
        },
        errors,
    )


def _supervise_command(repo: Path, *, max_workers: int) -> list[str]:
    worktree_parent = repo.parent / f"{repo.name}.swarm-worktrees"
    return [
        sys.executable,
        str(SWARM_PATH),
        "supervise",
        "--once",
        "--runner",
        "local",
        "--max-workers",
        str(max_workers),
        "--worktree-parent",
        str(worktree_parent),
        "--remote",
        "origin",
        "--base-branch",
        "main",
        "--executor-backend",
        "mock",
        "--codex-sandbox",
        "workspace-write",
    ]


def _run_cycle(
    repo: Path,
    cycle: int,
    *,
    max_workers: int,
    kill: bool,
    rng: random.Random,
) -> tuple[dict, str | None]:
    command = _supervise_command(repo, max_workers=max_workers)
    proc = subprocess.Popen(
        command,
        cwd=repo,
        env=_base_env(repo),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    delay: float | None = None
    killed = False
    kill_phase_seen = False
    if kill:
        # phase-synchronized kill: wait for the CYCLE'S OWN progress (a new
        # journal event) rather than wall-clock — scheduler speed must not
        # decide whether the kill lands (CI runners finish cycles faster
        # than any fixed delay).
        journal = repo / "reports" / "status" / "events" / "events.jsonl"
        baseline = journal.stat().st_size if journal.is_file() else 0
        deadline = time.monotonic() + 30.0
        jitter = rng.uniform(0.0, 0.05)
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                break
            current = journal.stat().st_size if journal.is_file() else 0
            if current > baseline:
                kill_phase_seen = True
                time.sleep(jitter)
                killed = _kill_group(proc)
                break
            time.sleep(0.02)
        delay = round(time.monotonic() - (deadline - 30.0), 3)
    output, _ = proc.communicate()
    record = {
        "cycle": cycle,
        "returncode": proc.returncode,
        "killed": killed,
        "kill_phase_seen": kill_phase_seen,
        "output_tail": (output or "")[-2000:],
    }
    if delay is not None:
        record["kill_delay_seconds"] = delay
    if kill and not killed and not kill_phase_seen:
        return record, f"supervisor_exited_before_kill:cycle={cycle}:rc={proc.returncode}"
    if kill and not killed and kill_phase_seen:
        # the cycle raced to completion between event and signal: report the
        # miss honestly; the caller retries the kill on the next cycle
        record["kill_missed_race"] = True
        return record, None
    if not kill and proc.returncode != 0:
        return record, f"supervisor_cycle_failed:cycle={cycle}:rc={proc.returncode}"
    return record, None


def _final_task_states(repo: Path) -> dict[str, str | None]:
    states: dict[str, str | None] = {}
    control = repo / ".orchestrator"
    for folder in (
        "backlog",
        "active",
        "integration_ready",
        "ready_for_review",
        "blocked",
        "done",
    ):
        for path in sorted((control / folder).glob("*.md")):
            if path.name == "README.md":
                continue
            try:
                text = path.read_text(encoding="utf-8")
            except OSError:
                continue
            frontmatter = parse_task_frontmatter(text)
            task_id = frontmatter.get("task_id") if isinstance(frontmatter, dict) else None
            if isinstance(task_id, str):
                states[task_id] = parse_status_value(text, "State")
    return dict(sorted(states.items()))


def _serial_merges(events: list[dict]) -> bool:
    inflight: str | None = None
    terminal = {
        "task_done",
        "merge_reverted",
        "merge_refused_operator_surface",
        "merge_refused_stale_lease",
        "merge_refused_non_ff",
    }
    for event in events:
        name = event.get("event")
        task_id = event.get("task_id")
        if name == "merge_started" and isinstance(task_id, str):
            if inflight is not None and inflight != task_id:
                return False
            inflight = task_id
        elif name in terminal and task_id == inflight:
            inflight = None
    return True


def _run_reconcile(repo: Path) -> tuple[dict, int, str]:
    cp = subprocess.run(
        [sys.executable, str(RECONCILE_PATH), "--json"],
        cwd=repo,
        check=False,
        capture_output=True,
        text=True,
    )
    try:
        payload = json.loads(cp.stdout)
    except json.JSONDecodeError:
        payload = {
            "ok": False,
            "findings": [
                {
                    "class": "reconcile_output_invalid",
                    "stdout_tail": cp.stdout[-1000:],
                    "stderr_tail": cp.stderr[-1000:],
                }
            ],
            "findings_count": 1,
        }
    return payload, cp.returncode, cp.stderr


def run_chaos(args: argparse.Namespace) -> dict:
    repo = Path(args.repo).expanduser().resolve()
    rng = random.Random(args.seed)
    errors: list[str] = []
    cycle_results: list[dict] = []
    initial_task_states = _final_task_states(repo)
    kills = {
        "supervisor": 0,
        "worker": 0,
    }

    worker_result: dict | None = None
    if args.kill_worker:
        worker_result, worker_errors = _inject_worker_kill(repo, rng)
        errors.extend(worker_errors)
        if worker_result.get("injected"):
            kills["worker"] = 1

    kill_pending = args.kill_supervisor_at_cycle
    for cycle in range(1, args.cycles + 1):
        kill_supervisor = kill_pending == cycle
        max_workers = 0 if args.kill_worker and cycle == 1 else 2
        result, error = _run_cycle(
            repo,
            cycle,
            max_workers=max_workers,
            kill=kill_supervisor,
            rng=rng,
        )
        cycle_results.append(result)
        if result["killed"]:
            kills["supervisor"] += 1
        elif kill_supervisor and result.get("kill_missed_race") and cycle < args.cycles:
            # the cycle completed between phase-event and signal: retry once
            # on the next cycle so a kill ALWAYS lands within the run
            kill_pending = cycle + 1
        if error:
            errors.append(error)

    events, malformed_count = swarm_events.read_events(repo)
    final_task_states = _final_task_states(repo)
    missing_tasks = sorted(set(initial_task_states) - set(final_task_states))
    blocked_tasks = sorted(
        task_id for task_id, state in final_task_states.items() if state == "blocked"
    )
    if missing_tasks:
        errors.append(f"tasks_missing_after_chaos:{','.join(missing_tasks)}")
    if blocked_tasks:
        errors.append(f"tasks_blocked_after_chaos:{','.join(blocked_tasks)}")
    serial_merges_ok = _serial_merges(events)
    if malformed_count:
        errors.append(f"malformed_journal_records:{malformed_count}")
    if not serial_merges_ok:
        errors.append("interleaved_merge_events")
    if args.kill_worker and worker_result and worker_result.get("task_id"):
        orphaned = any(
            event.get("event") == "task_orphaned"
            and event.get("task_id") == worker_result["task_id"]
            for event in events
        )
        if not orphaned:
            errors.append(f"worker_orphan_not_reaped:{worker_result['task_id']}")
    else:
        orphaned = False

    reconciliation, reconcile_returncode, reconcile_stderr = _run_reconcile(repo)
    if reconcile_returncode != 0 or not reconciliation.get("ok"):
        errors.append(f"reconciliation_failed:rc={reconcile_returncode}")

    return {
        "ok": not errors,
        "repo": str(repo),
        "seed": args.seed,
        "cycles_requested": args.cycles,
        "cycles_run": len(cycle_results),
        "kills_injected": kills,
        "worker_kill": worker_result,
        "worker_orphan_reaped": orphaned,
        "cycle_results": cycle_results,
        "initial_task_states": initial_task_states,
        "final_task_states": final_task_states,
        "live_claim_count": len(swarm_claims.read_claims(repo, "origin", fetch=False)),
        "serial_merges_ok": serial_merges_ok,
        "journal_event_count": len(events),
        "journal_malformed_count": malformed_count,
        "reconciliation": {
            "ok": reconciliation.get("ok", False),
            "findings": reconciliation.get("findings", []),
            "findings_count": reconciliation.get("findings_count", 0),
            "warnings": reconciliation.get("warnings", []),
            "warnings_count": reconciliation.get("warnings_count", 0),
            "checks": reconciliation.get("checks", {}),
            "stderr": reconcile_stderr[-1000:],
        },
        "errors": errors,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--cycles", required=True, type=int)
    parser.add_argument("--kill-supervisor-at-cycle", type=int, default=None)
    parser.add_argument("--kill-worker", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.cycles < 1:
        parser.error("--cycles must be >= 1")
    if args.kill_supervisor_at_cycle is not None and not (
        1 <= args.kill_supervisor_at_cycle <= args.cycles
    ):
        parser.error("--kill-supervisor-at-cycle must be within 1..--cycles")
    payload = run_chaos(args)
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(
            f"swarm_chaos:{'clean' if payload['ok'] else 'failed'} "
            f"cycles={payload['cycles_run']} kills={payload['kills_injected']} "
            f"events={payload['journal_event_count']}"
        )
        for error in payload["errors"]:
            print(f"error:{error}")
    return 0 if payload["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
