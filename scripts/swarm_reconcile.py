#!/usr/bin/env python3
"""Offline event-journal to repository-state reconciliation checks."""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
import subprocess
import sys
from typing import Iterable


_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

import swarm_claims
import swarm_events
from swarm_taskfile import parse_status_value
from swarm_taskfile import parse_task_frontmatter


RUN_MANIFEST_V2 = "research_swarm.runtime_run_manifest.v2"
REVIEW_LOG_V2 = "research_swarm.judge_review_log.v2"
MAX_REPORTED_ITEMS = 100


def _run_git(repo: Path, args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        check=False,
        capture_output=True,
        text=True,
    )


def _framework(repo: Path) -> dict:
    try:
        payload = json.loads((repo / "contracts/framework.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _repo_layout(repo: Path) -> tuple[Path, list[str], str, str, str]:
    framework = _framework(repo)
    engines = framework.get("execution_engines")
    routine = engines.get("routine_repo_tasks") if isinstance(engines, dict) else None
    control_raw = (
        routine.get("control_plane_root")
        if isinstance(routine, dict) and isinstance(routine.get("control_plane_root"), str)
        else ".orchestrator"
    )
    control = repo / control_raw

    states = framework.get("states")
    projection_raw = states.get("projection_dirs") if isinstance(states, dict) else None
    projections = (
        [Path(item).name for item in projection_raw if isinstance(item, str)]
        if isinstance(projection_raw, list)
        else []
    )
    if not projections:
        projections = [
            "backlog",
            "active",
            "integration_ready",
            "ready_for_review",
            "blocked",
            "done",
        ]

    bundle = framework.get("review_bundle")
    run_dir = (
        bundle.get("run_manifest_dir")
        if isinstance(bundle, dict) and isinstance(bundle.get("run_manifest_dir"), str)
        else "reports/status/swarm_runs"
    )
    review_dir = (
        bundle.get("judge_review_dir")
        if isinstance(bundle, dict) and isinstance(bundle.get("judge_review_dir"), str)
        else "reports/status/reviews"
    )
    roles = framework.get("roles")
    review_role = (
        roles.get("scientific_review_role")
        if isinstance(roles, dict) and isinstance(roles.get("scientific_review_role"), str)
        else "Judge"
    )
    return control, projections, run_dir, review_dir, review_role


def _worktree_roots(repo: Path) -> list[Path]:
    roots = [repo.resolve()]
    cp = _run_git(repo, ["worktree", "list", "--porcelain"])
    if cp.returncode != 0:
        return roots
    for line in cp.stdout.splitlines():
        if not line.startswith("worktree "):
            continue
        candidate = Path(line.partition(" ")[2]).resolve()
        if candidate not in roots:
            roots.append(candidate)
    return roots


def _read_json(path: Path) -> dict | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _parse_utc_iso(value: object) -> dt.datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        return dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _artifact_candidates(repo: Path, roots: Iterable[Path], raw_path: object) -> list[Path]:
    if not isinstance(raw_path, str) or not raw_path.strip():
        return []
    relative = Path(raw_path)
    if relative.is_absolute() or ".." in relative.parts:
        return []
    candidates: list[Path] = []
    for root in roots:
        candidate = (root / relative).resolve()
        try:
            candidate.relative_to(root.resolve())
        except ValueError:
            continue
        if candidate.is_file() and candidate not in candidates:
            candidates.append(candidate)
    return candidates


def _task_files(control: Path, projections: Iterable[str]) -> tuple[dict[str, dict], list[dict]]:
    tasks: dict[str, dict] = {}
    problems: list[dict] = []
    for state_dir in projections:
        folder = control / state_dir
        if not folder.is_dir():
            continue
        for path in sorted(folder.glob("*.md")):
            if path.name == "README.md":
                continue
            try:
                text = path.read_text(encoding="utf-8")
            except OSError as exc:
                problems.append({"class": "task_file_unreadable", "path": str(path), "detail": str(exc)})
                continue
            frontmatter = parse_task_frontmatter(text)
            task_id = frontmatter.get("task_id") if isinstance(frontmatter, dict) else None
            state = parse_status_value(text, "State")
            if not isinstance(task_id, str) or not task_id:
                problems.append({"class": "task_id_missing", "path": str(path)})
                continue
            if task_id in tasks:
                problems.append({"class": "duplicate_task_file", "task_id": task_id, "path": str(path)})
                continue
            tasks[task_id] = {"path": path, "state": state, "folder": state_dir}
    return tasks, problems


def _valid_run_manifest(payload: dict | None, task_id: str) -> bool:
    if not isinstance(payload, dict):
        return False
    task = payload.get("task")
    result = payload.get("result")
    return (
        payload.get("schema_version") == RUN_MANIFEST_V2
        and payload.get("provenance_class") == "executor_run"
        and isinstance(task, dict)
        and task.get("task_id") == task_id
        and isinstance(result, dict)
        and result.get("status") == "ok"
        and _parse_utc_iso(payload.get("generated_at_utc")) is not None
    )


def _valid_approval(payload: dict | None, task_id: str, review_role: str) -> bool:
    if not isinstance(payload, dict) or payload.get("schema_version") != REVIEW_LOG_V2:
        return False
    task = payload.get("task")
    reviewer = payload.get("reviewer")
    checks = payload.get("checks")
    decision = payload.get("decision")
    return (
        isinstance(task, dict)
        and task.get("task_id") == task_id
        and task.get("state_after") == "done"
        and isinstance(reviewer, dict)
        and reviewer.get("role") == review_role
        and isinstance(reviewer.get("session_id"), str)
        and bool(reviewer["session_id"].strip())
        and isinstance(checks, dict)
        and checks.get("gates_ok") is True
        and checks.get("outputs_ok") is True
        and checks.get("required_manifests_ok") is True
        and checks.get("review_bundle_ok") is True
        and isinstance(decision, dict)
        and decision.get("outcome") == "approve"
    )


def _collect_task_artifacts(
    roots: Iterable[Path],
    directory: str,
    task_id: str,
) -> list[tuple[Path, dict | None]]:
    collected: list[tuple[Path, dict | None]] = []
    seen: set[Path] = set()
    for root in roots:
        folder = root / directory
        if not folder.is_dir():
            continue
        for path in sorted(folder.glob(f"{task_id}_*.json")):
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            collected.append((resolved, _read_json(resolved)))
    return collected


def _event_task_id(event: dict) -> str | None:
    value = event.get("task_id")
    return value if isinstance(value, str) and value else None


def reconcile(repo: Path) -> dict:
    repo = repo.resolve()
    control, projections, run_dir, review_dir, review_role = _repo_layout(repo)
    roots = _worktree_roots(repo)
    tasks, task_problems = _task_files(control, projections)
    events, malformed_count = swarm_events.read_events(repo)
    events_by_task: dict[str, list[dict]] = {}
    for event in events:
        task_id = _event_task_id(event)
        if task_id is not None:
            events_by_task.setdefault(task_id, []).append(event)

    findings: list[dict] = []
    warnings: list[dict] = []
    checks: dict[str, dict[str, int]] = {
        name: {"checked": 0, "findings": 0, "warnings": 0}
        for name in (
            "journal_integrity",
            "task_done_bundle",
            "done_event_trail",
            "claim_ref_reconciliation",
            "run_finished_manifests",
            "merge_reverted_history",
            "spend_ledger",
        )
    }

    def add_finding(check: str, finding_class: str, **detail: object) -> None:
        checks[check]["findings"] += 1
        findings.append({"check": check, "class": finding_class, **detail})

    def add_warning(check: str, warning_class: str, **detail: object) -> None:
        checks[check]["warnings"] += 1
        warnings.append({"check": check, "class": warning_class, **detail})

    checks["journal_integrity"]["checked"] = len(events) + malformed_count
    if malformed_count:
        add_finding(
            "journal_integrity",
            "malformed_journal_records",
            count=malformed_count,
        )
    for problem in task_problems:
        add_finding("journal_integrity", str(problem.pop("class")), **problem)

    try:
        claims = swarm_claims.read_claims(repo, "origin", fetch=False)
    except Exception as exc:
        claims = {}
        add_finding(
            "claim_ref_reconciliation",
            "claim_refs_unreadable",
            detail=f"{type(exc).__name__}:{exc}",
        )
    live_claim_ids = set(claims)

    task_done_events = [event for event in events if event.get("event") == "task_done"]
    checks["task_done_bundle"]["checked"] = len(task_done_events)
    for event in task_done_events:
        task_id = _event_task_id(event)
        if task_id is None:
            add_finding("task_done_bundle", "task_done_missing_task_id")
            continue
        task = tasks.get(task_id)
        if task is None:
            add_finding("task_done_bundle", "task_done_missing_task", task_id=task_id)
        elif task.get("state") != "done":
            add_finding(
                "task_done_bundle",
                "task_done_state_mismatch",
                task_id=task_id,
                state=task.get("state"),
            )

        approvals = _collect_task_artifacts(roots, review_dir, task_id)
        if not any(_valid_approval(payload, task_id, review_role) for _, payload in approvals):
            add_finding("task_done_bundle", "done_without_approval", task_id=task_id)

        manifests = _collect_task_artifacts(roots, run_dir, task_id)
        if not any(_valid_run_manifest(payload, task_id) for _, payload in manifests):
            add_finding(
                "task_done_bundle",
                "done_without_valid_executor_run",
                task_id=task_id,
            )
        if task_id in live_claim_ids:
            add_finding("task_done_bundle", "done_with_live_claim", task_id=task_id)

    done_with_run_events = [
        task_id
        for task_id, task in tasks.items()
        if task.get("state") == "done"
        and any(
            event.get("event") == "run_started"
            for event in events_by_task.get(task_id, [])
        )
    ]
    checks["done_event_trail"]["checked"] = len(done_with_run_events)
    for task_id in done_with_run_events:
        trail = events_by_task[task_id]
        has_completion = any(event.get("event") == "task_done" for event in trail)
        has_approval = any(
            event.get("event") == "review_recorded" and event.get("outcome") == "approve"
            for event in trail
        )
        if not (has_completion or has_approval):
            add_finding(
                "done_event_trail",
                "done_without_completion_trail",
                task_id=task_id,
            )

    open_claim_events: dict[str, bool] = {}
    claim_created_ids: set[str] = set()
    for event in events:
        task_id = _event_task_id(event)
        if task_id is None:
            continue
        if event.get("event") == "claim_created":
            claim_created_ids.add(task_id)
            open_claim_events[task_id] = True
        elif event.get("event") in {"claim_released", "task_orphaned"}:
            open_claim_events[task_id] = False
    journal_open_ids = {task_id for task_id, is_open in open_claim_events.items() if is_open}
    checks["claim_ref_reconciliation"]["checked"] += len(journal_open_ids | live_claim_ids)
    for task_id in sorted(journal_open_ids - live_claim_ids):
        add_warning(
            "claim_ref_reconciliation",
            "open_claim_event_without_live_ref",
            task_id=task_id,
        )
    for task_id in sorted(live_claim_ids - journal_open_ids):
        add_warning(
            "claim_ref_reconciliation",
            "live_claim_without_open_event",
            task_id=task_id,
        )
        if task_id not in claim_created_ids and tasks.get(task_id, {}).get("state") == "done":
            add_finding(
                "claim_ref_reconciliation",
                "done_live_claim_without_claim_event",
                task_id=task_id,
            )

    run_finished = [event for event in events if event.get("event") == "run_finished"]
    checks["run_finished_manifests"]["checked"] = len(run_finished)
    resolved_runs: list[tuple[dict, dict | None]] = []
    for event in run_finished:
        task_id = _event_task_id(event)
        candidates = _artifact_candidates(repo, roots, event.get("run_manifest"))
        if not candidates:
            add_finding(
                "run_finished_manifests",
                "missing_run_manifest",
                task_id=task_id,
                run_manifest=event.get("run_manifest"),
            )
            resolved_runs.append((event, None))
            continue
        payloads = [_read_json(path) for path in candidates]
        payload = next((item for item in payloads if isinstance(item, dict)), None)
        if payload is None:
            add_finding(
                "run_finished_manifests",
                "invalid_run_manifest_json",
                task_id=task_id,
                run_manifest=event.get("run_manifest"),
            )
        resolved_runs.append((event, payload))

    reverted_events = [event for event in events if event.get("event") == "merge_reverted"]
    checks["merge_reverted_history"]["checked"] = len(reverted_events)
    head = _run_git(repo, ["rev-parse", "HEAD"])
    head_sha = head.stdout.strip() if head.returncode == 0 else "HEAD"
    for event in reverted_events:
        task_id = _event_task_id(event)
        pre_merge_sha = event.get("pre_merge_sha")
        if not isinstance(pre_merge_sha, str) or not pre_merge_sha:
            add_finding(
                "merge_reverted_history",
                "merge_reverted_missing_pre_merge_sha",
                task_id=task_id,
            )
            continue
        ancestor = _run_git(repo, ["merge-base", "--is-ancestor", pre_merge_sha, head_sha])
        if ancestor.returncode != 0:
            add_finding(
                "merge_reverted_history",
                "reverted_pre_merge_not_ancestor",
                task_id=task_id,
                pre_merge_sha=pre_merge_sha,
            )
            continue
        branch = event.get("branch")
        if not isinstance(branch, str) or not branch:
            continue
        branch_exists = _run_git(repo, ["rev-parse", "--verify", "--quiet", branch])
        if branch_exists.returncode != 0:
            continue
        commits = _run_git(repo, ["rev-list", f"{pre_merge_sha}..{branch}"])
        leaked = [
            sha
            for sha in commits.stdout.splitlines()
            if _run_git(repo, ["merge-base", "--is-ancestor", sha, head_sha]).returncode == 0
        ]
        if leaked:
            add_finding(
                "merge_reverted_history",
                "reverted_commits_in_base",
                task_id=task_id,
                commits=leaked[:10],
            )

    first_v2_index: int | None = None
    for index, (_, payload) in enumerate(resolved_runs):
        if isinstance(payload, dict) and payload.get("schema_version") == RUN_MANIFEST_V2:
            first_v2_index = index
            break
    spend_runs = resolved_runs[first_v2_index:] if first_v2_index is not None else []
    checks["spend_ledger"]["checked"] = len(spend_runs)
    for event, payload in spend_runs:
        task_id = _event_task_id(event)
        if not isinstance(payload, dict):
            add_finding(
                "spend_ledger",
                "spend_manifest_unavailable",
                task_id=task_id,
                run_manifest=event.get("run_manifest"),
            )
            continue
        executor = payload.get("executor")
        if isinstance(executor, dict) and executor.get("error") == "executor_skipped":
            continue
        usage = payload.get("usage")
        if not isinstance(usage, dict):
            add_finding("spend_ledger", "missing_usage", task_id=task_id)
            continue
        wall_clock = usage.get("wall_clock_seconds")
        if (
            not isinstance(wall_clock, (int, float))
            or isinstance(wall_clock, bool)
            or wall_clock < 0
        ):
            add_finding(
                "spend_ledger",
                "missing_wall_clock_seconds",
                task_id=task_id,
            )

    reported_findings = findings[:MAX_REPORTED_ITEMS]
    reported_warnings = warnings[:MAX_REPORTED_ITEMS]
    return {
        "ok": not findings,
        "repo": str(repo),
        "journal_event_count": len(events),
        "malformed_event_count": malformed_count,
        "task_count": len(tasks),
        "live_claim_count": len(live_claim_ids),
        "checks": checks,
        "findings": reported_findings,
        "findings_count": len(findings),
        "truncated_findings": len(findings) - len(reported_findings),
        "warnings": reported_warnings,
        "warnings_count": len(warnings),
        "truncated_warnings": len(warnings) - len(reported_warnings),
    }


def _print_human(payload: dict) -> None:
    status = "clean" if payload["ok"] else "findings"
    print(
        f"swarm_reconcile:{status} events={payload['journal_event_count']} "
        f"tasks={payload['task_count']} claims={payload['live_claim_count']}"
    )
    for name, counts in payload["checks"].items():
        print(
            f"- {name}: checked={counts['checked']} "
            f"findings={counts['findings']} warnings={counts['warnings']}"
        )
    for finding in payload["findings"]:
        print(f"finding:{finding['class']}:{json.dumps(finding, sort_keys=True)}")
    for warning in payload["warnings"]:
        print(f"warning:{warning['class']}:{json.dumps(warning, sort_keys=True)}")
    if payload["truncated_findings"]:
        print(f"findings_truncated:{payload['truncated_findings']}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    payload = reconcile(Path.cwd())
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        _print_human(payload)
    return 0 if payload["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
