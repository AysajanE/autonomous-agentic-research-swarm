#!/usr/bin/env python3.11
"""Narrated, zero-cost, offline demonstration of one full swarm cycle.

`make demo` runs this. It builds a throwaway research pack in a temporary
directory, gives two mock Workers the same repo, and shows what the framework
does with each of them:

  Act 1  a Worker that stays inside its declared scope    -> work is accepted
  Act 2  a Worker that reaches outside its declared scope -> work is refused
  Act 3  a Judge asked to approve its own session's work  -> review is refused

Nothing here calls an LLM, opens a network socket, spends money, or touches the
repository you ran it from. The `mock` executor backend replays a scripted
transcript through the *real* runtime, so every guardrail you see fire is the
same code that runs when a real Codex or Claude Code Worker is driving.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path

KERNEL_ROOT = Path(__file__).resolve().parents[1]
SWARM = KERNEL_ROOT / "scripts" / "swarm.py"
SWARM_INIT = KERNEL_ROOT / "scripts" / "swarm_init.py"

GREEN = "\033[32m"
RED = "\033[31m"
CYAN = "\033[36m"
DIM = "\033[2m"
BOLD = "\033[1m"
RESET = "\033[0m"

_COLOR = sys.stdout.isatty() and os.environ.get("NO_COLOR") is None


def c(text: str, color: str) -> str:
    return f"{color}{text}{RESET}" if _COLOR else text


def banner(title: str) -> None:
    print()
    print(c("=" * 72, CYAN))
    print(c(f"  {title}", BOLD + CYAN if _COLOR else CYAN))
    print(c("=" * 72, CYAN))


def say(text: str) -> None:
    print(textwrap.fill(text, width=72))


def show(label: str, body: str) -> None:
    print()
    print(c(f"--- {label} " + "-" * max(0, 68 - len(label)), DIM))
    for line in body.rstrip("\n").splitlines():
        print(c("  " + line, DIM))
    print(c("-" * 72, DIM))


def git(repo: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-c", "user.email=demo@example.invalid", "-c", "user.name=swarm-demo", *args],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )


def swarm(repo: Path, *args: str) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ, SWARM_REPO_ROOT=str(repo))
    return subprocess.run(
        [sys.executable, str(SWARM), *args],
        cwd=repo,
        env=env,
        capture_output=True,
        text=True,
    )


TASK_TEMPLATE = """---
task_schema: research_swarm.task.v2
task_id: {task_id}
title: "{title}"
workstream: W9
task_kind: ops
complexity_tier: S
success_criteria:
  - id: SC1
    statement: "reports/tables/demo_summary.md is written"
    verification: "make demo-verify"
budgets: {{max_wall_clock: 1h, max_tokens: 100000, max_cost_usd: 10}}
checkpoint_contract: none
recon_required: false
recon_waiver: ""
allow_network: false
role: Worker
priority: medium
dependencies: []
integration_ready_dependencies: []
requires_tools: ["python", "git"]
requires_env: []
allowed_paths:
  - "reports/tables/demo_summary.md"
disallowed_paths:
  - "contracts/"
outputs:
  - "reports/tables/demo_summary.md"
gates:
  - make demo-verify
stop_conditions:
  - "Contract ambiguity"
---

# Task {task_id} — {title}

## Context

{context}

## Status

- State: backlog
- Last updated: 2026-01-01

## Notes / Decisions
"""

GOOD_TRANSCRIPT = {
    "schema_version": "research_swarm.mock_transcript.v1",
    "actions": [
        {"note": "Read the task contract. Writing only the one file I own."},
        {
            "write": "reports/tables/demo_summary.md",
            "content": "# Demo Summary\n\n| metric | value |\n|---|---|\n| rows | 42 |\n",
        },
        {"set_task_state": "ready_for_review"},
    ],
    "returncode": 0,
    "stdout": "worker finished inside declared scope",
    "usage": {"input_tokens": 1200, "output_tokens": 300},
}

ROGUE_TRANSCRIPT = {
    "schema_version": "research_swarm.mock_transcript.v1",
    "actions": [
        {"note": "I have decided the project contract is wrong and will rewrite it."},
        {
            "write": "contracts/project.yaml",
            "content": "mode: rewritten_without_permission\n",
        },
    ],
    "returncode": 0,
    "stdout": "worker attempted an out-of-scope edit",
    "usage": {"input_tokens": 900, "output_tokens": 100},
}

VERIFY_TARGET = """
.PHONY: demo-verify
demo-verify:
\t@test -f reports/tables/demo_summary.md \\
\t  && echo "demo-verify: reports/tables/demo_summary.md present"
"""


def build_repo(root: Path) -> Path:
    repo = root / "demo-pack"
    subprocess.run(
        [sys.executable, str(SWARM_INIT), "--mode", "empirical", "--output", str(repo)],
        check=True,
        capture_output=True,
        text=True,
    )

    # An offline bare remote keeps the runtime's push path real without network.
    remote = root / "origin.git"
    subprocess.run(["git", "init", "-q", "--bare", str(remote)], check=True)

    git(repo, "init", "-q", "-b", "main")
    git(repo, "remote", "add", "origin", str(remote))

    with (repo / "Makefile").open("a", encoding="utf-8") as handle:
        handle.write(VERIFY_TARGET)

    (repo / ".orchestrator" / "backlog").mkdir(parents=True, exist_ok=True)
    (repo / ".orchestrator" / "mock_transcripts").mkdir(parents=True, exist_ok=True)

    (repo / ".orchestrator" / "backlog" / "T900_write_summary.md").write_text(
        TASK_TEMPLATE.format(
            task_id="T900",
            title="Write the demo summary table",
            context="A well-behaved Worker: writes exactly the file it declared.",
        ),
        encoding="utf-8",
    )
    (repo / ".orchestrator" / "backlog" / "T901_rogue_edit.md").write_text(
        TASK_TEMPLATE.format(
            task_id="T901",
            title="Write the demo summary table (rogue Worker)",
            context="A misbehaving Worker: tries to rewrite the project contract.",
        ),
        encoding="utf-8",
    )

    transcripts = repo / ".orchestrator" / "mock_transcripts"
    (transcripts / "T900.json").write_text(json.dumps(GOOD_TRANSCRIPT, indent=2), encoding="utf-8")
    (transcripts / "T901.json").write_text(json.dumps(ROGUE_TRANSCRIPT, indent=2), encoding="utf-8")

    git(repo, "add", "-A")
    git(repo, "commit", "-qm", "demo: scaffold plus two queued tasks")
    git(repo, "push", "-q", "origin", "main")
    return repo


def run_act(repo: Path, task_id: str) -> dict:
    result = swarm(repo, "run-task", "--task-id", task_id, "--executor-backend", "mock")
    payload: dict = {}
    for chunk in (result.stdout or "").split("\n{"):
        text = chunk if chunk.strip().startswith("{") else "{" + chunk
        try:
            candidate = json.loads(text)
        except json.JSONDecodeError:
            continue
        if isinstance(candidate, dict) and "state_after" in candidate:
            payload = candidate
    if not payload:
        print(result.stdout or "", result.stderr or "")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--keep",
        action="store_true",
        help="keep the temporary demo repository instead of deleting it",
    )
    args = parser.parse_args()

    root = Path(tempfile.mkdtemp(prefix="swarm-demo-"))
    try:
        banner("Autonomous Agentic Research Swarm — 60 second demo")
        say(
            "This runs one complete swarm cycle with a mock Worker. No API key, "
            "no network, no cost, and no changes to your own repository. "
            "Everything you are about to see is the real runtime."
        )
        print()
        say(f"Scratch repository: {root}")

        banner("Setup — create a fresh research pack")
        say(
            "swarm_init.py generates a contract-valid project 'pack': the "
            "control plane, the contracts, and the task folders. Then we queue "
            "two tasks for two Workers."
        )
        repo = build_repo(root)

        task_text = (repo / ".orchestrator" / "backlog" / "T900_write_summary.md").read_text()
        frontmatter = task_text.split("---")[1]
        keep = [
            line
            for line in frontmatter.splitlines()
            if line.startswith(("task_id", "role", "allowed_paths", "disallowed_paths", "outputs", "gates", "  - "))
        ]
        show("the unit of work: a task file (excerpt)", "\n".join(keep[:14]))
        say(
            "That YAML block is the contract. `allowed_paths` is the only place "
            "this Worker may write. `gates` is what must pass before the work "
            "counts. The Worker never negotiates these in chat."
        )

        status = swarm(repo, "status", "--no-fetch")
        show("swarm.py status", status.stdout)

        banner("Act 1 — a Worker that stays in scope")
        say(
            "T900's Worker writes exactly the one file it declared. Watch the "
            "runtime claim the task, run the Worker, run the declared gate, and "
            "promote the task."
        )
        result = run_act(repo, "T900")
        show(
            "run-task result",
            json.dumps(result, indent=2) if result else "(no result payload)",
        )

        artifact = repo / "reports" / "tables" / "demo_summary.md"
        if artifact.is_file():
            show("artifact the Worker produced", artifact.read_text())
            print(c("  PASS  in-scope work was accepted and left an artifact", GREEN))
        else:
            print(c("  the artifact was not written", RED))

        state_1 = result.get("state_after", "?")
        say(
            f"State moved {result.get('state_before', '?')} -> {state_1}, and a run "
            "manifest was written under reports/status/swarm_runs/. That manifest "
            "is the durable evidence: what ran, what it touched, what the gates said. "
            "The task is not done — it is queued for review."
        )

        banner("Act 2 — a Worker that reaches out of scope")
        say(
            "T901's Worker decides the project contract is wrong and tries to "
            "rewrite contracts/project.yaml — a path its task explicitly does "
            "not own. This is the failure mode that ruins unsupervised agent runs."
        )
        before = (repo / "contracts" / "project.yaml").read_text()
        result = run_act(repo, "T901")
        show(
            "run-task result",
            json.dumps(result, indent=2) if result else "(no result payload)",
        )
        after = (repo / "contracts" / "project.yaml").read_text()

        if before == after and "executor_failed" in result.get("blocked_reasons", []):
            print(c("  BLOCKED  the out-of-scope write was refused", GREEN))
            print(c("           contracts/project.yaml is byte-identical", GREEN))
        else:
            print(c("  the guardrail did not hold — please report this", RED))

        show("contracts/project.yaml (first lines, unchanged)", "\n".join(after.splitlines()[:3]))
        say(
            "The task was moved to `blocked` rather than merged. Nothing silently "
            "widened. A human decides what happens next."
        )

        banner("Act 3 — a Judge asked to approve its own session's work")
        say(
            "T900 is waiting in `ready_for_review`. Only a Judge can mark work "
            "done. We now ask the Judge to review the work this very session "
            "just produced."
        )
        judge = swarm(repo, "judge-task", "--task-id", "T900")
        verdict: dict = {}
        try:
            verdict = json.loads(judge.stdout or "{}")
        except json.JSONDecodeError:
            print(judge.stdout or "", judge.stderr or "")
        show("judge-task result", json.dumps(verdict, indent=2) if verdict else "(no payload)")

        review_rel = verdict.get("review_log")
        failures: list = []
        if review_rel and (repo / review_rel).is_file():
            review = json.loads((repo / review_rel).read_text())
            checks = review.get("checks")
            if isinstance(checks, dict):
                failures = checks.get("failures") or []
        if verdict.get("approved") is False:
            print(c("  REFUSED  the Judge did not approve the work, and said why", GREEN))
            for failure in failures:
                print(c(f"           - {failure}", GREEN))
        else:
            print(c("  the Judge approved — expected a refusal here", RED))
        say(
            "The Judge reruns the declared gates itself and checks the provenance "
            "chain: whether the reviewer is a different actor from the author, "
            "whether commits landed after the run manifest was sealed, whether "
            "the review bundle is complete. An agent cannot rubber-stamp itself. "
            "The refusal is written to a review log under reports/status/reviews/ "
            "instead of being lost in a chat scroll."
        )

        banner("What you just saw")
        for line in [
            "1. A task file — not a chat message — defined the work and its limits.",
            "2. The runtime claimed the task and ran a Worker against that contract.",
            "3. Declared gates decided whether the work counted.",
            "4. An out-of-scope write was refused, not merged.",
            "5. An agent was stopped from approving its own work.",
            "6. Every step left a durable artifact in the repository.",
        ]:
            print(f"  {line}")
        print()
        say(
            "Swap `--executor-backend mock` for `codex` (or run the Claude "
            "planner) and the same loop drives real AI agents in isolated git "
            "worktrees. Read the README section 'Run real agents' before you do."
        )
        print()
        if args.keep:
            print(f"Demo repository kept at: {repo}")
        return 0
    finally:
        if not args.keep:
            shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
