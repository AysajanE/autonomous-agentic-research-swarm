# Swarm Automation Runbook (tmux + unattended CLIs)

This runbook describes how to run the **supervisor loop** that picks ready tasks from
`.orchestrator/backlog/`, launches isolated worktrees, runs **Codex as Worker + Judge**, and (optionally)
opens GitHub PRs — without interactive permission prompts.

**Safety warning (non-negotiable):** unattended mode disables approval prompts. Only run this in an
external sandbox (VM/devcontainer/Codespaces) containing *only* this repo and no sensitive files.
See `AGENTS.md`.

**Governance (Option A):** the automation updates task `State:` and appends notes, but it does **not**
move task files between lifecycle folders. Use `python scripts/sweep_tasks.py` as the Planner sweep
step to align folders with `State:`.

## Prereqs

- `tmux` installed (recommended for overnight runs).
- CLIs installed and logged in:
  - `codex` (OpenAI Codex CLI)
  - `claude` (Claude Code) if you want planner selection via Claude
  - `gh` (GitHub CLI) if you want auto-PR creation
- Git identity configured for commits:
  - `git config --global user.name "..."` and `git config --global user.email "..."`

## One-time sanity check

Run in repo root:

```bash
make gate
make test
python scripts/swarm.py plan
```

## Start an unattended supervisor loop (recommended)

This starts a tmux session and launches a `supervisor` window that ticks every 5 minutes.

```bash
export SWARM_UNATTENDED_I_UNDERSTAND=1

python scripts/swarm.py tmux-start \
  --tmux-session swarm \
  --planner claude \
  --max-workers 2 \
  --interval-seconds 300 \
  --unattended \
  --create-pr
```

Notes:
- `--unattended` disables approval prompts (Codex uses `-a never`; Claude uses `--permission-mode bypassPermissions`).
- For ETL tasks (W1/W2), the runner enables Codex network access inside the workspace-write sandbox
  via a config override.
- Consider bounding individual worker runs with `--max-worker-seconds` (prevents runaway sessions).

## Planner sweep (move task files by State)

After merging PRs (or when updating lifecycle folders in your branch), run:

```bash
make sweep
```

Dry-run:

```bash
python scripts/sweep_tasks.py --dry-run
```

## Monitor

Attach to the session:

```bash
tmux attach -t swarm
```

List open PRs:

```bash
gh pr list
```

## Stop

```bash
tmux kill-session -t swarm
```

## Manual single-tick (no loop)

Start at most one ready task (spawns a tmux window per task by default):

```bash
python scripts/swarm.py tick --planner claude --max-workers 1 --unattended --create-pr
```

Or run locally (sequential, no tmux):

```bash
python scripts/swarm.py tick --runner local --planner heuristic --max-workers 1 --unattended
```
