# Swarm Automation Runbook (tmux + unattended CLIs)

This runbook is the **human-facing** “press go” guide for starting a fully autonomous swarm run:

- **Planner (control plane):** sweeps/moves tasks between lifecycle folders based on `State:`.
- **Worker + Judge (automation plane):** selects ready tasks, creates worktrees, runs Codex, runs gates, opens PRs, and (optionally) auto-merges.

The repo is the shared memory. Tasks live under `.orchestrator/`.

## Safety (read first; non‑negotiable)

Unattended mode disables approval prompts for agent CLIs.

Only run this in an **external sandbox** (Codespaces/VM/devcontainer) that contains **only this repo**
and **no secrets**. If you can’t say that confidently, do not run unattended mode.

Unattended mode is also guarded by a safety interlock:
- You must set `SWARM_UNATTENDED_I_UNDERSTAND=1` or the supervisor will refuse to run.

## Governance model (Option A — required for this repo)

This repo intentionally separates “state updates” from “moving task files”:

- `scripts/swarm.py` **does not** move task files between lifecycle folders.
  - It only updates task `State:` + appends to `## Notes / Decisions` in the task file.
- `scripts/sweep_tasks.py` **does** move task files (Planner action):
  - It reads each task’s `State:` and `git mv`s the file into the matching folder.

Why this matters for autonomy:
- Dependencies are considered satisfied when the dependency task has `State: done`
  (folder location does not affect dependency resolution).
- Therefore, the Planner sweep loop is **recommended for hygiene**, but not strictly required for dependency progress.

## Concurrency model (what the supervisor will/ won’t do)

To reduce collisions:
- The supervisor starts **at most one task per workstream at a time** (unless a task sets `parallel_ok: true` in frontmatter).
- Tasks should use **least‑privilege** `allowed_paths` (specific files/prefixes), not broad directories.

## Choose your autonomy level (pick one up front)

### A) Fully unattended (zero babysitting)

Requires **repo settings** that allow the automation to progress without humans:
- GitHub auto-merge enabled.
- No required human reviews (including CODEOWNERS) for the target branch.
- Whatever status checks are required for merge must be satisfiable automatically.
- Optional (hygiene): allow a Planner sweep loop to push lifecycle-folder moves to `main` (or implement a PR-based sweep policy).

### B) Unattended workers, human merges/sweeps

Use this if branch protection blocks direct pushes to `main`:
- Supervisor runs and opens PRs.
- Humans merge PRs and optionally run `make sweep` for folder hygiene.
- This is safer, but not “true unattended”.

This runbook documents both; the commands below default to (A).

## 0) One-time sandbox prerequisites (copy/paste checklist)

Run in a clean sandbox checkout of this repo:

```bash
git checkout main
git pull --ff-only origin main

make gate
make test

# hard requirements
command -v tmux
command -v codex
command -v gh

# optional (Planner selection via Claude)
command -v claude || true

gh auth status
codex --version
python --version
```

### Git identity (repo-local; recommended)

Automation commits (task state updates, and sweep commits). Configure identity **for this repo**:

```bash
git config user.name  "Aysajan Eziz"
git config user.email "aysajan1986@gmail.com"
git config --get user.name
git config --get user.email
```

Reason this is preferred over `--global` in a sandbox:
- It avoids modifying the sandbox’s global git identity (which might be shared across unrelated repos).

Alternative (recommended for team repos / longer runs):
- Use a dedicated bot identity (e.g., `swarm-bot` with a GitHub noreply email) so automation commits are clearly attributable.

### Codex configuration (recommended for swarm)

Create or update `~/.codex/config.toml` with the following recommended settings:

```toml
model = "gpt-5.2-codex"
model_reasoning_effort = "xhigh"  # Recommended for swarm automation
tool_output_token_limit = 25000
model_auto_compact_token_limit = 233000

[features]
ghost_commit = false
unified_exec = true
apply_patch_freeform = true
web_search_request = true
skills = true
shell_snapshot = true

[projects."/path/to/your/repo"]
trust_level = "trusted"
```

Important notes:
- `model_reasoning_effort = "xhigh"` is recommended for swarm automation to improve task execution quality
- Update the `[projects."/path/to/your/repo"]` section to match your actual repository path
- See the [Codex config reference](https://developers.openai.com/codex/config-reference) for more details

### Confirm repo is ready to start tasks

This prints the set of "ready" backlog tasks (dependencies satisfied, not already claimed):

```bash
python scripts/swarm.py plan
```

If `ready` is empty, inspect `.orchestrator/backlog/` and `.orchestrator/done/`.

## 1) Start the supervisor loop (Worker + Judge automation)

This creates a tmux session and starts a `supervisor` window that ticks every 5 minutes:

```bash
export SWARM_UNATTENDED_I_UNDERSTAND=1
tmux set-environment -g SWARM_UNATTENDED_I_UNDERSTAND 1

python scripts/swarm.py tmux-start \
  --tmux-session swarm \
  --planner claude \
  --codex-model gpt-5.2 \
  --claude-model opus \
  --max-workers 2 \
  --interval-seconds 300 \
  --unattended \
  --create-pr \
  --auto-merge \
  --final-state done \
  --max-worker-seconds 1800 \
  --max-review-seconds 300
```

What happens next:
- The supervisor selects ready tasks, creates worktrees (`../wt-T###` by default), and opens one tmux window per task.
- Each task window runs `scripts/swarm.py run-task ...`:
  - Codex runs as **Worker**.
  - `make gate` / `make test` run as deterministic **Judge** (per task frontmatter).
  - The task file `State:` is updated to `done` or `blocked`.
  - A PR is opened, and auto-merge is requested (if enabled and allowed by repo policy).

Notes:
- `--unattended` disables approval prompts (Codex uses `-a never`; Claude uses `--permission-mode bypassPermissions`).
- For ETL tasks (W1/W2), the runner enables network access inside the Codex workspace-write sandbox via a config override.
- Timeouts are drift control: if a worker times out, the task stays `State: active` with a note.
- Supervisor freshness: in unattended mode the supervisor hard-syncs its checkout to `origin/<base-branch>` each tick. Don’t do manual work in the supervisor checkout; use separate worktrees/branches.
- Recovery loop: in unattended mode the supervisor will periodically re-run a bounded “repair” pass for open task PRs that are failing checks or merge-conflicting and haven’t changed recently.
  - Defaults: `--repair-after-seconds 14400` (4h), `--max-repairs-per-tick 1`
  - Disable: set `--max-repairs-per-tick 0`

## 2) Start the Planner sweep loop (recommended; keeps lifecycle folders aligned)

The sweep loop must:
- keep `main` in sync with `origin/main`,
- move task files based on `State:` (`make sweep`),
- commit and push the sweep if it changed anything,
- fail loudly if push/pull fails (silent failure = silent stall).

Run this in the **same tmux session**:

```bash
tmux new-window -t swarm -n planner 'bash -lc '"'"'
set -euo pipefail
while true; do
  git fetch origin
  git checkout -f main
  git reset --hard origin/main

  make sweep

  if ! git diff --quiet; then
    git add -A
    git commit -m "Planner sweep: align task folders with State"
    git push origin main
  fi

  sleep 60
done
'"'"''
```

Branch protection note (important):
- If direct pushes to `main` are blocked, this loop will fail at `git push origin main`.
- If you still want continuous folder hygiene, you must either:
  - allow direct pushes to `main` from this automation identity, or
  - change your policy to a PR-based sweep (more complex/noisy), or
  - run sweeps manually.

## 3) Monitor / stop

Monitor:
- `tmux attach -t swarm`
- `gh pr list`
- Look for logs under `data/tmp/swarm_logs/` (untracked; local only).

Stop:

```bash
tmux kill-session -t swarm
```

## Troubleshooting (common “stall” causes)

- Missing unattended interlock:
  - Symptom: swarm refuses to run with `--unattended`.
  - Fix: `export SWARM_UNATTENDED_I_UNDERSTAND=1` (and restart).
- Git identity missing:
  - Symptom: commits fail in task windows or planner sweep.
  - Fix: set repo-local `git config user.name/user.email` (above).
- `gh` auth expired:
  - Symptom: PR creation/merge fails.
  - Fix: `gh auth login` and restart.
- Branch protection blocks sweeps:
  - Symptom: planner window errors on push; lifecycle folders won’t update automatically.
  - Fix: allow direct pushes for the sweep identity, switch to PR-based sweep, or skip sweeps and rely on `State:` for dependency progress.
- Stale / leftover worktrees:
  - Symptom: `Worktree path already exists: ../wt-T###`.
  - Fix: `git worktree list`, then `git worktree remove <path>` and delete the directory if needed.
