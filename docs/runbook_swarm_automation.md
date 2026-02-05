# Swarm Automation Runbook (press‑go guide)

This runbook is the **human-facing** step-by-step guide for starting and operating the automated research swarm in this repo.

It is designed for: **“I’m in a devcontainer / VM, I want to kickstart the swarm, and I don’t want surprises.”**

---

## What this automation actually does (read this once)

### Control plane vs automation plane

- **Control plane (Planner hygiene):**
  - Task files live under `.orchestrator/`.
  - Tasks have a `State:` field inside the file:
    `backlog | active | blocked | ready_for_review | done`.
  - **Only the Planner** is supposed to move task files between lifecycle folders.
  - The tool for that is: `python scripts/sweep_tasks.py` (also `make sweep`).

- **Automation plane (Worker + Judge):**
  - The supervisor is `python scripts/swarm.py`.
  - It selects “ready” tasks in `.orchestrator/backlog/` whose `State: backlog` **and** whose dependencies are already `State: done`.
  - For each selected task it:
    1) creates a **git worktree** in a sibling directory (default: `../wt-T###`)
    2) runs **Codex CLI** as Worker to implement the task
    3) runs **gates** (usually `make gate`, sometimes `make test`) as Judge
    4) enforces **path ownership** (only allowed_paths, no forbidden `.orchestrator/` edits except the task file + handoff notes)
    5) updates the task file’s `State:` and appends a note under `## Notes / Decisions`
    6) commits + pushes the branch
    7) optionally opens a PR and optionally requests auto-merge

### Important governance design (Option A — this repo’s rule)

This repo intentionally separates:
- **state updates** (automation does this),
from
- **moving task files between folders** (Planner does this).

Concretely:
- `scripts/swarm.py` updates the task file’s `State:` and notes, but **does not** `git mv` tasks between folders.
- `scripts/sweep_tasks.py` reads each task’s `State:` and **does** `git mv` the file into the matching folder.

Dependencies are satisfied when a dependency task has `State: done` **anywhere** (folder location does not affect dependency resolution).

---

## Safety (non‑negotiable)

Unattended mode disables approval prompts and will run code-changing actions automatically.

**Only run unattended mode in a sandbox** (devcontainer/VM/Codespaces) that contains:
- only this repo,
- no personal files,
- no production credentials,
- no sensitive environment variables.

Unattended mode has a safety interlock:
- you must set `SWARM_UNATTENDED_I_UNDERSTAND=1`
- otherwise `scripts/swarm.py` refuses to run with `--unattended`.

---

## The 10‑minute preflight (do this every time before a swarm run)

Run all commands from the **repo root**.

### 1) Update your base branch (avoid “stale main” problems)

```bash
git checkout main
git pull --ff-only origin main
````

If your repo default branch is not `main`, use that name everywhere below (`--base-branch`).

### 2) Run deterministic repo gates

```bash
make gate
make test
```

If `make gate` fails, do **not** start the swarm. Fix the gate failures first.

### 3) Verify required CLIs exist *in this environment*

The swarm defaults to `runner=tmux` for parallelism. If you don’t have tmux, use `--runner local` (covered below).

```bash
command -v python
command -v git

command -v codex
command -v gh

# optional (only needed if you want planner=claude)
command -v claude || true

# only needed if you use runner=tmux
command -v tmux || true
```

### 4) Verify Git identity (crucial: otherwise commits may silently fail)

`swarm.py` uses `git commit ...` with `check=False` in a couple spots, meaning it will not crash loudly if git identity is missing. That’s a feature (doesn’t kill the run), but it’s a **footgun** (you think you pushed work, but you didn’t).

Set repo-local identity:

```bash
git config user.name  "swarm-bot"
git config user.email "swarm-bot@users.noreply.github.com"

git config --get user.name
git config --get user.email
```

### 5) Verify GitHub auth (only if you plan to use `--create-pr` / `--auto-merge`)

```bash
gh auth status
gh repo view >/dev/null
```

Also verify you can push branches:

```bash
git push --dry-run origin HEAD
```

If `gh auth status` fails, fix it before running the swarm:

```bash
# interactive login
gh auth login
```

### 6) Verify Codex auth works (fast smoke test)

```bash
codex --version
```

Optional: run a tiny no-op prompt (should return quickly without doing anything harmful):

```bash
codex -a on-request exec --sandbox read-only -C . "Say 'codex_ok' and stop."
```

### 7) Confirm there are ready tasks

This prints a JSON object with:

* `done`: tasks whose `State: done`
* `claimed`: tasks already “claimed” by branches / PRs
* `ready`: backlog tasks whose dependencies are satisfied and not claimed

```bash
python scripts/swarm.py plan
```

If `ready` is empty, see **Troubleshooting → “No ready tasks”** below.

---

## Kickstart paths (choose one)

### Recommended “new team member” path (safe + low-friction)

This path assumes you want automation to open PRs, but you (a human) will review and merge.

**Key defaults:**

* no unattended mode
* `final-state=ready_for_review`
* `--create-pr` enabled
* no auto-merge
* `planner=heuristic` (deterministic; avoids Claude CLI brittleness)
* `runner=tmux` if available, otherwise `runner=local`

#### Step A — do a dry run (verifies selection logic without creating worktrees)

```bash
python scripts/swarm.py tick \
  --planner heuristic \
  --runner local \
  --max-workers 2 \
  --dry-run
```

You should see messages like:

* `[dry-run] would start T020: ...`

If you see errors here, fix them before continuing.

#### Step B — start the supervisor loop in tmux (recommended if tmux exists)

```bash
python scripts/swarm.py tmux-start \
  --tmux-session swarm \
  --planner heuristic \
  --max-workers 2 \
  --interval-seconds 300 \
  --create-pr \
  --final-state ready_for_review \
  --max-worker-seconds 1800 \
  --max-review-seconds 300 \
  --attach
```

What you should see next:

* tmux session `swarm` with a `supervisor` window
* additional windows created for each started task (named like `T020`, `T030`, etc.)
* each task window runs: `python scripts/swarm.py run-task --task-id ...`

If you do not have tmux, use the **runner=local loop** below.

#### Step B (no tmux) — run the loop sequentially in the current terminal

This won’t run tasks in parallel, but it’s simpler and works anywhere:

```bash
python scripts/swarm.py loop \
  --planner heuristic \
  --runner local \
  --max-workers 1 \
  --interval-seconds 300 \
  --create-pr \
  --final-state ready_for_review \
  --max-worker-seconds 1800 \
  --max-review-seconds 300
```

Stop with `Ctrl-C`.

#### Step C — review PRs and merge

```bash
gh pr list
```

When you merge PRs, new dependencies become satisfied and future ticks will pick up more tasks.

#### Step D — optional folder hygiene: run a sweep (manual)

Folder location is not required for dependency logic, but it’s useful for humans.

```bash
make sweep
git status
```

If it moved files:

```bash
git add -A
git commit -m "Planner sweep: align task folders with State"
git push origin main
```

If branch protection blocks pushes to `main`, use the PR-based sweep workflow below.

---

## Fully unattended (overnight) mode

This is “zero babysitting,” but it only works if your GitHub repo settings allow it.

### Hard requirements for true unattended merging

Almost always, you need:

* GitHub auto-merge enabled in repo settings
* no required human approvals that block merges
* required status checks that are satisfiable automatically
* an auth identity that can create PRs and request auto-merge

If you have CODEOWNERS-required reviews, you will not get true unattended merges without additional policy work.

### Step 1 — set the unattended interlock

```bash
export SWARM_UNATTENDED_I_UNDERSTAND=1
```

If you use tmux, the supervisor also propagates this into tmux automatically, but exporting it is still best practice.

### Step 2 — start unattended supervisor in tmux

**Important:** start with `final-state=ready_for_review` the first time you try unattended mode.
After you trust it, switch to `final-state=done`.

```bash
python scripts/swarm.py tmux-start \
  --tmux-session swarm \
  --planner heuristic \
  --max-workers 2 \
  --interval-seconds 300 \
  --unattended \
  --create-pr \
  --auto-merge \
  --final-state ready_for_review \
  --max-worker-seconds 1800 \
  --max-review-seconds 300 \
  --attach
```

Notes:

* `--unattended` makes Codex run with `-a never`.
* Tasks with frontmatter `allow_network: true` will get **network access enabled** inside the Codex workspace-write sandbox, but only if their workstream is allowlisted in `contracts/framework.json` (`network_workstreams`; default: `["W1","W2"]`).
* The supervisor loop hard-syncs its checkout to `origin/<base-branch>` each tick (to avoid “local main drift”).
* Each `run-task` writes a run manifest JSON under `reports/status/swarm_runs/` for audit/provenance.

### Step 3 — (optional) enable automated repair passes

By default, unattended mode will attempt bounded repair passes for open PRs that are:

* failing checks OR merge-conflicting
* and haven’t been updated recently (default 4 hours)

Defaults:

* `--repair-after-seconds 14400`
* `--max-repairs-per-tick 1`

To disable repair:

```bash
# in tmux-start add:
--max-repairs-per-tick 0
```

---

## Optional: Planner sweep loop (keeps lifecycle folders aligned)

Remember:

* dependencies are based on `State: done`, not folder
* sweeps are “hygiene for humans”

You have two options:

### Option 1: direct-push sweeps to `main` (only if branch protection allows)

In an additional tmux window:

```bash
tmux new-window -t swarm -n planner

tmux send-keys -t swarm:planner 'bash -lc "set -euo pipefail
while true; do
  git fetch origin
  git checkout -f main
  git reset --hard origin/main

  make sweep

  if ! git diff --quiet; then
    git add -A
    git commit -m \"Planner sweep: align task folders with State\"
    git push origin main
  fi

  sleep 60
done"' C-m
```

### Option 2: PR-based sweeps (works with branch protection; noisier but robust)

This creates a PR when there are sweep moves.

```bash
tmux new-window -t swarm -n planner

tmux send-keys -t swarm:planner 'bash -lc "set -euo pipefail
while true; do
  git fetch origin
  git checkout -f main
  git reset --hard origin/main

  make sweep

  if ! git diff --quiet; then
    BRANCH=planner-sweep-$(date -u +%Y%m%dT%H%M%SZ)
    git checkout -b $BRANCH
    git add -A
    git commit -m \"Planner sweep: align task folders with State\"
    git push -u origin $BRANCH

    gh pr create \
      --base main \
      --title \"Planner sweep\" \
      --body \"Automated sweep: move task files into lifecycle folders matching their State.\" || true

    # Optional if your repo supports it:
    gh pr merge --auto --squash --delete-branch || true
  fi

  sleep 300
done"' C-m
```

---

## Monitoring (what “healthy” looks like)

### 1) tmux

```bash
tmux attach -t swarm
```

Expected windows:

* `supervisor` (the loop)
* one window per active task (e.g., `T020`, `T030`, ...)
* optional `planner` window if you started it

### 2) PR queue

```bash
gh pr list
```

### 3) Logs (local, untracked)

* Worker “last message” and Judge review logs are written under:

  * `data/tmp/swarm_logs/` (gitignored)

If a task is blocked, the task file’s notes will include the review log path.

### 4) Worktrees

Worktrees are created by default as siblings to the repo (or under `--worktree-parent` if set).

List them:

```bash
git worktree list
```

---

## Recovery / Unblock playbook (most common “it doesn’t work” cases)

### Case A — “No ready tasks”

Run:

```bash
python scripts/swarm.py plan
```

Then:

1. Check backlog tasks exist:

   * `.orchestrator/backlog/*.md` (not README)

2. Check task states:

   * ready tasks must be in `.orchestrator/backlog/` and have `State: backlog`

3. Check dependencies:

   * a dependency is satisfied only when that dependency task has `State: done`
   * folder location doesn’t matter

4. Check “claimed”:

   * tasks are considered claimed if:

     * there’s an open PR whose head branch starts with `T###`
     * OR there is a remote branch `T###_*`
     * OR there’s a local worktree attached to a branch starting with `T###`

If a task is “claimed” but you want to rerun it, you likely need to close/delete the stale branch/PR.

### Case B — Task ends in `blocked` with `path_ownership_violation`

This means the Worker modified files outside the task’s `allowed_paths` or touched forbidden `.orchestrator/` files.

Where to look:

* the JSON printed at the end of `run-task` (it includes `ownership_failures`)
* the task file’s `## Notes / Decisions`

What to do:

1. Open the task branch worktree (e.g., `../wt-T030`)
2. `git status` → see what changed
3. revert forbidden changes, keep only allowed-path changes
4. rerun gates:

   ```bash
   make gate
   make test   # if the task declares it
   ```
5. commit + push, and re-run the task window if needed

### Case C — Task ends in `blocked` with `gates_failed`

The declared gates in the task frontmatter failed (typically `make gate` or `make test`).

Where to look:

* the task window output in tmux
* the PR checks (`gh pr checks <pr-url>`)

Fix in the task branch worktree, run the gates locally, then push.

### Case D — PR creation fails and kills the task window

`swarm.py run-task` calls `gh pr create` with `check=True`. If `gh` exists but isn’t authenticated or lacks permission, the task run will crash.

Fix:

```bash
gh auth status
gh auth login
```

Then rerun the task window / rerun `run-task` in that worktree.

### Case E — Worktree already exists (`Worktree path already exists: ../wt-T###`)

This happens if a previous run created the worktree and you didn’t remove it.

Fix:

```bash
git worktree list
git worktree remove ../wt-T020  # example
rm -rf ../wt-T020               # if needed
```

### Case F — “Claude planner” causes crashes

If you run with `--planner claude`:

* If `claude` is not installed, swarm falls back to heuristic.
* If `claude` IS installed but fails (auth/model mismatch), `tick` can crash.

Therefore:

* use `--planner heuristic` unless you’ve explicitly tested your Claude CLI.

Minimal test:

```bash
claude -p "Return JSON: {\"ok\": true}" --output-format json
```

---

## Cleanup (end of run)

### Stop the swarm

If using tmux:

```bash
tmux kill-session -t swarm
```

### Remove worktrees (optional but recommended)

```bash
git worktree list
# then remove the ones you created
git worktree remove ../wt-T020
git worktree remove ../wt-T030
```

### Delete stale remote branches (if needed)

If a task is stuck “claimed” by a dead branch:

```bash
git push origin --delete T020_some_branch_name
```

---

## Practical defaults (what we recommend as a team policy)

* Default to **heuristic planner** (deterministic; fewer moving parts).
* Default to **create PRs** (`--create-pr`), even in unattended mode, so:

  * other runs can see tasks are claimed
  * repair loops can find failing PRs
* Start with `final-state=ready_for_review` and no auto-merge.
* Turn on `--auto-merge` only after you trust:

  * your tasks are well-scoped
  * your branch protection settings won’t deadlock the run
* Keep `codex` auto-commit features OFF (e.g., `ghost_commit=false`) so swarm’s ownership checks (based on `git status`) remain meaningful.

---

## Reference: key commands

* Show ready tasks:

  * `python scripts/swarm.py plan`

* Start one tick (spawns worktrees + runs tasks once):

  * `python scripts/swarm.py tick --max-workers 2 --runner tmux --create-pr`

* Start continuous loop in tmux:

  * `python scripts/swarm.py tmux-start --tmux-session swarm ...`

* Move task files to match their `State:`:

  * `make sweep` (Planner action)

* Fast repo gates:

  * `make gate`
  * `make test`

---

## Diff summary (what I changed vs your current runbook)

- Made the runbook **operationally explicit** about what `swarm.py` *does vs does not do* (especially the Option A “State updates vs folder moves”).
- Added a **10‑minute preflight** that covers the real killers:
  - stale `main`, missing git identity, missing GH auth, tmux absence, and Codex auth.
- Added **three kickstart paths**:
  - safe PR-based human merge path (recommended),
  - non-tmux local loop (for devcontainers where tmux is missing),
  - fully unattended (with warnings and correct guardrails).
- Added **recovery playbooks** tied to the actual failure modes emitted by `swarm.py`:
  - `path_ownership_violation`, `gates_failed`, PR creation crash, stale worktrees, “claimed” tasks.
- Rewrote sweep loop snippets to be **copy/paste stable** (less brittle quoting), and added **PR-based sweep** as a first-class option.

If you want, paste the output of `python scripts/swarm.py plan` and (if you tried to run it) the last ~50 lines of the `supervisor` tmux window; I can pinpoint exactly where your current kickoff is breaking.
