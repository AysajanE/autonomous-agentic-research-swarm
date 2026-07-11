# Swarm Automation Runbook (default local execution path)

This is the Operator-facing guide for the default local swarm layer. Use it for routine repo delivery. For high-stakes synthesis work, use the reviewed staged-workflow-runner path instead.

## What automation owns

- reads `.orchestrator/` as the control plane
- selects ready tasks whose dependencies are satisfied
- creates isolated worktrees and launches role prompts
- writes durable run manifests to `reports/status/swarm_runs/`
- leaves `State:` authoritative in the task file and uses sweep for folder projection
- keeps verbose runtime logs under `data/tmp/swarm_logs/`

## Preflight every run

1. Sync the base branch:
   - `git checkout main`
   - `git pull --ff-only origin main`
   - replace `main` with your synchronized base branch if different
2. Run offline gates:
   - `make gate`
   - `make test`
3. Verify required tools:
   - always: `python`, `git`
   - optional for PR automation: `gh`
   - required for paper or release tasks: `quarto`
   - optional for parallel windows: `tmux`
4. Confirm git identity:
   - `git config --get user.name`
   - `git config --get user.email`
5. Confirm the repo is in a sandbox if unattended execution will be used.

## Task selection invariants

- `State:` inside the task file is authoritative.
- `dependencies` are satisfied only by `done`, except explicit `integration_ready_dependencies`.
- `integration_ready` is only for interface/export tasks in the allowlisted categories.
- `allow_network: true` is valid only for workstreams allowlisted by the active `contracts/pack.json` workflow configuration.
- Worker tasks do not own `reports/catalog.yaml`, paper build outputs, or release manifests.

## Start with a dry run

- `python scripts/swarm.py plan`
- `python scripts/swarm.py tick --planner heuristic --runner local --max-workers 1 --dry-run`

The dry run should show which tasks are `done`, `claimed`, and `ready` without creating worktrees or mutating task state.

## Start the default loop

For a simple local loop:

- `python scripts/swarm.py loop --planner heuristic --runner local --max-workers 1 --final-state ready_for_review`

If GitHub automation is configured, you may add `--create-pr`.

For tmux-supervised parallelism:

- `python scripts/swarm.py tmux-start --tmux-session swarm --planner heuristic --max-workers 2 --interval-seconds 300 --final-state ready_for_review --attach`

Keep `final-state` at `ready_for_review` until the runtime, gates, and prompts are trusted. Judge still owns `done`.

## Runtime expectations

- one task per branch/worktree
- one durable run manifest per artifact-producing task run
- one Judge review log per review event
- no network inside `make gate` or `make test`
- no `reports/catalog.yaml` regeneration outside the Operator release path
- no release assembly before upstream validation, figures/tables, and manuscript source exist

## Manual fallback

If automation is unavailable or a task needs a short repair:

- create the worktree manually
- run the role prompt manually
- keep the same task, manifest, and review-bundle rules
- if the run bypasses `scripts/swarm.py`, the Operator must still ensure a durable run manifest exists before `ready_for_review`

## Release candidate flow

After `T025` through `T070` are `done`:

1. Operator runs `python scripts/release_assembly.py --release-date YYYY-MM-DD --check`
2. Operator runs `quarto render reports/paper/index.qmd`
3. Operator confirms:
   - `reports/catalog.yaml`
   - `reports/paper/build/l2_l1_rent_working_paper.html`
   - `reports/paper/build/l2_l1_rent_working_paper.pdf`
   - `reports/paper/build/render_manifest.json`
   - `reports/status/releases/release_<YYYY-MM-DD>.json`
4. Judge performs the final release review

## Optional repair handling

Bounded repair passes are Operator-owned. If implemented in `scripts/swarm.py`, they must:

- target only failing or merge-conflicted task branches
- preserve task isolation and path ownership
- write new run manifests for repair attempts
- never bypass Judge review

## When not to use this path

Do not force high-stakes architecture rewrites, major replans, or release assessments through the ordinary Worker queue. Those belong on the reviewed staged-workflow-runner path under Operator ownership.

## Common failure modes

- `plan` shows nothing ready: inspect dependencies, stale claims, or missing manifests
- path ownership violation: revert out-of-scope edits and rerun gates
- gates failed: fix on the task branch and rerun the declared commands
- Quarto missing: do not advance `T070` or `T080`; block with the missing tool explicitly
