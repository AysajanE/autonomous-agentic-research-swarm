# Swarm Runbook (manual v1)

Use this runbook for normal repo task delivery. For architecture rewrites, major replans, and release assessments, use the reviewed staged-workflow-runner path instead of treating the work as an ordinary Worker task.

## Preflight

- Work inside a sandboxed environment that contains only this repo.
- Sync the base branch before starting.
- Run `make gate` and `make test`.
- Review `docs/protocol.md`, `contracts/project.yaml`, and `contracts/framework.json`.
- Verify required tools:
  - always: `git`, `python`
  - when paper or release tasks are in scope: `quarto`

## 1) Planner scopes the queue

- Create or update task files using the templates under `.orchestrator/templates/`.
- Keep `allowed_paths` narrow and keep dependencies aligned with the locked artifact DAG.
- Use `integration_ready` only for interface/export tasks that truly need early downstream consumption.

## 2) Operator prepares execution

- Verify git identity and, if needed, GitHub auth.
- Create one worktree per active task.
- Decide whether the run will be manual or via `scripts/swarm.py`.
- Keep verbose runtime logs in `data/tmp/swarm_logs/`.

Suggested worktree pattern:

    TASK_ID=T040
    git worktree add ../wt-${TASK_ID} -b ${TASK_ID}_short_name .

## 3) Worker executes exactly one task

- Run from the task worktree.
- Edit only allowed repo paths plus task `## Status` and `## Notes / Decisions`.
- Record reproduction commands, assumptions, and blockers.
- Stop instead of improvising on protocol or contract ambiguity.
- Write a handoff note when downstream tasks need durable guidance.

## 4) Judge reviews

- Rerun the declared gates.
- Verify outputs, raw and processed manifests, and the task success criteria.
- Confirm the review bundle:
  - task markdown
  - run manifest under `reports/status/swarm_runs/`
  - review log under `reports/status/reviews/`
  - handoff note if needed
- Set `State: done` only when the task is scientifically acceptable.

## 5) Planner or Operator sweeps lifecycle folders

- Run `python scripts/sweep_tasks.py`.
- Remember that folder placement is a projection; `State:` is authoritative.

## 6) Current empirical battle-test order

1. `T025` registry
2. `T030` growthepie vendor panel
3. `T035` on-chain L1 rent and canonical panel
4. `T040` metrics and tests
5. `T050` validation bundle
6. `T060` release figures and tables
7. `T070` Quarto manuscript source
8. `T080` Operator release assembly

## 7) Release closeout

After `T070` is done:

- Operator runs `python scripts/release_assembly.py --release-date YYYY-MM-DD --check`
- Operator runs `quarto render reports/paper/index.qmd`
- Judge verifies `reports/catalog.yaml`, paper build outputs, `render_manifest.json`, and `reports/status/releases/release_<YYYY-MM-DD>.json`

## Safety defaults

- Keep unattended execution inside sandboxed environments only.
- Prefer short runs and fresh sessions after merges or repeated gate failures.
- Do not bypass validation or review to unblock analysis or writing work.
