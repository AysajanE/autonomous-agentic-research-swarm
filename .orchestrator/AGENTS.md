# .orchestrator/AGENTS.md — Control Plane Rules

`.orchestrator/` is the file-based coordination layer for routine repo delivery.

## Who may change what

- **Planner only**
  - create, split, reorder, reopen, pause, or retire task files
  - edit `.orchestrator/workstreams.md`
  - edit anything under `.orchestrator/templates/`
  - move task files across `.orchestrator/{backlog,active,integration_ready,ready_for_review,blocked,done}/`

- **Operator**
  - run sweeps and reconcile folder placement to authoritative `State:`
  - update `## Status` and `## Notes / Decisions` on assigned Operator tasks
  - set `State: active` or `State: blocked` on operational grounds
  - set `State: ready_for_review` on Operator-owned tasks after declared outputs exist and declared gates pass
  - may not edit templates/workstreams or mark scientific work `done`

- **Worker**
  - edit only the assigned task file's `## Status` and `## Notes / Decisions`
  - write only within task `allowed_paths`
  - add a new note under `.orchestrator/handoff/`
  - may set `active`, `integration_ready`, `ready_for_review`, or `blocked`; never `done`

- **Judge**
  - rerun declared gates and verify outputs, manifests, and review-bundle completeness
  - edit task `## Status` and `## Notes / Decisions` during review
  - may set `done`, `active`, or `blocked`
  - does not move files between lifecycle folders

## Status discipline

- Always update `Last updated` in UTC date format (`YYYY-MM-DD`).
- Valid states are `backlog`, `active`, `integration_ready`, `ready_for_review`, `blocked`, and `done`.
- `integration_ready` is only for interface/export tasks that are explicitly allowed by downstream `integration_ready_dependencies`.
- `ready_for_review` means outputs exist, declared gates pass, required manifests exist, and a durable run manifest exists under `reports/status/swarm_runs/`.
- `blocked` must include the smallest actionable blocker and `@human` when human judgment is required.
- `done` requires Judge approval and a completed review bundle.

## Projection rules

- `State:` inside the task file is authoritative.
- Folder placement is a projection maintained by Planner and Operator.
- Worker and Judge runs must not `git mv` task files between lifecycle folders.

## History discipline

- Do not rewrite task history.
- Append notes; do not replace prior notes.
- Keep cross-task guidance in `.orchestrator/handoff/`, not inside unrelated task files.
