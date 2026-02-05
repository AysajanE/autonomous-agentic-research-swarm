# `.orchestrator/` — file-based coordination

This directory is the **single source of truth** for multi-agent coordination in this repo.

## How work flows

1. A **Planner** (human or agent) creates task files in `.orchestrator/backlog/`.
2. A **Worker** executes one assigned task (ideally in an isolated branch/worktree) and updates only:
   - `## Status`
   - `## Notes / Decisions`
3. The Planner periodically **sweeps** task files and moves them between:
   - `backlog/ → active/ → ready_for_review/ → done/` (happy path)
   - `backlog/ → active/ → integration_ready/` (optional integration checkpoint)
   - `backlog/ → active/ → blocked/` (blocked path)
   based on the task’s `State:`.

## State semantics

- `ready_for_review` means: **declared outputs exist** and **gates pass**.
- `integration_ready` (optional) means: **interfaces/contracts are exported** and downstream tasks may proceed (dependencies may treat this as “done enough”).

## Rules (mirrors `AGENTS.md`)

- Each task is a **single Markdown file**.
- **Only the Planner moves task files** across lifecycle folders.
- Workers update **only**:
  - `## Status`
  - `## Notes / Decisions`
- Cross-task handoffs go in `.orchestrator/handoff/` (short, durable notes).
- Prefer small, verifiable changes; run quality gates before calling a task done.

## Control plane mode (recommended): PR-synchronized

Workers typically operate in separate branches/worktrees, so task status updates are not expected
to be “real-time.” Status becomes visible when branches are pushed/PR’d (or when the Planner pulls).

## Contents

- `backlog/` — tasks not yet started
- `active/` — tasks currently being worked
- `integration_ready/` — optional integration checkpoint (interfaces exported; downstream unblocked)
- `ready_for_review/` — tasks awaiting Judge verification
- `blocked/` — blocked tasks (must include blocker note)
- `done/` — completed tasks
- `handoff/` — cross-task notes and integration hints
- `templates/` — task + handoff templates
- `workstreams.md` — project workstream definitions + ownership boundaries

## Templates

- Generic task: `.orchestrator/templates/task_template.md`
- W0 protocol/contracts tasks: `.orchestrator/templates/task_template_w0_protocol.md`
- W1/W2 ETL tasks: `.orchestrator/templates/task_template_w1_w2_etl.md`
- W8 modeling tasks: `.orchestrator/templates/task_template_w8_modeling.md`
- Hybrid bridge tasks: `.orchestrator/templates/task_template_hybrid_bridge.md`
