# .orchestrator — file-based coordination

`.orchestrator/` is the only shared coordination layer for routine repo delivery. One task Markdown file is one unit of planning, execution, review, and audit.

## How work flows

1. Planner defines tasks in `.orchestrator/backlog/`, keeps dependencies explicit, and maintains `.orchestrator/workstreams.md`.
2. Operator performs environment preflight, starts or supervises the local swarm, and keeps folder projection aligned with authoritative task state.
3. Worker or Operator executes one assigned task in one isolated branch/worktree and updates only `## Status` and `## Notes / Decisions`.
4. Judge reruns declared gates, verifies outputs and provenance, and either returns the task to `active`/`blocked` or marks it `done`.
5. Planner or Operator sweeps task files into the lifecycle folder that matches each task's `State:`.

## Task file contract

### Required frontmatter

- `task_id`
- `title`
- `workstream`
- `role`
- `priority`
- `dependencies`
- `allowed_paths`
- `outputs`
- `gates`
- `stop_conditions`

### Optional frontmatter

- `task_kind`
- `allow_network`
- `integration_ready_dependencies`
- `requires_tools`
- `requires_env`
- `instances`
- `experiment_spec`

### Required body sections

- `## Context`
- `## Inputs`
- `## Outputs`
- `## Success Criteria`
- `## Validation / Commands`
- `## Status`
- `## Notes / Decisions`

`State:` inside `## Status` is the authoritative lifecycle field.

## Lifecycle semantics

| State | Meaning | Typical setter |
|---|---|---|
| `backlog` | Planned work that is not yet claimed | Planner |
| `active` | Claimed work in progress in one isolated branch/worktree | Planner, Worker, Operator |
| `integration_ready` | Interface/export outputs exist and only explicitly allowlisted downstream tasks may consume them | Worker or Operator for eligible tasks |
| `ready_for_review` | Outputs exist, declared gates pass, required manifests exist, and a run manifest exists | Worker or Operator |
| `blocked` | The smallest actionable blocker is recorded | Planner, Worker, Operator, Judge |
| `done` | Judge-approved completion with a full review bundle | Judge |

## Dependency semantics

- `dependencies` are satisfied only by upstream tasks in `done`.
- The only exception is `integration_ready`: a downstream task may consume an upstream task in `integration_ready` only when the downstream task explicitly lists that task in `integration_ready_dependencies`.
- `integration_ready` is intended for W0, W3, W8, W9, and explicit bridge/interface tasks. It is not the default path for unvalidated empirical data products.

## Review bundle

A task may not reach `done` without the logical review bundle:

- the task Markdown file
- a matching run manifest under `reports/status/swarm_runs/`
- a matching Judge review log under `reports/status/reviews/`
- a handoff note under `.orchestrator/handoff/` when downstream guidance is needed

## Directory meanings

- `backlog/` — planned tasks not yet started
- `active/` — tasks in progress
- `integration_ready/` — interface/export checkpoint
- `ready_for_review/` — awaiting Judge review
- `blocked/` — blocked tasks
- `done/` — Judge-approved tasks
- `handoff/` — cross-task integration notes
- `templates/` — canonical task and handoff templates
- `workstreams.md` — ownership boundaries

## Execution path boundary

Use the local swarm layer (`scripts/swarm.py` plus this directory) for routine task execution. Use the reviewed staged-workflow-runner path only for high-stakes Operator synthesis work that should not be treated as a normal Worker task.
