# .orchestrator/AGENTS.md — Control Plane Rules

This directory is the repo’s coordination layer.

## Who may change what

- **Planner only**:
  - moves tasks across folders (`backlog/`, `active/`, `integration_ready/`, `ready_for_review/`, `blocked/`, `done/`)
  - edits `workstreams.md`
  - edits anything in `templates/`
- **Worker may**:
  - edit ONLY their assigned task file
  - edit ONLY `## Status` and `## Notes / Decisions`
  - add a new handoff note in `handoff/` using the template
- **Judge may**:
  - change task `State:` to `ready_for_review`, `integration_ready`, or `done` after gates pass (as applicable)
  - request revisions in `Notes / Decisions`

## Task claiming (if instructed by Planner)

Workers do NOT self-assign tasks unless the Planner explicitly says so.

## Status discipline

- Always update `Last updated` in UTC date (YYYY-MM-DD).
- `State` must be one of:
  `backlog | active | blocked | integration_ready | ready_for_review | done`
- `ready_for_review` means: declared outputs exist and gates pass.
- `integration_ready` (optional) means: interfaces exported; downstream tasks unblocked.
- If blocked: include `@human` and the smallest decision needed.

## No rewrites

Do not rewrite task context/history. Append notes, don’t rewrite them.
