---
task_id: T___
title: "<title>"
workstream: W__
role: Worker
priority: medium
dependencies: []
allowed_paths:
  - "<path>/"
disallowed_paths:
  - "docs/protocol.md"
  - "contracts/"
  - "registry/"
  - ".orchestrator/templates/"
  - ".orchestrator/workstreams.md"
  - "data/raw/"
outputs:
  - "<output path>"
gates:
  - "make gate"
stop_conditions:
  - "Contract ambiguity"
  - "Need credentials"
---

# Task T___ — <title>

## Context

Describe *why* this task exists and how it connects to the research plan/protocol.

## Assignment

- Workstream:
- Owner (agent/human):
- Suggested branch/worktree name:
- Allowed paths (edit/write):
- Disallowed paths:
- Stop conditions (escalate + block with `@human`):

## Inputs

- Docs:
- Data:
- Prior tasks / handoffs:
- External references (links):

## Outputs

- Code:
- Data artifacts (paths; note raw snapshots are append-only):
- Docs/figures:

## Success Criteria

- [ ] Clearly-defined outputs exist at the paths above
- [ ] Repro steps are documented (commands + expected outputs)
- [ ] Relevant quality gates/tests pass
- [ ] Any assumptions/limitations are recorded

## Validation / Commands

- `make gate`
- Add task-specific commands here.

## Worker edit rules

- **Workers edit only** `## Status` and `## Notes / Decisions`.
- **Workers do not move this file** between lifecycle folders; set `State:` and the Planner will sweep.

## Status

- State: backlog | active | blocked | ready_for_review | done
- Last updated: YYYY-MM-DD

## Notes / Decisions

- YYYY-MM-DD: <progress note, decision, or blocker; include `@human` if needed>
