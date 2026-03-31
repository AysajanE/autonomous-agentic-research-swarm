---
task_id: T___
title: "<title>"
workstream: W__
task_kind: ""  # protocol|registry|etl|metrics|validation|analysis|writing|bridge|model|ops
allow_network: false  # true requires workstream allowlist in contracts/framework.json
role: Worker  # use Operator only for W9 ops/release tasks
priority: medium
dependencies: []
integration_ready_dependencies: []
requires_tools:
  - "python"
  - "git"
requires_env: []
allowed_paths:
  - "<path/to/file_or_small_prefix>"
disallowed_paths:
  - "docs/protocol.md"
  - "contracts/"
  - "registry/"
  - ".orchestrator/templates/"
  - ".orchestrator/workstreams.md"
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

Describe why this task exists and which stage of the artifact DAG it advances.

## Assignment

- Workstream:
- Assigned role:
- Suggested branch/worktree name:
- Allowed paths:
- Disallowed paths:
- Stop conditions:

## Inputs

- Protocol / contracts:
- Upstream tasks / manifests:
- External references or systems:

## Outputs

- Code:
- Data / manifests:
- Reports / docs:

## Success Criteria

- [ ] Declared outputs exist at the paths above
- [ ] Reproduction commands are recorded
- [ ] Declared gates pass
- [ ] Assumptions and limitations are recorded

## Review Bundle Requirements

- [ ] If this task produces artifacts, a durable run manifest exists under `reports/status/swarm_runs/`
- [ ] Judge review is recorded under `reports/status/reviews/`
- [ ] Any downstream-critical guidance is captured in `.orchestrator/handoff/`

## Validation / Commands

- `make gate`
- Add task-specific commands here.

## Edit rules

- Workers and Operators edit only `## Status` and `## Notes / Decisions`.
- Planner and Operator handle folder moves via sweep or `git mv`.
- `integration_ready` may be used only for interface/export tasks named in downstream `integration_ready_dependencies`.

## Status

- State: backlog | active | integration_ready | ready_for_review | blocked | done
- Semantics:
  - `integration_ready`: interface/export task only; downstream allowlist required
  - `ready_for_review`: outputs exist, declared gates pass, required manifests exist, and a run manifest exists
  - `done`: Judge-approved
- Last updated: YYYY-MM-DD

## Notes / Decisions

- YYYY-MM-DD: <progress note, decision, or blocker; include `@human` when needed>
