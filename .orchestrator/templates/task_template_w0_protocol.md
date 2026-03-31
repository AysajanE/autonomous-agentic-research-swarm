---
task_id: T___
title: "<title>"
workstream: W0
task_kind: protocol
allow_network: false
role: Worker
priority: high
dependencies: []
integration_ready_dependencies: []
requires_tools:
  - "python"
  - "git"
requires_env: []
allowed_paths:
  - "docs/"
  - "contracts/"
disallowed_paths:
  - "src/"
  - "registry/"
  - "data/raw/"
outputs:
  - "docs/protocol.md"
  - "contracts/decisions.md"
gates:
  - "make gate"
stop_conditions:
  - "Definition ambiguity"
  - "Need credentials"
---

# Task T___ — <title> (W0 Protocol / Contracts)

## Context

Describe the smallest, testable protocol or contract change needed and why it matters.

## Assignment

- Workstream: W0 Protocol / Contracts
- Assigned role: Worker
- Suggested branch/worktree name:
- Allowed paths:
- Disallowed paths:
- Stop conditions:

## Inputs

- `docs/protocol.md`
- `contracts/project.yaml`
- `contracts/framework.json`
- the affected schema, dictionary, decision, model, or interface files
- prior task notes or handoffs

## Outputs

- updated protocol and contract files
- a required decision-log update in `contracts/decisions.md`
- changelog or version bumps if an interface contract changes

## Success Criteria

- [ ] The change is minimal, explicit, and testable
- [ ] `contracts/decisions.md` records the decision, rationale, and expected blast radius
- [ ] Any affected downstream interface or workflow is named explicitly
- [ ] `make gate` passes

## Review Bundle Requirements

- [ ] If the task changes a contract consumed downstream, the blast radius is captured in `.orchestrator/handoff/`
- [ ] A durable run manifest exists under `reports/status/swarm_runs/`
- [ ] Judge review is recorded under `reports/status/reviews/`

## Validation / Commands

- `make gate`

## Edit rules

- Workers edit only `## Status` and `## Notes / Decisions`.
- If the change implies task decomposition or workstream changes, hand that impact to Planner instead of editing `.orchestrator/templates/` or `.orchestrator/workstreams.md`.
- `integration_ready` is allowed only when the task exports an interface that named downstream tasks need before full completion.

## Status

- State: backlog | active | integration_ready | ready_for_review | blocked | done
- Semantics:
  - `integration_ready`: contract/interface export only; downstream allowlist required
  - `ready_for_review`: contract edits are complete, gates pass, and a run manifest exists
  - `done`: Judge-approved
- Last updated: YYYY-MM-DD

## Notes / Decisions

- YYYY-MM-DD: <progress note, decision, or blocker; include `@human` when needed>
