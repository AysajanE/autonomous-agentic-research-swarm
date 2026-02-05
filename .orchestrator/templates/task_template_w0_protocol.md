---
task_id: T___
title: "<title>"
workstream: W0
task_kind: protocol
allow_network: false
role: Worker
priority: high
dependencies: []
requires_tools:
  - "python"
  - "git"
requires_env: []
allowed_paths:
  - "docs/"
  - "contracts/"
  - ".orchestrator/"
disallowed_paths:
  - "src/"
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

# Task T___ — <title> (W0 Protocol/Contracts)

## Context

Describe the smallest, testable protocol/contract change needed and why it matters.

## Assignment

- Workstream: W0 Protocol/Contracts
- Owner (agent/human):
- Suggested branch/worktree name:
- Allowed paths (edit/write):
- Disallowed paths:
- Stop conditions (escalate + block with `@human`):

## Inputs

- `docs/protocol.md`
- `contracts/`
- Prior tasks / handoffs:
- External references (links):

## Outputs

- Updated contract/protocol files (canonical specs)
- Required decision log entry: `contracts/decisions.md`
- If an interface changes: update `contracts/CHANGELOG.md` and bump versions as needed

## Success Criteria

- [ ] Changes are minimal and unambiguous
- [ ] Definitions are testable (clear inclusion rules, units, tolerances)
- [ ] `contracts/decisions.md` records the decision + rationale + expected blast radius
- [ ] `make gate` passes

## Validation / Commands

- `make gate`

## Worker edit rules

- **Workers edit only** `## Status` and `## Notes / Decisions`.
- **Workers do not move this file** between lifecycle folders; set `State:` and the Planner will sweep.

## Status

- State: backlog | active | blocked | integration_ready | ready_for_review | done
- Semantics: `ready_for_review` => outputs exist + gates pass; `integration_ready` => interfaces exported; downstream unblocked (optional).
- Last updated: YYYY-MM-DD

## Notes / Decisions

- YYYY-MM-DD: <progress note, decision, or blocker; include `@human` if needed>
