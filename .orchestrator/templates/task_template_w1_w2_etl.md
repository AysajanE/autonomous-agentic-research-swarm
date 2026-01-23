---
task_id: T___
title: "<title>"
workstream: W__
role: Worker
priority: medium
dependencies: []
allowed_paths:
  - "src/etl/<script>.py"
  - "data/raw/<source>/"
  - "data/raw_manifest/<source>_"
  - "data/processed/<source>/"
disallowed_paths:
  - "docs/protocol.md"
  - "contracts/"
  - "registry/"
outputs:
  - "src/etl/<script>.py"
  - "data/raw/<source>/<YYYY-MM-DD>/..."
  - "data/raw_manifest/<source>_<YYYY-MM-DD>.json"
gates:
  - "make gate"
stop_conditions:
  - "Need credentials"
  - "Source instability / breaking changes"
---

# Task T___ — <title> (W1/W2 ETL)

## Context

Describe the source, what we’re pulling, and how it connects to downstream metrics/analysis.

## Assignment

- Workstream: W1 Data: off-chain / W2 Data: on-chain
- Owner (agent/human):
- Suggested branch/worktree name:
- Allowed paths (edit/write):
- Disallowed paths:
- Stop conditions (escalate + block with `@human`):

## Inputs

- `docs/protocol.md` (read-only; do not change definitions)
- `contracts/schemas/` (read-only unless explicitly assigned)
- Prior tasks / handoffs:
- External references (links, endpoints, docs):

## Outputs

- Raw snapshots (append-only): `data/raw/<source>/<YYYY-MM-DD>/...`
- Provenance manifest (tracked): `data/raw_manifest/<source>_<YYYY-MM-DD>.json`
- Normalized outputs (rebuildable): `data/processed/<source>/...`
- ETL code under `src/etl/`

## Success Criteria

- [ ] Raw snapshot is written to a dated folder (never overwritten)
- [ ] Manifest exists, includes hashes, and describes how to reproduce the pull
- [ ] Any transformations are code (no manual edits)
- [ ] `make gate` passes

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
