---
task_id: T___
title: "<title>"
workstream: W__
task_kind: etl
allow_network: true
role: Worker
priority: medium
dependencies: []
integration_ready_dependencies: []
requires_tools:
  - "python"
  - "git"
requires_env: []
allowed_paths:
  - "src/etl/<script>.py"
  - "data/raw/<source>/"
  - "data/raw_manifest/<source>_"
  - "data/processed/<source>/"
  - "data/processed_manifest/<name>_"
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
  - "Source instability or breaking changes"
---

# Task T___ — <title> (W1 / W2 ETL)

## Context

Describe the source, what is being pulled, and how it connects to downstream metrics, validation, or release work.

## Assignment

- Workstream:
- Assigned role:
- Suggested branch/worktree name:
- Allowed paths:
- Disallowed paths:
- Stop conditions:

## Inputs

- `docs/protocol.md`
- relevant files under `contracts/`
- upstream registry or manifest inputs
- external endpoint or node details

## Outputs

- raw snapshots: `data/raw/<source>/<YYYY-MM-DD>/...`
- raw provenance: `data/raw_manifest/<source>_<YYYY-MM-DD>.json`
- processed outputs under `data/processed/...`
- processed provenance under `data/processed_manifest/...`
- tracked samples when the task owns a sample surface

## Success Criteria

- [ ] Raw snapshots are append-only and written to a new dated folder
- [ ] The raw manifest includes hashes and the exact reproduction command
- [ ] Any processed output is created from code rather than manual edits
- [ ] Any committed processed lineage has a matching processed manifest
- [ ] `make gate` passes

## Review Bundle Requirements

- [ ] A durable run manifest exists under `reports/status/swarm_runs/`
- [ ] Any downstream lineage caveat is captured in `.orchestrator/handoff/`
- [ ] Judge review is recorded under `reports/status/reviews/`

## Validation / Commands

- `make gate`
- Add task-specific commands here.

## Edit rules

- Workers edit only `## Status` and `## Notes / Decisions`.
- ETL tasks should normally flow `backlog -> active -> ready_for_review -> done`.
- Do not use `integration_ready` for unvalidated empirical data outputs unless the task is explicitly scoped as an interface/export bridge and downstream tasks are allowlisted.

## Status

- State: backlog | active | integration_ready | ready_for_review | blocked | done
- Semantics:
  - `integration_ready`: exceptional interface/export case only
  - `ready_for_review`: outputs exist, manifests exist, gates pass, and a run manifest exists
  - `done`: Judge-approved
- Last updated: YYYY-MM-DD

## Notes / Decisions

- YYYY-MM-DD: <progress note, decision, or blocker; include `@human` when needed>
