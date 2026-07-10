---
task_id: T___
title: "<title>"
workstream: W8
task_kind: bridge
allow_network: false
role: Worker
priority: medium
dependencies: []
integration_ready_dependencies: []
requires_tools:
  - "python"
  - "git"
requires_env: []
allowed_paths:
  - "src/model/"
  - "contracts/instances/"
  - "reports/models/"
  - "tests/"
disallowed_paths:
  - "docs/protocol.md"
  - "registry/"
  - "data/raw/"
  - "data/processed/"
  - ".orchestrator/templates/"
  - ".orchestrator/workstreams.md"
outputs:
  - "src/model/<instance_generator_script>.py"
  - "contracts/instances/<instance_id>.json"
gates:
  - "make gate"
stop_conditions:
  - "Empirical-to-model interface ambiguity"
  - "Need to edit outside allowed paths"
---

# Task T___ — <title> (Hybrid bridge: empirical to modeling)

## Context

Define the contract boundary between empirical artifacts and modeling inputs. Downstream modeling work must consume a stable instance-manifest surface, not ad hoc empirical CSV paths.

## Assignment

- Workstream: W8 Modeling / Bridge
- Assigned role: Worker
- Suggested branch/worktree name:
- Allowed paths:
- Disallowed paths:
- Stop conditions:

## Bridge contract

- Hybrid interface: `contracts/hybrid_interface_v1.yaml`
- Input processed manifests:
  - `data/processed_manifest/<name>_<YYYY-MM-DD>.json`
- Generator script and command:
  - `src/model/<instance_generator_script>.py`
  - `python src/model/<instance_generator_script>.py --in ... --out ...`
- Output instance manifest:
  - `contracts/instances/<instance_id>.json`

## Outputs

- deterministic instance-generator code
- a bridge instance manifest with content-bound source manifests, generator command, generation time, content-bound outputs, and green pre-bridge validation records
- optional bridge report under `reports/models/`

## Success Criteria

- [ ] The bridge contract is explicit and consistent with `contracts/hybrid_interface_v1.yaml`
- [ ] Modeling tasks can reproduce the instance set from the named processed manifests and generator command
- [ ] No modeling task needs to read `data/processed/...` directly
- [ ] `make gate` passes

## Review Bundle Requirements

- [ ] The bridge output is eligible for `integration_ready` only after the instance manifest exists and named downstream tasks allowlist it
- [ ] A durable run manifest exists under `reports/status/swarm_runs/`
- [ ] Judge review is recorded under `reports/status/reviews/`

## Validation / Commands

- `make gate`
- Add task-specific bridge commands here.

## Edit rules

- Workers edit only `## Status` and `## Notes / Decisions`.
- Keep interface widening inside W0 contract review; do not invent new bridge fields in prose alone.

## Status

- State: backlog | active | integration_ready | ready_for_review | blocked | done
- Semantics:
  - `integration_ready`: allowed for this task type when the instance manifest is complete and downstream allowlists exist
  - `ready_for_review`: outputs exist, gates pass, and a run manifest exists
  - `done`: Judge-approved
- Last updated: YYYY-MM-DD

## Notes / Decisions

- YYYY-MM-DD: <progress note, decision, or blocker; include `@human` when needed>
