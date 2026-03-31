---
task_id: T___
title: "<title>"
workstream: W8
task_kind: model
allow_network: false
role: Worker
priority: medium
dependencies: []
integration_ready_dependencies: []
requires_tools:
  - "python"
  - "git"
  - "solver:<name>"
requires_env: []
instances:
  - "contracts/instances/<instance_set>/manifest.yaml"
experiment_spec: "contracts/experiments/<experiment>.yaml"
allowed_paths:
  - "src/model/"
  - "contracts/instances/"
  - "contracts/experiments/"
  - "reports/models/"
  - "tests/"
disallowed_paths:
  - "docs/protocol.md"
  - "registry/"
  - "data/raw/"
  - ".orchestrator/templates/"
  - ".orchestrator/workstreams.md"
outputs:
  - "reports/models/<experiment>/<run_id>/run_manifest.json"
  - "reports/models/<experiment>/<run_id>/results.json"
gates:
  - "make gate"
stop_conditions:
  - "Instance or experiment spec ambiguity"
  - "Need solver license or credentials"
  - "Need to edit outside allowed paths"
---

# Task T___ — <title> (W8 Modeling)

## Context

Describe the modeling objective and how it maps to `contracts/model_spec.md`.

## Assignment

- Workstream: W8 Modeling / Bridge
- Assigned role: Worker
- Suggested branch/worktree name:
- Allowed paths:
- Disallowed paths:
- Stop conditions:

## Inputs

- instance manifest(s): `contracts/instances/.../manifest.yaml`
- experiment spec: `contracts/experiments/<experiment>.yaml`
- solver requirement from `requires_tools`
- empirical input manifests if the task is part of a hybrid workflow

## Outputs

- run folder: `reports/models/<experiment>/<run_id>/`
- required modeling run manifest: `reports/models/<experiment>/<run_id>/run_manifest.json`
- required results: `reports/models/<experiment>/<run_id>/results.json`

## Success Criteria

- [ ] Instance and experiment contracts are referenced explicitly
- [ ] Solver name, version, and invocation are recorded
- [ ] The modeling run manifest records git SHA, commands, input contracts, and produced outputs
- [ ] `make gate` passes

## Review Bundle Requirements

- [ ] The modeling run manifest under `reports/models/...` exists
- [ ] A repo-level durable run manifest exists under `reports/status/swarm_runs/`
- [ ] Judge review is recorded under `reports/status/reviews/`

## Validation / Commands

- `make gate`
- Add task-specific modeling commands here.

## Edit rules

- Workers edit only `## Status` and `## Notes / Decisions`.
- Do not widen the instance or experiment contract without a W0-reviewed contract change.

## Status

- State: backlog | active | integration_ready | ready_for_review | blocked | done
- Semantics:
  - `integration_ready`: use only if the task exports a stable interface for named downstream tasks
  - `ready_for_review`: outputs exist, gates pass, and a run manifest exists
  - `done`: Judge-approved
- Last updated: YYYY-MM-DD

## Notes / Decisions

- YYYY-MM-DD: <progress note, decision, or blocker; include `@human` when needed>
