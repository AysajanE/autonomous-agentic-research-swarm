---
task_id: T___
title: "<title>"
workstream: W8
task_kind: model
allow_network: false
role: Worker
priority: medium
dependencies: []
requires_tools:
  - "python"
  - "git"
  - "solver:<name>" # e.g., solver:cbc
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
  - "Need solver license/credentials"
  - "Need to edit outside allowed paths"
---

# Task T___ — <title> (W8 Modeling)

## Context

Describe the modeling objective and how it maps to `contracts/model_spec.*`.

## Assignment

- Workstream:
- Owner (agent/human):
- Suggested branch/worktree name:
- Allowed paths (edit/write):
- Disallowed paths:
- Stop conditions (escalate + block with `@human`):

## Inputs

### Empirical / hybrid inputs (if applicable)

- Empirical input manifest(s): `data/processed_manifest/...` (paths)

### Modeling inputs (required)

- Instance set manifest(s) (frontmatter `instances`): `contracts/instances/.../manifest.yaml`
- Experiment spec (frontmatter `experiment_spec`): `contracts/experiments/<experiment>.yaml`
- Solver requirement (frontmatter `requires_tools`):

## Outputs

### Required output locations (suggested)

- Run folder: `reports/models/<experiment>/<run_id>/`
- Required run manifest: `reports/models/<experiment>/<run_id>/run_manifest.json`
- Required results: `reports/models/<experiment>/<run_id>/results.json`

### Run manifest requirements (required)

The run manifest must be machine-readable and include at minimum:
- task id + title
- git sha
- instance manifest path(s)
- experiment spec path
- solver name + version (and invocation command)
- reproduction command(s)
- list of produced output files (paths)

## Success Criteria

- [ ] Instance + experiment specs are referenced explicitly (no implicit assumptions)
- [ ] Solver requirement is satisfied (or task is blocked with an actionable note)
- [ ] Outputs exist under `reports/models/<experiment>/<run_id>/...`
- [ ] Run manifest exists and enables reproduction
- [ ] `make gate` passes

## Validation / Commands

- `make gate`
- Add task-specific commands here (e.g., `python -m src.model.run ...`).

## Worker edit rules

- **Workers edit only** `## Status` and `## Notes / Decisions`.
- **Workers do not move this file** between lifecycle folders; set `State:` and the Planner will sweep.

## Status

- State: backlog | active | blocked | integration_ready | ready_for_review | done
- Semantics: `ready_for_review` => outputs exist + gates pass; `integration_ready` => interfaces exported; downstream unblocked (optional).
- Last updated: YYYY-MM-DD

## Notes / Decisions

- YYYY-MM-DD: <progress note, decision, or blocker; include `@human` if needed>
