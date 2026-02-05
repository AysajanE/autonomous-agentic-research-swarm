---
task_id: T___
title: "<title>"
workstream: W8
task_kind: ops
allow_network: false
role: Worker
priority: medium
dependencies: []
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
  - "contracts/experiments/"
  - "registry/"
  - "data/raw/"
  - ".orchestrator/templates/"
  - ".orchestrator/workstreams.md"
outputs:
  - "src/model/<instance_generator_script>.py"
  - "contracts/instances/<instance_set>/manifest.yaml"
gates:
  - "make gate"
stop_conditions:
  - "Empirical-to-model interface ambiguity"
  - "Need to edit outside allowed paths"
---

# Task T___ — <title> (Hybrid bridge: empirical → modeling)

## Context

Define the contract boundary between empirical artifacts and modeling inputs. The goal is to make hybrid mode real:
downstream model runs should consume a stable **instance set manifest**, generated from explicit empirical manifests.

## Assignment

- Workstream:
- Owner (agent/human):
- Suggested branch/worktree name:
- Allowed paths (edit/write):
- Disallowed paths:
- Stop conditions (escalate + block with `@human`):

## Inputs

### Bridge contract (explicit, required)

Empirical input manifests → instance generator script → instance set manifest

- Empirical input manifest(s) (paths):
  - `data/processed_manifest/<name>_<YYYY-MM-DD>.json`
- Instance generator script (path + command):
  - Script: `src/model/<instance_generator_script>.py`
  - Command: `python src/model/<instance_generator_script>.py --in ... --out ...`
- Instance set manifest output (path):
  - `contracts/instances/<instance_set>/manifest.yaml`

### Notes

- The instance set manifest is the *interface surface* for modeling tasks: it must include enough metadata to trace back to the empirical manifests and reproduce instance generation deterministically.

## Outputs

- Instance generator script: `src/model/<instance_generator_script>.py`
- Instance set manifest: `contracts/instances/<instance_set>/manifest.yaml`
- (Optional) bridge report: `reports/models/<bridge_name>/<run_id>/...`

## Success Criteria

- [ ] Bridge contract is explicit (no “magic inputs”)
- [ ] Instance generator is deterministic (or explicitly parameterized)
- [ ] Instance set manifest can be regenerated from the empirical input manifests + generator command
- [ ] `make gate` passes

## Validation / Commands

- `make gate`
- Add task-specific bridge command(s) here.

## Worker edit rules

- **Workers edit only** `## Status` and `## Notes / Decisions`.
- **Workers do not move this file** between lifecycle folders; set `State:` and the Planner will sweep.

## Status

- State: backlog | active | blocked | integration_ready | ready_for_review | done
- Semantics: `ready_for_review` => outputs exist + gates pass; `integration_ready` => interfaces exported; downstream unblocked (optional).
- Last updated: YYYY-MM-DD

## Notes / Decisions

- YYYY-MM-DD: <progress note, decision, or blocker; include `@human` if needed>
