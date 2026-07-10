---
# Planner triage is mandatory before claim for every L task and any heuristic-flagged task (multi-input ETL, more than two outputs, or first backlog task in a workstream); M/L tasks with recon_required must then record at least three non-empty reconnaissance lines covering scope, risks, and decomposition pressure before promotion to ready_for_review.
task_schema: research_swarm.task.v2
task_id: T___  # replace with T###
title: "<concise task title>"
workstream: W__  # replace with a declared workstream
task_kind: analysis  # etl|analysis|validation|writing|lit_review|model|proof|bridge|ops|integrity_audit|repair
complexity_tier: S  # S|M|L; L requires progress_file
success_criteria:
  - id: SC1
    statement: "<observable completion statement>"
    verification: "<offline command or artifact pointer>"
budgets: {max_wall_clock: 1h, max_tokens: 100000, max_cost_usd: 10}
checkpoint_contract: none  # none|progress_file
recon_required: false  # true for M/L unless recon_waiver is non-empty
recon_waiver: ""  # explain an explicit false for M/L; leave empty otherwise
# triage: {status: confirmed, by: planner, note: "<why this task is bounded>"}  # required only when triage flags apply
constructed_by: ""  # validation only: T### for the construction task
inputs:
  - path: "<upstream manifest path>"  # `manifest` is also accepted
    sha256: "<64-character manifest sha256>"
    comparison_basis: false  # validation tasks need at least one true, disjoint from construction inputs
allow_network: false  # true requires a framework-allowlisted network workstream
role: Worker
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
  - make gate
stop_conditions:
  - "Contract ambiguity"
  - "Need credentials"
---

# Task T___ — <concise task title>

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
- Upstream tasks / manifests and pinned hashes:
- External references or systems:

## Reconnaissance

- Scope understanding:
- Risks and unknowns:
- Decomposition pressure assessment:
- Proposed bounded approach:

## Outputs

- Code:
- Data / manifests:
- Reports / docs:

## Success Criteria

- [ ] Each frontmatter success criterion is satisfied and verified
- [ ] Declared outputs exist at the paths above
- [ ] Reproduction commands are recorded
- [ ] Assumptions and limitations are recorded

## Review Bundle Requirements

- [ ] If this task produces artifacts, a durable run manifest exists under `reports/status/swarm_runs/`
- [ ] Judge review is recorded under `reports/status/reviews/`
- [ ] Any downstream-critical guidance is captured in `.orchestrator/handoff/`

## Validation / Commands

- `make gate`
- Add task-specific offline commands here.

## Edit rules

- Workers fill `## Reconnaissance` before implementation when required, then edit only `## Status` and `## Notes / Decisions`.
- Planner and Operator handle folder moves via sweep or `git mv`.
- `integration_ready` may be used only for interface/export tasks named in downstream `integration_ready_dependencies`.

## Status

- State: backlog
- Last updated: YYYY-MM-DD

## Notes / Decisions

- YYYY-MM-DD: <progress note, decision, or blocker; include `@human` when needed>
