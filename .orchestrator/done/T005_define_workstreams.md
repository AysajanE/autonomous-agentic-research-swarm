---
task_id: T005
title: "Define workstreams + ownership boundaries"
workstream: W0
role: Worker
priority: high
dependencies: [T000]
allowed_paths:
  - ".orchestrator/"
  - "docs/"
disallowed_paths:
  - "data/raw/"
outputs:
  - ".orchestrator/workstreams.md"
gates:
  - "make gate"
stop_conditions:
  - "Ownership dispute"
  - "Need to edit outside allowed paths"
---

# Task T005 — Define workstreams + ownership boundaries

## Context

Parallel work only scales when ownership boundaries are explicit. Define workstreams and which paths/artifacts each workstream owns.

## Assignment

- Workstream: W0 Protocol (coordination)
- Owner (agent/human):
- Suggested branch/worktree name: `w0-workstreams`
- Allowed paths (edit/write): `.orchestrator/`, `docs/`
- Disallowed paths: `data/raw/` (append-only)
- Stop conditions (escalate + block with `@human`):
  - Any dispute about which workstream owns a shared artifact (schemas, registries, definitions)

## Inputs

- `.orchestrator/templates/workstreams_template.md`
- `docs/end_to_end_research_plan.md` (phase decomposition)

## Outputs

- Updated `.orchestrator/workstreams.md` with:
  - Workstream names
  - “Owns paths” / “Does NOT own” rules
  - Expected outputs per workstream
  - Quality gates per workstream (even if stubbed initially)

## Success Criteria

- [ ] Workstreams cover the full research lifecycle end-to-end
- [ ] Ownership boundaries are strict enough to prevent clashes
- [ ] `make gate` passes

## Review Bundle Requirements

- [x] A durable run manifest exists under `reports/status/swarm_runs/`
- [x] Judge review is recorded under `reports/status/reviews/`
- [x] Legacy bootstrap completion evidence is backfilled to the v1 review-bundle contract

## Validation / Commands

- `make gate`

## Status

- State: done
- Last updated: 2026-03-31

## Notes / Decisions

- 2026-01-21: Task created (bootstrap).
- 2026-01-21: Populated `.orchestrator/workstreams.md` with initial workstreams and ownership boundaries.
- 2026-03-31: Backfilled review-bundle headings plus durable run/review artifacts so the task satisfies the v1 control-plane gate contract.
