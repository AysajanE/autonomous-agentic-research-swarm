---
task_id: T000
title: "Protocol lock (definitions first)"
workstream: W0
role: Worker
priority: high
dependencies: []
allowed_paths:
  - "docs/"
disallowed_paths:
  - "data/raw/"
outputs:
  - "docs/protocol.md"
gates:
  - "make gate"
stop_conditions:
  - "Definition ambiguity"
  - "Need to edit outside allowed paths"
---

# Task T000 — Protocol lock (definitions first)

## Context

Before any large-scale data collection or analysis, lock the definitions that all downstream tasks depend on (to prevent “metric shopping” and definition drift).

Primary reference docs:
- `docs/end_to_end_research_plan.md`
- `docs/end_to_end_data_collection_plan.md`
- `docs/protocol.md`

## Assignment

- Workstream: W0 Protocol
- Owner (agent/human):
- Suggested branch/worktree name: `w0-protocol-lock`
- Allowed paths (edit/write): `docs/`
- Disallowed paths: `data/raw/` (append-only), source code directories unless explicitly required
- Stop conditions (escalate + block with `@human`):
  - Any ambiguity in the primary metric definition, inclusion criteria, or regime cutoffs

## Inputs

- `docs/end_to_end_research_plan.md` (Phase 0)
- `docs/end_to_end_data_collection_plan.md` (Phase 0)

## Outputs

- Updated `docs/protocol.md` containing:
  - Primary metric name, formula, and units
  - Rollup inclusion criteria (and stable rollup identifiers)
  - Data source priority rules
  - Known regime dates (e.g., protocol upgrades) with timezone assumptions
  - Validation tolerances (acceptable error bands)

## Success Criteria

- [ ] `docs/protocol.md` is complete and unambiguous
- [ ] Downstream tasks can reference the protocol without re-interpreting definitions
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
- 2026-01-21: Filled `docs/protocol.md` with STR definition, inclusion criteria, source priority, regime dates, and validation tolerances.
- 2026-03-31: Backfilled review-bundle headings plus durable run/review artifacts so the task satisfies the v1 control-plane gate contract.
