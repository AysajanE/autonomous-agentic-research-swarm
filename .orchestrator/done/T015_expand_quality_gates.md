---
task_id: T015
title: "Expand quality gates (contract hardening)"
workstream: W0
role: Worker
priority: high
dependencies: [T010]
allowed_paths:
  - "scripts/"
  - "docs/"
  - ".orchestrator/"
  - "contracts/"
  - "reports/"
  - "src/"
  - "Makefile"
disallowed_paths:
  - "data/raw/"
outputs:
  - "scripts/quality_gates.py"
gates:
  - "make gate"
stop_conditions:
  - "Gate requires network calls"
  - "Non-deterministic behavior required"
---

# Task T015 — Expand quality gates (contract hardening)

## Context

Quality gates are the merge firewall. Expand gates beyond repo structure so the contract surface is hard to drift:
- protocol completeness
- workstreams completeness
- task hygiene

## Assignment

- Workstream: control-plane
- Owner (agent/human):
- Suggested branch/worktree name: `T015_expand_quality_gates`
- Allowed paths (edit/write): `scripts/`, `docs/`, `.orchestrator/`
- Disallowed paths: `data/raw/` (append-only)
- Stop conditions (escalate + block with `@human`):
  - Any proposed gate that requires network calls or non-deterministic behavior

## Inputs

- `scripts/quality_gates.py`

## Outputs

- Expanded `scripts/quality_gates.py` gates:
  - protocol completeness
  - workstreams completeness
  - task hygiene

## Success Criteria

- [x] Gates are fast and deterministic
- [x] Failures are actionable
- [x] `make gate` passes

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

- 2026-01-21: Implemented protocol/workstreams/task hygiene gates in `scripts/quality_gates.py`.
- 2026-03-31: Backfilled review-bundle headings plus durable run/review artifacts so the task satisfies the v1 control-plane gate contract.
