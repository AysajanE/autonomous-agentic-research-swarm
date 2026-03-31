---
task_id: T010
title: "Initialize reusable repo skeleton"
workstream: W0
role: Worker
priority: high
dependencies: [T000, T005]
allowed_paths:
  - ".orchestrator/"
  - "docs/"
  - "contracts/"
  - "scripts/"
  - "src/"
  - "data/"
  - "reports/"
  - ".github/"
  - "AGENTS.md"
  - "CLAUDE.md"
disallowed_paths:
  - "data/raw/"
outputs:
  - ".orchestrator/"
  - "contracts/"
  - "reports/"
  - "src/"
  - ".github/CODEOWNERS"
gates:
  - "make gate"
stop_conditions:
  - "Request to store large artifacts in git"
---

# Task T010 — Initialize reusable repo skeleton

## Context

Set up a reusable, agent-friendly repo skeleton that makes ownership boundaries explicit and makes outputs predictable/reviewable.

## Assignment

- Workstream: W0 Protocol / control-plane
- Owner (agent/human):
- Suggested branch/worktree name: `T010_initialize_repo_skeleton`
- Allowed paths (edit/write): `.orchestrator/`, `docs/`, `scripts/`, `src/`, `data/`, `reports/`, `.github/`, root files
- Disallowed paths: `data/raw/` (append-only; not committed)
- Stop conditions (escalate + block with `@human`):
  - Any request to store large artifacts in git

## Inputs

- `AGENTS.md`
- `.orchestrator/`

## Outputs

- Nested agent contracts:
  - `.orchestrator/AGENTS.md`
  - `docs/AGENTS.md`
  - `scripts/AGENTS.md`
  - `src/etl/AGENTS.md`, `src/validation/AGENTS.md`, `src/analysis/AGENTS.md`
  - `data/AGENTS.md`
  - `registry/AGENTS.md`
- Provenance + reporting structure:
  - `data/raw_manifest/`
  - `reports/validation/`
  - `reports/figures/`
  - `registry/CHANGELOG.md`
- Compatibility and review hints:
  - `CLAUDE.md`
  - `.github/CODEOWNERS`

## Success Criteria

- [x] Required directories exist and are tracked via small files/README/AGENTS
- [x] Ownership boundaries can be expressed via workstreams + task allowed paths
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

- 2026-01-21: Initialized reusable skeleton directories and nested `AGENTS.md` files.
- 2026-03-31: Backfilled review-bundle headings plus durable run/review artifacts so the task satisfies the v1 control-plane gate contract.
