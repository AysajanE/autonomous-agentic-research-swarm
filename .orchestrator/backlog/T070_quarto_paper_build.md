---
task_id: T070
title: "Quarto manuscript source and draft render for the L2-to-L1 rent paper"
workstream: W7
task_kind: writing
allow_network: false
role: Worker
priority: high
dependencies:
  - "T050"
  - "T060"
integration_ready_dependencies: []
requires_tools:
  - "python"
  - "git"
  - "quarto"
requires_env: []
allowed_paths:
  - "reports/paper/_quarto.yml"
  - "reports/paper/index.qmd"
  - "reports/paper/references.bib"
disallowed_paths:
  - "docs/protocol.md"
  - "contracts/"
  - "src/"
  - "reports/paper/build/"
  - "reports/catalog.yaml"
outputs:
  - "reports/paper/_quarto.yml"
  - "reports/paper/index.qmd"
  - "reports/paper/references.bib"
gates:
  - "make gate"
stop_conditions:
  - "Validated figures or tables are missing"
  - "Need to change protocol or metric definitions"
  - "Quarto is unavailable"
---

# Task T070 — Quarto manuscript source and draft render for the L2-to-L1 rent paper

## Context

The paper is a first-class release surface. This task owns manuscript source only: Quarto configuration, narrative structure, and bibliography. Final rendered artifacts remain an Operator release surface under T080.

## Assignment

- Workstream: W7 Writing
- Assigned role: Worker
- Suggested branch/worktree name: `T070_quarto_paper_source`
- Allowed paths: paper source files only
- Stop conditions: block with `@human` rather than changing definitions or inventing missing evidence in prose

## Inputs

- `docs/protocol.md`
- `contracts/project.yaml`
- `reports/validation/rollup_panel_validation.md`
- `reports/validation/l1_rent_decomposition_validation.md`
- `reports/validation/cross_source_reconciliation.md`
- `reports/figures/str_ecosystem_timeseries.svg`
- `reports/figures/str_post_dencun_regimes.svg`
- `reports/tables/str_regime_summary.md`

## Outputs

- `reports/paper/_quarto.yml`
- `reports/paper/index.qmd`
- `reports/paper/references.bib`

## Success Criteria

- [ ] The manuscript contains sections for question, data/protocol, validation, results, and provenance/limitations
- [ ] Figures are linked from `../figures/` and tables are included from `../tables/`
- [ ] Quarto configuration targets the locked build directory and output basename
- [ ] `quarto render reports/paper/index.qmd --to html` succeeds as a draft render
- [ ] This task does not commit final paper build artifacts under `reports/paper/build/`
- [ ] `make gate` passes

## Review Bundle Requirements

- [ ] A durable run manifest exists under `reports/status/swarm_runs/`
- [ ] Any unresolved writing caveat is captured in `.orchestrator/handoff/`
- [ ] Judge review is recorded under `reports/status/reviews/`

## Validation / Commands

- `make gate`
- `quarto render reports/paper/index.qmd --to html`

## Status

- State: backlog
- Last updated: 2026-03-29

## Notes / Decisions

- 2026-03-29: New v1 task added so Quarto manuscript source is part of the core battle-test path instead of a placeholder.
