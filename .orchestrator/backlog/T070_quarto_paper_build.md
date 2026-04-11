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
- State: done
- Last updated: 2026-04-11
## Notes / Decisions

- 2026-03-29: New v1 task added so Quarto manuscript source is part of the core battle-test path instead of a placeholder.
- 2026-04-11: Claimed by local swarm runtime on branch `T070_quarto_paper_build` (base branch `main`).
- 2026-04-11: Rewrote `reports/paper/index.qmd` from the earlier methods placeholder into a results-forward working paper tied to the validated `2026-04-09` bundle. The manuscript now includes the required question, data/protocol, validation, results, and provenance/limitations sections; links figures from `../figures/`; and includes the locked regime table from `../tables/str_regime_summary.md`.
- 2026-04-11: Updated `reports/paper/_quarto.yml` to keep the locked `build/` output directory while targeting the downstream release basenames `l2_l1_rent_working_paper.html` and `l2_l1_rent_working_paper.pdf`. Refreshed `reports/paper/references.bib` to cite the protocol, project contract, validation bundle, release figures/table, and the W6 caveat note used in manuscript prose.
- 2026-04-11: Reproduction commands re-verified in this worktree: `make gate`; `tmpdir="$(mktemp -d /tmp/t070_quarto_check.XXXXXX)"`; `homedir="$(mktemp -d /tmp/t070_home.XXXXXX)"`; `cp -R reports "$tmpdir/"`; `env HOME="$homedir" quarto render "$tmpdir/reports/paper/index.qmd" --to html`.
- 2026-04-11: Outcome summary: `make gate` passed; the draft Quarto render succeeded in a `/tmp` mirror and produced `/tmp/t070_quarto_check.UwoqfM/reports/paper/build/l2_l1_rent_working_paper.html`. The temp mirror preserved the same relative `reports/paper`, `reports/figures`, and `reports/tables` layout as the repo so include paths and the locked basename were exercised without writing to the Operator-owned `reports/paper/build/` surface.
- 2026-04-11: Durable local-swarm run manifest already exists at `reports/status/swarm_runs/T070_20260411T150935Z.json`. This repair pass only re-verified the existing paper-source outputs and corrected the runtime notes; final in-repo paper build artifacts remain T080/Operator-owned.
- 2026-04-11: Remaining caveats and downstream guidance are captured in superseding handoff note `.orchestrator/handoff/H070_paper_source_handoff_2026-04-11_reverify.md`.
- 2026-04-11: Runtime passed: outputs, gates, manifests, and run manifest are present. Ready for Judge review. Run manifest: reports/status/swarm_runs/T070_20260411T153246Z.json
- 2026-04-11: Judge approved; review log: reports/status/reviews/T070_20260411T153559Z.json
