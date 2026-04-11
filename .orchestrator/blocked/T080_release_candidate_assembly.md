---
task_id: T080
title: "Operator release candidate assembly for the L2-to-L1 rent paper"
workstream: W9
task_kind: ops
allow_network: false
role: Operator
priority: high
dependencies:
  - "T025"
  - "T030"
  - "T035"
  - "T040"
  - "T050"
  - "T060"
  - "T070"
integration_ready_dependencies: []
requires_tools:
  - "python"
  - "git"
  - "quarto"
requires_env: []
allowed_paths:
  - "reports/catalog.yaml"
  - "reports/paper/build/"
  - "reports/status/releases/"
disallowed_paths:
  - "docs/protocol.md"
  - "contracts/"
  - "src/"
  - "reports/paper/index.qmd"
outputs:
  - "reports/catalog.yaml"
  - "reports/paper/build/l2_l1_rent_working_paper.html"
  - "reports/paper/build/l2_l1_rent_working_paper.pdf"
  - "reports/paper/build/render_manifest.json"
  - "reports/status/releases/release_YYYY-MM-DD.json"
gates:
  - "make gate"
  - "python scripts/release_assembly.py --as-of YYYY-MM-DD --check"
stop_conditions:
  - "Successful producing run manifests are missing"
  - "Quarto render fails"
  - "Required validation or release artifacts are missing"
  - "Need to change scientific definitions"
---

# Task T080 — Operator release candidate assembly for the L2-to-L1 rent paper

## Context

This is the Operator-owned final assembly task. It compiles the catalog from successful run manifests, renders the final Quarto outputs, and writes the release manifest that ties the full empirical lineage together for Judge review.

## Assignment

- Workstream: W9 Ops / Release
- Assigned role: Operator
- Suggested branch/worktree name: `T080_release_candidate`
- Allowed paths: release-only shared surfaces
- Stop conditions: block instead of bypassing missing manifests, failed validation, or failed paper renders

## Inputs

- `contracts/project.yaml`
- `contracts/framework.json`
- Successful run manifests under `reports/status/swarm_runs/` for `T025` through `T070`
- Judge-approved upstream task states
- Release figures, tables, validation reports, and paper source from upstream tasks

## Outputs

- Compiled release index: `reports/catalog.yaml`
- Final paper build:
  - `reports/paper/build/l2_l1_rent_working_paper.html`
  - `reports/paper/build/l2_l1_rent_working_paper.pdf`
  - `reports/paper/build/render_manifest.json`
- Final release manifest: `reports/status/releases/release_<YYYY-MM-DD>.json`

## Success Criteria

- [ ] `reports/catalog.yaml` is compiled from successful run manifests and released artifacts, not hand-edited as task prose
- [ ] `python scripts/release_assembly.py --as-of YYYY-MM-DD --check` succeeds
- [ ] `quarto render reports/paper/index.qmd` emits the locked HTML, PDF, and render manifest outputs
- [ ] The release manifest references exact raw manifests, processed manifests, validation artifacts, figures, tables, paper outputs, compiled catalog, and git SHA
- [ ] The task is handed to Judge at `ready_for_review`; Operator does not mark it `done`
- [ ] `make gate` passes

## Review Bundle Requirements

- [ ] A durable run manifest exists under `reports/status/swarm_runs/` for the release assembly run
- [ ] Any release caveat is captured in `.orchestrator/handoff/`
- [ ] Judge review is recorded under `reports/status/reviews/`

## Validation / Commands

- `make gate`
- `python scripts/release_assembly.py --as-of YYYY-MM-DD --check`
- `quarto render reports/paper/index.qmd`

## Status

- State: blocked
- Last updated: 2026-04-11

## Notes / Decisions

- 2026-03-29: New v1 Operator task added to make catalog compilation, paper build, and release manifest assembly first-class release work.
- 2026-04-11: @human Blocked on a stale release-assembly gate contract before task launch. This task, related runbooks, and `contracts/framework.json` still reference `python scripts/release_assembly.py --as-of YYYY-MM-DD --check`, but the current `scripts/release_assembly.py` CLI accepts `--release-date`, not `--as-of`. Local swarm `run-task` and `judge-task` execute declared task gates literally, so T080 cannot pass runtime or Judge cleanly until the gate contract is aligned to the current CLI or the script restores a compatible `--as-of` alias.
