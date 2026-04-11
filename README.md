# Autonomous Agentic Research Swarm

This repository is a repo-native research operating system for the current L2-to-L1 rent analysis project. v1 is designed to carry one real project from definition lock to a reproducible working-paper release.

## Locked empirical artifact DAG

`registry -> raw snapshots/manifests -> processed datasets/manifests -> validation -> figures/tables -> Quarto paper -> release manifest`

The current project is not releaseable until that full path exists.

## Default execution paths

- **Local swarm** (`scripts/swarm.py` + `.orchestrator/`) is the default engine for routine repo task execution, deterministic gates, and normal multi-agent delivery.
- **Reviewed staged-workflow-runner path** is for high-stakes Operator-owned synthesis work such as architecture rewrites, major replans, and release assessments.
- The paper substrate is **Quarto-backed Markdown** under `reports/paper/`.

## Four-role operating model

- **Operator** — runtime preflight, worktree/tmux supervision, repair handling, sweeps, run/review/release logging, catalog refresh, and release assembly.
- **Planner** — task decomposition, dependency wiring, workstream ownership, and lifecycle projection.
- **Worker** — one assigned task, one isolated worktree, one explicit output contract.
- **Judge** — reruns gates, verifies outputs and provenance, and is the only role allowed to mark work `done`.

## Current battle-test queue

1. `T025` — populate `registry/rollup_registry_v1.csv` with evidence-backed in-scope rows.
2. `T030` — pull growthepie snapshots, write raw manifests, normalize the vendor panel, and commit a tiny deterministic sample.
3. `T035` — build the authoritative on-chain L1 rent path, write processed manifests, and materialize the canonical `daily_rollup_panel`.
4. `T040` — lock STR math in `src/analysis/metrics_str.py` with sample-only tests.
5. `T050` — validate the canonical panel, L1 rent decomposition, and cross-source reconciliation.
6. `T060` — generate release figures and tables from validated artifacts only.
7. `T070` — write Quarto manuscript source and confirm a draft render path.
8. `T080` — Operator release assembly: compile `reports/catalog.yaml`, render final paper artifacts, and write the release manifest.

## What counts as a release candidate

A release candidate must include all of the following:

- an evidence-backed `registry/rollup_registry_v1.csv`
- raw manifests for growthepie and the on-chain L1 rent pull
- processed manifests for the vendor panel, L1 rent decomposition, and canonical rollup panel
- validation JSON/Markdown outputs
- release figures and tables
- Quarto paper source plus rendered HTML/PDF and `render_manifest.json`
- `reports/catalog.yaml` compiled from successful run manifests
- `reports/status/releases/release_<YYYY-MM-DD>.json`

A sample figure alone is not battle-test success.

## Repository map

- `.orchestrator/` — file-based control plane, task queue, templates, and handoffs
- `contracts/` — project instance contract, framework policy, empirical definitions, and hybrid/modeling interfaces
- `docs/` — protocol lock, runbooks, and role prompts
- `registry/` — versioned rollup universe contract
- `data/raw_manifest/` and `data/processed_manifest/` — tracked provenance for raw and processed artifacts
- `src/etl/`, `src/validation/`, `src/analysis/`, `src/model/` — code split by responsibility
- `reports/` — validation outputs, figures, tables, Quarto paper source/build, catalog, and release manifests
- `tests/` — fast offline tests on tracked samples

## Quickstart

1. Read `AGENTS.md`.
2. Review `docs/protocol.md`, `contracts/project.yaml`, and `contracts/framework.json`.
3. Inspect `.orchestrator/workstreams.md` and the live backlog under `.orchestrator/backlog/`.
4. Run `make gate` and `make test` on the base branch.
5. Use `docs/runbook_swarm.md` for manual execution or `docs/runbook_swarm_automation.md` for the default local swarm path.
6. When upstream tasks are done, run the Operator release path with `python scripts/release_assembly.py --release-date YYYY-MM-DD --check`.

## Mode coverage

- **Empirical** is the active mode for this repo and the only mode that may be claimed as end-to-end ready after the battle-test release succeeds.
- **Modeling** remains contract-ready through `contracts/model_spec.md`, `contracts/instances/`, and `contracts/experiments/`, but it is not yet battle-tested here.
- **Hybrid** remains contract-ready through `contracts/hybrid_interface_v1.yaml`; modeling tasks may consume only explicit instance manifests, not ad hoc empirical CSV paths.
