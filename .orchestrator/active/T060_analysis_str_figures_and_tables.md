---
task_id: T060
title: "Analysis: release STR figures and tables from validated artifacts"
workstream: W6
task_kind: analysis
allow_network: false
role: Worker
priority: high
dependencies:
  - "T040"
  - "T050"
integration_ready_dependencies: []
requires_tools:
  - "python"
  - "git"
requires_env: []
allowed_paths:
  - "src/analysis/build_str_release_outputs.py"
  - "reports/figures/str_ecosystem_timeseries.svg"
  - "reports/figures/str_post_dencun_regimes.svg"
  - "reports/tables/str_regime_summary.csv"
  - "reports/tables/str_regime_summary.md"
disallowed_paths:
  - "docs/protocol.md"
  - "contracts/"
  - "src/etl/"
  - "src/analysis/metrics_str.py"
  - "data/raw/"
  - "reports/catalog.yaml"
outputs:
  - "src/analysis/build_str_release_outputs.py"
  - "reports/figures/str_ecosystem_timeseries.svg"
  - "reports/figures/str_post_dencun_regimes.svg"
  - "reports/tables/str_regime_summary.csv"
  - "reports/tables/str_regime_summary.md"
gates:
  - "make gate"
stop_conditions:
  - "Validated inputs are missing or failing"
  - "Need to change metric definitions"
---

# Task T060 — Analysis: release STR figures and tables from validated artifacts

## Context

This task turns validated empirical artifacts into the minimum release analysis bundle. It replaces the old sample-only figure slice with stable figures and tables that the paper can include directly.

## Assignment

- Workstream: W6 Analysis
- Assigned role: Worker
- Suggested branch/worktree name: `T060_str_release_outputs`
- Allowed paths: one release-output builder script plus the locked figure/table filenames
- Stop conditions: block with `@human` if the requested outputs require protocol or validation changes

## Inputs

- `reports/validation/rollup_panel_validation.json`
- `reports/validation/l1_rent_decomposition_validation.json`
- `reports/validation/cross_source_reconciliation.json`
- `data/processed_manifest/daily_rollup_panel_<YYYY-MM-DD>.json`
- `data/processed_manifest/daily_l1_rent_decomposition_<YYYY-MM-DD>.json`
- `src/analysis/metrics_str.py`

## Outputs

- Analysis code: `src/analysis/build_str_release_outputs.py`
- Figures:
  - `reports/figures/str_ecosystem_timeseries.svg`
  - `reports/figures/str_post_dencun_regimes.svg`
- Tables:
  - `reports/tables/str_regime_summary.csv`
  - `reports/tables/str_regime_summary.md`

## Success Criteria

- [ ] The script reads validated local artifacts only and makes no network calls
- [ ] Both figures and both table formats are generated at the locked filenames
- [ ] The Markdown table fragment is suitable for direct inclusion in the paper
- [ ] Output filenames are stable and deterministic across reruns
- [ ] `make gate` passes

## Review Bundle Requirements

- [ ] A durable run manifest exists under `reports/status/swarm_runs/`
- [ ] Any interpretation caveat needed by T070 is captured in `.orchestrator/handoff/`
- [ ] Judge review is recorded under `reports/status/reviews/`

## Validation / Commands

- `make gate`
- `python src/analysis/build_str_release_outputs.py --sample`
- `python src/analysis/build_str_release_outputs.py --as-of YYYY-MM-DD`

## Status
- State: active
- Last updated: 2026-04-11
## Notes / Decisions

- 2026-03-29: v1 rewrite expands T060 from one sample figure to the minimum release figure-and-table bundle.
- 2026-04-11: Claimed by local swarm runtime on branch T060_analysis_str_figures_and_tables.
- 2026-04-11: Blocked under the task stop conditions. The current `2026-04-09` canonical surfaces in this worktree are not safe for downstream W6 release analysis: an independent panel-vs-decomposition coherence check on `data/processed/panels/daily_rollup_panel.csv` and `data/processed/l1_rent/daily_l1_rent_decomposition.csv` found `129` dates where daily summed panel rent differs from decomposition total, with `max_abs_diff=2.9148600184379916 ETH` on `2022-05-11`.
- 2026-04-11: `make gate` also fails upstream at `review_bundle_integrity` because completed tasks `T049` and `T052` declare `data/processed/l1_rent/daily_rollup_rent_components.csv`, but that file is missing in this worktree. The current validation reports still cite that missing component surface in their provenance, so the requested release outputs would rely on inconsistent validated inputs.
- 2026-04-11: @human unblock needed: provide a coherence-clean, reviewable canonical artifact bundle for W6 consumption, including a restored `data/processed/l1_rent/daily_rollup_rent_components.csv` and a panel/decomposition pair that agrees at the locked tolerance, or explicitly direct T060 to consume a different validated as-of surface.
- 2026-04-11: @human Runtime blocked: gates_failed, missing_outputs, task_marked_blocked. Run manifest: reports/status/swarm_runs/T060_20260411T113419Z.json. outputs=src/analysis/build_str_release_outputs.py=missing_file; reports/figures/str_ecosystem_timeseries.svg=missing_file; reports/figures/str_post_dencun_regimes.svg=missing_file; reports/tables/str_regime_summary.csv=missing_file; reports/tables/str_regime_summary.md=missing_file
- 2026-04-11: Operator root-cause review supersedes the earlier blocker rationale. The true upstream blocker was repo materialization: `data/processed/l1_rent/daily_rollup_rent_components.csv` matched its tracked `2026-04-09` processed manifest locally but remained git-ignored and therefore disappeared in fresh worktrees. That file is now tracked on `main`, and `scripts/quality_gates.py` now hard-fails review bundles when declared `data/processed/` file outputs are git-ignored or untracked in a real git worktree.
- 2026-04-11: Resume T060 using the repaired W2/W5 contract: `daily_rollup_panel.csv` is vendor-keyed by protocol when both `l2_fees_eth` and `rent_paid_eth` exist, `daily_rollup_rent_components.csv` may be a strict superset of panel keys, and the locked internal coherence check is component daily totals versus `daily_l1_rent_decomposition.csv`, not panel daily totals versus decomposition.
