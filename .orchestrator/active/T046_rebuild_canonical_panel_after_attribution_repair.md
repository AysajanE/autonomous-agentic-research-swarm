---
task_id: T046
title: "Rebuild canonical L1 rent outputs after attribution-hook repair"
workstream: W2
task_kind: etl
allow_network: true
role: Worker
priority: high
dependencies:
  - "T035"
  - "T045"
integration_ready_dependencies: []
requires_tools:
  - "python"
  - "git"
requires_env: []
allowed_paths:
  - "src/etl/build_l1_rent_panel.py"
  - "data/raw/l1_rent/"
  - "data/raw_manifest/l1_rent_"
  - "data/processed/l1_rent/"
  - "data/processed/panels/"
  - "data/processed_manifest/daily_"
  - "data/samples/l1_rent/"
  - "data/samples/panels/"
disallowed_paths:
  - "docs/protocol.md"
  - "contracts/"
  - "registry/"
outputs:
  - "src/etl/build_l1_rent_panel.py"
  - "data/raw/l1_rent/<YYYY-MM-DD>/..."
  - "data/raw_manifest/l1_rent_<YYYY-MM-DD>.json"
  - "data/processed/l1_rent/daily_l1_rent_decomposition.csv"
  - "data/processed/panels/daily_rollup_panel.csv"
  - "data/processed_manifest/daily_l1_rent_decomposition_<YYYY-MM-DD>.json"
  - "data/processed_manifest/daily_rollup_panel_<YYYY-MM-DD>.json"
  - "data/samples/l1_rent/daily_l1_rent_decomposition_sample.csv"
  - "data/samples/panels/daily_rollup_panel_sample.csv"
gates:
  - "make gate"
stop_conditions:
  - "Need protocol or contract changes"
  - "Upstream attribution evidence remains missing after T045"
  - "Source instability or breaking API changes"
---

# Task T046 — Rebuild canonical L1 rent outputs after attribution-hook repair

## Context

T035 completed, but T050 exposed that the canonical panel still omits vendor-covered rollup-days because the upstream attribution hooks are incomplete. Once T045 repairs the registry inputs, this task reruns the authoritative pipeline and checks whether the repaired attribution closes the canonical coverage and rent gaps.

## Assignment

- Workstream: W2 Data: on-chain
- Assigned role: Worker
- Suggested branch/worktree name: `T046_rebuild_canonical_panel`
- Allowed paths: `src/etl/build_l1_rent_panel.py` and the canonical l1-rent/raw/processed surfaces
- Stop conditions: block with `@human` instead of widening source priority, protocol semantics, or registry rules

## Inputs

- `docs/protocol.md`
- `contracts/project.yaml`
- `registry/rollup_registry_v1.csv`
- `.orchestrator/backlog/T045_repair_registry_attribution_hooks_for_reconciliation.md`
- `.orchestrator/handoff/H050_canonical_validation_reconciliation_blocker.md`

## Outputs

- Updated authoritative raw/processed artifacts and manifests for the rerun date
- Any ETL repair needed to consume the updated attribution hooks
- Updated canonical samples if the rerun changes tracked rows

## Success Criteria

- [ ] The authoritative rerun uses the repaired registry attribution inputs without silent fallback to stale coverage
- [ ] Any processed output lineage has matching processed manifests
- [ ] T050's dominant coverage gaps are re-measured and documented after the rerun
- [ ] `make gate` passes

## Review Bundle Requirements

- [ ] A durable run manifest exists under `reports/status/swarm_runs/`
- [ ] Any remaining attribution caveat needed by T050 is captured in `.orchestrator/handoff/`
- [ ] Judge review is recorded under `reports/status/reviews/`

## Validation / Commands

- `make gate`
- `python src/etl/build_l1_rent_panel.py --run-date YYYY-MM-DD`

## Status
- State: blocked
- Last updated: 2026-04-08
## Notes / Decisions

- 2026-04-08: Added as the canonical rebuild step after T045. This task should not start until the registry attribution-hook repair is complete.
- 2026-04-08: Claimed by local swarm runtime on branch T046_rebuild_canonical_panel_after_attribution_repair.
- 2026-04-08: Read the current T045 outputs from `.orchestrator/done/T045_repair_registry_attribution_hooks_for_reconciliation.md` and `.orchestrator/handoff/H045_registry_reconciliation_attribution_hooks.md`. The input path named in this task (`.orchestrator/handoff/H050_canonical_validation_reconciliation_blocker.md`) is not present in this worktree; the H045 handoff is the durable upstream repair note that exists.
- 2026-04-08: Confirmed the repaired registry rows are present, but this worktree does not currently have the local Growthepie/vendor ETL surfaces needed for a canonical rebuild (`data/processed/growthepie/vendor_daily_rollup_panel.csv` is missing, and only `data/raw_manifest/growthepie_2026-04-01.json` exists).
- 2026-04-08: Validation run: `make gate` passed.
- 2026-04-08: Reproduction attempts: `python src/etl/build_l1_rent_panel.py --run-date 2026-04-08` failed with `required growthepie raw manifest is missing: /Users/aeziz-local/Research/wt-T046/data/raw_manifest/growthepie_2026-04-08.json`; `python src/etl/build_l1_rent_panel.py --run-date 2026-04-01` failed with `raw manifest already exists for this run date: /Users/aeziz-local/Research/wt-T046/data/raw_manifest/l1_rent_2026-04-01.json. Manifests are append-only; do not overwrite prior provenance.`
- 2026-04-08: No code or data outputs were changed under the task's allowed paths because proceeding would require either (a) fresh Growthepie inputs for a new run date or (b) edits outside `allowed_paths` to regenerate the missing Growthepie local surfaces.
- 2026-04-08: @human unblock needed: provide a fresh Growthepie input bundle for a new canonical rerun date (at minimum a matching `data/raw_manifest/growthepie_<YYYY-MM-DD>.json` plus the local `data/processed/growthepie/vendor_daily_rollup_panel.csv`), or explicitly revise the task contract to allow decoupling the canonical rerun date from the existing `2026-04-01` Growthepie snapshot / regenerating Growthepie local surfaces.
- 2026-04-08: @human Runtime blocked: missing_outputs, task_marked_blocked. Run manifest: reports/status/swarm_runs/T046_20260408T165337Z.json. outputs=data/raw/l1_rent/<YYYY-MM-DD>/...=missing_or_empty_dir; data/processed/l1_rent/daily_l1_rent_decomposition.csv=missing_file; data/processed/panels/daily_rollup_panel.csv=missing_file
