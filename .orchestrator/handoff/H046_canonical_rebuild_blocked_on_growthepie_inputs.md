# H046 — Canonical Rebuild Blocked on Growthepie Inputs

Date: 2026-04-08
Task: T046

## Summary

- The upstream registry repair from T045 is present and readable in this worktree.
- The canonical rerun is blocked by missing Growthepie inputs, not by an ETL regression in `src/etl/build_l1_rent_panel.py`.

## What was verified

- `make gate` passed in `/Users/aeziz-local/Research/wt-T046`.
- The repaired registry/handoff artifacts exist:
  - `.orchestrator/done/T045_repair_registry_attribution_hooks_for_reconciliation.md`
  - `.orchestrator/handoff/H045_registry_reconciliation_attribution_hooks.md`
- The task input path `.orchestrator/handoff/H050_canonical_validation_reconciliation_blocker.md` does not exist in this worktree.
- Current Growthepie surfaces in this worktree:
  - Present: `data/raw_manifest/growthepie_2026-04-01.json`
  - Present: `data/processed_manifest/vendor_daily_rollup_panel_2026-04-01.json`
  - Missing local ETL surface: `data/processed/growthepie/vendor_daily_rollup_panel.csv`
  - Missing fresh manifest for a new rerun date: no `data/raw_manifest/growthepie_2026-04-08.json`

## Commands and outcomes

- `make gate`
  - Passed.
- `python src/etl/build_l1_rent_panel.py --run-date 2026-04-08`
  - Failed immediately: missing `data/raw_manifest/growthepie_2026-04-08.json`.
- `python src/etl/build_l1_rent_panel.py --run-date 2026-04-01`
  - Failed immediately: `data/raw_manifest/l1_rent_2026-04-01.json` already exists and the manifests are append-only.

## Why this blocks T046

- `build_l1_rent_panel.py` currently requires a Growthepie raw manifest with the same `--run-date` as the canonical rerun.
- The only Growthepie raw manifest available in this worktree is for `2026-04-01`.
- Reusing `--run-date 2026-04-01` is disallowed because the canonical `l1_rent` raw/processed manifests for that date already exist and are append-only.
- Regenerating the missing Growthepie local ETL surfaces would write outside T046's `allowed_paths`.

## Smallest unblock needed

- Provide a fresh Growthepie input bundle for a new rerun date:
  - `data/raw_manifest/growthepie_<YYYY-MM-DD>.json`
  - local `data/processed/growthepie/vendor_daily_rollup_panel.csv`
- Or explicitly revise the task/allowed-path contract so T046 may decouple the canonical rerun date from the existing `2026-04-01` Growthepie snapshot or regenerate the Growthepie local surfaces itself.
