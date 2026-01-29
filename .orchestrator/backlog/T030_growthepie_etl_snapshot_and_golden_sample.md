---
task_id: T030
title: "growthepie ETL: snapshot exports + create golden sample panel"
workstream: W1
role: Worker
priority: high
dependencies:
  - "T020"
allowed_paths:
  - "src/etl/growthepie_fetch.py"
  - "data/raw/growthepie/"
  - "data/raw_manifest/growthepie_"
  - "data/processed/growthepie/"
  - "data/samples/growthepie/"
disallowed_paths:
  - "docs/protocol.md"
  - "contracts/"
  - "registry/"
  - "src/analysis/"
  - "src/validation/"
outputs:
  - "src/etl/growthepie_fetch.py"
  - "data/raw/growthepie/YYYY-MM-DD/..."
  - "data/raw_manifest/growthepie_YYYY-MM-DD.json"
  - "data/processed/growthepie/vendor_daily_rollup_panel.csv"
  - "data/samples/growthepie/vendor_daily_rollup_panel_sample.csv"
gates:
  - "make gate"
stop_conditions:
  - "Need credentials"
  - "Source instability / breaking changes"
---

# Task T030 — growthepie ETL: snapshot exports + create golden sample panel

## Context

Per `docs/protocol.md`, growthepie exports are the **primary** source for the STR denominator (`L2Fees`) and a **secondary** candidate for vendor `rent_paid/profit` series.

This task builds a reproducible ETL that:
- snapshots raw growthepie exports (append-only),
- records provenance in `data/raw_manifest/`,
- produces a normalized “vendor daily rollup panel” CSV for local analysis,
- and commits a tiny golden sample CSV under `data/samples/` for deterministic tests.

## Inputs

- `docs/protocol.md` (read-only): primary metric units + source priority
- `contracts/schemas/panel_schema_str_v1.yaml` (read-only): expected fields for the STR panel
- growthepie API:
  - `https://api.growthepie.com/v1/master.json`
  - `https://api.growthepie.com/v1/export/{metric_key}.json`

## Outputs

- ETL code: `src/etl/growthepie_fetch.py`
  - Must be the only place that performs network calls for growthepie.
  - Should support a `--run-date YYYY-MM-DD` argument to control snapshot folder naming.
  - Should fetch at least: master.json + the daily exports needed for STR (fees, rent_paid, profit, txcount), preferring ETH-native series when available.
- Raw snapshots (append-only; not committed): `data/raw/growthepie/<YYYY-MM-DD>/...`
- Provenance manifest (tracked): `data/raw_manifest/growthepie_<YYYY-MM-DD>.json`
  - Use `python scripts/make_raw_manifest.py ...` and ensure it includes file hashes and the exact repro command.
- Local processed artifact (not committed): `data/processed/growthepie/vendor_daily_rollup_panel.csv`
  - Minimal columns should align with the schema contract from T020 (date, rollup_id, fees/rent/profit, units explicit).
- Golden sample (tracked): `data/samples/growthepie/vendor_daily_rollup_panel_sample.csv`
  - Must be tiny (seconds to load).
  - Choose a fixed, documented date range and a small rollup subset.
  - Include the same columns as the processed panel (subset of rows only).

## Success Criteria

- [ ] Running the ETL writes a dated snapshot under `data/raw/growthepie/<run-date>/` without overwriting existing snapshots
- [ ] Manifest exists under `data/raw_manifest/` and validates via `make gate`
- [ ] Processed panel CSV is produced deterministically from the raw snapshot
- [ ] Golden sample CSV is committed and is stable (fixed date range + rollups documented in-file or in a small README next to it)
- [ ] `make gate` passes

## Validation / Commands

- `make gate`
- Example (replace placeholders):
  - `python src/etl/growthepie_fetch.py --run-date 2026-01-22`
  - `python scripts/make_raw_manifest.py growthepie data/raw/growthepie/2026-01-22 --as-of 2026-01-22 -- python src/etl/growthepie_fetch.py --run-date 2026-01-22`

## Status
- State: done
- Last updated: 2026-01-29
## Notes / Decisions

- 2026-01-22: Task created (Planner) as first real W1 ETL vertical slice.


- 2026-01-29: Claimed by swarm runner; starting worker (branch: T030_growthepie_etl_snapshot_and_golden_sample).


- 2026-01-29: Judge: gates ok; ownership ok. Review log: /home/vscode/swarm-worktrees/wt-T030/data/tmp/swarm_logs/T030_20260129T154706Z_judge_review.txt
