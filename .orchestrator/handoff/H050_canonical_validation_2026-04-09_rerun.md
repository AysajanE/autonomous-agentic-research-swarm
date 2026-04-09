# Handoff H050 — Canonical validation rerun on 2026-04-08 manifests

## Summary (1–3 sentences)

Re-ran `T050` against the synced `2026-04-08` processed manifests and refreshed the stable validation report bundle under `reports/validation/`. The canonical bundle remains blocked, but the failure surface changed materially from the stale `2026-04-01` run: vendor profit identity now passes, while source coverage and matched-key rent reconciliation still fail and are now grouped by rollup/date range in the reports.

## What changed / what exists now

- Files/paths:
  - `src/validation/validate_str_pipeline.py`
  - `reports/validation/rollup_panel_validation.json`
  - `reports/validation/rollup_panel_validation.md`
  - `reports/validation/l1_rent_decomposition_validation.json`
  - `reports/validation/l1_rent_decomposition_validation.md`
  - `reports/validation/cross_source_reconciliation.json`
  - `reports/validation/cross_source_reconciliation.md`
- Outputs produced:
  - Canonical rollup panel validation now reports `1,125` vendor-only rollup-days with grouped concentration in `zksync_era` (`538`), `arbitrum` (`368`), and `linea` (`219`); there are no authoritative-only keys.
  - L1 rent decomposition validation passes cleanly for `1,558` dates through `2026-04-07`.
  - Cross-source reconciliation still fails even on matched keys:
    - vendor profit identity: `pass` with `0` violations
    - matched-row aggregate rent delta: `10.667051775329645%`
    - monthly aggregate violations above `10%`: `32`
    - rollup-month violations above `10%`: `247`
    - daily outliers above `10%`: `6,926`
  - The report now preserves monthly reconciliation statistics even when key coverage fails, so downstream owners can see that early-2022 `optimism` deltas remain materially above tolerance after excluding unmatched rows.

## How to reproduce / verify

- Commands:
  - `python -m py_compile src/validation/validate_str_pipeline.py`
  - `python src/validation/validate_str_pipeline.py --sample`
  - `python src/validation/validate_str_pipeline.py --as-of 2026-04-08`
  - `make gate`
- Expected results:
  - `py_compile` exits `0`.
  - `--sample` exits `0`.
  - `--as-of 2026-04-08` exits `1` and refreshes the three canonical report pairs at the stable paths above.
  - `make gate` passes.

## Assumptions / risks

- The canonical blocker is now narrower than the stale 2026-04-01 diagnostic: vendor profit semantics are no longer the issue on the current vendor extract.
- Key coverage mismatch is not the only remaining gap. Even after restricting to matched rows, the rent reconciliation still exceeds the protocol ceiling, so upstream fixes should not stop at aligning the row universe.
- This rerun was executed directly in the worktree, not via a fresh `scripts/swarm.py` runtime, so no new durable swarm run manifest was written.

## Open questions / next steps

- Confirm whether the vendor-only `arbitrum`, `zksync_era`, and `linea` rollup-days are expected to be absent from the canonical panel.
- If the coverage mismatch is not intentional, fix the upstream registry/attribution/row-omission logic and rerun `python src/validation/validate_str_pipeline.py --as-of 2026-04-08`.
- After key coverage aligns, inspect the matched-key monthly rent deltas that still exceed the `10%` target band, starting with the early-2022 `optimism` months surfaced in `reports/validation/cross_source_reconciliation.{json,md}`.
- Before any future review attempt, Operator should record a durable swarm run manifest with the commands and outcomes above.
