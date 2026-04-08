# Handoff H050 — Canonical STR validation blocked on reconciliation failures

## Summary (1–3 sentences)

The canonical `python src/validation/validate_str_pipeline.py --as-of 2026-04-01` run now resolves the hydrated local manifests and refreshes the stable report bundle, but the bundle is not review-ready. The authoritative panel and vendor panel diverge on key coverage, the vendor `profit_eth` identity fails materially, and the matched-key aggregate rent delta is `10.707572667673433%`, so downstream analysis and writing should remain blocked on T050.

## What changed / what exists now

- Files/paths:
  - `reports/validation/rollup_panel_validation.json`
  - `reports/validation/rollup_panel_validation.md`
  - `reports/validation/l1_rent_decomposition_validation.json`
  - `reports/validation/l1_rent_decomposition_validation.md`
  - `reports/validation/cross_source_reconciliation.json`
  - `reports/validation/cross_source_reconciliation.md`
- Outputs produced:
  - `rollup_panel_validation` status `fail`
  - `l1_rent_decomposition_validation` status `pass`
  - `cross_source_reconciliation` status `fail`
  - This note supersedes the earlier missing-CSV blocker as the current canonical state for T050.

## How to reproduce / verify

- Commands:
  - `python src/validation/validate_str_pipeline.py --as-of 2026-04-01`
  - `make gate`
- Expected results:
  - Canonical validator exit code `1`
  - `rollup_panel_validation` fails on `authoritative_vs_vendor_key_coverage` with `1,158` vendor-only keys
  - `l1_rent_decomposition_validation` passes all checks for `1,551` dates
  - `cross_source_reconciliation` fails on `vendor_profit_identity` (`543` violating rows) and `monthly_cross_source_reconciliation`
  - `make gate` passes

## Assumptions / risks

- Key-coverage mismatch is concentrated in `arbitrum` (`368` dates, `2022-01-01` to `2023-01-03`), `zksync_era` (`540` dates, `2023-03-24` to `2026-01-30`), `linea` (`248` dates, `2023-07-13` to `2026-03-26`), and `taiko` (`2` dates, `2025-11-29` to `2025-11-30`).
- Vendor profit identity violations are concentrated in `starknet` (`508` rows, max abs diff `82.418787225` ETH, `2024-02-26` to `2025-10-17`), with smaller clusters in `zksync_era` (`29` rows, max abs diff `1.0336628657304` ETH) and `linea` (`6` rows, max abs diff `62.46171051244441` ETH).
- On matched keys, vendor rent totals `87,176.30702620494` ETH versus authoritative `78,744.6648188145` ETH, which is an absolute delta of `10.707572667673433%` and therefore above the protocol’s `5–10%` tolerance target.
- This run was not emitted through a fresh `scripts/swarm.py` execution, so Operator will need to capture a durable run manifest before any later review attempt.

## Open questions / next steps

- Decide whether the vendor-versus-authoritative key mismatches reflect expected universe/coverage differences or an upstream attribution/registry bug.
- Decide whether vendor `profit_eth` remains contract-compatible for `starknet`, `linea`, and `zksync_era`; if not, adjust the producing logic or document the source-definition change through the proper W0 path.
- After upstream reconciliation is resolved, rerun `python src/validation/validate_str_pipeline.py --as-of 2026-04-01` and confirm the stable canonical report bundle passes before advancing downstream analysis or writing tasks.
