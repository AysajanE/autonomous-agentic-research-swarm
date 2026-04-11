# H060 — Release output caveats for T070

Date: 2026-04-11
Task: T060

## Summary

- The release figures and regime table were built from the validated `2026-04-09` empirical surface with `python src/analysis/build_str_release_outputs.py --as-of 2026-04-09`.
- The generated outputs are:
  - `reports/figures/str_ecosystem_timeseries.svg`
  - `reports/figures/str_post_dencun_regimes.svg`
  - `reports/tables/str_regime_summary.csv`
  - `reports/tables/str_regime_summary.md`

## Writing caveats

- The authoritative `daily_rollup_panel.csv` remains a vendor-keyed missingness surface by protocol:
  - panel rows exist only when both `l2_fees_eth` and `rent_paid_eth` exist
  - the component surface may therefore contain canonical-only rent keys that are absent from the panel
- On the current validated surface:
  - `daily_rollup_panel.csv` has `12,434` rows
  - `daily_rollup_rent_components.csv` has `12,563` rows
  - the additional `129` component-only keys are expected protocol-missingness, not a live validation failure
- The locked internal coherence claim for prose is:
  - daily summed component `rent_paid_eth` matches `daily_l1_rent_decomposition.l1_total_rent_eth` within tolerance
  - panel daily totals need not equal decomposition totals because the panel excludes rollup-days with missing vendor fee denominators
- The post-Dencun regime figure shades contiguous blob-fee floor runs using the protocol definition:
  - dates `>= 2024-03-13`
  - blob base fee within `1.05 ×` the post-Dencun minimum for runs of at least `7` consecutive days
- The current validation bundle still carries one documented non-gating monthly vendor reconciliation residual:
  - `2025-05`
  - `abs_delta_eth = 18.358717091619667`
  - `pct_difference = 12.369423293164906%`
  - this remains diagnostic context for the manuscript, not a release-blocking failure
