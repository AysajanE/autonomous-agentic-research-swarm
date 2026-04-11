# H060 — Release analysis blocked on upstream canonical integrity

Date: 2026-04-11
Task: T060

## Summary

- T060 did not generate `src/analysis/build_str_release_outputs.py` or the locked STR release figures/tables.
- The block is upstream, not a W6 plotting issue:
  - the current canonical panel and decomposition surfaces in this worktree are internally inconsistent
  - `make gate` fails because a required upstream component surface is missing

## Evidence

- Independent coherence check on the live canonical processed files:
  - inputs:
    - `data/processed/panels/daily_rollup_panel.csv`
    - `data/processed/l1_rent/daily_l1_rent_decomposition.csv`
  - result:
    - `129` dates where `sum(panel.rent_paid_eth)` differs from `l1_total_rent_eth`
    - `max_abs_diff = 2.9148600184379916 ETH`
    - worst date: `2022-05-11`
- `make gate` result:
  - `review_bundle_integrity` failed
  - failure text:
    - `.orchestrator/done/T049_emit_rollup_day_rent_components_for_auditability.md:missing_outputs:data/processed/l1_rent/daily_rollup_rent_components.csv=missing_file`
    - `.orchestrator/done/T052_repair_starknet_shared_sharp_allocation_in_canonical_etl.md:missing_outputs:data/processed/l1_rent/daily_rollup_rent_components.csv=missing_file`
- Direct filesystem check confirms:
  - `data/processed/l1_rent/daily_rollup_rent_components.csv` is absent in this worktree

## Impact

- W6 may not responsibly publish release figures/tables from these surfaces.
- The current STR validation reports are not a sufficient downstream publication basis in this worktree because they point to a missing rent-component artifact and the live panel/decomposition pair still fails the internal coherence check described in `H055`.

## Reproduction

- `make gate`
- `python - <<'PY'`
- `import pandas as pd`
- `panel = pd.read_csv('data/processed/panels/daily_rollup_panel.csv')`
- `decomp = pd.read_csv('data/processed/l1_rent/daily_l1_rent_decomposition.csv')`
- `by_day = panel.groupby('date_utc', as_index=False)['rent_paid_eth'].sum().rename(columns={'rent_paid_eth': 'panel_rent_eth'})`
- `merged = by_day.merge(decomp[['date_utc', 'l1_total_rent_eth']], on='date_utc', how='outer').fillna(0)`
- `merged['abs_diff'] = (merged['panel_rent_eth'] - merged['l1_total_rent_eth']).abs()`
- `print(int((merged['abs_diff'] > 1e-9).sum()), merged['abs_diff'].max(), merged.sort_values('abs_diff', ascending=False).head(1).to_dict('records'))`
- `PY`

## Smallest Unblocker

- Provide a coherence-clean canonical artifact bundle for W6, including:
  - a restored `data/processed/l1_rent/daily_rollup_rent_components.csv`
  - a canonical `daily_rollup_panel.csv` and `daily_l1_rent_decomposition.csv` pair that agrees by day within the locked tolerance
- Or explicitly authorize T060 to target a different validated as-of surface.
