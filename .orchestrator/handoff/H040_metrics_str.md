# H040 — STR metrics module handoff

Date: 2026-04-08
Task: T040

## Outputs

- `src/analysis/metrics_str.py`
- `tests/test_metrics_str.py`

## Module contract

- `compute_rollup_str(panel)` returns the input panel's complete rows plus a `str` column.
- `compute_ecosystem_str(panel)` groups complete rows by `date_utc`, sums `rent_paid_eth` and `l2_fees_eth`, and adds daily ecosystem `str`.
- Required columns are `date_utc`, `rollup_id`, `l2_fees_eth`, and `rent_paid_eth`.
- Missingness follows the locked row-omission rule: rows missing either metric column are excluded from both numerator and denominator.
- Denominator-zero follows the locked `NaN` rule at both rollup-day and ecosystem-day levels.

## Verification

- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest tests/test_metrics_str.py`
- `make gate`
- `make test`

## Caveat

- Plain `pytest tests/test_metrics_str.py` is not reliable on this workstation because host-level auto-loaded `web3` plugins crash before test collection. Use the command above or `make test`.
