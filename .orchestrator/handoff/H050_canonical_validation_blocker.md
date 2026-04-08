# Handoff H050 — Canonical STR validation blocked on missing processed CSVs

## Summary (1–3 sentences)

Implemented the STR validation bundle under `src/validation/validate_str_pipeline.py` and confirmed it passes on the tracked sample CSVs. The canonical `--as-of 2026-04-01` run is blocked because all three processed CSVs referenced by the committed manifests are absent from this worktree, so the stable validation reports currently record that blocked canonical state.

## What changed / what exists now

- Files/paths:
  - `src/validation/validate_str_pipeline.py`
  - `reports/validation/rollup_panel_validation.json`
  - `reports/validation/rollup_panel_validation.md`
  - `reports/validation/l1_rent_decomposition_validation.json`
  - `reports/validation/l1_rent_decomposition_validation.md`
  - `reports/validation/cross_source_reconciliation.json`
  - `reports/validation/cross_source_reconciliation.md`
- Validator behavior:
  - `--sample` reads the tracked sample CSVs and runs schema/key coverage, STR compatibility, decomposition identity, vendor profit identity, and monthly reconciliation checks.
  - `--as-of YYYY-MM-DD` resolves the three processed manifests, requires the manifest-backed CSV outputs to exist locally, and writes blocked reports with the minimal next step when they do not.
- Current report state:
  - The three stable report pairs under `reports/validation/` reflect the blocked canonical `2026-04-01` run, not the passing sample run.

## How to reproduce / verify

- Commands:
  - `python -m py_compile src/validation/validate_str_pipeline.py`
  - `python src/validation/validate_str_pipeline.py --sample`
  - `python src/validation/validate_str_pipeline.py --as-of 2026-04-01`
  - `make gate`
- Observed outcomes:
  - `py_compile` passed.
  - `--sample` exited `0`; monthly aggregate rent reconciliation was `2.08%`, top-rollup monthly deltas were all below `4%`, vendor profit identity had `0` violations, and decomposition total rent matched the component sum exactly on all sampled days.
  - `--as-of 2026-04-01` exited `2`; blocked because these manifest-backed outputs are missing locally:
    - `data/processed/growthepie/vendor_daily_rollup_panel.csv`
    - `data/processed/l1_rent/daily_l1_rent_decomposition.csv`
    - `data/processed/panels/daily_rollup_panel.csv`
  - `make gate` passed.

## Assumptions / risks

- The sample run only proves the validation code path and offline checks; it does not satisfy the release-firewall requirement for canonical artifacts.
- Downstream analysis/writing tasks should treat T050 as blocked until the canonical processed CSVs are materialized and the `--as-of` run passes.
- This work was not recorded through a new `scripts/swarm.py` runtime run, so Operator must capture a durable run manifest separately before review.

## Open questions / next steps

- Restore or materialize the three processed CSVs named above for `2026-04-01`, or rerun the producing ETL in a workspace that contains them.
- After the files exist locally, rerun `python src/validation/validate_str_pipeline.py --as-of 2026-04-01` and confirm the stable report bundle switches from `blocked` to a passing or actionable failing validation state.
- Operator should record the command history and outcomes in a durable swarm run manifest before sending the task for review.
