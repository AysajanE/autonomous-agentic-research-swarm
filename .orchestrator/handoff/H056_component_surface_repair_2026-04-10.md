# H056 — Component-surface universe repair and validation follow-through

Date: 2026-04-10
Role: Operator

## Purpose

Carry out the narrow post-audit repair sequence after the scientific-integrity review:

1. repair the authoritative rollup-day component surface so it no longer inherits vendor-only missingness
2. update validation so it enforces the repaired authoritative invariants
3. rerun validation on the repaired `2026-04-09` surface

## Repair branches

- W2 repair branch: `T049_component_surface_universe_repair`
  - commits:
    - `61891d9` — `T049: repair component surface key coverage`
    - `87ddf9d` — `T049: widen component sample coverage`
    - `b25f4c0` — `T049: refresh component sample artifact`
- W5 repair branch: `T050_component_surface_validation_repair`
  - commits:
    - `f4975ef` — cherry-pick of the W2 key-coverage repair
    - `a448b7c` — cherry-pick of the W2 sample-coverage repair
    - `0ec6c03` — `T050: validate repaired component surface coverage`

## What changed

### W2 ETL

- `src/etl/build_l1_rent_panel.py` no longer emits `daily_rollup_rent_components.csv` by iterating only `vendor_rows`.
- The component surface now covers:
  - all panel keys, including zero-rent panel rows where vendor `l2_fees_eth` exists
  - all canonical on-chain rent keys from `rollup_daily`
- Net result on the `2026-04-09` surface:
  - panel rows remain `12,434`
  - component rows increase to `12,563`
  - the additional `129` keys are canonical-only rent rows previously omitted from the component artifact
- The tracked component sample was widened from panel-style sample keys to all component rows on the sample dates so sample-mode validation remains coherent.

### W5 validation

- `src/validation/validate_str_pipeline.py` now treats the component surface as a superset of the panel instead of requiring exact key equality.
- The validator now enforces the missing internal coherence check:
  - daily sum of `daily_rollup_rent_components.rent_paid_eth`
  - must equal `daily_l1_rent_decomposition.l1_total_rent_eth`
  - within the locked identity tolerance

## Verification

On the repaired W2 surface:

- panel keys missing from components: `0`
- panel/component overlapping rent mismatches: `0`
- decomposition/component daily mismatches: `0`
- component-only keys: `129`
- `make gate`: pass

On the repaired W5 surface:

- `python src/validation/validate_str_pipeline.py --sample`: pass
- `python src/validation/validate_str_pipeline.py --as-of 2026-04-09`: pass
- `make gate`: pass

The canonical validation bundle remains green after the repair:

- `reports/validation/rollup_panel_validation.{json,md}`: pass
- `reports/validation/l1_rent_decomposition_validation.{json,md}`: pass
- `reports/validation/cross_source_reconciliation.{json,md}`: pass

## Residual not fixed by this repair

- `make test` still fails on the repaired W5 branch because `tests/test_metrics_str.py` expected values are stale relative to the updated sample panel.
- This is the previously identified W4 test-fixture drift, not a new W2/W5 regression.

## Operator conclusion

- The authoritative component-surface integrity defect identified in `H055` is repaired on the live W2/W5 repair branches.
- The repo is no longer blocked by the specific component-universe issue.
- Remaining cleanup is narrower:
  - reconcile or refresh the W4 STR sample test expectations
  - decide when to review / integrate the repair branches into the mainline task topology
