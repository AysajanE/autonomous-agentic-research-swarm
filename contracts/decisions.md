# Decision log (contract)

Chronological log of key project decisions (definitions, inclusions, tolerances, model choices).

Policy:
- If a decision affects results, record it here (with rationale and expected impact).

## Decisions

- 2026-01-23 — Lock STR schema + edge-case rules (owner: @human)
  - Decision:
    - STR panel schema is defined in `contracts/schemas/panel_schema_str_v1.yaml`.
    - Decomposition schema stub is defined in `contracts/schemas/panel_schema_decomp_v1.yaml`.
    - ETH is the canonical unit for fee/rent/profit series in contracts (`*_eth` fields are ETH, not wei).
    - Denominator-zero rule: if `Σ_i L2Fees_{i,t} == 0`, then `STR_t = NaN` (undefined).
    - Missingness rule: if either `L2Fees_{i,t}` or `RentPaid_{i,t}` is missing for a rollup-day, exclude that rollup-day from both numerator and denominator sums for ecosystem aggregates.
    - Canonical rollup key is `rollup_id` (registry-backed in `registry/rollup_registry_v1.csv`).
    - Vendor identity tolerance uses an explicit ETH-based formula (see `docs/protocol.md`).
    - Regime classification includes an explicit blob-fee floor regime definition (see `docs/protocol.md`).
  - Rationale:
    - Prevents “metric shopping” and schema drift by locking names/units/edge cases before ETL/metrics work scales.
  - Expected impact:
    - Downstream ETL/metrics/validation tasks can rely on stable field names and deterministic handling of zeros/missingness.
  - Links/refs:
    - `docs/protocol.md`
    - `contracts/data_dictionary.md`
    - `contracts/schemas/panel_schema_str_v1.yaml`
    - `contracts/schemas/panel_schema_decomp_v1.yaml`

- 2026-02-05 — Add framework config + hybrid interface contracts (owner: @human)
  - Decision:
    - Framework-level policy/config lives in `contracts/framework.json` (mode, features, required paths, prompt templates, network workstreams).
    - Hybrid projects must define an explicit empirical→modeling interface in `contracts/hybrid_interface_v1.yaml` (or JSON equivalent), including which processed datasets feed instance generation and how to reproduce instance sets.
  - Rationale:
    - Removes domain-specific assumptions from framework gates and prevents “hybrid = two parallel projects” by enforcing a defined boundary.
  - Expected impact:
    - `scripts/quality_gates.py` and `scripts/swarm.py` can be reused across empirical/modeling/hybrid projects with only config/contract changes.
