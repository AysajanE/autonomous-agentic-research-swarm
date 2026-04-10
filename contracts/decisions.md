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

- 2026-04-10 — Lock canonical rent vs vendor benchmark policy (owner: @human)
  - Decision:
    - Canonical `daily_rollup_panel.rent_paid_eth` remains the release-truth numerator for STR and is defined by authoritative on-chain attributable Ethereum L1 fee accounting.
    - growthepie `rent_paid` remains a secondary vendor benchmark used for reconciliation and triangulation, not a source that can silently redefine canonical `RentPaid`.
    - A canonical-vs-vendor key-universe mismatch is a release blocker unless a W0-reviewed exception is recorded.
    - Matched-key benchmark divergence above tolerance is release-blocking only when it remains unexplained after component-level audit of the canonical on-chain surface.
    - The project now requires a rollup-day rent component audit surface so validation can distinguish integrity failures from benchmark-definition differences.
    - The `daily_rollup_rent_components` audit surface carries two parallel decompositions of the same canonical total: tx-family components and fee-class components. Each decomposition must reconcile independently to `rent_paid_eth`; they are not meant to be summed together.
  - Rationale:
    - Post-repair `T050` evidence for `2026-04-09` eliminated the old key-universe mismatch (`mismatched_key_count = 0`) but still showed a matched-key aggregate gap concentrated in a few rollups, especially `starknet` and `taiko`.
    - Vendor implementation evidence shows growthepie `rent_paid_eth` is assembled from a curated economics mapping and cost components that are not identical to literal canonical on-chain fee accounting for every chain.
    - The first `T049` implementation pass exposed that a single additive identity over both component families would double-count canonical rent and make the audit surface internally inconsistent.
  - Expected impact:
    - Validation must separate schema/coverage failures from benchmark divergences.
    - Canonical ETL work gains an explicit component-audit surface instead of relying on replay logs for root-cause diagnosis.
    - Release gating no longer depends on forcing vendor parity when the benchmark difference is evidence-backed and methodologically explained.
  - Links/refs:
    - `docs/protocol.md`
    - `contracts/data_dictionary.md`
    - `contracts/project.yaml`
    - `.orchestrator/handoff/H048_t050_contract_resolution_blocker.md`
