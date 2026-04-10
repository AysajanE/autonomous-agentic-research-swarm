# Data Dictionary (contract)

This file is the canonical reference for tables/fields/units/keys used in the project.

## Tables

### daily_rollup_panel

- Purpose: Analysis-ready daily rollup panel used to compute Settlement Take Rate (STR).
- Primary key: (`date_utc`, `rollup_id`)
- Grain: daily × rollup (UTC)
- Row inclusion rule: rows exist **iff** both `l2_fees_eth` and `rent_paid_eth` are present (missingness is represented by omitting the row, not by nulls).
- Source(s):
  - Primary denominator (`l2_fees_eth`): growthepie (ETH-native series)
  - Authoritative numerator (`rent_paid_eth`): on-chain computed canonical series
  - Related vendor benchmark surfaces (`rent_paid_eth`, `profit_eth`): growthepie vendor panel and validation reports only; secondary cross-checks rather than the source of canonical `rent_paid_eth`
- Chain-specific rule:
  - For rollups with shared settlement infrastructure, `rent_paid_eth` includes only the direct-exclusive attributable surface unless a reviewed allocation model is locked.
  - For Starknet specifically, canonical `rent_paid_eth` excludes raw generic SHARP verifier-stack fees and is benchmarked against the direct-exclusive state-update settlement surface under the current contract.

#### Fields

| Field | Type | Units | Nullable | Description |
|---|---|---|---|---|
| `date_utc` | date | YYYY-MM-DD (UTC) | no | UTC date for daily aggregation |
| `rollup_id` | string | slug | no | Stable rollup identifier (see `registry/rollup_registry_v1.csv`) |
| `l2_fees_eth` | number | ETH | no | Total user fees paid on the rollup for `date_utc` (ETH-native) |
| `rent_paid_eth` | number | ETH | no | Authoritative on-chain attributable Ethereum L1 fee accounting for settlement/DA/proofs for `date_utc` (ETH-native); for shared-settlement rollups this is the direct-exclusive attributable surface unless a reviewed allocation model is locked |
| `profit_eth` | number | ETH | yes | Optional vendor-derived benchmark field; used only for sanity checks when contract-compatible and not part of the canonical STR numerator |
| `txcount` | integer | count | yes | Transaction count (if provided) |

### daily_rollup_rent_components

- Purpose: Rollup-day audit surface that decomposes canonical `rent_paid_eth` into component families for validation, reconciliation, and benchmark attribution.
- Primary key: (`date_utc`, `rollup_id`)
- Grain: daily × rollup (UTC)
- Source(s): on-chain computed canonical series (authoritative)
- Identity rule:
  - tx-family columns (`batch_submissions_eth`, `proof_submissions_eth`, `state_updates_eth`) must sum exactly to `rent_paid_eth`
  - fee-class columns (`blob_fee_burn_eth`, `execution_base_fee_burn_eth`, `execution_priority_fee_eth`) must separately sum exactly to `rent_paid_eth`
- Chain-specific rule:
  - This table contains only components that are inside canonical `rent_paid_eth`; excluded shared-cost diagnostics do not belong in the canonical component totals.
  - For Starknet under the current contract, raw shared SHARP verifier-stack fees are excluded from the canonical component surface until a reviewed allocation model exists.

#### Fields

| Field | Type | Units | Nullable | Description |
|---|---|---|---|---|
| `date_utc` | date | YYYY-MM-DD (UTC) | no | UTC date for daily aggregation |
| `rollup_id` | string | slug | no | Stable rollup identifier (see `registry/rollup_registry_v1.csv`) |
| `batch_submissions_eth` | number | ETH | no | Canonical execution/blob fees attributed to batch-submission transactions for the rollup-day; excludes shared-settlement raw fees that are outside canonical `rent_paid_eth` |
| `proof_submissions_eth` | number | ETH | no | Canonical execution/blob fees attributed to proof or custom settlement submissions for the rollup-day; excludes shared-settlement raw fees that are outside canonical `rent_paid_eth` |
| `state_updates_eth` | number | ETH | no | Canonical execution/blob fees attributed to state-update transactions for the rollup-day; for Starknet under the current contract this is the direct-exclusive benchmark-compatible settlement surface |
| `blob_fee_burn_eth` | number | ETH | no | Canonical EIP-4844 blob fee burn attributed to the rollup-day |
| `execution_base_fee_burn_eth` | number | ETH | no | Canonical EIP-1559 execution-layer base fee burn attributed to the rollup-day |
| `execution_priority_fee_eth` | number | ETH | no | Canonical execution-layer priority fees attributed to the rollup-day |
| `rent_paid_eth` | number | ETH | no | Canonical rollup-day total; must equal both the tx-family subtotal and the fee-class subtotal for the same row |

### daily_l1_rent_decomposition

- Purpose: Daily Ethereum L1 rent components used for burn vs tips and blob vs calldata analysis.
- Primary key: (`date_utc`)
- Grain: daily (UTC)
- Source(s): on-chain computed series (authoritative)

#### Fields

| Field | Type | Units | Nullable | Description |
|---|---|---|---|---|
| `date_utc` | date | YYYY-MM-DD (UTC) | no | UTC date for daily aggregation |
| `l1_base_fee_burn_eth` | number | ETH | no | ETH burned via EIP-1559 base fee (execution layer) |
| `l1_blob_fee_burn_eth` | number | ETH | no | ETH burned via EIP-4844 blob base fee |
| `l1_priority_fee_eth` | number | ETH | no | ETH paid as priority fees (tips) |
| `l1_total_rent_eth` | number | ETH | no | Total L1 rent (burn + tips); must equal sum of components |
| `l1_blob_gas_used` | integer | blob gas | yes | Total blob gas used (optional cross-check field) |
| `l1_calldata_gas_used` | integer | gas | yes | Total calldata gas proxy (optional cross-check field) |
| `l1_blob_base_fee_gwei` | number | gwei | yes | Blob base fee level (optional; used for regime classification) |

### <future_table_name>

- Purpose:
- Primary key:
- Grain:
- Source(s):

#### Fields

| Field | Type | Units | Nullable | Description |
|---|---|---|---|---|
|  |  |  |  |  |
