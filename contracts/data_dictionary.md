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
  - Vendor series (`rent_paid_eth`, `profit_eth`): growthepie (secondary; on-chain may supersede)

#### Fields

| Field | Type | Units | Nullable | Description |
|---|---|---|---|---|
| `date_utc` | date | YYYY-MM-DD (UTC) | no | UTC date for daily aggregation |
| `rollup_id` | string | slug | no | Stable rollup identifier (see `registry/rollup_registry_v1.csv`) |
| `l2_fees_eth` | number | ETH | no | Total user fees paid on the rollup for `date_utc` (ETH-native) |
| `rent_paid_eth` | number | ETH | no | Total fees paid by the rollup to Ethereum L1 for settlement/DA/proofs for `date_utc` (ETH-native) |
| `profit_eth` | number | ETH | yes | Vendor-provided profit series; used only for sanity checks (`profit ≈ fees − rent`) |
| `txcount` | integer | count | yes | Transaction count (if provided) |

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
