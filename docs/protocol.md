# Protocol Lock (Phase 0)

This file is the canonical definition set for this repo. If definitions conflict, stop and mark the relevant task `blocked` with `@human`.

## Research mode

- Mode: empirical

## Primary metric

- Name: Settlement Take Rate (STR)
- Formula: `STR_t = (Σ_i RentPaid_{i,t}) / (Σ_i L2Fees_{i,t})`
- Frequency: daily
- Units: Unitless ratio (0–1+); primary computation uses **ETH-native series** (USD series is secondary for interpretation only).

Definitions:
- `L2Fees_{i,t}`: total fees paid by users on rollup *i* on day *t*.
- `RentPaid_{i,t}`: total fees paid by rollup *i* to Ethereum L1 for settlement/DA/proofs on day *t*.
- Aggregation: the sum is over the **in-scope rollup universe** for day *t*.

## Rollup inclusion criteria

In-scope rollups must:
- Be an L2 rollup (optimistic or ZK) that posts data to **Ethereum L1 mainnet**.
- Have a stable identifier in the project universe and be attributable in at least one primary data source.

Out of scope:
- Non-Ethereum DA/settlement chains (may be discussed as competition, but excluded from STR computation).
- Sidechains that do not settle to Ethereum L1.

Time window:
- Start: 2022-01-01 (UTC)
- End: run date (UTC), daily frequency

## Data source priority

When sources disagree for the same concept, prefer (highest to lowest):
1. **On-chain computed** Ethereum L1 costs (authoritative for `RentPaid` and its decomposition).
2. **growthepie** exports (primary for `L2Fees`; secondary for vendor `rent_paid/profit` series).
3. **L2BEAT** costs series (triangulation / sanity check).

Rules:
- Prefer ETH-native series; convert to USD only using an explicit price series and document the source.
- Changing source priority requires a Workstream W0 task and explicit review.

## Known regime dates

Daily regime boundaries are evaluated in **UTC**.

- Dencun / EIP-4844 activation on Ethereum mainnet: 2024-03-13 (UTC)
  - Treat dates `>= 2024-03-13` as **post-Dencun** for daily panels.
- Analysis start date: 2022-01-01 (UTC)

## Validation tolerances

Unless overridden by a task:

- Accounting identity (vendor series): `profit ≈ fees − rent_paid`
  - Tolerance: 0.5–1.0% (unit-dependent; document the chosen tolerance)
- Cross-source reconciliation (monthly aggregates, top rollups):
  - Target tolerance: 5–10%; otherwise explain and document the cause
- Blob usage cross-check (sample month):
  - `blobGasUsed` tolerance: ≤1% between Blobscan and on-chain daily aggregation
- Price inputs (monthly averages):
  - Tolerance: 1–2% between primary and secondary price sources
