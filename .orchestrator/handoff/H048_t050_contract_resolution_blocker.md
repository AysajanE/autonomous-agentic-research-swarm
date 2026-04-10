# H048 — T050 contract-resolution blocker

Date: 2026-04-10

## Summary

The current `T050` blocker is no longer a broad canonical data-collection failure.

After the repaired `T046` rebuild and the clean `2026-04-09` rerun of `T050`:

- `mismatched_key_count = 0`
- `matched_row_count = 12434`
- vendor total rent = `132,310.156968212 ETH`
- authoritative total rent = `144,655.91426627047 ETH`
- aggregate matched-key difference = `8.534567950905517%`

This means the old rollup-day universe mismatch has been materially repaired. The remaining failure is a methodology conflict on matched keys.

## Dominant evidence

- Residual absolute mismatch is highly concentrated:
  - `starknet`: `13,626.815081 ETH` absolute, about `85.9%` of total absolute mismatch
  - `taiko`: `1,573.240609 ETH` absolute, about `9.9%`
  - excluding only `starknet` drops aggregate difference to about `0.995%`
  - excluding `starknet` and `taiko` drops it to about `0.213%`
- The largest `starknet` monthly gaps line up with canonical batch-submission cost families rather than random missing rows:
  - `2023-11`: net canonical-over-vendor `2150.323791 ETH`
  - `2023-05`: `1686.165898 ETH`
  - `2023-09`: `1544.309800 ETH`
- The `taiko` gap is directionally opposite: vendor exceeds canonical, especially in late 2024 blob-era months.

## Vendor methodology findings

Public growthepie documentation is too high-level to reproduce `rent_paid` from first principles, but the vendor implementation evidence is enough to prove the benchmark object differs from literal canonical on-chain cost accounting:

- growthepie economics uses a chain-specific curated mapping of transactions that count for economics/DA
- their backend computes:
  - `rent_paid_eth = cost_l1_raw_eth + ethereum_blobs_eth`
  - `profit_eth = fees_paid_eth - (ethereum_blobs_eth + celestia_blobs_eth + eigenda_blobs_eth + cost_l1_raw_eth + l1_settlement_custom_eth)`
- therefore vendor `profit_eth` subtracts at least one settlement-like component that vendor `rent_paid_eth` does not include
- Starknet has a separate `l1_settlement_custom_eth` adapter for proof costs
- the public Starknet economics mapping includes `updateState*` and the blob producer, but not the large batch-submission family that dominates the canonical excess
- the Taiko vendor pipeline explicitly filters known false positives such as `withdrawBond`

## Implication

The repo is currently comparing two different measurement objects:

1. canonical on-chain attributable L1 fee accounting
2. growthepie vendor economics `rent_paid`

`T050` cannot be made scientifically stable until the repo locks which one is release truth and what role the other plays.

## Required path forward

1. `T048` must lock the canonical rent contract and vendor benchmark policy in W0.
2. `T049` should expose rollup-day rent components so future mismatches can be resolved without another expensive replay loop.
3. `T050` should only be resumed after `T048`, and ideally after `T049`, so validation can distinguish:
   - integrity failures
   - key-coverage failures
   - benchmark divergences allowed or disallowed by the locked contract
