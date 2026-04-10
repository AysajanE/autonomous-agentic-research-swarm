# H052 — Starknet root cause: shared SHARP cost over-attribution (2026-04-10)

## Summary

The remaining Starknet `T050` benchmark failure is not a missing sender/selector issue. It is a methodology error in the canonical Starknet attribution model.

Canonical Starknet currently charges the full raw Ethereum tx fees of generic SHARP verifier-stack contracts to Starknet:

- `registerContinuousMemoryPage`
- `registerContinuousPageBatch`
- `verifyMerkle`
- `verifyFRI`
- `verifyProofAndRegister`

That is not scientifically valid as a Starknet-specific `rent_paid_eth` measure unless those costs are Starknet-exclusive. The available protocol evidence indicates they are shared / amortized SHARP costs instead.

## Evidence

- Refreshed vendor Starknet total: `2221.2945702734282 ETH`
- Refreshed canonical Starknet `state_updates_eth`: `2221.2945677904236 ETH`
- Difference: `2.483004664100008e-06 ETH`

- Refreshed canonical Starknet total: `15848.10965125149 ETH`
- Canonical excess over vendor: `13626.815080978062 ETH`
- Canonical `batch_submissions_eth + proof_submissions_eth`: `13626.815083461057 ETH`

So:

- `vendor Starknet rent ≈ canonical state_updates_eth`
- `canonical Starknet excess ≈ canonical batch_submissions_eth + proof_submissions_eth`

The post-repair validator in `wt-T050-final` reports the same result: Starknet is an explained methodology difference because its total delta matches `batch_submissions_eth + proof_submissions_eth` within floating noise.

## Why this matters

The local Starknet tracked tx universe includes SHARP-side addresses and methods, not just the Starknet Core `updateState*` contract.

From the local L2BEAT Starknet tracked transaction snapshot:

- `batchSubmissions`
  - `registerContinuousMemoryPage`
  - `registerContinuousPageBatch`
- `proofSubmissions`
  - `verifyMerkle`
  - `verifyFRI`
  - `verifyProofAndRegister`
- `stateUpdates`
  - `updateState`
  - `updateStateKzgDA`

That means the current generic tracked-tx model is measuring SHARP infrastructure costs directly as if they were rollup-exclusive Starknet costs.

## Official-source basis

- Starknet SHARP protocol docs describe SHARP as a shared prover / aggregator and say Starknet pays only its relative share of proof verification:
  - https://docs.starknet.io/learn/protocol/sharp
- Starknet cost docs describe an allocation model:
  - fixed cost per SHARP train
  - fixed cost per Starknet block
  - memory-page and state-update components
  - proof verification amortized across trains
  - https://community.starknet.io/t/starknet-costs-and-fees/113853
- Data-availability docs confirm the memory-page registry is part of the proof / public-memory machinery:
  - https://community.starknet.io/t/data-availability-with-eip4844/113065

## Vendor-side methodology evidence

growthepie’s Starknet economics mapping includes only `updateState*` and Starknet blob-producer surfaces for `rent_paid_eth`, while proofs are loaded separately as `l1_settlement_custom_eth`.

Relevant local files:

- `/tmp/gtp-dna/economics_da/economics_mapping.yml`
- `/tmp/gtp-backend/backend/src/adapters/adapter_starknet_proof.py`
- `/tmp/gtp-backend/backend/src/db_connector.py`

## Root cause

The Starknet blocker is that canonical is applying a raw tx-attribution method to a shared-settlement architecture that requires allocation.

This is why:

- the Starknet gap remained after all sender/selector repairs
- Taiko repair widened the aggregate without changing Starknet itself
- vendor Starknet aligns exactly with `state_updates_eth`
- the residual Starknet delta is exactly the SHARP-side tx families

## Required fix

Do not keep Starknet on the generic tracked-tx method for SHARP-side `batchSubmissions` / `proofSubmissions`.

Instead:

1. keep direct raw attribution for Starknet-exclusive `updateState*` and blob state-diff publication
2. replace SHARP-side raw Starknet attribution with a Starknet-specific allocated shared-settlement model
3. lock that model in W0 before changing the ETL
