# H052 — Starknet root cause: shared SHARP cost over-attribution (2026-04-10)

## Executive conclusion

The remaining Starknet `T050` benchmark failure is **not** a missing sender/selector problem like Taiko. It is a **methodology error** in the canonical Starknet attribution model.

Canonical Starknet currently charges the **full raw Ethereum tx fees** of generic SHARP contracts to Starknet:

- `MemoryPageFactRegistry` page registrations (`registerContinuousMemoryPage`, `registerContinuousPageBatch`)
- `VerifyMerkle`
- `VerifyFRI`
- `VerifyProofAndRegister`

That is not scientifically valid as a Starknet-specific `rent_paid_eth` measure, because Starknet’s own protocol documentation says these SHARP costs are **shared and amortized**, not directly attributable 1:1 from raw on-chain tx totals.

## Local evidence

### 1. The vendor series is exactly the Starknet `state_updates_eth` component

From the refreshed `wt-T049` component surface:

- vendor Starknet total: `2221.2945702734282 ETH`
- canonical Starknet `state_updates_eth` total: `2221.2945677904236 ETH`
- difference: `2.483004664100008e-06 ETH`

This is effectively exact equality.

### 2. The canonical Starknet excess is exactly the SHARP-side components

From the same surface:

- canonical Starknet total: `15848.10965125149 ETH`
- canonical excess over vendor: `13626.815080978062 ETH`
- canonical `batch_submissions_eth + proof_submissions_eth`: `13626.815083461057 ETH`

Again, this matches to floating noise.

So the Starknet mismatch is mathematically:

`vendor Starknet rent ≈ state_updates_eth`

`canonical Starknet excess ≈ batch_submissions_eth + proof_submissions_eth`

### 3. The tracked Starknet non-state-update surfaces are generic SHARP contracts

The local L2BEAT tracked snapshot in
`data/raw/l1_rent/2026-04-09/l2beat/starknet/tracked_transactions.json`
contains:

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

The non-state-update contract surfaces in the snapshot are:

- `0xe583bcde0160b637330b27a3ea1f3c02ba2ec460`
- `0xfd14567eaf9ba941cb8c8a94eec14831ca7fd1b4`
- `0x32a91ff604ab2adcd832e91d68b2f3f25358fdad`
- `0x634dcf4f1421fc4d95a968a559a450ad0245804c`
- `0x30efaaa99f8efe310d9fdc83072e2a04c093d400`
- `0xdef8a3b280a54ee7ed4f72e1c7d6098ad8df44fb`
- `0x47312450b3ac8b5b8e247a6bb6d523e7605bdb60`

These are SHARP verifier-stack contracts, not the Starknet Core contract.

## Official protocol evidence

### 1. SHARP is shared and Starknet pays only a relative share of onchain verification

Official Starknet SHARP documentation:

- https://docs.starknet.io/learn/protocol/sharp

Key point: SHARP aggregates multiple Cairo programs, and for Starknet this means sending a single proof for multiple blocks and paying onchain verification cost only according to Starknet’s **relative share** in that proof.

### 2. Starknet’s own cost model is not “sum all SHARP tx fees”

Official Starknet costs post:

- https://community.starknet.io/t/starknet-costs-and-fees/113853

The Starknet team defines:

- **fixed per SHARP train**: about `6M gas` per train
  - `Verify FRI`
  - `Verify Merkle`
  - `Verify Proof and Register`
- **fixed per Starknet block**: about `215K gas` per block
  - `23K` gas fact registration
  - `56K` gas for **SHARP memory page 0**
  - `136K` gas for **State Update**

This is an **allocation model**:

- proof verification costs are per-train and must be amortized by train size
- only **memory page 0** is called out as the fixed per-block memory-page cost
- the cost model is not “charge Starknet the full raw gas of every MemoryPageFactRegistry / Merkle / FRI / VerifyProof tx”

### 3. Data availability docs confirm the memory-page registry is generic proof/public-memory machinery

- https://community.starknet.io/t/data-availability-with-eip4844/113065

The current mechanism explicitly sends memory pages to `MemoryPageFactRegistry` as part of the public-memory proof flow. This is proof/data-availability machinery, not a simple Starknet-owned settlement inbox.

### 4. External contract identity matches the local tracked surfaces

- `MemoryPageFactRegistry`: https://www.codeslaw.app/contracts/ethereum/0xe583bcde0160b637330b27a3ea1f3c02ba2ec460
- `SHARP Verifier`: https://www.codeslaw.app/contracts/ethereum/0x47312450b3ac8b5b8e247a6bb6d523e7605bdb60

These addresses and methods match the local Starknet tracked-call universe.

## Why earlier “Starknet was resolved” was misleading

What was resolved earlier was **diagnosis**, not **scientific closure**.

The component audit successfully proved that the benchmark gap was not random:

- vendor Starknet tracks `state_updates_eth`
- canonical excess tracks `batch_submissions_eth + proof_submissions_eth`

That explained the mismatch, but it did **not** establish that canonical’s Starknet methodology was correct.

The new investigation shows it is not enough to say “vendor excludes proofs.” The deeper issue is:

**canonical Starknet is using a raw-tx attribution method on a shared SHARP architecture that requires allocation.**

## Once-and-for-all fix

### Do not keep Starknet on the generic tracked-tx method for `batchSubmissions` / `proofSubmissions`

That method is acceptable for rollups whose settlement costs are paid on rollup-specific contracts.
It is not acceptable for Starknet’s shared SHARP verifier stack.

### Replace Starknet SHARP attribution with a Starknet-specific allocation model

For Starknet:

1. Keep direct raw on-chain attribution for:
   - `updateState`
   - `updateStateKzgDA`
   - blob state-diff publication

2. Stop charging full raw SHARP verifier-stack tx fees directly to Starknet for:
   - `registerContinuousMemoryPage`
   - `registerContinuousPageBatch`
   - `verifyMerkle`
   - `verifyFRI`
   - `verifyProofAndRegister`

3. Introduce a Starknet-specific allocated settlement component built from official sources:
   - preferred: Starknet official SHARP pricing / allocation reports
   - fallback: official fixed-cost formulas plus train-size allocation, if the pricing report is unavailable historically

4. Split the output contractually if needed:
   - direct exclusive L1 settlement / DA cost
   - allocated shared SHARP settlement cost

5. Update benchmark logic so growthepie `rent_paid_eth` is compared against the Starknet direct-exclusive surface, while the allocated shared SHARP settlement remains visible in components / costs / caveats.

## Why this is the right root cause

This diagnosis simultaneously explains all observed facts:

- why the Starknet delta is huge
- why it begins when SHARP-side tracked calls appear
- why vendor rent matches only `state_updates_eth`
- why the delta is exactly the SHARP-side tx families
- why the issue persisted after all sender/selector repair work
- why Taiko repair widened the aggregate without changing Starknet itself

The Starknet problem is not a missing contract window. It is that the current canonical method is **measuring the wrong Starknet-specific object** on a shared settlement architecture.
