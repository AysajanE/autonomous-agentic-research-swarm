## T049 blocker: parallel component identities need W0 clarification

Date: 2026-04-10
Role: Worker
Worktree: `/Users/aeziz-local/Research/wt-T049`
Branch: `T049_emit_rollup_day_rent_components`

### Root cause

`T049` currently requires one output surface, `daily_rollup_rent_components`, to include:

- tx-family components:
  - `batch_submissions_eth`
  - `proof_submissions_eth`
  - `state_updates_eth`
- fee-class components:
  - `blob_fee_burn_eth`
  - `execution_base_fee_burn_eth`
  - `execution_priority_fee_eth`

The task file and `contracts/data_dictionary.md` also state that the component columns must sum exactly to canonical `rent_paid_eth`.

That identity cannot hold as written. The tx-family columns are one full decomposition of canonical rent, and the fee-class columns are another full decomposition of the same canonical rent. Summing all component columns together would double-count the total.

### Why this blocks implementation

Implementing the file without clarifying the contract would force one of three scientifically invalid outcomes:

1. emit both decompositions and violate the stated identity rule
2. suppress one decomposition while pretending to satisfy the required coverage
3. redefine one set of columns to mean something other than the current contract text

All three would silently change metric semantics.

### Smallest clean unblocker

Approve a narrow W0 contract clarification:

- tx-family columns must sum to `rent_paid_eth`
- fee-class columns must separately sum to `rent_paid_eth`

After that clarification, `T049` can proceed cleanly by carrying subtype through the checkpoint/raw paths and emitting a single audit artifact with two explicit parallel identities.

### Current WIP state

Partial non-committed code work already in `src/etl/build_l1_rent_panel.py`:

- added `subtype` to `BlockscoutTx` and `BlobscanTx`
- bumped `PARTITION_CHECKPOINT_COMPAT_VERSION` to `4`
- added `COMPONENT_HEADERS`
- added `ROLLUP_SUBTYPE_TO_COMPONENT_FIELD`
- wired stored Blockscout page parsing to accept/validate subtype

No clean run was launched from this branch because the contract ambiguity was identified before the component emission logic was completed.
