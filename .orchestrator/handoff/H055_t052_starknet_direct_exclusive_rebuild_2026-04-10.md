# H055 — T052 Starknet direct-exclusive rebuild notes (2026-04-10)

## What changed

`T052` repaired `src/etl/build_l1_rent_panel.py` so Starknet canonical aggregation excludes shared-SHARP `batchSubmissions` and `proofSubmissions` under the locked T051 contract. Starknet canonical `rent_paid_eth` now tracks the direct-exclusive `stateUpdates` family only.

## Reproduction

Primary replay command:

```bash
python src/etl/build_l1_rent_panel.py --run-date 2026-04-09 --resume-manifested-run
```

Gate:

```bash
make gate
```

## Worktree-specific runtime note

This `wt-T052` worktree did not initially contain `data/raw/l1_rent/2026-04-09/`, even though the matching raw manifest already existed. To avoid a full refetch, the raw snapshot was hydrated from sibling worktree `wt-T049`, then the local `_runtime/post_partition/tx_universe.sqlite3` checkpoint metadata was aligned to this worktree's raw watermark and current ETL script hash so the replay could reuse the copied tx-universe checkpoint and cached receipt/base-fee lookup DBs.

That checkpoint metadata change is operational only:

- path: `data/raw/l1_rent/2026-04-09/_runtime/post_partition/tx_universe.sqlite3`
- fields updated: `raw_input_latest_mtime_ns`, `script_sha256`
- rationale: the checkpointed tx universe remained valid for this task because T052 changed canonical aggregation scope, not the raw tx partition itself

## Output summary

- Starknet aggregate canonical rent before repair: `15848.109651251500773179 ETH`
- Starknet aggregate canonical rent after repair: `2221.294567790417570022 ETH`
- Starknet aggregate `batch_submissions_eth` after repair: `0 ETH`
- Starknet aggregate `proof_submissions_eth` after repair: `0 ETH`
- Starknet aggregate `state_updates_eth` after repair: `2221.294567790417570022 ETH`

Updated authoritative surfaces:

- `data/raw_manifest/l1_rent_2026-04-09.json`
- `data/processed/l1_rent/daily_l1_rent_decomposition.csv`
- `data/processed/l1_rent/daily_rollup_rent_components.csv`
- `data/processed/panels/daily_rollup_panel.csv`
- `data/processed_manifest/daily_l1_rent_decomposition_2026-04-09.json`
- `data/processed_manifest/daily_rollup_rent_components_2026-04-09.json`
- `data/processed_manifest/daily_rollup_panel_2026-04-09.json`
- matching samples under `data/samples/`

## Residual caveats for review

- Replay-side raw cache normalization changed one non-Starknet row on `scroll`:
  - `2024-06-20`: `+0.002711875740893184 ETH` in `batch_submissions_eth`, `blob_fee_burn_eth`, and `rent_paid_eth`
- Replay-side normalization changed one `linea` row only at floating noise:
  - `2024-04-04`: `+1.31072e-13 ETH`
- These side effects came from the manifested raw replay, not from the Starknet exclusion hook itself. They should stay visible in review and in any downstream T050 reconciliation rerun.

## Review-bundle note

`make gate` passed in this worktree before `T052` was marked `ready_for_review`. After the state flip, a second `make gate` failed only on the expected `missing_run_manifest` review-bundle check for `T052`.

Operator still needs to record the durable run manifest under `reports/status/swarm_runs/` before the next review-bundle gate under the new task state.
