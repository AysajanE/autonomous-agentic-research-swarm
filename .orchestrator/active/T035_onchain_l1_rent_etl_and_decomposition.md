---
task_id: T035
title: "On-chain L1 rent extraction, decomposition, and canonical panel build"
workstream: W2
task_kind: etl
allow_network: true
role: Worker
priority: high
dependencies:
  - "T025"
  - "T030"
integration_ready_dependencies: []
requires_tools:
  - "python"
  - "git"
requires_env: []
allowed_paths:
  - "src/etl/build_l1_rent_panel.py"
  - "data/raw/l1_rent/"
  - "data/raw_manifest/l1_rent_"
  - "data/processed/l1_rent/"
  - "data/processed/panels/"
  - "data/processed_manifest/daily_l1_rent_decomposition_"
  - "data/processed_manifest/daily_rollup_panel_"
  - "data/samples/l1_rent/"
  - "data/samples/panels/"
disallowed_paths:
  - "docs/protocol.md"
  - "contracts/"
  - "registry/"
  - "src/analysis/"
  - "src/validation/"
outputs:
  - "src/etl/build_l1_rent_panel.py"
  - "data/raw/l1_rent/YYYY-MM-DD/..."
  - "data/raw_manifest/l1_rent_YYYY-MM-DD.json"
  - "data/processed/l1_rent/daily_l1_rent_decomposition.csv"
  - "data/processed/panels/daily_rollup_panel.csv"
  - "data/processed_manifest/daily_l1_rent_decomposition_YYYY-MM-DD.json"
  - "data/processed_manifest/daily_rollup_panel_YYYY-MM-DD.json"
  - "data/samples/l1_rent/daily_l1_rent_decomposition_sample.csv"
  - "data/samples/panels/daily_rollup_panel_sample.csv"
gates:
  - "make gate"
stop_conditions:
  - "Need credentials"
  - "On-chain attribution is ambiguous"
  - "Required growthepie manifest or registry inputs are missing"
---

# Task T035 — On-chain L1 rent extraction, decomposition, and canonical panel build

## Context

This task creates the authoritative rent path for release. It turns raw on-chain L1 inputs into the daily L1 rent decomposition and the canonical `daily_rollup_panel` that combines registry-backed rollup attribution, authoritative `rent_paid_eth`, and growthepie `l2_fees_eth`.

## Assignment

- Workstream: W2 Data: on-chain
- Assigned role: Worker
- Suggested branch/worktree name: `T035_l1_rent_panel`
- Allowed paths: one on-chain ETL script plus raw/manifests/processed/sample paths for the L1 rent and canonical panel surfaces
- Stop conditions: block with `@human` instead of guessing attribution, source priority, or chain-specific rules

## Inputs

- `docs/protocol.md`
- `contracts/data_dictionary.md`
- `contracts/decisions.md`
- `contracts/schemas/panel_schema_decomp_v1.yaml`
- `contracts/schemas/panel_schema_str_v1.yaml`
- `registry/rollup_registry_v1.csv`
- `data/raw_manifest/growthepie_<YYYY-MM-DD>.json`
- `data/processed_manifest/vendor_daily_rollup_panel_<YYYY-MM-DD>.json`

## Outputs

- ETL code: `src/etl/build_l1_rent_panel.py`
- Append-only raw snapshots: `data/raw/l1_rent/<YYYY-MM-DD>/...`
- Raw provenance: `data/raw_manifest/l1_rent_<YYYY-MM-DD>.json`
- Processed decomposition: `data/processed/l1_rent/daily_l1_rent_decomposition.csv`
- Canonical panel: `data/processed/panels/daily_rollup_panel.csv`
- Processed provenance:
  - `data/processed_manifest/daily_l1_rent_decomposition_<YYYY-MM-DD>.json`
  - `data/processed_manifest/daily_rollup_panel_<YYYY-MM-DD>.json`
- Tracked samples:
  - `data/samples/l1_rent/daily_l1_rent_decomposition_sample.csv`
  - `data/samples/panels/daily_rollup_panel_sample.csv`

## Success Criteria

- [ ] Raw on-chain pulls are append-only and have a matching raw manifest
- [ ] `daily_l1_rent_decomposition.csv` is produced deterministically from local raw inputs
- [ ] `daily_rollup_panel.csv` uses growthepie `l2_fees_eth` and authoritative on-chain `rent_paid_eth`
- [ ] The canonical panel manifest cites both the growthepie raw manifest and the on-chain raw manifest
- [ ] Tracked samples exist for the decomposition and canonical panel surfaces
- [ ] `make gate` passes

## Review Bundle Requirements

- [ ] A durable run manifest exists under `reports/status/swarm_runs/`
- [ ] Any attribution or lineage caveat needed by T050 is captured in `.orchestrator/handoff/`
- [ ] Judge review is recorded under `reports/status/reviews/`

## Validation / Commands

- `make gate`
- `python src/etl/build_l1_rent_panel.py --run-date YYYY-MM-DD`
- `python scripts/make_raw_manifest.py l1_rent data/raw/l1_rent/YYYY-MM-DD --as-of YYYY-MM-DD -- python src/etl/build_l1_rent_panel.py --run-date YYYY-MM-DD`

## Status
- State: active
- Last updated: 2026-04-04
## Notes / Decisions

- 2026-03-29: New v1 task added to make the on-chain rent path authoritative before metrics, validation, or release analysis.
- 2026-04-01: Operator activated isolated worktree `/Users/aeziz-local/Research/wt-T035` on branch `T035_l1_rent_panel` to execute the authoritative on-chain rent path end to end and supervise it through Judge review.
- 2026-04-01: @human Runtime blocked: executor_timeout, path_ownership_violation, missing_outputs, missing_required_manifests. Run manifest: reports/status/swarm_runs/T035_20260401T135106Z.json. ownership=.orchestrator/backlog/T025_populate_rollup_registry.md[committed]=orchestrator_write_forbidden; .orchestrator/backlog/T030_growthepie_etl_snapshot_and_vendor_panel.md[committed]=orchestrator_write_forbidden; .orchestrator/backlog/T035_onchain_l1_rent_etl_and_decomposition.md[committed]=orchestrator_write_forbidden; .orchestrator/done/T025_populate_rollup_registry.md[committed]=orchestrator_write_forbidden; .orchestrator/done/T030_growthepie_etl_snapshot_and_vendor_panel.md[committed]=orchestrator_write_forbidden; data/AGENTS.md[committed]=outside_allowed_paths; scripts/quality_gates.py[committed]=outside_allowed_paths; src/etl/AGENTS.md[committed]=outside_allowed_paths; tests/test_quality_gates_processed_manifests.py[committed]=outside_allowed_paths outputs=data/raw_manifest/l1_rent_YYYY-MM-DD.json=missing_file; data/processed/l1_rent/daily_l1_rent_decomposition.csv=missing_file; data/processed/panels/daily_rollup_panel.csv=missing_file; data/processed_manifest/daily_l1_rent_decomposition_YYYY-MM-DD.json=missing_file; data/processed_manifest/daily_rollup_panel_YYYY-MM-DD.json=missing_file; data/samples/l1_rent/daily_l1_rent_decomposition_sample.csv=missing_file; data/samples/panels/daily_rollup_panel_sample.csv=missing_file manifests=missing_raw_manifest_file,missing_processed_manifest_file
- 2026-04-01: Updated `src/etl/build_l1_rent_panel.py` to make Blockscout resume append-only and faster against the existing partial snapshot. The script now reads cached `page_size`, can resume from an older 250-row cache into new 1000-row requests without skipping rows, and `python -m py_compile src/etl/build_l1_rent_panel.py` passed.
- 2026-04-01: Blocked on required registry attribution inputs before any authoritative panel can be produced. `python src/etl/build_l1_rent_panel.py --run-date 2026-04-01` now fails fast with `required registry attribution inputs are missing for pre-Dencun rollups: scroll[active_pre_dencun=2023-10-17..2024-03-12, growthepie_pre_dencun_rows=148]`.
- 2026-04-01: Reproduction commands: `python -m py_compile src/etl/build_l1_rent_panel.py`; `python src/etl/build_l1_rent_panel.py --run-date 2026-04-01`. Outcome: compile passed; ETL exited with the explicit missing-registry-input blocker above. `make gate` was not run because the task stop condition triggered before declared outputs/manifests could be produced.
- 2026-04-01: Files changed: `src/etl/build_l1_rent_panel.py`. Files created: none. Local-only partial raw snapshot remains under `data/raw/l1_rent/2026-04-01/` from the resumed fetch attempts, but no raw/processed manifests or canonical outputs were written for this blocked run.
- 2026-04-01: @human Should Scroll's pre-Dencun window (`2023-10-17` through `2024-03-12`) be attributed via additional registry evidence (for example submission contract-based hooks), or should Scroll be explicitly excluded from the canonical panel for that interval? The current registry has `batcher_addresses_json=[]`, so proceeding would otherwise silently omit 148 growthepie-covered rollup-days from the authoritative panel.
- 2026-04-01: @human Runtime blocked: path_ownership_violation, missing_outputs, missing_required_manifests. Run manifest: reports/status/swarm_runs/T035_20260401T152122Z.json. ownership=.orchestrator/backlog/T035_onchain_l1_rent_etl_and_decomposition.md[committed]=orchestrator_write_forbidden; reports/status/swarm_runs/T035_20260401T135106Z.json[committed]=outside_allowed_paths outputs=data/raw_manifest/l1_rent_YYYY-MM-DD.json=missing_file; data/processed/l1_rent/daily_l1_rent_decomposition.csv=missing_file; data/processed/panels/daily_rollup_panel.csv=missing_file; data/processed_manifest/daily_l1_rent_decomposition_YYYY-MM-DD.json=missing_file; data/processed_manifest/daily_rollup_panel_YYYY-MM-DD.json=missing_file; data/samples/l1_rent/daily_l1_rent_decomposition_sample.csv=missing_file; data/samples/panels/daily_rollup_panel_sample.csv=missing_file manifests=missing_raw_manifest_file,missing_processed_manifest_file
- 2026-04-01: Repaired `src/etl/build_l1_rent_panel.py` so pre-Dencun rollups without `batcher_addresses_json` can use registry-backed `evidence_url` L2BEAT `trackedTransactions` function-call hooks. The ETL now snapshots `data/raw/l1_rent/2026-04-01/l2beat/scroll/tracked_transactions.json`, validates Scroll against pre-Dencun `batchSubmissions` / `stateUpdates` selectors on `0xa13BAF47339d63B743e7Da8741db5456DAc1E556`, and can fetch pre-Dencun Blockscout pages by `to` address plus selector when it reaches that part of the run.
- 2026-04-01: Added Blockscout resilience beyond the original repair: cached page metadata now records `result_count`, selector-filtered pages can resume safely, and retry exhaustion on later pages now splits the remaining timestamp window instead of hard-failing when the timeout happens after page 1. `python -m py_compile src/etl/build_l1_rent_panel.py` passed after both changesets.
- 2026-04-01: Reproduction commands run in this worktree: `python -m py_compile src/etl/build_l1_rent_panel.py`; `python src/etl/build_l1_rent_panel.py --run-date 2026-04-01`; `python src/etl/build_l1_rent_panel.py --run-date 2026-04-01 --blockscout-page-size 2000`. Outcome: compile passed; the ETL is now active and resumable, but the full historical Blockscout backfill has not completed yet, so no raw manifest, processed outputs, processed manifests, samples, or `make gate` result exist yet.
- 2026-04-01: Safe checkpoint after manual stop: append-only raw snapshot `data/raw/l1_rent/2026-04-01/` contains 1108 files, including the cached Scroll L2BEAT evidence above, but has not yet reached `blockscout/txlist_to/` or any processed surfaces. The resumed run was still traversing dense pre-Dencun Arbitrum sender history (latest observed continuation reached December 2023) when stopped intentionally for handoff.
- 2026-04-01: Resume command from the current checkpoint: `python src/etl/build_l1_rent_panel.py --run-date 2026-04-01 --blockscout-page-size 2000`. Assumption: the existing partial snapshot under `data/raw/l1_rent/2026-04-01/` remains in place so the script can keep resuming append-only instead of restarting from scratch.
- 2026-04-01: @human Runtime blocked: missing_outputs, missing_required_manifests. Run manifest: reports/status/swarm_runs/T035_20260401T155307Z.json. outputs=data/raw_manifest/l1_rent_YYYY-MM-DD.json=missing_file; data/processed/l1_rent/daily_l1_rent_decomposition.csv=missing_file; data/processed/panels/daily_rollup_panel.csv=missing_file; data/processed_manifest/daily_l1_rent_decomposition_YYYY-MM-DD.json=missing_file; data/processed_manifest/daily_rollup_panel_YYYY-MM-DD.json=missing_file; data/samples/l1_rent/daily_l1_rent_decomposition_sample.csv=missing_file; data/samples/panels/daily_rollup_panel_sample.csv=missing_file manifests=missing_raw_manifest_file,missing_processed_manifest_file
- 2026-04-01: Resumed the append-only checkpoint with `python src/etl/build_l1_rent_panel.py --run-date 2026-04-01 --blockscout-page-size 2000 --blobscan-page-size 2000` and kept the same raw snapshot in place instead of restarting. The in-flight run advanced past the prior Arbitrum checkpoint, completed Starknet's pre-Dencun Blockscout history plus post-Dencun Blobscan fetches, and has now moved on to Optimism pre-Dencun Blockscout history.
- 2026-04-01: Repaired `src/etl/build_l1_rent_panel.py` again to make Blobscan page-1 instability less destructive under the larger resume page size. Added bounded cooldown retries before fallback/splitting (`BLOBSCAN_INSTABILITY_RETRY_DELAY_SECONDS=30`, `BLOBSCAN_INSTABILITY_RETRY_ROUNDS=2`) so transient 502/503/timeouts do not immediately explode into pathological recursive window splits. `python -m py_compile src/etl/build_l1_rent_panel.py` passed after this change.
- 2026-04-01: Current live checkpoint while still `active`: raw snapshot `data/raw/l1_rent/2026-04-01/` has reached 2213 files. Observed completed/near-completed branches include Arbitrum Blockscout/Blobscan, Starknet Blockscout (`319` pages for `0x2c169...`, `27` pages for `0xf6b0...`) and Starknet Blobscan (`55` pages for `0x2c169...`, `25` pages for `0xf6b0...`). Latest frontier is Optimism Blockscout sender history (`0x688724...`) with dense-window continuation logs in May-July 2022. No raw manifest, processed outputs, processed manifests, samples, or `make gate` result exist yet because the ETL has not reached its terminal write phase.
- 2026-04-01: Later active checkpoint from the same append-only run: raw snapshot `data/raw/l1_rent/2026-04-01/` reached 2817 files. Optimism pre-Dencun Blockscout sender history completed through the Dencun cutoff at `2024-03-12` with `549` cached txlist pages, and the run moved into Optimism Blobscan from `2024-03-13` onward.
- 2026-04-01: The current frontier is Optimism post-Dencun Blobscan (`0x688724...`). Cached Blobscan pages for Optimism have started landing (`5` files observed so far), and repeated page-1/page-2 Blobscan stalls in March-June 2024 are being handled by the new cooldown + page-size fallback logic rather than hard-failing the ETL. No stop condition has triggered, but the ETL has not yet reached manifests, processed outputs, samples, or `make gate`.
- 2026-04-02: @human Runtime blocked: missing_outputs, missing_required_manifests. Run manifest: reports/status/swarm_runs/T035_20260401T163917Z.json. outputs=data/raw_manifest/l1_rent_YYYY-MM-DD.json=missing_file; data/processed/l1_rent/daily_l1_rent_decomposition.csv=missing_file; data/processed/panels/daily_rollup_panel.csv=missing_file; data/processed_manifest/daily_l1_rent_decomposition_YYYY-MM-DD.json=missing_file; data/processed_manifest/daily_rollup_panel_YYYY-MM-DD.json=missing_file; data/samples/l1_rent/daily_l1_rent_decomposition_sample.csv=missing_file; data/samples/panels/daily_rollup_panel_sample.csv=missing_file manifests=missing_raw_manifest_file,missing_processed_manifest_file
- 2026-04-02: Reclassified the task to `active` and continued from the existing append-only checkpoint instead of restarting. Updated `src/etl/build_l1_rent_panel.py` so the eventual fetch/processed manifests record the actual CLI flags used when non-default retry, timeout, or page-size parameters are supplied. `python -m py_compile src/etl/build_l1_rent_panel.py` passed after the change.
- 2026-04-02: Resume commands exercised from the same `data/raw/l1_rent/2026-04-01/` checkpoint: `python -u src/etl/build_l1_rent_panel.py --run-date 2026-04-01 --blockscout-page-size 2000 --blobscan-page-size 2000`; `python -u src/etl/build_l1_rent_panel.py --run-date 2026-04-01 --blockscout-page-size 2000 --blobscan-page-size 2000 --timeout-seconds 20 --retries 2`; exploratory tighter resume `python -u src/etl/build_l1_rent_panel.py --run-date 2026-04-01 --blockscout-page-size 2000 --blobscan-page-size 2000 --timeout-seconds 15 --retries 1`. Outcome: the 20s/2-retry configuration materially reduced dead time versus the original 45s/4-retry default while preserving the same append-only snapshot; the 15s/1-retry run was started only briefly and interrupted during replay before any durable improvement over 20s/2 could be established.
- 2026-04-02: Current live checkpoint after the resumed sweeps: `data/raw/l1_rent/2026-04-01/` has grown from the earlier 2824-file checkpoint to 3573 files. The resumed ETL cleared the long Optimism Blobscan backlog, advanced through additional rollups beyond the original checkpoint, and the raw tree now includes active `blockscout/` or `blobscan/` branches for `arbitrum`, `starknet`, `optimism`, `linea`, `zksync_era`, and `base`. The latest observed foreground frontier reached Base's post-Dencun Blobscan slice starting at `2024-03-13`.
- 2026-04-02: The run has not reached terminal write surfaces yet. `data/raw/l1_rent/2026-04-01/blockscout/txlist_to/`, `data/raw/l1_rent/2026-04-01/rpc/receipts/`, and `data/raw/l1_rent/2026-04-01/rpc/blocks/` are still absent, so no raw manifest, processed CSVs, processed manifests, samples, or `make gate` result exist yet. Recommended next resume command from this checkpoint: `python -u src/etl/build_l1_rent_panel.py --run-date 2026-04-01 --blockscout-page-size 2000 --blobscan-page-size 2000 --timeout-seconds 20 --retries 2`.
- 2026-04-02: Attempted to leave a detached background resume inside this worktree using `nohup` and then `setsid`, but each detached process exited immediately before any durable log output was written. No independent background ETL process was left running after those attempts, so the task remains `active` but currently requires an explicit resume command in a later session.
- 2026-04-02: Operator supervision note: `reports/status/swarm_runs/T035_20260401T163917Z.json` was a false blocked projection caused by a premature Worker exit while the ETL was still mid-run. Relaunched from `/Users/aeziz-local/Research/wt-T035` at approximately `2026-04-02 07:27` America/Toronto with the existing append-only checkpoint preserved. The replacement Worker resumed `python src/etl/build_l1_rent_panel.py --run-date 2026-04-01 --blockscout-page-size 2000 --blobscan-page-size 2000`, and active progress resumed on Arbitrum Blobscan recovery pages (`2025-09-01__2025-09-30_page-0004.json` through `page-0006.json` observed by `2026-04-02 07:31:55`). Operational state is `active`, not blocked.
- 2026-04-02: Repaired a latent terminal-phase ETL bug before the raw sweep could reach manifest writing. `build_processed_manifest(...)` had been reading `args` out of scope, which would have crashed after writing the raw manifest and CSV outputs; updated the helper to accept the concrete command string from `main(...)` instead. `python -m py_compile src/etl/build_l1_rent_panel.py` passed after the fix.
- 2026-04-02: Tightened Blobscan transport recovery inside `src/etl/build_l1_rent_panel.py` so persistent page-1 instability no longer idles for repeated 30s sleeps or gets trapped above a `500`-row floor. The ETL now uses `BLOBSCAN_INSTABILITY_RETRY_DELAY_SECONDS=10`, `BLOBSCAN_INSTABILITY_RETRY_ROUNDS=1`, and `BLOBSCAN_MIN_PAGE_SIZE=100`, allowing `1000 -> 500 -> 250 -> 125 -> 100` fallback before time-splitting the window. `python -m py_compile src/etl/build_l1_rent_panel.py` passed after the change.
- 2026-04-02: Direct Blobscan probes against the failing October 2025 Arbitrum interval confirmed the issue was transport instability rather than missing data: both `from=` and `rollups=` queries returned payloads on ad hoc requests, and smaller page sizes (`500`, `250`, `125`, `100`) were all accepted on demand. Based on that evidence, the Worker kept the scientific attribution logic unchanged and repaired only the page-size/cooldown behavior within the allowed ETL path.
- 2026-04-02: Current live checkpoint from the same append-only raw snapshot: `data/raw/l1_rent/2026-04-01/` has grown to 3958 files. The repaired run resumed with `python -u src/etl/build_l1_rent_panel.py --run-date 2026-04-01 --blockscout-page-size 2000 --blobscan-page-size 1000 --timeout-seconds 20 --retries 2`, pushed the Optimism Blobscan branch to 619 cached files with windows extending through `2026-03`, and then advanced the active frontier onward into Base pre-Dencun Blockscout / post-Dencun Blobscan recovery. No terminal outputs exist yet: `data/raw_manifest/l1_rent_2026-04-01.json`, processed CSVs, processed manifests, samples, and `make gate` remain pending until the ETL reaches its terminal write phase.
- 2026-04-02: Practical runtime handoff after continued progress: the append-only raw snapshot reached 4255 files, including 297 cached Base Blobscan pages, but the ETL was still far from the final receipts/base-fee/output phase. Because repeated `nohup`/`setsid` backgrounds died immediately in this environment, the Worker moved the same resume command into a detached `tmux` session instead. Verified live handoff details: session `t035_l1_rent`; command `python -u src/etl/build_l1_rent_panel.py --run-date 2026-04-01 --blockscout-page-size 2000 --blobscan-page-size 1000 --timeout-seconds 20 --retries 2`; log `data/raw/l1_rent/2026-04-01/resume_20260402T200330Z_tmux.log`; `tmux list-panes -t t035_l1_rent -F '#{pane_pid} #{pane_current_command} #{pane_dead}'` returned a live `python3.13` pane; the log grew from 11 KB to 24 KB during the verification soak. Task state remains `active`; no raw manifest, processed outputs, processed manifests, sample CSVs, or `make gate` result exist yet.
- 2026-04-02: Continued supervising the same detached `tmux` run instead of starting a second ETL process. The append-only raw snapshot has now grown to 4335 files, and the live frontier is still advancing through cached-page-size backfill gaps rather than idling: recent new files landed under `data/raw/l1_rent/2026-04-01/blobscan/optimism/0x6887246668a3b87f54deb3b94ba47a6f63f32985/` for split `2024-09-01..2024-09-23` windows while older Base and Arbitrum Blobscan branches remained intact. The run is still pre-terminal: `data/raw/l1_rent/2026-04-01/blockscout/txlist_to/`, `data/raw/l1_rent/2026-04-01/rpc/receipts/`, and `data/raw/l1_rent/2026-04-01/rpc/blocks/` are absent, so no raw manifest, processed CSVs, processed manifests, sample CSVs, or `make gate` result exist yet.
- 2026-04-02: Added a forward repair to `src/etl/build_l1_rent_panel.py` for the repeated Blockscout `offset=2000` month-window page-1 timeouts seen during the active resume. On future resume/restart, `fetch_blockscout_tx_window(...)` now degrades page size before recursive time-splitting (`2000 -> 1000 -> 500 -> 250`, floor `BLOCKSCOUT_MIN_PAGE_SIZE=250`) instead of only bisecting time windows. The live `tmux` process was left running on the old code to preserve in-flight work; `python -m py_compile src/etl/build_l1_rent_panel.py` passed after the edit so the improved path is ready if the detached process later needs to be restarted from the same checkpoint.
- 2026-04-02: @human Runtime blocked: missing_outputs, missing_required_manifests. Run manifest: reports/status/swarm_runs/T035_20260402T112730Z.json. outputs=data/raw_manifest/l1_rent_YYYY-MM-DD.json=missing_file; data/processed/l1_rent/daily_l1_rent_decomposition.csv=missing_file; data/processed/panels/daily_rollup_panel.csv=missing_file; data/processed_manifest/daily_l1_rent_decomposition_YYYY-MM-DD.json=missing_file; data/processed_manifest/daily_rollup_panel_YYYY-MM-DD.json=missing_file; data/samples/l1_rent/daily_l1_rent_decomposition_sample.csv=missing_file; data/samples/panels/daily_rollup_panel_sample.csv=missing_file manifests=missing_raw_manifest_file,missing_processed_manifest_file
- 2026-04-03: Operator restored `State: active` on operational grounds after verifying that the detached `tmux` ETL never stopped. At `2026-04-03 07:10 EDT`, session `t035_l1_rent` still held live PID `77650` running `python -u src/etl/build_l1_rent_panel.py --run-date 2026-04-01 --blockscout-page-size 2000 --blobscan-page-size 1000 --timeout-seconds 20 --retries 2`, the append-only raw snapshot had grown from 5886 files at the last April 2 evening checkpoint to 14157 files, and the newest writes were landing under `data/raw/l1_rent/2026-04-01/blobscan/base/0x5050f69a9786f081509234f1a7f4684b5e5b76c9/` for split January 2026 windows. The `blocked` state from the prior swarm run manifest is therefore stale relative to live execution; the task remains pre-terminal because `data/raw_manifest/l1_rent_2026-04-01.json`, processed CSVs/manifests/samples, `data/raw/l1_rent/2026-04-01/blockscout/txlist_to/`, and `data/raw/l1_rent/2026-04-01/rpc/{receipts,blocks}/` are still absent.
- 2026-04-03: Repaired `src/etl/build_l1_rent_panel.py` so Blobscan canonicalizes the provider alias `world` to the repo slug `worldchain` while preserving the existing ambiguity guard for every other Blobscan rollup mismatch. The same patch also maps canonical repo slug `worldchain` back to Blobscan query filter `world` when the ETL uses `rollups=` filters. Validation run: `python -m py_compile src/etl/build_l1_rent_panel.py` passed; a direct module check confirmed `canonicalize_blobscan_rollup_id('world') == 'worldchain'`, `blobscan_rollup_filter_value('worldchain') == 'world'`, and `normalize_blobscan_tx(...)` now returns `rollup_id='worldchain'` for Blobscan rows labeled `world`.
- 2026-04-03: Resumed the append-only ETL from the existing `data/raw/l1_rent/2026-04-01/` checkpoint with the requested stable flags in detached `tmux` session `t035_l1_rent`: `python -u src/etl/build_l1_rent_panel.py --run-date 2026-04-01 --blockscout-page-size 2000 --blobscan-page-size 1000 --timeout-seconds 20 --retries 2`. New resume log: `data/raw/l1_rent/2026-04-01/resume_20260403T145736Z_world_alias.log`. The raw tree grew from 15898 files before restart to 15900 files during the verification soak, including new append-only Optimism Blockscout cache file `data/raw/l1_rent/2026-04-01/blockscout/txlist/optimism/0x6887246668a3b87f54deb3b94ba47a6f63f32985/2024-01-01__2024-01-31_page-0001.json`, showing the repaired run is writing new raw pages instead of failing at the old Worldchain mismatch.
- 2026-04-03: Verified the Worldchain alias repair on-disk by running the same ETL helper path against the preserved snapshot for Worldchain's configured batcher `0xdbbe3d8c2d2b22a2611c5a94a9a12c2fcd49eb29` over `2026-03-01` through `2026-04-01` with the same Blobscan page size / retry / timeout settings. That append-only verification created `data/raw/l1_rent/2026-04-01/blobscan/worldchain/0xdbbe3d8c2d2b22a2611c5a94a9a12c2fcd49eb29/` with 9 cached files; the first page `2026-03-01__2026-03-31_page-0001.json` stores both top-level `rollup_id` and transaction `rollup_id` as `worldchain` while the recorded Blobscan URL still queries by `from=0xdbbe3d8c2d2b22a2611c5a94a9a12c2fcd49eb29`. The task remains `active` because the long historical ETL has not yet reached raw manifest / processed output / sample / `make gate` surfaces; `make gate` was therefore not run in this repair pass.
- 2026-04-03: @human Runtime blocked: missing_outputs, missing_required_manifests. Run manifest: reports/status/swarm_runs/T035_20260403T145543Z.json. outputs=data/raw_manifest/l1_rent_YYYY-MM-DD.json=missing_file; data/processed/l1_rent/daily_l1_rent_decomposition.csv=missing_file; data/processed/panels/daily_rollup_panel.csv=missing_file; data/processed_manifest/daily_l1_rent_decomposition_YYYY-MM-DD.json=missing_file; data/processed_manifest/daily_rollup_panel_YYYY-MM-DD.json=missing_file; data/samples/l1_rent/daily_l1_rent_decomposition_sample.csv=missing_file; data/samples/panels/daily_rollup_panel_sample.csv=missing_file manifests=missing_raw_manifest_file,missing_processed_manifest_file
- 2026-04-04: Operator investigated the new stop in `data/raw/l1_rent/2026-04-01/resume_20260403T145736Z_world_alias.log` and confirmed a real source-side RPC contract change rather than another attribution failure. The preserved run had advanced to later blob-only chains (`lisk`, `ink`, `soneium`, `unichain`) and then stopped at the receipt/base-fee enrichment boundary with HTTP `413` from `https://eth.blockscout.com/api/eth-rpc`: `"Payload Too Large. Max batch size is 5"`. Direct live probes against the same public endpoint reproduced the behavior: 5-call JSON-RPC batches succeed, 6-call batches fail with the same `413`, while the ETL still defaulted to `RPC_BATCH_SIZE = 100`.
- 2026-04-04: Repaired `src/etl/build_l1_rent_panel.py` to match the live provider contract instead of repeatedly restarting into the same stop: set default `RPC_BATCH_SIZE = 5`, reject non-positive `--rpc-batch-size`, and clamp larger requested batch sizes down to 5 with an explicit log message. Validation: `python -m py_compile src/etl/build_l1_rent_panel.py` passed after the change. Resume command launched from the preserved append-only checkpoint in detached `tmux` session `t035_l1_rent`: `python -u src/etl/build_l1_rent_panel.py --run-date 2026-04-01 --blockscout-page-size 2000 --blobscan-page-size 1000 --rpc-batch-size 5 --timeout-seconds 20 --retries 2`. New live log: `data/raw/l1_rent/2026-04-01/resume_20260404T111627Z_rpc5.log`. Operational state is `active`: the ETL is running again, but it is still re-walking cached dense Optimism Blockscout windows before it reaches the receipt/base-fee phase, so `data/raw/l1_rent/2026-04-01/blockscout/{receipts,block_base_fees}/`, the raw manifest, processed outputs/manifests, sample CSVs, and `make gate` remain pending.
- 2026-04-04: Operator investigated the next stop in `data/raw/l1_rent/2026-04-01/resume_20260404T111627Z_rpc5.log` and isolated it to Blobscan terminal-window instability on Base rather than missing chain coverage. The run preserved progress through `data/raw/l1_rent/2026-04-01/blobscan/base/.../2024-09-03T011333Z__2024-09-03T012812Z_page-0001.json`, then recursively split a later Base window down to the exact one-second slice `2024-09-03T01:28:13Z..2024-09-03T01:28:13Z` and still hit repeated HTTP `503` at `ps=100`, causing a fatal `source instability or breaking API changes` stop. Direct live probes against the exact failing URL immediately after the stop returned `200 {"transactions":[]}` for both `from=` and `rollups=base&categories=rollup` query shapes, so the root cause is a brittle ETL stop condition on transient one-second Blobscan failures, not missing data.
- 2026-04-04: Repaired `src/etl/build_l1_rent_panel.py` to treat exact one-second Blobscan windows as recoverable transport instability instead of terminal source failure. Added dedicated terminal-window retry constants, and when page-1 Blobscan fetches fail at a one-second window with no captured rows, the ETL now waits and retries the same exact scope before declaring the source broken. Validation: `python -m py_compile src/etl/build_l1_rent_panel.py` passed; a direct module probe of the exact previously failing Base second (`2024-09-03T01:28:13Z..2024-09-03T01:28:14Z`) now returns `rows 0` through `fetch_blobscan_window(...)` instead of raising. Operational state remains `active`; the next step is to resume the same append-only checkpoint again with the unchanged stable flags and continue toward the receipts/base-fee phase.
