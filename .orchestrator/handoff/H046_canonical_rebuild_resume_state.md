# H046 — Canonical Rebuild Resume State

Date: 2026-04-08
Task: T046

## Current state

- T046 is still `active`.
- The `2026-04-08` canonical rerun is no longer blocked on missing Growthepie inputs.
- The seeded rerun has not finished yet, so the `2026-04-08` raw manifest and processed manifests are not written and `make gate` has not been rerun on final outputs.

## What changed in this worker pass

- Seeded `data/raw/l1_rent/2026-04-08/` from the preserved T035 raw cache under `/Users/aeziz-local/Research/.deleting-wt-T035-20260408/data/raw/l1_rent/2026-04-01/` using the repaired T035 raw manifest as the canonical file list.
- Patched `src/etl/build_l1_rent_panel.py` so seeded cache reuse works against the preserved raw snapshot:
  - legacy offset-naive cached timestamps are normalized to UTC
  - complete enclosing Blockscout cached windows can be reused for narrower continuation windows
  - continuation fetches preserve the stored cached page size instead of forcing a new page shape mid-window
  - uncached page-1 Blockscout windows can use the exact-window BigQuery backfill path directly
- Mirrored `~/.config/gcloud/` into `/tmp/codex-gcloud-auth/` so `bq` can run with an active authenticated config from a writable directory inside the sandbox.

## Reproduction / resume commands

- Seed the `2026-04-08` raw snapshot from the preserved T035 manifest-driven file list:
  - `jq -r '.files[].path' data/raw_manifest/l1_rent_2026-04-01.json | sed 's#^data/raw/l1_rent/2026-04-01/##' > /tmp/l1_rent_seed_files.txt`
  - `rsync -a --ignore-existing --files-from=/tmp/l1_rent_seed_files.txt /Users/aeziz-local/Research/.deleting-wt-T035-20260408/data/raw/l1_rent/2026-04-01/ data/raw/l1_rent/2026-04-08/`
- Mirror gcloud config into a writable temp dir:
  - `mkdir -p /tmp/codex-gcloud-auth`
  - `rsync -a ~/.config/gcloud/ /tmp/codex-gcloud-auth/`
- Resume the authoritative rerun:
  - `CLOUDSDK_CONFIG=/tmp/codex-gcloud-auth python src/etl/build_l1_rent_panel.py --run-date 2026-04-08 --blockscout-page-size 250`

## Most recent observed behavior

- The live rerun is successfully backfilling missing exact Blockscout windows via BigQuery, including uncached `2026-04-01..2026-04-07` tail windows for rollups such as `optimism`, `zksync_era`, and `linea`.
- Blobscan reuse is working for many cached post-Dencun months.
- The run is now spending time on bounded BigQuery/backfill work for repaired historical sender-hook addresses rather than refetching the entire history from live Blockscout.
