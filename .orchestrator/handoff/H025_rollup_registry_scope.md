# H025 — Rollup Registry v1 Scope And Exclusions

Date: 2026-03-31
Task: T025

## Inclusion filter used

The populated `registry/rollup_registry_v1.csv` intentionally tracks the ETL-facing release universe needed by T030 and T035, not every chain visible on L2 dashboards.

Included rows had to satisfy all of:

- present in `https://api.growthepie.com/v1/master.json`
- `deployment == PROD`
- `supported_metrics` contains both `fees` and `rent_paid`
- `technology` contains `Rollup`
- `da_layer` starts with `Ethereum`
- `l2beat_link` points to a specific `https://l2beat.com/scaling/projects/...` page

## Included rollups

- `arbitrum`
- `starknet`
- `optimism`
- `zksync_era`
- `linea`
- `base`
- `scroll`
- `mode`
- `taiko`
- `worldchain`
- `lisk`
- `ink`
- `soneium`
- `unichain`

## Address-hook notes

- `batcher_addresses_json` comes from L2BEAT permission-role accounts when the project page exposes a Sequencer, Operator, or Validator account list.
- `scroll` is intentionally `[]`: the L2BEAT page exposes tracked submission contracts and governance actors, but not a distinct current batcher/operator account list.
- `mode`, `worldchain`, and `lisk` also expose proposer accounts on L2BEAT; those were not copied because this field is batcher-only.

## Held-out candidates

- `loopring`: growthepie currently lacks a `fees` metric, so T030 cannot supply the primary denominator.
- `ronin`: growthepie marks it `deployment=DEV` and the linked L2BEAT URL is nonspecific; the legacy 2021 launch date also conflicts with a clean rollup start-date interpretation.
- `zircuit`: growthepie uses a noncanonical `deployment=ZIRCUIT` value instead of `PROD`; held out rather than guessing whether it should be treated as production scope.
- `mantle`, `manta`, `fraxtal`, `celo`, `metis`, `gravity`, `arbitrum_nova`: excluded by protocol because DA is not Ethereum L1 mainnet.
- `blast`, `zora`, and similar L2BEAT-visible projects absent from growthepie chain coverage were not added to this ETL-facing v1 universe.

## Reproduction

- `curl -s https://api.growthepie.com/v1/master.json`
- `curl -s https://l2beat.com/scaling/projects/<slug>` and parse `window.__SSR_DATA__`

If T030 or T035 need any held-out chain added later, treat it as a registry change request rather than widening the universe ad hoc in ETL code.
