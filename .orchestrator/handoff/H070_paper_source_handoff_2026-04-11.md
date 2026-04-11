# H070 — Quarto paper source handoff for T080

Date: 2026-04-11
Task: T070

## Summary

- Updated paper source files only:
  - `reports/paper/_quarto.yml`
  - `reports/paper/index.qmd`
  - `reports/paper/references.bib`
- The manuscript is now aligned to the locked L2-to-L1 rent release candidate rather than the older methods placeholder.
- It cites the validated `2026-04-09` surface, embeds:
  - `../figures/str_ecosystem_timeseries.svg`
  - `../figures/str_post_dencun_regimes.svg`
  - `../tables/str_regime_summary.md`
- `_quarto.yml` now targets the locked downstream basenames:
  - `build/l2_l1_rent_working_paper.html`
  - `build/l2_l1_rent_working_paper.pdf`

## Verification run

- `make gate` passed in the repo worktree.
- Draft render was verified in a `/tmp` mirror rather than in `reports/paper/build/`, because `reports/paper/build/` is Operator-owned and this worker run was not allowed to leave release-surface artifacts behind.
- Verified render command:
  - `env HOME=/tmp/t070_home quarto render /tmp/t070_quarto_check/reports/paper/index.qmd --to html`
- Verified output:
  - `/tmp/t070_quarto_check/reports/paper/build/l2_l1_rent_working_paper.html`

## Operator caveats

- On this macOS machine, Quarto 1.9.36 uses `HOME/Library/Caches/quarto/...` for its Sass cache. Under the current sandbox, rendering with the default `HOME=/Users/aeziz-local` failed with:
  - `ERROR: unable to open database file`
- Rendering succeeded once `HOME` was redirected to a writable `/tmp` location.
- For T080, prefer:
  - `env HOME=/tmp/t080_home quarto render reports/paper/index.qmd`
  - or another writable HOME/cache path inside the runtime sandbox
- `reports/paper/README.md` still references the older `build/index.html` surface. This worker task could not update that file, but T080 should follow the higher-precedence task/contract surfaces that lock `l2_l1_rent_working_paper.html` and `.pdf`.

## Remaining boundary

- This run was not executed through the local swarm runtime, so there is no durable `reports/status/swarm_runs/T070_*.json` manifest yet.
- T080 or an Operator/local-swarm rerun should record the durable run manifest before review.
