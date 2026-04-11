# H070 — Quarto paper source handoff for T080 (reverified)

Date: 2026-04-11
Task: T070
Supersedes: `.orchestrator/handoff/H070_paper_source_handoff_2026-04-11.md` for runtime-provenance guidance

## Summary

- Verified that the existing paper source outputs remain the correct T070 deliverables:
  - `reports/paper/_quarto.yml`
  - `reports/paper/index.qmd`
  - `reports/paper/references.bib`
- The manuscript remains aligned to the locked L2-to-L1 rent release candidate and cites the validated `2026-04-09` surface.
- The paper source still embeds:
  - `../figures/str_ecosystem_timeseries.svg`
  - `../figures/str_post_dencun_regimes.svg`
  - `../tables/str_regime_summary.md`
- `_quarto.yml` still targets the locked downstream basenames:
  - `build/l2_l1_rent_working_paper.html`
  - `build/l2_l1_rent_working_paper.pdf`

## Verification

- Durable local-swarm run manifest exists:
  - `reports/status/swarm_runs/T070_20260411T150935Z.json`
- Re-verification commands run in this repair pass:
  - `make gate`
  - `tmpdir="$(mktemp -d /tmp/t070_quarto_check.XXXXXX)"`
  - `homedir="$(mktemp -d /tmp/t070_home.XXXXXX)"`
  - `cp -R reports "$tmpdir/"`
  - `env HOME="$homedir" quarto render "$tmpdir/reports/paper/index.qmd" --to html`
- Re-verification outcomes:
  - `make gate` passed.
  - Draft render succeeded in a `/tmp` mirror and produced `/tmp/t070_quarto_check.UwoqfM/reports/paper/build/l2_l1_rent_working_paper.html`.

## Operator Caveats

- Final in-repo paper build artifacts under `reports/paper/build/` remain T080/Operator-owned; this worker task intentionally verified the draft render in `/tmp` instead of writing to the release surface.
- On this macOS machine, Quarto 1.9.36 uses `HOME/Library/Caches/quarto/...` for its Sass cache. Rendering with the default repo user home can fail under sandboxed permissions with `ERROR: unable to open database file`.
- For T080, prefer a writable home/cache path inside the runtime sandbox, for example:
  - `env HOME=/tmp/t080_home quarto render reports/paper/index.qmd`
- `reports/paper/README.md` still references the older `build/index.html` surface. This task could not edit that file, so T080 should continue to follow the higher-precedence task and contract surfaces that lock `l2_l1_rent_working_paper.html` and `.pdf`.
