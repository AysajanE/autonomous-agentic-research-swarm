# H080 — Release candidate assembly handoff

Date: 2026-04-11
Task: T080

## Summary

- Final release surfaces were assembled on branch `T080_release_candidate` in worktree `/Users/aeziz-local/Research/wt-T080`.
- Canonical paper build outputs now live at:
  - `reports/paper/build/l2_l1_rent_working_paper.html`
  - `reports/paper/build/l2_l1_rent_working_paper.pdf`
  - `reports/paper/build/render_manifest.json`
- Release assembly outputs were written at:
  - `reports/status/releases/release_2026-04-11.json`
  - `reports/catalog.yaml`

## Operational Notes

- On this macOS host, Quarto 1.9.36 needs a writable temporary `HOME` because it writes cache state under `HOME/Library/Caches/quarto/...`.
- The successful render command was:
  - `env HOME=<tmp-home> quarto render reports/paper/`
- Rendering `reports/paper/index.qmd` directly is not sufficient for the locked release surface because it bypasses the project-level `output-dir: build` behavior for the HTML artifact. Rendering the `reports/paper/` project directory is the correct T080 command.
- `.gitignore` required repair before assembly because it still unignored the legacy `reports/paper/build/index.html` path while ignoring the canonical T080 HTML/PDF/render-manifest outputs.
- The repair switched the tracked release build surface to:
  - `reports/paper/build/l2_l1_rent_working_paper.html`
  - `reports/paper/build/l2_l1_rent_working_paper.pdf`
  - `reports/paper/build/render_manifest.json`
- Transient `reports/paper/index_files/` render spillover is now ignored and should not be committed.

## Validation

- `python scripts/release_assembly.py --release-date 2026-04-11 --write`
- `python scripts/release_assembly.py --release-date 2026-04-11 --check`
- `make gate`
