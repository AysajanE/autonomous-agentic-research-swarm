# `reports/paper/build/`

Tracked durable paper render artifacts live here.

## Expected contents for v1

- `index.html` — self-contained HTML render of `reports/paper/index.qmd`

## Policy

- Regenerate the build surface with `quarto render reports/paper/`.
- Commit only durable paper artifacts required by the release surface.
- Keep the namespace clean: for v1, only `README.md` plus `index.html` should be present.
- Do not commit transient asset directories. The Quarto config embeds resources so the locked Stage 1 release layer hashes a single HTML artifact.
- If `index.qmd`, `references.bib`, or `_quarto.yml` changes, refresh `index.html` before running paper or release checks.