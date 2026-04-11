# `reports/paper/build/`

Tracked durable paper render artifacts live here.

## Expected contents for the release candidate

- `l2_l1_rent_working_paper.html` — canonical self-contained HTML render
- `l2_l1_rent_working_paper.pdf` — canonical PDF render
- `render_manifest.json` — render provenance for the T080 build

## Policy

- Regenerate the build surface with `quarto render reports/paper/`.
- Commit only durable paper artifacts required by the release surface.
- Keep the namespace clean: before T080 this directory may contain only
  `README.md`; after T080 it should contain `README.md` plus the three
  canonical build outputs above.
- Do not commit transient asset directories. The Quarto config embeds resources
  into the HTML render.
- `scripts/release_assembly.py` reports `paper.status = pending_stage2` until
  all three canonical build outputs exist together.
- Legacy draft artifacts such as `index.html` are not canonical release
  artifacts and should not be committed back into this namespace.
