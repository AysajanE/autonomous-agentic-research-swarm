# `reports/paper/`

Quarto-backed manuscript sources for the repository's L2-to-L1 rent paper plus
the Operator-owned release build namespace that T080 materializes.

## Scope and ownership

`reports/paper/` is downstream of the locked protocol, contracts, and validated
release artifacts.

- T070 owns manuscript source only:
  - `reports/paper/_quarto.yml`
  - `reports/paper/index.qmd`
  - `reports/paper/references.bib`
- T080 owns the tracked release-candidate build namespace under
  `reports/paper/build/`.

Do not use paper work to reopen protocol definitions, registry scope, runtime
semantics, or release-catalog policy.

## Files

- `_quarto.yml` — Quarto project configuration for the release-candidate paper
- `index.qmd` — manuscript source tied to the validated release bundle
- `references.bib` — repo-local bibliography entries used by the manuscript
- `build/README.md` — build namespace policy and canonical T080 outputs

## Render workflow

From the repository root, render the paper with:

`quarto render reports/paper/`

The canonical T080 build outputs are:

- `reports/paper/build/l2_l1_rent_working_paper.html`
- `reports/paper/build/l2_l1_rent_working_paper.pdf`
- `reports/paper/build/render_manifest.json`

## Writing rules

- Keep scientific definitions aligned with `docs/protocol.md`,
  `contracts/data_dictionary.md`, and `contracts/decisions.md`.
- Do not introduce figures or tables that are not backed by durable artifacts
  under `reports/figures/` or `reports/tables/`.
- Treat the manuscript as downstream of the repo's contracts-first workflow
  rather than as a place to redefine metric semantics.
- Keep source/release boundaries explicit: T070 writes source, T080 writes
  rendered build artifacts and release surfaces.

## Release integration

`scripts/release_assembly.py` keeps the historical paper-status vocabulary:

- `pending_stage2` until all three canonical T080 build outputs exist
- `present` once the HTML, PDF, and `render_manifest.json` are all materialized

Legacy draft artifacts such as `reports/paper/build/index.html` are not part of
the canonical release-candidate contract and should not be used to claim
`paper.status = present`.
