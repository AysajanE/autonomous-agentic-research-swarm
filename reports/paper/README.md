# `reports/paper/`

Quarto-backed manuscript sources and durable rendered outputs for the repository's current L2-to-L1 rent analysis project.

## Scope and ownership

This directory is the Stage 2 continuation-owned paper layer.

It sits downstream of the locked:

- protocol and contract surfaces
- runtime and review semantics
- Stage 1 continuation release/catalog layer

Do not use paper work to reopen:

- Stage 1 diagnosis
- Stage 2 architecture
- Stage 3 foundations
- Stage 4 runtime/gate semantics
- Stage 1 continuation release assembly, catalog, or release-status behavior

## Files

- `_quarto.yml` — Quarto project configuration for the paper surface
- `index.qmd` — manuscript source
- `references.bib` — repo-local bibliography entries used by the manuscript
- `build/index.html` — durable self-contained HTML render
- `build/README.md` — build namespace policy and expected durable outputs

## Render workflow

From the repository root, render the paper with:

`quarto render reports/paper/`

Expected durable output for v1:

- `reports/paper/build/index.html`

## Build namespace policy

Treat `reports/paper/build/` as a tracked durable artifact surface, not as scratch space.

For v1, keep the namespace clean:

- `README.md`
- `index.html`

The locked Stage 1 release layer hashes every non-README artifact under `reports/paper/build/`, so transient side files should not be committed there.

## Writing rules

- Keep scientific definitions aligned with `docs/protocol.md`, `contracts/data_dictionary.md`, and `contracts/decisions.md`.
- Do not introduce figures or tables that are not backed by durable artifacts under `reports/figures/` or `reports/tables/`.
- If the repository still lacks populated result artifacts, keep the manuscript honest: describe the protocol, provenance, runtime, and release surface without inventing unattested empirical estimates.
- Treat the manuscript as downstream of the repo's contracts-first workflow rather than as a place to redefine metric semantics.

## Release integration

The locked Stage 1 release layer already defines paper status semantics:

- `pending_stage2` when `reports/paper/build/` has no non-README artifacts
- `present` when the build namespace contains durable render artifacts

This packet materializes `reports/paper/build/index.html`, so follow-on release previews and release writes should report `paper.status = present` without changing any Stage 1 release semantics.