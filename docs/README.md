# Documentation Index

- **`architecture/`** — cross-cutting design docs:
  - `RESEARCH_AND_CITATIONS.md` — **start here for "why."** Every
    non-obvious decision in this repo (dataset licenses/schemas, model
    stack corrections, low-VRAM mechanism choices), what was actually
    verified vs. assumed, against what real source, and what changed as
    a result. Includes a citation index for quick lookup.
  - `DIRECTORY_LAYOUT.md` — why this repo is structured the way it is,
    the old→new package mapping, the one real cross-package dependency.
  - `SYNC_DESIGN.md` — the verified data-forge ↔ training sync contract
    (directory names, field names, join keys), checked against real
    running code on both sides, not just cross-referenced as text.
- **`data-forge/`** — data pipeline docs (source registry, architecture,
  completeness audit trail).
- **`training/`** — points to `../training/README.md` (the package's own
  README is the primary doc; nothing duplicated here).
- **`inference/`** — points to `../inference/README.md` likewise.
- **`review/`** — an independent, phase-by-phase design-sync audit
  (data↔training↔inference consistency, hyperparameters vs. published
  precedent, real vs. synthetic data tracing, VRAM/orchestrator
  correctness) with every fix applied during the review verified
  against the repo's real test suite where one exists. Start at
  `review/README.md` for the index; `review/07_consolidated_citations.md`
  is the paper-ready bibliography plus the full open-action rollup.

Each package's own `README.md` is the source of truth for that package —
this `docs/` tree holds cross-cutting design docs that don't belong to
any one package, plus data-forge's docs (which were migrated here
verbatim from its own former `docs/` directory, unchanged, since
data-forge itself wasn't split).
