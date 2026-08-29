# The Master Dataset Registry (Source Ledger)

This is the definitive list of data sources the Data-Forge is configured to ingest. This satisfies the EU AI Act / data provenance requirements.

| Dataset Name | Source / Repo ID | Domain | License Status | Pipeline Usage |
| :--- | :--- | :--- | :--- | :--- |
| **PD12M** | `Spawning/PD12M` (HuggingFace) | General Visual | CC0 / Public Domain (CDLA-Permissive-2.0) | Primary backbone for continuous renderer (Z-Image/Qwen-Image-Edit) aesthetics and composition. Metadata-only repo (parquet + image-URL column) — fetched via the `download_mode: "url_list"` path in `fetcher.py`, sampled to 200K rows (see `configs/datasets.yaml`; the PRD's actual training target is 100K-500K curated samples total across all sources, not the full 12.4M superset). |
| **CC12M** | `google-research-datasets/conceptual_12m` (HuggingFace) | General Visual | Unverified / Scraped | Supplementary diversity. Strictly filtered by Safety & Aesthetic gates. Same metadata-only ingestion path as PD12M, plus a headerless-TSV fallback (`caption`, `url` columns) for its canonical raw distribution format; sampled to 150K rows. |
| **RICO** | `interactionmining.org/rico` | Mobile UI | Academic / Unverified | Base UI layout distribution. Routed to `excluded_pending_review` by License Agent unless verified. |
| **CLAY** | `google-research-datasets/clay` (GitHub) | Mobile UI | Academic / Archived | Denoised RICO. Preferred over raw RICO for structural JSON training. |
| **Enrico** | `luileito/enrico` (GitHub) | Mobile UI | Academic | High-quality UI subset. Used primarily for **Held-Out Eval** and Verifier calibration. |
| **WebUI** | `biglab/webui-350k` (HuggingFace) | Web/Desktop UI | Per source COPYRIGHT.txt | Extends UI coverage beyond mobile to complex web dashboards and landing pages. |
| **Screen2Words** | `bevaya/RICO-Screen2Words` (HuggingFace) | UI Captions | CC-BY-4.0 | Dense caption training pairs for the Planner and Recaption Agent. |
| **UICrit** | `google-research-datasets/uicrit` (GitHub) | UI Critique | CC BY-ND | **Critical:** Seeds the DPO preference-pair training signal (`ui_critique/`) and calibrates the Tier-1 Audit Agent + Gemma 4 Critic Tier rubrics. NoDerivs license — flag for manual legal review. |
| **Synthetic Self-Play** | *Generated Locally by Krisna* | Agentic Edits | Internal | Iterative design trajectories (Bad Design -> Critique -> Fix) generated post-M1. |
| **Commissioned / rights-cleared gap-filling** | *Not yet sourced* | Domain gaps neither PD12M/CC12M nor the UI datasets cover (PRD §8.1) | N/A | **Intentionally out of scope for data-forge.** This is a manual sourcing/licensing process (commissioning or licensing specific imagery), not an automatable fetch — there is deliberately no `datasets.yaml` entry or fetch path for it. Noted here explicitly so its absence reads as a documented scope boundary, not an overlooked source. |

## Domain Tagging

Records are tagged `ui_first` or `general_design` (`data_forge/data/domain_tagger.py`) via exact `source_dataset` match against the set above, falling back to a structural heuristic (≥2 overlapping UI element types from Stage 6's extraction) only for unrecognized sources. `rico_core`, `rico_semantic`, `clay`, `enrico`, `webui`, `screen2words`, and `uicrit` are all explicitly `ui_first` sources — all seven must stay in that source list, or records from any missing one silently fall through to the structure-based heuristic instead of being reliably tagged, undermining the `ui_first_ratio` enforcement in `s07_routing.py` before it even runs.

## Known Limitation: `ui_first_ratio` Is Enforced Per-Chunk, Not Corpus-Wide

`ShardRouter.route()` (`data_forge/data/shard_router.py`) enforces the configured `ui_first_ratio` (default 0.70) against whichever chunk of records is passed into a single call — it correctly caps the majority domain down and accepts a smaller total routed set when the minority domain can't supply enough to hit the ratio exactly (verified by `tests/test_shard_router.py::test_overflow_excluded`). What it does *not* do is enforce the ratio cumulatively across chunks against a fixed, corpus-wide training-pool size cap — each chunk is ratio-balanced independently. If different chunks have meaningfully different `ui_first`:`general_design` compositions, the corpus-wide realized ratio can drift from the configured target even though every individual chunk is internally correct. Addressing this would require a persistent, cross-chunk counter (e.g. tracked in the manifest) rather than the current fresh-per-call `total`; left as an open item for a future revision rather than expanded in scope here.
