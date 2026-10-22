# The Master Dataset Registry (Source Ledger)

> [!IMPORTANT]
> **Canonical Literature Survey**: See [LITERATURE_SURVEY.md](LITERATURE_SURVEY.md) for the formal theoretical analysis, methodology evaluation, and academic credits for all prior research underpinning Data-Forge.
>
> This document is the source-audit ledger distinguishing active training sources from quarantined catalog entries. A source is not training-approved merely because it appears in this document; the runtime registry and immutable receipt gates (`audit_registry.py`) are authoritative.

## Current Audit State & Calibrated 356,721-Record Production Portfolio

The production training and evaluation portfolio is bounded at **356,721 records (~357K records / ~480 GB)**, strictly derived from the **13 active sources** defined in `datasets.yaml` and verified by `audit_registry.py`. This preserves **~520 GB** unallocated safety buffer below the $1.0\text{ TB}$ workstation NVMe ceiling, guaranteeing deterministic pipeline execution in **~14–15 hours** on a single NVIDIA A6000 (48GB VRAM) and enforcing the 18h intake ceiling. 

> [!NOTE]
> **Photographic Prior Preservation Verdict**: General stock photography downloads (e.g. PD12M, CC12M, LAION) are **quarantined and eliminated**. As codified in `training/configs/polish_qwen_image21_lora.yaml` and enforced by `qwen_trainer_contract.py`, the base model weights remain completely frozen (`base_weights_frozen: true`). Consequently, base photorealism, skin tones, and natural lighting manifolds are permanently retained in $W_0$ without requiring DreamBooth-style prior preservation regularizers (`with_prior_preservation: false`) or burning bandwidth and disk space on unconditioned stock photos.

Run:
```powershell
python scripts/data-forge/audit_registry.py
```
before every fetch. The command is read-only and reports enabled/quarantined sources, effective counts, policy violations, and environment-independent source blockers.

---

## Active Certified Training & Evaluation Registry (13 Sources)

| Domain & Pillar | Dataset Name | Source / Repo ID | License Status | Role & Configured Allocation (`sample_size`) |
| :--- | :--- | :--- | :--- | :--- |
| **Pillar 1: Mobile UI** | **RICO Semantic** | `rico_semantic` / Android | Verified Open | $66,000$ pairs (sample cap). Android semantic screen hierarchies, bounding boxes, and component categories ($18.5\%$). |
| **Pillar 2: Editorial & Web** | **DocLayNet** | `docling-project/DocLayNet` | CDLA-Permissive-1.0 | $80,000$ pairs (sample cap). Enterprise multi-column documents, annual reports, manuals, and publication spreads. |
| **Pillar 2: Editorial & Web** | **Figma2Code** | `xcodemind/Figma2Code` | MIT | $2,817$ pairs. Clean production web UI screenshots paired with design tokens, buttons, and layout metadata. Total Pillar 2: $82,817$ pairs ($23.2\%$). |
| **Pillar 3: Graphic Design** | **Crello** | `cyberagent/crello` | CDLA-Permissive-2.0 | $25,000$ pairs (sample cap). Commercial marketing banners, social templates, and structured layer geometries. |
| **Pillar 3: Graphic Design** | **PKU-PosterLayout** | `creative-graphic-design/PKU-PosterLayout`| CC-BY-SA-4.0 | $10,000$ pairs (sample cap). Commercial presentation and marketing poster layout pairs with visual anchors. |
| **Pillar 3: Graphic Design** | **ABO E-Commerce** | `wardana/amazon-berkeley-objects`| CC-BY-NC-4.0 | $35,000$ pairs (sample cap). Commercial e-commerce product cards, metadata, and container layouts. Total Pillar 3: $70,000$ pairs ($19.6\%$). |
| **Pillar 4: Instruction & DPO** | **MagicBrush** | `osunlp/MagicBrush` | CC-BY-4.0 | $10,500$ pairs. Ground-truth human-annotated image editing triplets $(I_{\text{source}}, \text{instruction}, I_{\text{target}})$ with spatial masks. |
| **Pillar 4: Instruction & DPO** | **Pick-a-Pic v2** | `pickapic_v2` (DPO stream) | CDLA-Permissive-2.0 | $50,000$ pairs (sample cap). Pairwise human aesthetic preference comparisons for flow alignment. |
| **Pillar 4: Instruction & DPO** | **HPDv2** | `hpdv2` (DPO stream) | Apache-2.0 | $50,000$ pairs (sample cap). Human preference aesthetic comparisons. Total Pillar 4: $110,500$ pairs ($31.0\%$). |
| **Evaluation Suite** | **TASTE Benchmark** | `taste` (Eval track) | Evaluation only | $14,500$ screens. Held-out multi-axis expert design rating benchmark (isolated from training). |
| **Evaluation Suite** | **UI-Vision** | `ui_vision` (Eval track) | MIT | $10,000$ screens. Desktop application visual grounding and layout benchmark. |
| **Evaluation Suite** | **PartiPrompts** | `partiprompts` (Eval track) | Permissive | $1,632$ prompts. Fine-grained typography and challenging compositional prompts. |
| **Evaluation Suite** | **ScreenSpot** | `screenspot` (Eval track) | Apache-2.0 | $1,272$ screens. Mobile/web GUI visual element grounding and bounding box benchmark. Total Eval: $27,404$ records ($7.7\%$). |
| **TOTAL ACTIVE PORTFOLIO** | **All 13 Active Sources** | **`datasets.yaml` Active Set** | **Verified Provenance** | **$356,721$ records ($329,317$ training $+ 27,404$ eval / ~480 GB)** |

---

## Quarantined & Retired Sources (Negative Audit Log)

| Dataset | Previous Status | Quarantine Reason |
| :--- | :--- | :--- |
| **PubLayNet** | Candidate | **Quarantined**: Severe monochrome LaTeX academic bias. Induces drab aesthetic collapse in generative diffusion backbones. |
| **DocBank** | Candidate | **Quarantined**: Monochrome arXiv paper distribution. Irrelevant to visual UI and modern graphic design. |
| **Raw RICO** | Candidate | **Quarantined**: $65\%$ intra-corpus duplicate screens, broken bounding boxes, and obsolete 2017 Android Marshmallow styling. Fully replaced by CLAY. |
| **WebSight & AtomBlock** | Candidate | **Quarantined**: Synthetic LLM-generated HTML/CSS with severe flex-wrap bugs, overlapping divs, and zero human aesthetic ground truth. |
| **PD12M & CC12M** | Candidate | **Quarantined**: Unfiltered general web crawl. Succeeded by curated LAION $\ge 6.25$ and Open Images v7 with explicit prior preservation conditioning. |

### Vetted but not yet integrated

| Candidate | Repo | Why not wired in yet |
|---|---|---|
| **GameLabel-10K** | `Jonathan-Zhou/GameLabel-10k` (confirmed real, Apache 2.0, arXiv:2409.19830) | **INTEGRATED.** Real, general-domain (not UI-specific) image-preference pairs (~10K labels, 7K prompts) collected via mobile-game crowdsourcing, already demonstrated improving Flux-Schnell via DPO/LoRA fine-tuning. Column schema now confirmed directly against the live dataset (HF's own dataset-viewer output, not guessed): `prompt`, `img0_votes`/`img1_votes` (int 0-5, a vote COUNT from up to 5 players — not a fixed 0/1 label), `img0_encoding`/`img1_encoding` (base64 JPEG wrapped in a stray Python `b'...'` bytes-repr string, not the standard HF `{"bytes": ...}` image-feature dict). Neither `preference_pair` nor `hpdv2_ranked_list` fit this shape, so a dedicated `download_mode: "gamelabel_csv"` adapter was added (`fetcher.py::_fetch_gamelabel_csv`/`_decode_gamelabel_image`), reading via the `refs/convert/parquet` revision Hugging Face auto-generates for this repo rather than streaming the raw 2.26GB `data.csv` directly, deriving a win/lose label from vote counts (dropping ties, including 0-0), and preserving the vote margin per pair for potential downstream weighting. See `datasets.yaml`'s `gamelabel_10k` entry and `tests/data_forge/test_fetch_gamelabel.py` (tie-dropping, malformed-row handling, and label/margin correctness all verified against real pandas/PIL, not just asserted). Supplements Stage-1 (general) DPO, does not close the Stage-2 (UI-domain) gap DesignSense-10k/DesignPref are meant for.
| **Synthetic Self-Play** | *Generated Locally by Krisna* | Agentic Edits | Internal | Iterative design trajectories (Bad Design -> Critique -> Fix) generated post-M1. |
| **Commissioned / rights-cleared gap-filling** | *Not yet sourced* | Domain gaps neither PD12M/CC12M nor the UI datasets cover (PRD §8.1) | N/A | **Intentionally out of scope for data-forge.** This is a manual sourcing/licensing process (commissioning or licensing specific imagery), not an automatable fetch — there is deliberately no `datasets.yaml` entry or fetch path for it. Noted here explicitly so its absence reads as a documented scope boundary, not an overlooked source. |

## Domain Tagging

Records are tagged `ui_first` or `general_design` (`data-forge/src/data_forge/data/domain_tagger.py`) via exact `source_dataset` match against the set above, falling back to a structural heuristic (≥2 overlapping UI element types from Stage 6's extraction) only for unrecognized sources. `rico_core`, `rico_semantic`, `clay`, `enrico`, `webui`, `screen2words`, `screenspot`, `seeclick`, `android_control`, `mind2web`, and `uicrit` are all explicitly `ui_first` sources — all eleven must stay in that source list, or records from any missing one silently fall through to the structure-based heuristic instead of being reliably tagged, undermining the `ui_first_ratio` enforcement in `s07_routing.py` before it even runs.

## Known Limitation: `ui_first_ratio` Is Enforced Per-Chunk, Not Corpus-Wide

`ShardRouter.route()` (`data-forge/src/data_forge/data/shard_router.py`) enforces the configured `ui_first_ratio` (default 0.70) against whichever chunk of records is passed into a single call — it correctly caps the majority domain down and accepts a smaller total routed set when the minority domain can't supply enough to hit the ratio exactly (verified by `tests/test_shard_router.py::test_overflow_excluded`). What it does *not* do is enforce the ratio cumulatively across chunks against a fixed, corpus-wide training-pool size cap — each chunk is ratio-balanced independently. If different chunks have meaningfully different `ui_first`:`general_design` compositions, the corpus-wide realized ratio can drift from the configured target even though every individual chunk is internally correct. Addressing this would require a persistent, cross-chunk counter (e.g. tracked in the manifest) rather than the current fresh-per-call `total`; left as an open item for a future revision rather than expanded in scope here.
